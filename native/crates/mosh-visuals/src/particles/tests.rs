//! Offscreen GPU tests for the particle layer. No window is opened: every frame renders into a
//! `LayerTarget` and is read back.
//!
//! The tests the brief asks for are `lit_after_a_few_frames`,
//! `kick_raises_brightness_and_spread`, `silence_settles`, `stable_under_noise` (1 000 frames, plus
//! the `#[ignore]`d 5 000), `count_follows_p0` (and its CPU twin `count_scales_with_p0`),
//! `burst_brightness_is_limited_to_one_per_400ms` and, in every test here, zero validation
//! messages. The GPU budget is measured by `tests/particles_budget.rs::gpu_ms_per_draw`, which is a
//! separate binary so it has the GPU to itself, and `style_previews` writes the PNGs under
//! `native/target/style-previews/particles/`.

use super::preview;
use super::*;

use mosh_core::{EventFlags, TransportSource};
use mosh_render::{FrameRecorder, Gpu, GpuPreference, LayerTarget, Pipeline};
use std::path::PathBuf;

/// A small internal size for the functional tests (a layer of a 640x360 output at scale 0.5).
const TEST_WIDTH: u32 = 320;
const TEST_HEIGHT: u32 = 180;

/// The size the brief budgets at.
const BUDGET_WIDTH: u32 = 960;
const BUDGET_HEIGHT: u32 = 540;

fn gpu() -> Gpu {
    let gpu = Gpu::new(GpuPreference::Integrated)
        .or_else(|_| Gpu::new(GpuPreference::ByName(String::new())))
        .expect("a Vulkan device");
    println!(
        "adapter: {} | validation layers: {} | timestamp period: {} ns",
        gpu.adapter_name(),
        if gpu.validation_enabled() {
            "on"
        } else {
            "unavailable"
        },
        gpu.timestamp_period_ns()
    );
    gpu
}

fn assert_no_validation(gpu: &Gpu) {
    let messages = gpu.validation_messages();
    println!(
        "validation: {} message(s), layer {}",
        messages.len(),
        if gpu.validation_enabled() {
            "enabled"
        } else {
            "unavailable"
        }
    );
    assert!(
        messages.is_empty(),
        "{} validation message(s):\n{}",
        messages.len(),
        messages.join("\n")
    );
}

fn frame(level: u8, bands: [f32; 3], onset: f32) -> AnalysisFrame {
    AnalysisFrame {
        time_s: 1.0,
        ppq: 4.25,
        beat_phase: 0.25,
        bar_phase: 0.4,
        beat_index: 7,
        bar_index: 1,
        tempo_bpm: 128.0,
        playing: true,
        source: TransportSource::Plugin,
        bands,
        mel: [0.0; 24],
        onset,
        loudness_db: -12.0,
        loudness: 0.7,
        brightness: 0.5,
        intensity: 0.6,
        level,
        events: EventFlags::empty(),
    }
}

fn quiet() -> AnalysisFrame {
    let mut f = frame(1, [0.0, 0.0, 0.0], 0.0);
    f.playing = false;
    f
}

fn loud() -> AnalysisFrame {
    frame(4, [0.9, 0.8, 0.7], 0.0)
}

/// An onset with the low band up: a kick.
fn kick() -> AnalysisFrame {
    frame(3, [0.9, 0.5, 0.4], 0.9)
}

fn render(
    gpu: &Gpu,
    particles: &mut Particles,
    layer: usize,
    dst: &LayerTarget,
    frame: &AnalysisFrame,
    params: &ClipParams,
    t: f32,
) {
    let mut recorder = FrameRecorder::new(gpu);
    particles
        .draw(&mut recorder, layer, dst, frame, params, t)
        .expect("particle draw");
    gpu.wait_frame(recorder.submit());
}

/// What a read-back says about the picture.
#[derive(Default)]
struct Light {
    /// Sum of the RGB channels, 0..765 per pixel: a monotonic "how much light".
    total: u64,
    /// Pixels above a visible threshold.
    lit: usize,
    /// Brightness-weighted mean distance from the centre, in screen heights.
    mean_r: f32,
    /// The share of the light more than 0.3 screen heights from the centre: how far out the
    /// field reaches, which is what a kick's shockwave changes.
    outer: f32,
}

fn light_stats(pixels: &[u8], width: u32, height: u32) -> Light {
    let aspect = f64::from(width) / f64::from(height);
    let mut total = 0u64;
    let mut lit = 0usize;
    let mut weight = 0.0f64;
    let mut radius = 0.0f64;
    let mut outer = 0.0f64;
    for (index, pixel) in pixels.chunks(4).enumerate() {
        let luma = u64::from(pixel[0]) + u64::from(pixel[1]) + u64::from(pixel[2]);
        total += luma;
        if luma > 24 {
            lit += 1;
        }
        let x = (index % width as usize) as f64;
        let y = (index / width as usize) as f64;
        let dx = (x / f64::from(width) - 0.5) * aspect;
        let dy = y / f64::from(height) - 0.5;
        let r = (dx * dx + dy * dy).sqrt();
        let w = luma as f64;
        weight += w;
        radius += w * r;
        if r > 0.3 {
            outer += w;
        }
    }
    Light {
        total,
        lit,
        mean_r: if weight > 0.0 {
            (radius / weight) as f32
        } else {
            0.0
        },
        outer: if weight > 0.0 {
            (outer / weight) as f32
        } else {
            0.0
        },
    }
}

fn picture(gpu: &Gpu, width: u32, height: u32, dst: &LayerTarget) -> Light {
    light_stats(&gpu.read_back(dst), width, height)
}

/// Mean absolute per-channel difference, 0..255: how much moved between two read-backs.
fn mean_difference(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len(), "read-backs of different sizes");
    let total: f32 = a
        .chunks(4)
        .zip(b.chunks(4))
        .map(|(left, right)| {
            (0..3)
                .map(|channel| (f32::from(left[channel]) - f32::from(right[channel])).abs())
                .sum::<f32>()
        })
        .sum();
    total / (a.len() / 4 * 3) as f32
}

/// The uniform row `index` of a frame's block, for the CPU-only tests.
fn row(block: &[u8], index: usize) -> [f32; 4] {
    let mut values = [0.0f32; 4];
    for (slot, value) in values.iter_mut().enumerate() {
        let at = index * 16 + slot * 4;
        *value = f32::from_le_bytes(block[at..at + 4].try_into().expect("four bytes"));
    }
    values
}

#[test]
fn the_uniform_block_matches_the_prelude() {
    assert_eq!(UNIFORM_BYTES, 192);
    let rows = PRELUDE.matches(": vec4f,").count();
    assert_eq!(rows, UNIFORM_ROWS, "the prelude's uniform rows");
    println!("uniform block: {UNIFORM_ROWS} rows, {UNIFORM_BYTES} bytes");
}

#[test]
fn count_scales_with_p0_and_intensity() {
    let count = |p0: f32, intensity: f32| {
        let mut params = ClipParams::default();
        params.p[0] = p0;
        let mut f = loud();
        f.intensity = intensity;
        let block = Plan::build(&f, &params, 0.0, DEFAULT_DT, false, false, (960, 540));
        row(&block, 1)[2] as u32
    };
    let (low, mid, high) = (count(0.0, 0.5), count(0.5, 0.5), count(1.0, 0.5));
    assert!(low > 0, "the default count must not be zero");
    assert!(mid > low && high > mid, "{low} -> {mid} -> {high}");
    assert_eq!(count(1.0, 1.0), MAX_PARTICLES, "p0 = 1 fills the state");
    let quiet_half = count(0.5, 0.0);
    assert!(
        quiet_half < mid,
        "intensity scales the count: {quiet_half} at 0, {mid} at 0.5"
    );
    println!("count: p0 0 -> {low}, 0.5 -> {mid}, 1 -> {high}; intensity 0 -> {quiet_half}");
}

#[test]
fn the_step_length_is_clamped() {
    assert_eq!(step_dt(None, 1.0), DEFAULT_DT);
    assert_eq!(step_dt(Some(1.0), 1.0), DEFAULT_DT, "no delta at all");
    assert_eq!(
        step_dt(Some(1.0), 0.5),
        DEFAULT_DT,
        "a clock that stepped back"
    );
    assert_eq!(step_dt(Some(1.0), f32::NAN), DEFAULT_DT);
    assert_eq!(step_dt(Some(1.0), 9.0), DT_MAX, "a stall does not teleport");
    assert_eq!(step_dt(Some(1.0), 1.000_001), DT_MIN);
    let sixty = step_dt(Some(1.0), 1.0 + 1.0 / 60.0);
    assert!((sixty - 1.0 / 60.0).abs() < 1e-5, "60 fps: {sixty}");
}

#[test]
fn burst_brightness_is_limited_to_one_per_400ms() {
    let floor = mosh_render::FLASH_INTERVAL.as_secs_f32();
    assert!((floor - 0.4).abs() < 1e-6, "the renderer's floor is 400 ms");
    let mut gate = FlashGate::default();
    let mut allowed = Vec::new();
    // A 16th-note kick pattern at 180 BPM: a kick every 83 ms for two seconds.
    let mut t = 0.0f32;
    while t < 2.0 {
        if gate.allow(t) {
            allowed.push(t);
        }
        t += 1.0 / 12.0;
    }
    assert!(allowed.len() >= 2, "{allowed:?}");
    for pair in allowed.windows(2) {
        assert!(
            pair[1] - pair[0] >= floor - 1e-4,
            "bursts {:.3}s apart: {allowed:?}",
            pair[1] - pair[0]
        );
    }
    println!(
        "24 kicks in 2 s: {} brightness bursts at {:?}",
        allowed.len(),
        allowed
    );

    // A clock that steps backwards must not unlock a burst.
    let mut gate = FlashGate::default();
    assert!(gate.allow(10.0));
    assert!(!gate.allow(2.0), "backwards");
    assert!(!gate.allow(10.3), "300 ms later");
    assert!(gate.allow(10.5), "500 ms later");
}

#[test]
fn lit_after_a_few_frames() {
    let gpu = gpu();
    let mut particles = Particles::new(&gpu, TEST_WIDTH, TEST_HEIGHT).expect("particles");
    let dst = LayerTarget::new_single(&gpu, TEST_WIDTH, TEST_HEIGHT);
    let params = ClipParams::default();
    for step in 0..6 {
        render(
            &gpu,
            &mut particles,
            0,
            &dst,
            &loud(),
            &params,
            step as f32 / 60.0,
        );
    }
    let light = picture(&gpu, TEST_WIDTH, TEST_HEIGHT, &dst);
    assert!(light.lit > 200, "the layer must be lit: {}", light.lit);
    assert!(light.total > 0, "and not black");
    println!(
        "6 loud frames at {TEST_WIDTH}x{TEST_HEIGHT}: {} lit pixels, total {}",
        light.lit, light.total
    );
    assert_no_validation(&gpu);
}

#[test]
fn kick_raises_brightness_and_spread() {
    let gpu = gpu();
    let mut particles = Particles::new(&gpu, TEST_WIDTH, TEST_HEIGHT).expect("particles");
    let dst = LayerTarget::new_single(&gpu, TEST_WIDTH, TEST_HEIGHT);
    let params = ClipParams::default();

    // Settle in silence, so the kick is the only thing that moves the field.
    for step in 0..40 {
        render(
            &gpu,
            &mut particles,
            0,
            &dst,
            &quiet(),
            &params,
            step as f32 / 60.0,
        );
    }
    let before = picture(&gpu, TEST_WIDTH, TEST_HEIGHT, &dst);

    render(&gpu, &mut particles, 0, &dst, &kick(), &params, 40.0 / 60.0);
    let burst = picture(&gpu, TEST_WIDTH, TEST_HEIGHT, &dst);
    assert!(
        burst.total > before.total,
        "a kick must raise the light: {} -> {}",
        before.total,
        burst.total
    );

    // Six quiet frames: the shockwave has moved the field but the burst flash has gone.
    for step in 41..47 {
        render(
            &gpu,
            &mut particles,
            0,
            &dst,
            &quiet(),
            &params,
            step as f32 / 60.0,
        );
    }
    let after = picture(&gpu, TEST_WIDTH, TEST_HEIGHT, &dst);
    assert!(
        after.outer > before.outer * 1.15 || after.mean_r > before.mean_r * 1.05,
        "a kick must spread the field: radius {:.4} -> {:.4}, outer share {:.3} -> {:.3}",
        before.mean_r,
        after.mean_r,
        before.outer,
        after.outer
    );
    println!(
        "kick: light {} -> {} (+{:.0}%), radius {:.4} -> {:.4}, outer share {:.3} -> {:.3}",
        before.total,
        burst.total,
        100.0 * (burst.total as f64 / before.total.max(1) as f64 - 1.0),
        before.mean_r,
        after.mean_r,
        before.outer,
        after.outer
    );
    assert_no_validation(&gpu);
}

#[test]
fn silence_settles() {
    let gpu = gpu();
    let mut particles = Particles::new(&gpu, TEST_WIDTH, TEST_HEIGHT).expect("particles");
    let dst = LayerTarget::new_single(&gpu, TEST_WIDTH, TEST_HEIGHT);
    let params = ClipParams::default();

    let mut peak = Light::default();
    for step in 0..90 {
        let input = if step % 12 == 0 { kick() } else { loud() };
        render(
            &gpu,
            &mut particles,
            0,
            &dst,
            &input,
            &params,
            step as f32 / 60.0,
        );
        let light = picture(&gpu, TEST_WIDTH, TEST_HEIGHT, &dst);
        if light.total > peak.total {
            peak = light;
        }
    }
    assert!(peak.total > 0, "loud input must light the layer");

    for step in 0..240 {
        render(
            &gpu,
            &mut particles,
            0,
            &dst,
            &quiet(),
            &params,
            (90 + step) as f32 / 60.0,
        );
    }
    let settled = picture(&gpu, TEST_WIDTH, TEST_HEIGHT, &dst);
    assert!(
        settled.total * 2 < peak.total,
        "240 silent frames must settle the field: peak {}, after {}",
        peak.total,
        settled.total
    );

    // And it stops moving: two consecutive silent frames are nearly the same picture.
    let before = gpu.read_back(&dst);
    render(
        &gpu,
        &mut particles,
        0,
        &dst,
        &quiet(),
        &params,
        330.0 / 60.0,
    );
    let after = gpu.read_back(&dst);
    let moved = mean_difference(&before, &after);
    assert!(
        moved < 2.0,
        "a settled field must not move: {moved:.3} mean difference"
    );
    println!(
        "silence: peak light {} -> {} ({:.1}%), consecutive frames differ by {moved:.3}",
        peak.total,
        settled.total,
        100.0 * settled.total as f64 / peak.total as f64
    );
    assert_no_validation(&gpu);
}

#[test]
fn count_follows_p0() {
    let gpu = gpu();
    let mut particles = Particles::new(&gpu, TEST_WIDTH, TEST_HEIGHT).expect("particles");
    let sparse = LayerTarget::new_single(&gpu, TEST_WIDTH, TEST_HEIGHT);
    let dense = LayerTarget::new_single(&gpu, TEST_WIDTH, TEST_HEIGHT);

    let mut few = ClipParams::default();
    few.p[0] = 0.0;
    let mut many = ClipParams::default();
    many.p[0] = 1.0;

    for step in 0..8 {
        let t = step as f32 / 60.0;
        render(&gpu, &mut particles, 0, &sparse, &loud(), &few, t);
        render(&gpu, &mut particles, 1, &dense, &loud(), &many, t);
    }
    let low = picture(&gpu, TEST_WIDTH, TEST_HEIGHT, &sparse);
    let high = picture(&gpu, TEST_WIDTH, TEST_HEIGHT, &dense);
    assert!(low.lit > 0, "the sparse field must draw something");
    // Twice the particles is not twice the lit *pixels*: a denser field overlaps itself more, and
    // the light flux is normalised so a denser field is not simply brighter (the table pass's
    // `spread * spread * motion.w`). Both measures still have to move, though.
    assert!(
        high.lit > low.lit * 3 / 2,
        "p0 = 1 must draw visibly more particles: {} -> {} lit pixels",
        low.lit,
        high.lit
    );
    assert!(
        high.total > low.total * 3 / 2,
        "p0 = 1 must draw more light: {} -> {}",
        low.total,
        high.total
    );
    println!(
        "count: p0 0 -> {} lit pixels / {} light, p0 1 -> {} / {} ({:.1}x lit, {:.1}x light)",
        low.lit,
        low.total,
        high.lit,
        high.total,
        high.lit as f64 / low.lit as f64,
        high.total as f64 / low.total as f64
    );
    assert_no_validation(&gpu);
}

#[test]
fn stable_under_noise() {
    run_noise(1_000, 3);
}

#[test]
#[ignore = "5 000 noise frames: by hand, the gate runs the 1 000-frame version"]
fn stable_under_noise_5000() {
    run_noise(5_000, 4);
}

fn run_noise(frames: usize, seed: u64) {
    let gpu = gpu();
    let mut particles = Particles::new(&gpu, TEST_WIDTH, TEST_HEIGHT).expect("particles");
    let dst = LayerTarget::new_single(&gpu, TEST_WIDTH, TEST_HEIGHT);
    let check = probe_pipeline(&gpu);
    let mut state_probe: Option<LayerTarget> = None;
    let mut trail_probe: Option<LayerTarget> = None;
    let mut random = Random::new(seed);
    let mut worst_state = 0u8;
    let mut worst_trail = 0u8;

    for index in 0..frames {
        let input = random.frame();
        let params = random.params();
        render(
            &gpu,
            &mut particles,
            0,
            &dst,
            &input,
            &params,
            index as f32 / 60.0,
        );
        let layer = &particles.layers[0];
        let state_probe = state_probe.get_or_insert_with(|| {
            LayerTarget::new(&gpu, layer.state.width(), layer.state.height())
        });
        let trail_probe = trail_probe.get_or_insert_with(|| {
            LayerTarget::new(&gpu, layer.trail.width(), layer.trail.height())
        });
        if index % 50 == 49 || index + 1 == frames {
            let (bad, magnitude) = probe(&gpu, &check, state_probe, &layer.state);
            assert_eq!(
                bad, 0,
                "NaN or infinity in the particle state at frame {index}"
            );
            let (trail_bad, trail_magnitude) = probe(&gpu, &check, trail_probe, &layer.trail);
            assert_eq!(
                trail_bad, 0,
                "NaN or infinity in the trail at frame {index}"
            );
            // Position and velocity are at most ~3 screen heights; 40 bytes is 5.
            assert!(
                magnitude <= 40,
                "the state ran to {:.2} at frame {index}",
                f32::from(magnitude) * 32.0 / 255.0
            );
            // The draw pass clamps the trail to 6; 64 bytes is 8.
            assert!(
                trail_magnitude <= 64,
                "the trail ran to {:.2} at frame {index}",
                f32::from(trail_magnitude) * 32.0 / 255.0
            );
            worst_state = worst_state.max(magnitude);
            worst_trail = worst_trail.max(trail_magnitude);
        }
    }
    println!(
        "{frames} noise frames: no NaN or infinity; worst magnitude {:.2} in the state, \
         {:.2} in the trail",
        f32::from(worst_state) * 32.0 / 255.0,
        f32::from(worst_trail) * 32.0 / 255.0
    );
    assert_no_validation(&gpu);
}

/// The health probe's pipeline (`particles/check.wgsl`): one input, no uniform.
fn probe_pipeline(gpu: &Gpu) -> Pipeline {
    const SOURCE: &str = include_str!("check.wgsl");
    let words = mosh_shaders::compile_wgsl(SOURCE).expect("check.wgsl compiles");
    Pipeline::fullscreen(gpu, &words, 0, 1)
        .expect("check pipeline")
        .with_name("particles.check")
}

/// Runs the probe over `src`. Returns `(NaN or infinity pixels, worst magnitude byte)`.
fn probe(gpu: &Gpu, pipeline: &Pipeline, dst: &LayerTarget, src: &LayerTarget) -> (u16, u8) {
    assert_eq!(
        (dst.width(), dst.height()),
        (src.width(), src.height()),
        "probe target size"
    );
    let mut recorder = FrameRecorder::new(gpu);
    recorder
        .draw(pipeline, dst, &[], &[src])
        .expect("check draw");
    gpu.wait_frame(recorder.submit());
    let pixels = gpu.read_back(dst);
    let mut bad = 0u16;
    let mut worst = 0u8;
    for pixel in pixels.chunks(4) {
        if pixel[0] > 0 {
            bad += 1;
        }
        worst = worst.max(pixel[1]);
    }
    (bad, worst)
}

/// The previews: four settings at the budget size, written under `native/target/style-previews/`
/// (inside the build directory, never committed). The PR lists them.
#[test]
fn style_previews() {
    // Long enough that the last kick (at 43) is 14 frames behind the capture: the burst has
    // spread and the field is on its way back to the middle, which is the look a still should
    // show rather than a frame of the burst itself.
    const FRAMES: usize = 57;

    let gpu = gpu();
    let directory: PathBuf =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/style-previews/particles");
    let mut written = Vec::new();
    let variants: [(&str, [f32; 8], bool); 4] = [
        ("01-sparks-default", [0.0; 8], false),
        (
            "02-crisp-glitter",
            [0.55, 0.12, 0.95, 0.8, 0.35, 0.35, 0.22, 0.0],
            false,
        ),
        (
            "03-soft-dust",
            [0.85, 0.90, 0.25, 0.15, 0.20, 0.95, 0.55, 0.85],
            false,
        ),
        (
            "04-kick-burst",
            [0.35, 0.35, 0.7, 1.0, 0.7, 0.5, 0.1, 0.2],
            true,
        ),
    ];
    for (name, p, burst) in variants {
        let mut particles = Particles::new(&gpu, BUDGET_WIDTH, BUDGET_HEIGHT).expect("particles");
        let dst = LayerTarget::new_single(&gpu, BUDGET_WIDTH, BUDGET_HEIGHT);
        let params = ClipParams {
            p,
            palette: 2,
            seed: 7,
        };
        for step in 0..FRAMES {
            let input = if step % 20 == 3 { kick() } else { loud() };
            render(
                &gpu,
                &mut particles,
                0,
                &dst,
                &input,
                &params,
                step as f32 / 60.0,
            );
        }
        // The burst variant is captured six frames after a kick: the shockwave has carried the
        // field outward and it is still lit, rather than a frame of the burst itself.
        if burst {
            render(
                &gpu,
                &mut particles,
                0,
                &dst,
                &kick(),
                &params,
                FRAMES as f32 / 60.0,
            );
            for step in 0..6 {
                render(
                    &gpu,
                    &mut particles,
                    0,
                    &dst,
                    &loud(),
                    &params,
                    (FRAMES + 1 + step) as f32 / 60.0,
                );
            }
        }
        let pixels = gpu.read_back(&dst);
        let light = light_stats(&pixels, BUDGET_WIDTH, BUDGET_HEIGHT);
        assert!(light.lit > 500, "{name}: {} lit pixels", light.lit);
        let path = preview::write(&directory, name, BUDGET_WIDTH, BUDGET_HEIGHT, &pixels);
        println!("{name}: {} lit pixels -> {path}", light.lit);
        written.push(path);
    }
    assert_eq!(written.len(), 4);
    assert_no_validation(&gpu);
}

/// xorshift64*, so the noise sequence is the same on every run.
struct Random(u64);

impl Random {
    fn new(seed: u64) -> Random {
        Random(seed | 1)
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    fn unit(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }

    fn frame(&mut self) -> AnalysisFrame {
        AnalysisFrame {
            time_s: f64::from(self.next_u32() % 600) * 0.1,
            ppq: f64::from(self.next_u32() % 256),
            beat_phase: self.unit(),
            bar_phase: self.unit(),
            beat_index: i64::from(self.next_u32() % 512),
            bar_index: i64::from(self.next_u32() % 128),
            tempo_bpm: 60.0 + 120.0 * self.unit(),
            playing: !self.next_u32().is_multiple_of(4),
            source: TransportSource::Plugin,
            bands: [self.unit(), self.unit(), self.unit()],
            mel: [0.0; 24],
            onset: self.unit(),
            loudness_db: -60.0 + 60.0 * self.unit(),
            loudness: self.unit(),
            brightness: self.unit(),
            intensity: self.unit(),
            level: 1 + (self.next_u32() % 4) as u8,
            events: EventFlags::empty(),
        }
    }

    fn params(&mut self) -> ClipParams {
        let mut p = [0.0f32; 8];
        for value in &mut p {
            *value = self.unit();
        }
        ClipParams {
            p,
            palette: (self.next_u32() % 8) as u16,
            seed: u64::from(self.next_u32()),
        }
    }
}
