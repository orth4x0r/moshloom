//! Offscreen GPU tests for the fluid visualizer (Task 2.4). No window is opened: everything
//! renders into layer targets and is read back.
//!
//! The four tests the plan asks for are `dye_rises_after_onset`, `silence_decays`,
//! `velocity_available_after_draw` and `stable_under_noise` (1 000 frames by default, plus
//! `#[ignore]`d `stable_under_noise_10000`). `gpu_ms_per_draw` prints the number the PR body
//! quotes.

use super::*;

use mosh_core::{EventFlags, TransportSource};
use mosh_render::GpuPreference;

/// A small internal size for the functional tests: a fluid layer's sim cost is fixed by the
/// 256x144 grid, but the present and flow passes are drawn at this size.
const TEST_WIDTH: u32 = 320;
const TEST_HEIGHT: u32 = 180;

/// `fluidvis_check` reports the largest magnitude it saw as `value / 32`; anything above this
/// byte means a channel ran past 10 (velocity is clamped to 6, dye to 4 in the passes).
const MAGNITUDE_LIMIT: u8 = 80;

fn gpu() -> Gpu {
    let gpu = Gpu::new(GpuPreference::Integrated)
        .or_else(|_| Gpu::new(GpuPreference::ByName(String::new())))
        .expect("a Vulkan device");
    println!(
        "adapter: {} | validation layers: {}",
        gpu.adapter_name(),
        if gpu.validation_enabled() {
            "on"
        } else {
            "unavailable"
        }
    );
    gpu
}

fn assert_no_validation(gpu: &Gpu) {
    let messages = gpu.validation_messages();
    assert!(
        messages.is_empty(),
        "{} validation message(s):\n{}",
        messages.len(),
        messages.join("\n")
    );
    println!(
        "validation: {} message(s), layer {}",
        messages.len(),
        if gpu.validation_enabled() {
            "enabled"
        } else {
            "unavailable"
        }
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

/// The three named frames the tests use.
fn quiet() -> AnalysisFrame {
    let mut f = frame(1, [0.0, 0.0, 0.0], 0.0);
    f.playing = false;
    f.onset = 0.0;
    f
}

fn loud() -> AnalysisFrame {
    frame(4, [0.9, 0.8, 0.7], 0.0)
}

/// An onset with the low band up: a kick.
fn kick() -> AnalysisFrame {
    frame(3, [0.9, 0.5, 0.4], 0.9)
}

/// Records, submits and waits for one fluid frame.
fn render(
    gpu: &Gpu,
    fluid: &mut FluidVisualizer,
    dst: &LayerTarget,
    frame: &AnalysisFrame,
    params: &ClipParams,
    t: f32,
) {
    let mut recorder = FrameRecorder::new(gpu);
    fluid
        .draw(&mut recorder, dst, frame, params, t)
        .expect("fluid draw");
    gpu.wait_frame(recorder.submit());
}

/// Sum of the dye target's RGB bytes: a coarse but monotonic dye mass.
fn dye_mass(gpu: &Gpu, fluid: &FluidVisualizer) -> u64 {
    gpu.read_back(&fluid.dye)
        .chunks(4)
        .map(|pixel| u64::from(pixel[0]) + u64::from(pixel[1]) + u64::from(pixel[2]))
        .sum()
}

/// The `fluidvis_check` pipeline: test-only, so the show never pays for it.
fn check_pipeline(gpu: &Gpu) -> Pipeline {
    let words = mosh_shaders::spirv("fluidvis_check").expect("fluidvis_check compiled");
    Pipeline::fullscreen(gpu, words, 16, 2)
        .expect("check pipeline")
        .with_name("fluid check")
}

fn check_uniform() -> [u8; 16] {
    pack::<4, 16>([
        1.0 / GRID_WIDTH as f32,
        1.0 / GRID_HEIGHT as f32,
        0.0,
        0.0,
    ])
}

/// Runs the health probe over the dye and the velocity field. Returns `(nan or inf pixels,
/// worst magnitude byte)`.
fn probe(
    gpu: &Gpu,
    fluid: &FluidVisualizer,
    check: &LayerTarget,
    pipeline: &Pipeline,
) -> (usize, u8) {
    let mut recorder = FrameRecorder::new(gpu);
    recorder
        .draw(
            pipeline,
            check,
            &check_uniform(),
            &[&fluid.dye, &fluid.velocity_a],
        )
        .expect("check draw");
    gpu.wait_frame(recorder.submit());

    let pixels = gpu.read_back(check);
    let mut bad = 0;
    let mut worst = 0u8;
    for pixel in pixels.chunks(4) {
        if pixel[0] > 127 {
            bad += 1;
        }
        worst = worst.max(pixel[1]);
    }
    (bad, worst)
}

#[test]
fn uniform_block_sizes_match_the_shaders() {
    // The seven extracted passes all declare `FUB`; `fluid.rs` writes 64 bytes for it.
    for name in [
        "fluid_advect",
        "fluid_force",
        "fluid_curl",
        "fluid_vorticity",
        "fluid_divergence",
        "fluid_jacobi",
        "fluid_gradient",
    ] {
        assert_eq!(
            mosh_shaders::uniform_bytes(name),
            Some(FUB_BYTES as u32),
            "{name}: uniform block size"
        );
    }
    assert_eq!(
        mosh_shaders::uniform_bytes("fluidvis_dye"),
        Some(DYE_BYTES as u32)
    );
    assert_eq!(
        mosh_shaders::uniform_bytes("fluidvis_present"),
        Some(PRESENT_BYTES as u32)
    );
    assert_eq!(
        mosh_shaders::uniform_bytes("fluidvis_flow"),
        Some(FLOW_BYTES as u32)
    );
    println!("uniform blocks: {FUB_BYTES} + {DYE_BYTES} + {PRESENT_BYTES} + {FLOW_BYTES} bytes");
}

#[test]
fn adapter_renames_the_entry_point_and_the_bindings() {
    let words = mosh_shaders::spirv("fluid_advect").expect("fluid advect");
    let adapted = adapt_module(words, &FUB_BINDINGS, FRAGMENT_ENTRY);
    assert!(
        adapted.len() > words.len(),
        "the entry point name grew from `fs` to `fs_main`"
    );

    // Walk the adapted module: one fragment OpEntryPoint named `fs_main`, and bindings in
    // mosh-render's layout (images at the odd numbers, one sampler at 2).
    const OP_ENTRY_POINT: u16 = 15;
    const OP_DECORATE: u16 = 71;
    let mut entry: Option<String> = None;
    let mut bindings = Vec::new();
    let mut index = 5;
    while index < adapted.len() {
        let count = (adapted[index] >> 16) as usize;
        let opcode = (adapted[index] & 0xffff) as u16;
        assert!(count > 0, "zero-length instruction");
        if opcode == OP_ENTRY_POINT {
            let mut bytes: Vec<u8> = Vec::new();
            for word in &adapted[index + 3..index + count] {
                bytes.extend_from_slice(&word.to_le_bytes());
                if bytes.contains(&0) {
                    break;
                }
            }
            let text = String::from_utf8_lossy(&bytes);
            entry = Some(text.trim_end_matches('\0').to_string());
        }
        if opcode == OP_DECORATE && adapted[index + 2] == 33 {
            bindings.push(adapted[index + 3]);
        }
        index += count;
    }
    assert_eq!(entry.as_deref(), Some(FRAGMENT_ENTRY));
    bindings.sort_unstable();
    assert_eq!(bindings, vec![0, 1, 2, 3, 5, 7], "bind group 0 remapped");
    println!("adapted bindings: {bindings:?}, entry point {entry:?}");
}

#[test]
fn dye_rises_after_onset() {
    let gpu = gpu();
    let mut fluid = FluidVisualizer::new(&gpu, TEST_WIDTH, TEST_HEIGHT).expect("fluid");
    let dst = LayerTarget::new(&gpu, TEST_WIDTH, TEST_HEIGHT);
    let params = ClipParams::default();

    for step in 0..4 {
        render(&gpu, &mut fluid, &dst, &quiet(), &params, step as f32 / 60.0);
    }
    let before = dye_mass(&gpu, &fluid);

    render(&gpu, &mut fluid, &dst, &kick(), &params, 4.0 / 60.0);
    let after = dye_mass(&gpu, &fluid);

    assert_eq!(before, 0, "silence must not inject dye");
    assert!(
        after > before,
        "an onset frame must raise the dye: {before} -> {after}"
    );
    println!("dye after one kick: {before} -> {after} (sum of RGB bytes)");

    // And the dye is the picture: a second of held note at level 4 has to reach the layer target.
    for step in 0..60 {
        render(
            &gpu,
            &mut fluid,
            &dst,
            &loud(),
            &params,
            (5 + step) as f32 / 60.0,
        );
    }
    let picture = gpu.read_back(&dst);
    let mean: f64 = picture
        .chunks(4)
        .map(|pixel| {
            (f64::from(pixel[0]) + f64::from(pixel[1]) + f64::from(pixel[2])) / (3.0 * 255.0)
        })
        .sum::<f64>()
        / (picture.len() / 4) as f64;
    assert!(mean > 0.01, "the dye must be visible, mean luminance {mean}");
    println!("dye picture after 60 loud frames: mean luminance {mean:.3}");
    assert_no_validation(&gpu);
}

#[test]
fn silence_decays() {
    let gpu = gpu();
    let mut fluid = FluidVisualizer::new(&gpu, TEST_WIDTH, TEST_HEIGHT).expect("fluid");
    let dst = LayerTarget::new(&gpu, TEST_WIDTH, TEST_HEIGHT);
    let params = ClipParams::default();

    // A minute of loud input, kicks every 15 frames.
    let mut peak = 0u64;
    for step in 0..120 {
        let input = if step % 15 == 0 { kick() } else { loud() };
        render(&gpu, &mut fluid, &dst, &input, &params, step as f32 / 60.0);
        peak = peak.max(dye_mass(&gpu, &fluid));
    }
    assert!(peak > 0, "loud input must produce dye");

    // Then 600 silent frames: the fade has to take the dye down towards zero.
    for step in 0..600 {
        render(&gpu, &mut fluid, &dst, &quiet(), &params, step as f32 / 60.0);
    }
    let after = dye_mass(&gpu, &fluid);
    assert!(
        after * 5 < peak,
        "600 silent frames must fade the dye: peak {peak}, after {after}"
    );
    println!(
        "dye decay over 600 silent frames: peak {peak} -> {after} ({:.1}%)",
        100.0 * after as f64 / peak as f64
    );
    assert_no_validation(&gpu);
}

#[test]
fn velocity_available_after_draw() {
    let gpu = gpu();
    let mut fluid = FluidVisualizer::new(&gpu, TEST_WIDTH, TEST_HEIGHT).expect("fluid");
    let dst = LayerTarget::new(&gpu, TEST_WIDTH, TEST_HEIGHT);
    let params = ClipParams::default();

    assert!(
        fluid.velocity().is_none(),
        "no field before the simulation has run"
    );

    render(&gpu, &mut fluid, &dst, &quiet(), &params, 0.0);
    let velocity = fluid.velocity().expect("field after one draw");
    assert_eq!(
        (velocity.width(), velocity.height()),
        (TEST_WIDTH, TEST_HEIGHT),
        "the field is at the layers' internal size"
    );
    let still = gpu.read_back(velocity);
    assert!(
        still.chunks(4).all(|pixel| pixel[0] == 0 && pixel[1] == 0),
        "a silent first frame has no velocity"
    );

    render(&gpu, &mut fluid, &dst, &kick(), &params, 1.0 / 60.0);
    let velocity = fluid.velocity().expect("field after the kick");
    let moved = gpu.read_back(velocity);
    let moving = moved
        .chunks(4)
        .filter(|pixel| pixel[0] > 0 || pixel[1] > 0)
        .count();
    let strongest = moved
        .chunks(4)
        .map(|pixel| pixel[0].max(pixel[1]))
        .max()
        .unwrap_or(0);
    assert!(
        moving > 100,
        "a kick must move pixels in the field, {moving} did"
    );
    assert!(strongest >= 200, "the kick peak is {strongest}");
    println!(
        "velocity field {}x{}: {moving} non-zero bytes, peak {strongest}",
        velocity.width(),
        velocity.height()
    );
    assert_no_validation(&gpu);
}

#[test]
fn stable_under_noise() {
    run_noise(1_000, 3);
}

#[test]
#[ignore = "10 000 noise frames: by hand, the gate runs the 1 000-frame version"]
fn stable_under_noise_10000() {
    run_noise(10_000, 4);
}

fn run_noise(frames: usize, seed: u64) {
    let gpu = gpu();
    let mut fluid = FluidVisualizer::new(&gpu, TEST_WIDTH, TEST_HEIGHT).expect("fluid");
    let dst = LayerTarget::new(&gpu, TEST_WIDTH, TEST_HEIGHT);
    let check = LayerTarget::new(&gpu, GRID_WIDTH, GRID_HEIGHT);
    let pipeline = check_pipeline(&gpu);
    let mut random = Random::new(seed);
    let mut worst = 0u8;

    for index in 0..frames {
        let input = random.frame();
        let params = random.params();
        render(
            &gpu,
            &mut fluid,
            &dst,
            &input,
            &params,
            index as f32 / 60.0,
        );
        if index % 100 == 99 || index + 1 == frames {
            let (bad, magnitude) = probe(&gpu, &fluid, &check, &pipeline);
            assert_eq!(bad, 0, "NaN or infinity in dye or velocity at frame {index}");
            assert!(
                magnitude <= MAGNITUDE_LIMIT,
                "dye or velocity ran to {} at frame {index} (limit {MAGNITUDE_LIMIT})",
                u32::from(magnitude) * 32 / 255
            );
            worst = worst.max(magnitude);
        }
    }
    println!(
        "{frames} noise frames: no NaN, worst magnitude byte {worst} of {MAGNITUDE_LIMIT} allowed"
    );
    assert_no_validation(&gpu);
}

#[test]
fn gpu_ms_per_draw() {
    // The reference layer size the design budgets at (2 ms render CPU, 6 ms GPU per frame for
    // three layers plus an overlay, at 1920x1080).
    const WIDTH: u32 = 1280;
    const HEIGHT: u32 = 720;
    const FRAMES: usize = 30;

    let gpu = gpu();
    let mut fluid = FluidVisualizer::new(&gpu, WIDTH, HEIGHT).expect("fluid");
    let dst = LayerTarget::new(&gpu, WIDTH, HEIGHT);
    let params = ClipParams::default();
    let iterations = Plan::build(
        &loud(),
        &params,
        0.0,
        [WIDTH as f32, HEIGHT as f32],
        [WIDTH as f32, HEIGHT as f32],
    )
    .iterations;
    let passes = 9 + iterations as usize;

    // Two warm-up frames, then the measured ones.
    for step in 0..2 {
        render(&gpu, &mut fluid, &dst, &loud(), &params, step as f32 / 60.0);
    }

    let mut total = 0.0f32;
    let mut per_pass: Vec<(String, f32)> = Vec::new();
    for step in 0..FRAMES {
        let input = if step % 8 == 0 { kick() } else { loud() };
        let mut recorder = FrameRecorder::new(&gpu);
        fluid
            .draw(&mut recorder, &dst, &input, &params, step as f32 / 60.0)
            .expect("fluid draw");
        let timings = gpu.wait_frame(recorder.submit());
        assert_eq!(
            timings.per_pass_ms.len(),
            passes,
            "one timing per recorded pass"
        );
        total += timings.frame_ms;
        for (index, (name, milliseconds)) in timings.per_pass_ms.iter().enumerate() {
            if per_pass.len() <= index {
                per_pass.push((name.clone(), 0.0));
            }
            per_pass[index].1 += milliseconds;
        }
    }

    let mean = total / FRAMES as f32;
    assert!(mean.is_finite() && mean > 0.0, "fluid draw took {mean} ms");
    println!(
        "fluid: {mean:.3} ms per draw at {WIDTH}x{HEIGHT} internal ({passes} passes, {iterations} jacobi)"
    );
    for (name, milliseconds) in &per_pass {
        println!("  {name}: {:.3} ms", milliseconds / FRAMES as f32);
    }
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
