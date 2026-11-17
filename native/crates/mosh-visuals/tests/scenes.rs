//! Task 2.1: the 16 web scenes as Vulkan layer visualizers.
//!
//! Every test here is offscreen and small on purpose (the reference machine is shared with other
//! workers): 160x90 for correctness, 960x540 only for the one timing report.

use mosh_core::{AnalysisFrame, ClipParams};
use mosh_render::{FrameRecorder, Gpu, GpuPreference, LayerTarget};
use mosh_shaders::VISUALIZERS;
use mosh_visuals::scenes::Scenes;

const W: u32 = 160;
const H: u32 = 90;

fn gpu() -> Gpu {
    Gpu::new(GpuPreference::Integrated).expect("a Vulkan device on the integrated GPU")
}

/// The 16 scenes, in `VISUALIZERS` order; `fluid` is V-fluid's (Task 2.4), not a scene module.
fn scene_ids() -> Vec<&'static str> {
    VISUALIZERS
        .iter()
        .map(|visualizer| visualizer.id)
        .filter(|id| *id != "fluid")
        .collect()
}

/// Mean of the three colour channels over the whole frame, 0..1.
fn mean_luminance(pixels: &[u8]) -> f32 {
    let sum: u64 = pixels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pixel| u64::from(pixel[0]) + u64::from(pixel[1]) + u64::from(pixel[2]))
        .sum();
    let count = (pixels.len() / 4) as f32;
    sum as f32 / (count * 3.0 * 255.0)
}

/// The largest value in any colour channel (alpha excluded: a black frame is opaque).
fn brightest_channel(pixels: &[u8]) -> u8 {
    pixels
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|pixel| &pixel[..3])
        .copied()
        .max()
        .unwrap_or(0)
}

/// Pixels that differ by more than `tolerance` in any colour channel.
fn differing_pixels(a: &[u8], b: &[u8], tolerance: u8) -> usize {
    a.as_chunks::<4>()
        .0
        .iter()
        .zip(b.as_chunks::<4>().0.iter())
        .filter(|(left, right)| (0..3).any(|c| left[c].abs_diff(right[c]) > tolerance))
        .count()
}

/// Three intensity levels: a quiet bar, a mid bar and a drop.
fn frame_at(level: usize) -> AnalysisFrame {
    let (band, pulse): (f32, f32) = match level {
        0 => (0.18, 0.25),
        1 => (0.5, 0.6),
        _ => (0.9, 1.0),
    };
    AnalysisFrame {
        time_s: 12.0,
        ppq: 16.0,
        beat_phase: (1.0 - pulse).clamp(0.0, 1.0),
        bar_phase: 0.25,
        beat_index: 16,
        bar_index: 4,
        tempo_bpm: 120.0,
        playing: true,
        bands: [band, band * 0.8, band * 0.6],
        mel: std::array::from_fn(|k| band * (1.0 - k as f32 / 30.0)),
        onset: pulse,
        loudness: band,
        level: level as u8 + 1,
        intensity: band,
        ..AnalysisFrame::default()
    }
}

/// A clip whose parameters look like one procedural seed's worth: speed, density, hue, variant
/// and palette all move, which is what "3 seeds" means for a scene.
fn params_for(seed: u64) -> ClipParams {
    let r = |k: u64| {
        ((seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(k * 1442695040888963407))
            >> 40) as f32
            / (1u64 << 24) as f32
    };
    ClipParams {
        p: [
            0.7 + r(1) * 0.8,     // p0 speed
            0.3 + r(2) * 0.5,     // p1 density
            r(3),                 // p2 hue
            (r(4) * 3.0).floor(), // p3 variant
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        palette: match seed % 3 {
            0 => 0, // 0 = the scene's default palette, as engine.c resolves it
            1 => 3,
            _ => 6,
        },
        seed,
    }
}

fn draw(
    scenes: &mut Scenes,
    gpu: &Gpu,
    dst: &LayerTarget,
    id: &str,
    frame: &AnalysisFrame,
    params: &ClipParams,
    t: f32,
) {
    let mut recorder = FrameRecorder::new(gpu);
    scenes
        .draw(&mut recorder, dst, id, frame, params, t)
        .unwrap_or_else(|error| panic!("draw {id}: {error}"));
    gpu.wait_frame(recorder.submit());
}

fn assert_no_validation(gpu: &Gpu) {
    let messages = gpu.validation_messages();
    assert!(
        messages.is_empty(),
        "{} validation messages: {messages:#?}",
        messages.len()
    );
}

/// Each of the 16 scenes at 3 intensity levels and 3 seeds: mean luminance > 0.02, read back.
#[test]
fn every_scene_renders_lit() {
    let gpu = gpu();
    let mut scenes = Scenes::new(&gpu).expect("scenes");
    let ids = scene_ids();
    assert_eq!(ids.len(), 16, "16 scene visualizers, got {ids:?}");

    let frames = [frame_at(0), frame_at(1), frame_at(2)];
    let targets: Vec<LayerTarget> = (0..9).map(|_| LayerTarget::new(&gpu, W, H)).collect();

    let mut dark = Vec::new();
    let mut report = Vec::new();
    for id in &ids {
        for (level, frame) in frames.iter().enumerate() {
            for seed in 0..3u64 {
                let dst = &targets[level * 3 + seed as usize];
                let params = params_for(seed);
                draw(
                    &mut scenes,
                    &gpu,
                    dst,
                    id,
                    frame,
                    &params,
                    2.0 + seed as f32 * 0.7,
                );
                let luminance = mean_luminance(&gpu.read_back(dst));
                report.push(format!(
                    "{id:<12} level {level} seed {seed}: mean luminance {luminance:.4}"
                ));
                if luminance <= 0.02 {
                    dark.push(format!("{id} level {level} seed {seed}: {luminance:.4}"));
                }
            }
        }
    }
    for line in &report {
        println!("{line}");
    }
    assert!(
        dark.is_empty(),
        "{} of {} scene draws were not lit (> 0.02):\n{}",
        dark.len(),
        report.len(),
        dark.join("\n")
    );
    assert_no_validation(&gpu);
}

/// An id that is not a visualizer, a custom name nobody registered, and the empty string: black
/// pixels, no panic, no validation message.
#[test]
fn unknown_scene_is_black_not_panic() {
    let gpu = gpu();
    let mut scenes = Scenes::new(&gpu).expect("scenes");
    let dst = LayerTarget::new(&gpu, W, H);
    let frame = frame_at(1);
    let params = params_for(1);

    for id in ["nonsense", "custom:nobody", "", "custom:", "fluid"] {
        draw(&mut scenes, &gpu, &dst, id, &frame, &params, 1.0);
        let pixels = gpu.read_back(&dst);
        assert_eq!(pixels.len(), (W * H * 4) as usize);
        let brightest = brightest_channel(&pixels);
        assert_eq!(
            brightest, 0,
            "{id:?} must draw black, brightest channel {brightest}"
        );
    }
    assert_no_validation(&gpu);
}

/// A custom scene compiles and draws; a broken one is rejected with its line, and the last good
/// version keeps drawing.
#[test]
fn custom_scene_compiles_and_broken_keeps_previous() {
    let gpu = gpu();
    let mut scenes = Scenes::new(&gpu).expect("scenes");
    let dst = LayerTarget::new(&gpu, W, H);
    let frame = frame_at(1);
    let params = params_for(0);

    // A web-style custom scene: `web_prelude()`'s `U`, `lut()` and `h2()` are all it uses, so this
    // is a shader the web app would run unchanged.
    let good = r#"
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy / U.res;
  let c = lut(uv.x * 0.7 + uv.y * 0.3 + U.hue) * (0.4 + 0.6 * h2(floor(pos.xy / 8.)));
  return vec4f(c, 1.);
}
"#;
    scenes
        .set_custom(&gpu, "mine", good)
        .expect("a valid custom scene registers");
    draw(&mut scenes, &gpu, &dst, "custom:mine", &frame, &params, 1.0);
    let drawn = gpu.read_back(&dst);
    let luminance = mean_luminance(&drawn);
    assert!(
        luminance > 0.02,
        "the custom scene must be lit, mean luminance {luminance:.4}"
    );

    // `nope` is on line 3 of the body the human typed, the same shader `gpu_test.mjs` rejects.
    let broken = "@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {\n  let q = XY(pos);\n  return fin(vec3f(q.x, nope, 0.));\n}";
    let error = scenes
        .set_custom(&gpu, "mine", broken)
        .expect_err("a broken custom scene must not register");
    assert!(
        error.contains("line 3") || error.contains("3:"),
        "the error names the line in the body, got {error:?}"
    );

    draw(&mut scenes, &gpu, &dst, "custom:mine", &frame, &params, 1.0);
    assert_eq!(
        gpu.read_back(&dst),
        drawn,
        "a broken compile keeps the previous version drawing"
    );

    // A name nobody registered is black, and a broken first compile registers nothing.
    assert!(
        scenes
            .set_custom(
                &gpu,
                "other",
                "@fragment fn fs() -> @location(0) vec4f { return nope; }"
            )
            .is_err()
    );
    draw(
        &mut scenes,
        &gpu,
        &dst,
        "custom:other",
        &frame,
        &params,
        1.0,
    );
    let black = gpu.read_back(&dst);
    assert_eq!(
        brightest_channel(&black),
        0,
        "a failed registration draws black"
    );
    assert_no_validation(&gpu);
}

/// `p0` (speed) is the scene's clock: 0 freezes it, 1 runs it, and at least one scene's pixels
/// differ between the two.
#[test]
fn scene_params_change_pixels() {
    let gpu = gpu();
    let mut scenes = Scenes::new(&gpu).expect("scenes");
    let frame = frame_at(2);
    let mut frozen_params = params_for(1);
    frozen_params.p[0] = 0.0;
    let mut moving_params = frozen_params;
    moving_params.p[0] = 1.0;

    let frozen = LayerTarget::new(&gpu, W, H);
    let moving = LayerTarget::new(&gpu, W, H);
    let mut changed = Vec::new();
    for id in scene_ids() {
        draw(&mut scenes, &gpu, &frozen, id, &frame, &frozen_params, 3.0);
        draw(&mut scenes, &gpu, &moving, id, &frame, &moving_params, 3.0);
        let count = differing_pixels(&gpu.read_back(&frozen), &gpu.read_back(&moving), 2);
        println!("{id:<12} p0 0 vs 1: {count} of {} pixels differ", W * H);
        if count > 0 {
            changed.push((id, count));
        }
    }
    assert!(
        !changed.is_empty(),
        "p0 must move at least one scene: {changed:?}"
    );
    assert_no_validation(&gpu);
}

/// Per-scene GPU time at 960x540, from `GpuTimings`, on the integrated GPU. Prints; asserts only
/// that every scene produced a number and that nothing was reported by the validation layers.
#[test]
fn scene_gpu_times_at_960x540() {
    let gpu = gpu();
    let mut scenes = Scenes::new(&gpu).expect("scenes");
    let dst = LayerTarget::new(&gpu, 960, 540);
    let frame = frame_at(2);
    let params = params_for(2);

    // Warm-up: 30 frames at this size, so the iGPU is at a steady clock before anything is
    // measured. A cold GPU reports 3-4x these numbers, and this machine is shared.
    for pass in 0..30 {
        let mut recorder = FrameRecorder::new(&gpu);
        scenes
            .draw(
                &mut recorder,
                &dst,
                "tunnel",
                &frame,
                &params,
                1.0 + pass as f32 / 60.0,
            )
            .expect("warm-up draw");
        gpu.wait_frame(recorder.submit());
    }

    let mut rows = Vec::new();
    for id in scene_ids() {
        let mut samples = Vec::new();
        for pass in 0..20 {
            let t = 4.0 + pass as f32 / 60.0;
            let mut recorder = FrameRecorder::new(&gpu);
            scenes
                .draw(&mut recorder, &dst, id, &frame, &params, t)
                .expect("draw");
            let timings = gpu.wait_frame(recorder.submit());
            if pass < 2 {
                continue; // the first passes of a scene warm its own caches
            }
            let ms = timings
                .per_pass_ms
                .iter()
                .find(|(name, _)| name == id)
                .map(|(_, ms)| *ms)
                .unwrap_or_else(|| {
                    panic!("no timestamp for the {id} pass: {:?}", timings.per_pass_ms)
                });
            samples.push(ms);
        }
        samples.sort_by(f32::total_cmp);
        // The machine is shared with other workers, so both ends are reported: the minimum is the
        // pass's own cost, the median what it costs while someone else is using the GPU too.
        rows.push((id, samples[0], samples[samples.len() / 2]));
    }
    println!("GPU ms per scene pass at 960x540 on {}", gpu.adapter_name());
    println!("  {:<12} {:>7} {:>7}", "scene", "min", "median");
    for (id, min, median) in &rows {
        println!("  {id:<12} {min:>7.3} {median:>7.3}");
    }
    let total_min: f32 = rows.iter().map(|(_, min, _)| *min).sum();
    let total_median: f32 = rows.iter().map(|(_, _, median)| *median).sum();
    println!("  {:<12} {total_min:>7.3} {total_median:>7.3}", "sum");
    assert!(
        rows.iter()
            .all(|(_, min, median)| *min > 0.0 && *median > 0.0),
        "every scene must have a GPU time: {rows:?}"
    );
    assert_no_validation(&gpu);
}
