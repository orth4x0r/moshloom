//! The particle layer's GPU budget, on its own: 960x540, the default clip, five passes.
//!
//! It lives in `tests/` rather than beside the functional tests on purpose. `cargo test` runs one
//! test *target* at a time but the tests inside a target in parallel, and every other offscreen
//! test in this crate submits its own frames: a timing measured next to them is the machine's, not
//! the layer's (measuring the layer alongside the rest of the lib's tests read 1.3-2.4 ms for the
//! same code that reads 0.9 ms alone). This binary has one test, so it has the GPU to itself.
//!
//! No window is opened. The number is printed in every build; the 1 ms assertion is release-only,
//! because a debug build's CPU can starve the submit (the GPU work itself is identical).

use mosh_core::{AnalysisFrame, ClipParams, EventFlags, TransportSource};
use mosh_render::{FrameRecorder, Gpu, GpuPreference, LayerTarget};
use mosh_visuals::particles::Particles;

/// The size the brief budgets at: a 1920x1080 output with the layer's internal scale at 0.5.
const WIDTH: u32 = 960;
const HEIGHT: u32 = 540;
const FRAMES: usize = 40;
/// sim, table, buckets, draw, present.
const PASSES: usize = 5;

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

fn loud() -> AnalysisFrame {
    frame(4, [0.9, 0.8, 0.7], 0.0)
}

fn kick() -> AnalysisFrame {
    frame(3, [0.9, 0.5, 0.4], 0.9)
}

fn render(
    gpu: &Gpu,
    particles: &mut Particles,
    dst: &LayerTarget,
    input: &AnalysisFrame,
    params: &ClipParams,
    t: f32,
) {
    let mut recorder = FrameRecorder::new(gpu);
    particles
        .draw(&mut recorder, 0, dst, input, params, t)
        .expect("particle draw");
    gpu.wait_frame(recorder.submit());
}

#[test]
fn gpu_ms_per_draw() {
    let gpu = gpu();
    let started = std::time::Instant::now();
    let mut particles = Particles::new(&gpu, WIDTH, HEIGHT).expect("particles");
    println!(
        "particles: 5 pipelines built in {:.1} ms (shaders compiled once, in new)",
        started.elapsed().as_secs_f32() * 1000.0
    );
    let dst = LayerTarget::new_single(&gpu, WIDTH, HEIGHT);

    // The default clip is the one the budget is quoted (and asserted) for; the full-density clip
    // is printed so the price of `p0 = 1` is on record rather than assumed.
    let default = measure(
        &gpu,
        &mut particles,
        &dst,
        ClipParams::default(),
        "default p0",
    );
    let mut heavy = ClipParams::default();
    heavy.p[0] = 1.0;
    measure(&gpu, &mut particles, &dst, heavy, "p0 = 1");
    let messages = gpu.validation_messages();
    println!("validation: {} message(s)", messages.len());
    assert!(messages.is_empty(), "{}", messages.join("\n"));
    if !cfg!(debug_assertions) {
        assert!(
            default < 1.0,
            "the layer's budget is 1 ms at {WIDTH}x{HEIGHT}: {default:.3} ms"
        );
    }
}

/// Renders `FRAMES` frames of one clip and prints its median, mean and best, plus one line per
/// pass. Returns the median.
fn measure(
    gpu: &Gpu,
    particles: &mut Particles,
    dst: &LayerTarget,
    params: ClipParams,
    label: &str,
) -> f32 {
    // Warm-up: the GPU's clocks and the pass's first descriptors. This iGPU idles at a low clock
    // and takes about a second of submitted frames to reach its working clock, and a cold run
    // measures 2-2.5x the warm one (0.88 ms against 0.36 ms for the same code), so the loop is
    // long on purpose.
    for step in 0..60 {
        render(gpu, particles, dst, &loud(), &params, step as f32 / 60.0);
    }

    let mut timings = Vec::with_capacity(FRAMES);
    let mut per_pass: Vec<(String, f32)> = Vec::new();
    for step in 0..FRAMES {
        let input = if step % 8 == 0 { kick() } else { loud() };
        let mut recorder = FrameRecorder::new(gpu);
        particles
            .draw(
                &mut recorder,
                0,
                dst,
                &input,
                &params,
                (step + 4) as f32 / 60.0,
            )
            .expect("particle draw");
        let frame = gpu.wait_frame(recorder.submit());
        assert_eq!(
            frame.per_pass_ms.len(),
            PASSES,
            "one timing per recorded pass"
        );
        timings.push(frame.frame_ms);
        for (index, (name, milliseconds)) in frame.per_pass_ms.iter().enumerate() {
            if per_pass.len() <= index {
                per_pass.push((name.clone(), 0.0));
            }
            per_pass[index].1 += milliseconds;
        }
    }

    let best = timings.iter().copied().fold(f32::INFINITY, f32::min);
    let mean = timings.iter().sum::<f32>() / timings.len() as f32;
    timings.sort_by(|a, b| a.total_cmp(b));
    let median = timings[timings.len() / 2];
    println!(
        "particles ({label}): {median:.3} ms median, {mean:.3} ms mean, {best:.3} ms best per \
         layer at {WIDTH}x{HEIGHT} ({PASSES} passes)"
    );
    for (name, milliseconds) in &per_pass {
        println!("  {name}: {:.3} ms", milliseconds / FRAMES as f32);
    }
    assert!(median.is_finite() && median > 0.0, "median {median} ms");
    median
}
