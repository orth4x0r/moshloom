//! The control window's two pictures of the show, on a real device and without a window: the
//! stage preview (the presented stack composited again at 96 px and read back) and the clip
//! thumbnails (one scene frame at 96x54 per look).
//!
//! Offscreen only: no swapchain, no present. What is measured is exactly what the render thread
//! adds after a stage frame when the control window asks for a picture. Run with `--nocapture`
//! for the numbers.

use std::time::Instant;

use mosh_app::editor::model::{catalog, new_clip};
use mosh_app::preview::{
    PREVIEW_FPS, PreviewTap, THUMB_SIZE, ThumbRequest, ThumbTap, has_thumbnail, thumb_frame,
    thumb_key,
};
use mosh_core::{BlendMode, ClipParams, LayerState};
use mosh_render::{FrameRecorder, Gpu, GpuPreference, LayerTarget};
use mosh_visuals::{LayerSlot, Visuals};

fn gpu() -> Gpu {
    Gpu::new(GpuPreference::Integrated)
        .or_else(|_| Gpu::new(GpuPreference::Discrete))
        .expect("a Vulkan device")
}

fn layer(index: usize, clip: u32, visualizer: &str, blend: BlendMode, opacity: f32) -> LayerState {
    LayerState {
        layer: index,
        clip: Some(clip),
        visualizer: Some(visualizer.to_string()),
        params: ClipParams {
            p: [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            palette: 0,
            seed: 1,
        },
        effects: Vec::new(),
        transition: None,
        blend,
        opacity,
        seed: 1,
    }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// How bright a picture is: the mean of its colour channels, 0..255.
fn mean_level(rgba: &[u8]) -> f64 {
    let sum: u64 = rgba
        .as_chunks::<4>().0.iter()
        .map(|pixel| u64::from(pixel[0]) + u64::from(pixel[1]) + u64::from(pixel[2]))
        .sum();
    sum as f64 / (rgba.len() / 4 * 3).max(1) as f64
}

#[test]
fn the_stage_preview_costs_a_number() {
    let gpu = gpu();
    // The app's shape at 1080p: layers at half size, composited to the output.
    let mut visuals = Visuals::new(&gpu, 960, 540).expect("visuals");
    let mut slots = [
        LayerSlot::new(&gpu, 0, 960, 540),
        LayerSlot::new(&gpu, 1, 960, 540),
    ];
    let states = [
        layer(0, 1, "tunnel", BlendMode::Normal, 1.0),
        layer(1, 2, "spectrum", BlendMode::Add, 0.6),
    ];
    let frame = thumb_frame();
    let mut tap = PreviewTap::new(&gpu, (1920, 1080)).expect("preview tap");
    assert_eq!(tap.size(), (96, 54));
    assert_eq!(tap.stage(), (1920, 1080));

    let mut times = Vec::new();
    let mut last = None;
    for index in 0..60 {
        // A stage frame: the layers are drawn and submitted, as `Stage::present` does first.
        let mut recorder = FrameRecorder::new(&gpu);
        for (slot, state) in slots.iter_mut().zip(&states) {
            visuals
                .draw_layer(&mut recorder, slot, state, &frame, 12.0 + index as f32 / 60.0)
                .expect("draw the layer");
        }
        let ticket = recorder.submit();
        gpu.wait_frame(ticket);

        // The preview: the same stack once more, small, and back to the CPU.
        let layers: Vec<(&LayerTarget, BlendMode, f32)> = slots
            .iter()
            .zip(&states)
            .map(|(slot, state)| (slot.output(), state.blend, state.opacity))
            .collect();
        let started = Instant::now();
        let picture = tap.capture(&gpu, &layers).expect("capture");
        times.push(started.elapsed().as_secs_f64() * 1_000.0);
        last = Some(picture);
    }
    let picture = last.expect("a preview");
    assert_eq!((picture.width, picture.height), (96, 54));
    assert_eq!(picture.rgba.len(), 96 * 54 * 4);
    assert!(picture.rgba.as_chunks::<4>().0.iter().all(|pixel| pixel[3] == 255));
    let level = mean_level(&picture.rgba);
    assert!(level > 4.0, "the preview is black (mean level {level:.2})");

    // The first captures include pipeline warm-up; the steady state is what a set pays.
    let mut steady = times[5..].to_vec();
    steady.sort_by(f64::total_cmp);
    let median = percentile(&steady, 0.5);
    let p95 = percentile(&steady, 0.95);
    println!(
        "stage preview, 2 layers at 960x540 -> 96x54, offscreen, {}: capture median {median:.3} ms, p95 {p95:.3} ms, first {:.3} ms; at {PREVIEW_FPS} fps that is {:.1} ms of render thread per second",
        gpu.adapter_name(),
        times[0],
        median * f64::from(PREVIEW_FPS)
    );
    // A 60 fps stage frame at 1080p leaves about 10 ms of its 16.7 ms slot unused; the capture
    // has to fit in that. The bound is a guard against it growing back (it was 22 ms at 256 px),
    // loose enough for a machine that is running other tests at the same time; the number above
    // is the result.
    assert!(
        median < 10.0,
        "a preview capture took {median:.3} ms: more than the slack a 60 fps frame leaves"
    );
    let messages = gpu.validation_messages();
    assert!(messages.is_empty(), "validation: {messages:?}");
}

#[test]
fn every_scene_gets_a_thumbnail() {
    let gpu = gpu();
    let mut visuals = Visuals::new(&gpu, 960, 540).expect("visuals");
    let mut tap = ThumbTap::new(&gpu);
    let mut times = Vec::new();
    let mut dark = Vec::new();
    let mut drawn = 0;
    for entry in catalog() {
        let clip = new_clip(1, entry.id);
        let request = ThumbRequest {
            key: thumb_key(entry.id, &clip.params),
            visualizer: entry.id.to_string(),
            params: clip.params,
        };
        let started = Instant::now();
        let picture = tap
            .render(&gpu, &mut visuals, &request)
            .unwrap_or_else(|error| panic!("{}: {error}", entry.id));
        let ms = started.elapsed().as_secs_f64() * 1_000.0;
        match picture {
            Some(picture) => {
                assert!(has_thumbnail(entry.id));
                assert_eq!((picture.width, picture.height), THUMB_SIZE);
                assert_eq!(
                    picture.rgba.len(),
                    (THUMB_SIZE.0 * THUMB_SIZE.1 * 4) as usize
                );
                drawn += 1;
                times.push(ms);
                if mean_level(&picture.rgba) < 1.0 {
                    dark.push(entry.id);
                }
            }
            None => assert!(!has_thumbnail(entry.id), "{} has no thumbnail", entry.id),
        }
    }
    times.sort_by(f64::total_cmp);
    println!(
        "clip thumbnails, {}x{}, offscreen: {drawn} drawn, median {:.3} ms, worst {:.3} ms each (once per look); dark: {dark:?}",
        THUMB_SIZE.0,
        THUMB_SIZE.1,
        percentile(&times, 0.5),
        times.last().copied().unwrap_or_default()
    );
    // A thumbnail is one frame with no history. Scenes that draw their history (Spectrum is the
    // mel history, Fluid an advected field) can be dark here; in the app they share the stage's
    // tables and are not. Stubs of other lanes may be black too. So dark ones are reported, and
    // the test requires that nearly all of the 16 web scenes are lit.
    let classic_dark: Vec<&str> = dark
        .iter()
        .copied()
        .filter(|id| {
            catalog()
                .iter()
                .any(|entry| entry.id == *id && entry.group == "Classic")
        })
        .collect();
    assert!(classic_dark.len() <= 2, "black thumbnails: {classic_dark:?}");
    assert!(drawn >= 16);
    let messages = gpu.validation_messages();
    assert!(messages.is_empty(), "validation: {messages:?}");
}
