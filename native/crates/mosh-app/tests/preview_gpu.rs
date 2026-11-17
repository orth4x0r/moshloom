//! Clip thumbnails on a real device, without a window: what one costs the frame that carries
//! it, and that every scene gets a picture.
//!
//! In the app a thumbnail is one more scene pass at the end of a stage frame's own recorder,
//! read back by `mosh_render::AsyncReadback` inside the present submission. There is no stage
//! here, so the readback is driven through its offscreen path (`record_target`, the same
//! commands in a submission of its own); the scene pass is recorded exactly as the app records
//! it. Run with `--release -- --nocapture` for the numbers.

use std::time::{Duration, Instant};

use mosh_app::editor::model::{catalog, new_clip};
use mosh_app::preview::{THUMB_SIZE, ThumbQueue, ThumbRequest, has_thumbnail, thumb_frame, thumb_key};
use mosh_core::{AnalysisFrame, BlendMode, ClipParams, LayerState};
use mosh_render::{AsyncReadback, FrameRecorder, Gpu, GpuPreference, LayerTarget};
use mosh_visuals::{LayerSlot, Visuals};

fn gpu() -> Gpu {
    Gpu::new(GpuPreference::Integrated)
        .or_else(|_| Gpu::new(GpuPreference::Discrete))
        .expect("a Vulkan device")
}

fn live_frame(index: u32) -> AnalysisFrame {
    AnalysisFrame {
        time_s: 12.0 + f64::from(index) / 60.0,
        ppq: 16.0 + f64::from(index) / 30.0,
        beat_phase: 0.4,
        bar_phase: 0.3,
        tempo_bpm: 120.0,
        playing: true,
        bands: [0.5, 0.4, 0.3],
        mel: [0.4; 24],
        loudness: 0.5,
        intensity: 0.5,
        level: 2,
        ..AnalysisFrame::default()
    }
}

fn median_ms(values: &mut [Duration]) -> f64 {
    values.sort();
    values[values.len() / 2].as_secs_f64() * 1_000.0
}

/// How bright a picture is: the mean of its colour channels, 0..255.
fn mean_level(rgba: &[u8]) -> f64 {
    let sum: u64 = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pixel| u64::from(pixel[0]) + u64::from(pixel[1]) + u64::from(pixel[2]))
        .sum();
    sum as f64 / (rgba.len() / 4 * 3).max(1) as f64
}

/// A show opens with every scene in it: each stage frame carries one thumbnail, and what that
/// adds to the frame's own CPU time (recording one scene pass and the readback's blits, and
/// picking up the picture recorded two frames ago) stays far inside the 2 ms a frame has.
#[test]
fn a_thumbnail_rides_in_a_frame_for_a_fraction_of_its_budget() {
    let gpu = gpu();
    let mut visuals = Visuals::new(&gpu, 960, 540).expect("visuals");
    let mut slot = LayerSlot::new(&gpu, 0, 960, 540);
    let stage_layer = LayerState {
        layer: 0,
        clip: Some(1),
        visualizer: Some("tunnel".to_string()),
        params: ClipParams {
            p: [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            palette: 0,
            seed: 1,
        },
        effects: Vec::new(),
        transition: None,
        blend: BlendMode::Normal,
        opacity: 1.0,
        seed: 1,
    };
    let target = LayerTarget::new_single(&gpu, THUMB_SIZE.0, THUMB_SIZE.1);
    let mut readback = AsyncReadback::new(&gpu, THUMB_SIZE.0, THUMB_SIZE.1).expect("readback");

    // Every look the browser shows, queued in one go, as when a show opens.
    let mut queue = ThumbQueue::new();
    let mut names = std::collections::HashMap::new();
    for entry in catalog() {
        let clip = new_clip(1, entry.id);
        if !has_thumbnail(entry.id) {
            continue;
        }
        let key = thumb_key(entry.id, &clip.params);
        names.insert(key, entry.id);
        queue.push(ThumbRequest {
            key,
            visualizer: entry.id.to_string(),
            params: clip.params,
        });
    }
    let asked = queue.len();
    assert!(asked >= 16, "the 16 web scenes at least: {asked}");

    let allocations = gpu.allocation_count();
    let (mut plain, mut carrying) = (Vec::new(), Vec::new());
    let mut pictures = Vec::new();
    let mut drawn_in_frame = Vec::new();
    for index in 0..200u32 {
        let live = live_frame(index);
        let t = live.time_s as f32;
        let request = queue.next_frame();
        drawn_in_frame.push(request.is_some());

        // The frame's own CPU work: pick up a finished picture, record the layer, and (in a
        // frame that carries one) record the thumbnail's scene pass.
        let started = Instant::now();
        let ready = readback.begin_frame();
        let mut recorder = FrameRecorder::new(&gpu);
        visuals
            .draw_layer(&mut recorder, &mut slot, &stage_layer, &live, t)
            .expect("the stage's layer");
        if let Some(request) = request.as_ref() {
            visuals
                .scenes
                .draw(
                    &mut recorder,
                    &target,
                    &request.visualizer,
                    &thumb_frame(&live),
                    &request.params,
                    t,
                )
                .unwrap_or_else(|error| panic!("{}: {error}", request.visualizer));
        }
        let mut cpu = started.elapsed();
        gpu.wait_frame(recorder.submit());
        // In the app the readback's commands go into the present submission; here they get one
        // of their own, and only the time spent recording them is the frame's.
        if let Some(request) = request.as_ref() {
            cpu += readback
                .record_target(&gpu, &target, request.key)
                .expect("record the readback");
            carrying.push(cpu);
        } else {
            plain.push(cpu);
        }
        if let Some(picture) = ready {
            pictures.push(picture);
        }
    }

    // One per frame, none in the warm-up, all of them delivered, nothing allocated for them.
    let first = drawn_in_frame.iter().position(|drawn| *drawn).expect("a thumbnail");
    assert_eq!(first as u32, mosh_app::preview::THUMB_WARMUP_FRAMES);
    assert_eq!(drawn_in_frame.iter().filter(|drawn| **drawn).count(), asked);
    assert!(drawn_in_frame[first..first + asked].iter().all(|drawn| *drawn));
    assert_eq!(pictures.len(), asked);
    assert_eq!(gpu.allocation_count(), allocations, "no allocation per thumbnail");

    let mut dark = Vec::new();
    for picture in &pictures {
        assert_eq!((picture.width, picture.height), THUMB_SIZE);
        assert_eq!(picture.rgba.len(), (THUMB_SIZE.0 * THUMB_SIZE.1 * 4) as usize);
        if mean_level(&picture.rgba) < 1.0 {
            dark.push(names[&picture.tag]);
        }
    }

    let worst = carrying.iter().max().copied().unwrap_or_default().as_secs_f64() * 1_000.0;
    let (plain, carrying) = (median_ms(&mut plain), median_ms(&mut carrying));
    println!(
        "clip thumbnails, {}x{}, {}: {asked} drawn one per frame; frame CPU median {plain:.3} ms without, {carrying:.3} ms carrying one (+{:.3} ms), worst frame {worst:.3} ms; dark: {dark:?}",
        THUMB_SIZE.0,
        THUMB_SIZE.1,
        gpu.adapter_name(),
        carrying - plain
    );
    // The budget is 2 ms for the whole frame; the thumbnail's share has to be a small part of
    // it. (An unoptimised build records several times slower; the bound is for the shipped one.)
    let budget = if cfg!(debug_assertions) { 4.0 } else { 1.0 };
    assert!(
        carrying - plain < budget,
        "a thumbnail added {:.3} ms to a frame",
        carrying - plain
    );

    // A thumbnail is one frame with no history of its own. A scene that draws the spectrum's
    // history (Spectrum) shows what the stage's audio has been; stubs of other lanes may still
    // be black. So dark ones are reported, and nearly all of the 16 web scenes must be lit.
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
    let messages = gpu.validation_messages();
    assert!(messages.is_empty(), "validation: {messages:?}");
}
