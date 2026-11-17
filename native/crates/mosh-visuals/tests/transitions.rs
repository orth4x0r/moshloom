//! Task 2.3 gate, transitions half: a crossfade midpoint is the average, a cut is instant, and
//! strobe's full-frame flashes go through the shared 400 ms `FlashLimiter` — counted both from
//! the limiter and from the luminance of the pictures that were actually written.

mod common;

use std::time::{Duration, Instant};

use common::*;
use mosh_core::{ActiveTransition, AnalysisFrame, TransitionKind};
use mosh_render::{FlashLimiter, FrameRecorder, LayerTarget};
use mosh_visuals::transitions::TransitionPass;

const W: u32 = 128;
const H: u32 = 72;

fn transition(kind: TransitionKind, progress: f32) -> ActiveTransition {
    ActiveTransition {
        kind,
        progress,
        from: Some(1),
    }
}

/// Reads every pixel of `pixels` back as `expected` within `tolerance`.
fn assert_flat(pixels: &[u8], expected: [i32; 3], tolerance: i32, what: &str) {
    for (index, pixel) in pixels.chunks(4).enumerate() {
        for channel in 0..3 {
            let difference = i32::from(pixel[channel]) - expected[channel];
            assert!(
                difference.abs() <= tolerance,
                "{what}: pixel {index} channel {channel} is {} (expected {} +-{tolerance})",
                pixel[channel],
                expected[channel]
            );
        }
    }
}

#[test]
fn crossfade_midpoint_is_average() {
    let gpu = gpu();
    let mut transitions = TransitionPass::new(&gpu).expect("transitions");
    let solid = solid_pipeline(&gpu);
    let from = LayerTarget::new(&gpu, W, H);
    let src = LayerTarget::new(&gpu, W, H);
    let dst = LayerTarget::new(&gpu, W, H);
    draw_solid(&gpu, &solid, &from, [0.2, 0.2, 0.2, 1.0]);
    draw_solid(&gpu, &solid, &src, [0.8, 0.8, 0.8, 1.0]);

    {
        let mut recorder = FrameRecorder::new(&gpu);
        transitions
            .begin(&mut recorder, 0, &from)
            .expect("save the from picture");
        gpu.wait_frame(recorder.submit());
    }

    let frame = AnalysisFrame::default();
    let mut flash = FlashLimiter::new();
    let mut recorder = FrameRecorder::new(&gpu);
    let applied = transitions
        .apply(
            &mut recorder,
            0,
            &src,
            &dst,
            &transition(TransitionKind::Crossfade, 0.5),
            &frame,
            &mut flash,
        )
        .expect("crossfade");
    gpu.wait_frame(recorder.submit());
    assert!(applied, "a crossfade at progress 0.5 draws");

    let out = gpu.read_back(&dst);
    // smoothstep(0.5) is 0.5, so the midpoint is (0.2 + 0.8) / 2 = 0.5 -> 128 of 255.
    assert_flat(&out, [128, 128, 128], 2, "crossfade midpoint");
    println!(
        "crossfade 0.2 -> 0.8 at 0.5: mean luma {:.2}",
        mean_luma(&out)
    );
    assert!(
        flash.last_allowed().is_none(),
        "a crossfade is not a flash and must not touch the limiter"
    );
    assert_no_validation(&gpu);
}

#[test]
fn cut_is_instant() {
    let gpu = gpu();
    let mut transitions = TransitionPass::new(&gpu).expect("transitions");
    let solid = solid_pipeline(&gpu);
    let from = LayerTarget::new(&gpu, W, H);
    let src = LayerTarget::new(&gpu, W, H);
    let dst = LayerTarget::new(&gpu, W, H);
    draw_solid(&gpu, &solid, &from, [1.0, 0.0, 0.0, 1.0]);
    draw_solid(&gpu, &solid, &src, [0.0, 0.0, 1.0, 1.0]);

    {
        let mut recorder = FrameRecorder::new(&gpu);
        transitions
            .begin(&mut recorder, 0, &from)
            .expect("save the from picture");
        gpu.wait_frame(recorder.submit());
    }

    let frame = AnalysisFrame::default();
    let mut flash = FlashLimiter::new();
    // A cut shows the *new* picture on its first frame, however early its progress is.
    for progress in [0.0f32, 0.01, 0.99] {
        let mut recorder = FrameRecorder::new(&gpu);
        let applied = transitions
            .apply(
                &mut recorder,
                0,
                &src,
                &dst,
                &transition(TransitionKind::Cut, progress),
                &frame,
                &mut flash,
            )
            .expect("cut");
        gpu.wait_frame(recorder.submit());
        assert!(applied, "a cut draws at progress {progress}");
        let out = gpu.read_back(&dst);
        assert_flat(&out, [0, 0, 255], 1, &format!("cut at progress {progress}"));
    }
    println!("cut: the new picture from progress 0.0 on");
    assert_no_validation(&gpu);
}

/// Datamosh and Melt both hold the old picture at progress 0 (Datamosh displaced, Melt sagging)
/// and are the new picture by the end; a finished transition draws nothing, and neither does one
/// whose `begin` never ran.
#[test]
fn datamosh_and_melt_blend_and_finish() {
    let gpu = gpu();
    let mut transitions = TransitionPass::new(&gpu).expect("transitions");
    let solid = solid_pipeline(&gpu);
    let pattern = pattern_pipeline(&gpu);
    let from = LayerTarget::new(&gpu, W, H);
    let src = LayerTarget::new(&gpu, W, H);
    let dst = LayerTarget::new(&gpu, W, H);
    draw_pattern(&gpu, &pattern, &from);
    draw_solid(&gpu, &solid, &src, [0.1, 0.1, 0.1, 1.0]);
    let from_luma = mean_luma(&gpu.read_back(&from));
    let to_luma = mean_luma(&gpu.read_back(&src));

    {
        let mut recorder = FrameRecorder::new(&gpu);
        transitions
            .begin(&mut recorder, 0, &from)
            .expect("save the from picture");
        gpu.wait_frame(recorder.submit());
    }

    let frame = AnalysisFrame::default();
    let mut flash = FlashLimiter::new();
    for kind in [TransitionKind::Datamosh, TransitionKind::Melt] {
        let mut draw = |progress: f32| -> (bool, Vec<u8>) {
            let mut recorder = FrameRecorder::new(&gpu);
            let applied = transitions
                .apply(
                    &mut recorder,
                    0,
                    &src,
                    &dst,
                    &transition(kind, progress),
                    &frame,
                    &mut flash,
                )
                .unwrap_or_else(|error| panic!("{kind:?} at {progress}: {error}"));
            gpu.wait_frame(recorder.submit());
            (applied, gpu.read_back(&dst))
        };

        let (applied, start) = draw(0.0);
        assert!(applied, "{kind:?}: progress 0.0 draws the old picture");
        let (applied, nearly_done) = draw(0.9);
        assert!(applied, "{kind:?}: progress 0.9 draws");
        let start_luma = mean_luma(&start);
        let end_luma = mean_luma(&nearly_done);
        println!(
            "{kind:?}: progress 0.0 luma {start_luma:.1} (old {from_luma:.1}, \
             {} of {} pixels moved), progress 0.9 luma {end_luma:.1} (new {to_luma:.1})",
            differing_pixels(&start, &gpu.read_back(&from), 2),
            W * H
        );
        assert!(
            (start_luma - from_luma).abs() < 12.0,
            "{kind:?} at 0.0 is not the old picture ({start_luma:.1} vs {from_luma:.1})"
        );
        assert!(
            (end_luma - to_luma).abs() < (start_luma - to_luma).abs(),
            "{kind:?} at 0.9 is not closer to the new picture ({end_luma:.1} vs {to_luma:.1})"
        );

        // A finished transition leaves the layer to show `src` itself.
        let before = gpu.read_back(&dst);
        let mut recorder = FrameRecorder::new(&gpu);
        let applied = transitions
            .apply(
                &mut recorder,
                0,
                &src,
                &dst,
                &transition(kind, 1.0),
                &frame,
                &mut flash,
            )
            .expect("finished transition");
        gpu.wait_frame(recorder.submit());
        assert!(!applied, "{kind:?} at progress 1.0 must draw nothing");
        assert_eq!(
            differing_pixels(&before, &gpu.read_back(&dst), 0),
            0,
            "{kind:?} at progress 1.0 wrote to the destination"
        );
    }

    // Datamosh displaces the old picture; Melt at progress 0 does not.
    let mut recorder = FrameRecorder::new(&gpu);
    transitions
        .apply(
            &mut recorder,
            0,
            &src,
            &dst,
            &transition(TransitionKind::Datamosh, 0.0),
            &frame,
            &mut flash,
        )
        .expect("datamosh");
    gpu.wait_frame(recorder.submit());
    let displaced = differing_pixels(&gpu.read_back(&dst), &gpu.read_back(&from), 2);
    assert!(
        displaced * 100 >= (W * H) as usize,
        "datamosh displaced only {displaced} of {} pixels of the old picture",
        W * H
    );

    // No saved picture (a layer whose clip never changed): there is nothing to blend from.
    let mut recorder = FrameRecorder::new(&gpu);
    let applied = transitions
        .apply(
            &mut recorder,
            5,
            &src,
            &dst,
            &transition(TransitionKind::Crossfade, 0.5),
            &frame,
            &mut flash,
        )
        .expect("a transition with no from picture");
    gpu.wait_frame(recorder.submit());
    assert!(!applied, "a transition without begin must not draw");
    println!("datamosh displacement: {displaced} of {} pixels", W * H);
    assert_no_validation(&gpu);
}

/// One second of 60 fps frames asking for 16 flashes per second: the limiter allows at most three
/// flash starts, and the read-back shows at most three luminance onsets.
#[test]
fn strobe_respects_flash_limit() {
    let gpu = gpu();
    let mut transitions = TransitionPass::new(&gpu).expect("transitions");
    let solid = solid_pipeline(&gpu);
    let from = LayerTarget::new(&gpu, W, H);
    let src = LayerTarget::new(&gpu, W, H);
    let dst = LayerTarget::new(&gpu, W, H);
    // White on both sides: the only thing that can change the picture is the flash itself.
    draw_solid(&gpu, &solid, &from, [1.0, 1.0, 1.0, 1.0]);
    draw_solid(&gpu, &solid, &src, [1.0, 1.0, 1.0, 1.0]);
    let neutral = mean_luma(&gpu.read_back(&from));

    {
        let mut recorder = FrameRecorder::new(&gpu);
        transitions
            .begin(&mut recorder, 0, &from)
            .expect("save the from picture");
        gpu.wait_frame(recorder.submit());
    }

    let frame = AnalysisFrame::default();
    let mut flash = FlashLimiter::new();
    let frame_budget = Duration::from_micros(16_666);
    let start = Instant::now();
    let mut last_allowed = flash.last_allowed();
    let mut starts = 0usize;
    let mut onsets = 0usize;
    let mut flashing_before = false;
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;

    for index in 0..60 {
        let frame_start = Instant::now();
        {
            let mut recorder = FrameRecorder::new(&gpu);
            let applied = transitions
                .apply(
                    &mut recorder,
                    0,
                    &src,
                    &dst,
                    &transition(TransitionKind::Strobe, 0.25),
                    &frame,
                    &mut flash,
                )
                .expect("strobe");
            assert!(applied, "frame {index}: a strobe at progress 0.25 draws");
            gpu.wait_frame(recorder.submit());
        }
        let luma = mean_luma(&gpu.read_back(&dst));
        // Count flash starts the limiter allowed...
        if flash.last_allowed() != last_allowed {
            starts += 1;
            last_allowed = flash.last_allowed();
        }
        // ... and flash starts the picture shows, as a departure from the neutral luma.
        let flashing = (luma - neutral).abs() > 40.0;
        if flashing && !flashing_before {
            onsets += 1;
        }
        flashing_before = flashing;
        lo = lo.min(luma);
        hi = hi.max(luma);
        if let Some(rest) = frame_budget.checked_sub(frame_start.elapsed()) {
            std::thread::sleep(rest);
        }
    }
    let elapsed = start.elapsed();
    println!(
        "strobe at 16 asks/s: {starts} flash starts from the limiter, {onsets} luminance onsets, \
         luma {} .. {} over {:.3} s ({} frames)",
        lo.round(),
        hi.round(),
        elapsed.as_secs_f32(),
        60
    );
    assert!(starts >= 1, "the first flash is always allowed");
    assert!(
        starts <= 3,
        "{starts} flash starts in {:.3} s: the 400 ms floor allows at most 3",
        elapsed.as_secs_f32()
    );
    assert!(
        onsets <= 3,
        "{onsets} luminance onsets in {:.3} s: at most 3 flash starts reached the picture",
        elapsed.as_secs_f32()
    );
    assert!(
        hi - lo > 40.0,
        "the strobe never changed the picture (luma {} .. {})",
        lo.round(),
        hi.round()
    );
    assert_no_validation(&gpu);
}
