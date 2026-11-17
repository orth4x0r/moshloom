//! Task 2.3 gate, effects half: the eight-slot pass is the identity at amount 0, moves pixels at
//! amount 1, refuses to draw when there is nothing to apply, and costs what it costs.

mod common;

use common::*;
use mosh_core::{AnalysisFrame, EffectKind, EffectSlot};
use mosh_render::{FrameRecorder, LayerTarget};
use mosh_visuals::effects::EffectChain;

/// Small on purpose: the lane's tests run beside other lanes' on one machine.
const W: u32 = 128;
const H: u32 = 72;

/// Every effect this lane owns, in the order the web app names its uniform components.
const ALL: [EffectKind; 8] = [
    EffectKind::Fold,
    EffectKind::Mirror,
    EffectKind::Streak,
    EffectKind::Crush,
    EffectKind::Zoom,
    EffectKind::Spin,
    EffectKind::Echo,
    EffectKind::Negative,
];

fn slot(kind: EffectKind, amount: f32, param: f32) -> EffectSlot {
    EffectSlot {
        kind,
        amount,
        param,
    }
}

/// The picture after one effects draw: record, submit, read back.
fn apply_once(
    chain: &mut EffectChain,
    gpu: &mosh_render::Gpu,
    source: &LayerTarget,
    dest: &LayerTarget,
    slots: &[EffectSlot],
) -> (bool, Vec<u8>) {
    let frame = AnalysisFrame::default();
    let mut recorder = FrameRecorder::new(gpu);
    let applied = chain
        .apply(&mut recorder, source, dest, slots, &frame, 0.0)
        .expect("effects draw");
    gpu.wait_frame(recorder.submit());
    (applied, gpu.read_back(dest))
}

#[test]
fn each_effect_at_zero_is_identity() {
    let gpu = gpu();
    let mut chain = EffectChain::new(&gpu).expect("effects");
    let source = LayerTarget::new(&gpu, W, H);
    let dest = LayerTarget::new(&gpu, W, H);
    draw_pattern(&gpu, &pattern_pipeline(&gpu), &source);

    // The reference chain has one effect at full amount. Every other effect, added at amount 0,
    // has to leave it byte for byte alone: "zero is the identity" is checked in a chain that is
    // actually drawing, not in the early return.
    let reference_slots = [slot(EffectKind::Fold, 1.0, 0.3)];
    let (_, reference) = apply_once(&mut chain, &gpu, &source, &dest, &reference_slots);
    println!("reference (fold at 1.0): luma {:.1}", mean_luma(&reference));

    for kind in ALL {
        let slots = [slot(kind, 0.0, 0.7), reference_slots[0]];
        let (applied, out) = apply_once(&mut chain, &gpu, &source, &dest, &slots);
        assert!(applied, "{kind:?} at 0 beside fold at 1.0 still draws");
        let differing = differing_pixels(&reference, &out, 1);
        println!(
            "{kind:?} at amount 0: {differing} of {} pixels differ",
            W * H
        );
        assert_eq!(
            differing, 0,
            "{kind:?} at amount 0 moved {differing} pixels of the reference chain"
        );
    }
    assert_no_validation(&gpu);
}

#[test]
fn each_effect_changes_pixels_at_full() {
    let gpu = gpu();
    let mut chain = EffectChain::new(&gpu).expect("effects");
    let source = LayerTarget::new(&gpu, W, H);
    let dest = LayerTarget::new(&gpu, W, H);
    draw_pattern(&gpu, &pattern_pipeline(&gpu), &source);
    let source_pixels = gpu.read_back(&source);

    for kind in ALL {
        let slots = [slot(kind, 1.0, 0.5)];
        // Two frames: the second is the one echo needs, because its history is the first.
        let (_, first) = apply_once(&mut chain, &gpu, &source, &dest, &slots);
        let (_, second) = apply_once(&mut chain, &gpu, &source, &dest, &slots);
        let differing = differing_pixels(&source_pixels, &second, 2);
        println!(
            "{kind:?} at amount 1.0: {differing} of {} pixels differ (mean {:.1}, frame 1 mean {:.1})",
            W * H,
            mean_difference(&source_pixels, &second),
            mean_luma(&first),
        );
        assert!(
            differing * 20 >= (W * H) as usize,
            "{kind:?} at amount 1.0 changed only {differing} of {} pixels",
            W * H
        );
    }
    assert_no_validation(&gpu);
}

#[test]
fn effects_return_false_when_empty() {
    let gpu = gpu();
    let mut chain = EffectChain::new(&gpu).expect("effects");
    let source = LayerTarget::new(&gpu, W, H);
    let dest = LayerTarget::new(&gpu, W, H);
    draw_pattern(&gpu, &pattern_pipeline(&gpu), &source);

    // Paint the destination so a refused apply is visible if it writes anyway.
    draw_solid(&gpu, &solid_pipeline(&gpu), &dest, [0.25, 0.5, 0.75, 1.0]);
    let painted = gpu.read_back(&dest);

    let frame = AnalysisFrame::default();
    let mut recorder = FrameRecorder::new(&gpu);
    for (what, slots) in [
        ("no slots", Vec::new()),
        ("one zeroed slot", vec![slot(EffectKind::Fold, 0.0, 0.5)]),
        (
            "two zeroed slots",
            vec![
                slot(EffectKind::Fold, 0.0, 0.5),
                slot(EffectKind::Echo, 0.0, 0.0),
            ],
        ),
        // Mosh is the mosh lane's job, not this one's.
        ("a mosh slot alone", vec![slot(EffectKind::Mosh, 1.0, 0.5)]),
    ] {
        let applied = chain
            .apply(&mut recorder, &source, &dest, &slots, &frame, 0.0)
            .expect("empty effects draw");
        assert!(!applied, "{what}: apply must report nothing to do");
    }
    gpu.wait_frame(recorder.submit());
    let after = gpu.read_back(&dest);
    assert_eq!(
        differing_pixels(&painted, &after, 0),
        0,
        "a refused apply wrote to the destination"
    );

    // ... and the same call with a live amount does write, so the check above is not vacuous.
    let (applied, out) = apply_once(
        &mut chain,
        &gpu,
        &source,
        &dest,
        &[slot(EffectKind::Fold, 1.0, 0.5)],
    );
    assert!(applied, "a fold at 1.0 draws");
    assert!(
        differing_pixels(&painted, &out, 1) > 0,
        "the fold wrote nothing"
    );
    println!(
        "empty apply left all {} pixels of the painted destination alone; a live fold changed {}",
        W * H,
        differing_pixels(&painted, &out, 1)
    );
    assert_no_validation(&gpu);
}

/// The number the plan asks for: GPU ms of the whole lane, all eight effects at once, at the
/// layer size the reference project uses (`internal_scale` 0.5 of 1920x1080).
#[test]
fn effect_pass_gpu_ms_at_960x540() {
    let gpu = gpu();
    let mut chain = EffectChain::new(&gpu).expect("effects");
    let source = LayerTarget::new(&gpu, 960, 540);
    let dest = LayerTarget::new(&gpu, 960, 540);
    draw_pattern(&gpu, &pattern_pipeline(&gpu), &source);
    let slots: Vec<EffectSlot> = ALL.iter().map(|kind| slot(*kind, 1.0, 0.5)).collect();

    let mut best = f32::MAX;
    let mut total = 0.0f32;
    let mut frames = 0;
    for index in 0..5 {
        let frame = AnalysisFrame::default();
        let mut recorder = FrameRecorder::new(&gpu);
        assert!(
            chain
                .apply(&mut recorder, &source, &dest, &slots, &frame, 0.0)
                .expect("effects draw"),
            "the measured draw must not be skipped"
        );
        let timings = gpu.wait_frame(recorder.submit());
        let milliseconds = timings
            .per_pass_ms
            .iter()
            .find(|(name, _)| name == "fx_effects")
            .map(|(_, ms)| *ms)
            .unwrap_or(timings.frame_ms);
        println!(
            "frame {index}: fx_effects {milliseconds:.4} ms (frame {:.4} ms, passes {:?})",
            timings.frame_ms,
            timings
                .per_pass_ms
                .iter()
                .map(|(name, ms)| format!("{name} {ms:.4}"))
                .collect::<Vec<_>>()
        );
        if milliseconds > 0.0 {
            best = best.min(milliseconds);
            total += milliseconds;
            frames += 1;
        }
    }
    println!(
        "fx_effects, 8 effects at 960x540: best {best:.4} ms, mean {:.4} ms over {frames} frame(s)",
        if frames > 0 {
            total / frames as f32
        } else {
            0.0
        }
    );
    assert!(
        frames > 0 && best.is_finite() && best > 0.0,
        "the device reported no GPU timestamp for the effect pass"
    );
    assert_no_validation(&gpu);
}

/// The lane as the app drives it: `Visuals::draw_layer` runs scene -> mosh -> effects -> transition
/// and hands the transition its "from" picture on the frame a clip changes. The scenes and the
/// mosh pass are other lanes' stubs today, so this asserts the chain and its hazards (the effects
/// pass reads its own destination as echo history; the transition reads its saved copy while
/// writing a different target), not the picture.
#[test]
fn facade_runs_effects_and_a_transition() {
    use mosh_core::{ActiveTransition, BlendMode, ClipParams, LayerState, TransitionKind};
    use mosh_visuals::{LayerSlot, Stage, Visuals};

    let gpu = gpu();
    let mut visuals = Visuals::new(&gpu, W, H).expect("visuals");
    let mut layer_slot = LayerSlot::new(&gpu, 0, W, H);
    let frame = AnalysisFrame::default();
    let state = |clip: u32, transition: TransitionKind, progress: f32| LayerState {
        layer: 0,
        clip: Some(clip),
        visualizer: Some("tunnel".to_string()),
        params: ClipParams::default(),
        effects: ALL.iter().map(|kind| slot(*kind, 1.0, 0.5)).collect(),
        transition: Some(ActiveTransition {
            kind: transition,
            progress,
            from: Some(clip.saturating_sub(1)),
        }),
        blend: BlendMode::Normal,
        opacity: 1.0,
        seed: 0,
    };

    // Frame 1: the first clip, no "from" picture yet.
    {
        let mut recorder = FrameRecorder::new(&gpu);
        visuals
            .draw_layer(
                &mut recorder,
                &mut layer_slot,
                &state(1, TransitionKind::Crossfade, 0.25),
                &frame,
                0.0,
            )
            .expect("first frame");
        gpu.wait_frame(recorder.submit());
    }
    let first_stage = layer_slot.output_stage;
    // Frame 2: a clip change, so `begin` saves frame 1's picture and the crossfade blends it.
    {
        let mut recorder = FrameRecorder::new(&gpu);
        visuals
            .draw_layer(
                &mut recorder,
                &mut layer_slot,
                &state(2, TransitionKind::Crossfade, 0.5),
                &frame,
                0.0,
            )
            .expect("second frame");
        gpu.wait_frame(recorder.submit());
    }
    println!(
        "facade: frame 1 ended at {first_stage:?}, frame 2 at {:?} ({} bytes read back)",
        layer_slot.output_stage,
        gpu.read_back(layer_slot.output()).len()
    );
    assert_eq!(
        layer_slot.output_stage,
        Stage::Transition,
        "the transition did not draw"
    );
    assert_no_validation(&gpu);
}
