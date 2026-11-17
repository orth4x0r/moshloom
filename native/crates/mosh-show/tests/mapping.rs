//! Intensity-level ranges, the beat pulse and manual parameter moves.

mod common;

use common::*;
use mosh_core::{ControlEvent, EventFlags, LaunchQuantize, ParamRange, ShowMode};

#[test]
fn level_selects_param_range() {
    let mut c = clip(1);
    c.params.p = [0.5; 8];
    // Level 1 squeezes parameter 0 to 0, level 4 opens it to 1, position 0.5 lands midway.
    c.ranges[0][0] = ParamRange { lo: 0.0, hi: 0.2 };
    c.ranges[3][0] = ParamRange { lo: 0.8, hi: 1.0 };
    c.ranges[1][0] = ParamRange { lo: 0.4, hi: 0.6 };

    let mut show = mosh_show::Show::new(project(
        ShowMode::Session,
        vec![c],
        grid(&[&[Some(1)]], LaunchQuantize::Bar),
    ));
    show.update(&frame(0.0, 1, EventFlags::BAR), &[ControlEvent::LaunchClip { layer: 0, scene: 0 }]);

    let plan = show.update(&frame(4.0, 1, EventFlags::empty()), &[]);
    assert!((plan.layers[0].params.p[0] - 0.1).abs() < 1e-6, "level 1 range");

    let plan = show.update(&frame(8.0, 2, EventFlags::empty()), &[]);
    assert!((plan.layers[0].params.p[0] - 0.5).abs() < 1e-6, "level 2 range");

    let plan = show.update(&frame(12.0, 4, EventFlags::empty()), &[]);
    assert!((plan.layers[0].params.p[0] - 0.9).abs() < 1e-6, "level 4 range");

    // Parameters the clip does not constrain keep their authored position.
    assert!((plan.layers[0].params.p[1] - 0.5).abs() < 1e-6);
}

#[test]
fn beat_pulse_moves_params_within_the_range() {
    let mut c = clip(1);
    c.params.p = [0.5; 8];
    c.pulse_depth = [0.25; 4];

    let mut show = mosh_show::Show::new(project(
        ShowMode::Session,
        vec![c],
        grid(&[&[Some(1)]], LaunchQuantize::Bar),
    ));
    show.update(&frame(0.0, 2, EventFlags::BAR), &[ControlEvent::LaunchClip { layer: 0, scene: 0 }]);

    // On the beat the pulse is 1, halfway through it is exp(-2).
    let on_beat = show.update(&frame(4.0, 2, EventFlags::BAR), &[]).layers[0].params.p[0];
    let mid_beat = show.update(&frame(4.5, 2, EventFlags::empty()), &[]).layers[0].params.p[0];
    assert!((on_beat - (0.5 + 0.25)).abs() < 1e-6, "pulse peak was {on_beat}");
    let want = 0.5 + 0.25 * (-2.0f32).exp();
    assert!((mid_beat - want).abs() < 1e-6, "pulse decay was {mid_beat}, want {want}");
    assert!(on_beat > mid_beat, "the pulse must decay through the beat");
}

#[test]
fn params_are_clamped_to_the_range() {
    let mut c = clip(1);
    c.params.p = [0.9; 8];
    c.pulse_depth = [0.5; 4];
    c.ranges[1][0] = ParamRange { lo: 0.0, hi: 0.4 };

    let mut show = mosh_show::Show::new(project(
        ShowMode::Session,
        vec![c],
        grid(&[&[Some(1)]], LaunchQuantize::Bar),
    ));
    show.update(&frame(0.0, 2, EventFlags::BAR), &[ControlEvent::LaunchClip { layer: 0, scene: 0 }]);
    let plan = show.update(&frame(4.0, 2, EventFlags::empty()), &[]);
    // base 0.9 + 0.5 pulse clamps to 1.0, the top of the level's range.
    assert!((plan.layers[0].params.p[0] - 0.4).abs() < 1e-6);
}

#[test]
fn transition_runs_for_its_beats() {
    let a = clip(1);
    let mut b = clip(2);
    b.transition_in = mosh_core::Transition { kind: mosh_core::TransitionKind::Crossfade, beats: 4.0 };
    let mut show = mosh_show::Show::new(project(
        ShowMode::Session,
        vec![a, b],
        grid(&[&[Some(1)], &[Some(2)]], LaunchQuantize::Bar),
    ));
    show.update(&frame(0.0, 2, EventFlags::BAR), &[ControlEvent::LaunchClip { layer: 0, scene: 0 }]);
    // A Cut by default, and nothing to cut from: no transition on the first clip.
    let plan = show.update(&frame(4.0, 2, EventFlags::BAR), &[]);
    assert_eq!(plan.layers[0].transition, None, "an empty layer needs no transition");

    let plan =
        show.update(&frame(8.0, 2, EventFlags::BAR), &[ControlEvent::LaunchClip { layer: 0, scene: 1 }]);
    let t = plan.layers[0].transition.expect("a crossfade was asked for");
    assert_eq!(t.from, Some(1));
    assert_eq!(t.kind, mosh_core::TransitionKind::Crossfade);
    assert_eq!(t.progress, 0.0);

    let plan = show.update(&frame(10.0, 2, EventFlags::empty()), &[]);
    let t = plan.layers[0].transition.expect("the transition ended early");
    assert!((t.progress - 0.5).abs() < 1e-6, "progress was {}", t.progress);

    let plan = show.update(&frame(12.0, 2, EventFlags::BAR), &[]);
    assert_eq!(plan.layers[0].transition, None, "the transition outlived its beats");
}

#[test]
fn flash_is_requested_on_a_drop_downbeat_and_on_a_strobe() {
    // A drop mid-bar is not a flash; a drop on the downbeat is.
    let mut show = mosh_show::Show::new(project(
        ShowMode::Session,
        vec![clip(1)],
        grid(&[&[Some(1)]], LaunchQuantize::Bar),
    ));
    show.update(&frame(0.0, 2, EventFlags::BAR), &[ControlEvent::LaunchClip { layer: 0, scene: 0 }]);
    let plan = show.update(&frame(2.0, 2, EventFlags::DROP), &[]);
    assert!(!plan.flash_requested, "a mid-bar drop flashed");
    let plan = show.update(&frame(4.0, 2, EventFlags::BAR | EventFlags::DROP), &[]);
    assert!(plan.flash_requested, "a drop on the downbeat did not ask for a flash");
    let plan = show.update(&frame(6.0, 2, EventFlags::empty()), &[]);
    assert!(!plan.flash_requested, "the flash request outlived its frame");

    // A strobe transition asks for one as it starts.
    let mut strobe = clip(2);
    strobe.transition_in =
        mosh_core::Transition { kind: mosh_core::TransitionKind::Strobe, beats: 2.0 };
    let mut show = mosh_show::Show::new(project(
        ShowMode::Session,
        vec![clip(1), strobe],
        grid(&[&[Some(1)], &[Some(2)]], LaunchQuantize::Bar),
    ));
    show.update(&frame(0.0, 2, EventFlags::BAR), &[ControlEvent::LaunchClip { layer: 0, scene: 0 }]);
    let plan = show.update(&frame(4.0, 2, EventFlags::BAR), &[]);
    assert!(!plan.flash_requested, "no strobe yet");
    let plan =
        show.update(&frame(8.0, 2, EventFlags::BAR), &[ControlEvent::LaunchClip { layer: 0, scene: 1 }]);
    assert!(plan.flash_requested, "a strobe transition did not ask for a flash");
}
