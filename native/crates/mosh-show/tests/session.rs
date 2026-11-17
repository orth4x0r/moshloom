//! Session grid behaviour: quantized launches, scenes, follow actions, lock and autopilot.

mod common;

use common::*;
use mosh_core::{ControlEvent, EventFlags, FollowAction, LaunchQuantize, ParamRange, ShowMode};

fn bar() -> EventFlags {
    EventFlags::BAR
}

#[test]
fn launch_waits_for_next_bar() {
    let mut show = mosh_show::Show::new(project(
        ShowMode::Session,
        vec![clip(1), clip(2)],
        grid(&[&[Some(1)], &[Some(2)]], LaunchQuantize::Bar),
    ));

    // Nothing is playing in Session mode until something is launched.
    let plan = show.update(&frame(0.0, 2, EventFlags::empty()), &[]);
    assert_eq!(ids(&plan), vec![None]);

    // A launch mid-bar is queued, not applied.
    let plan = show.update(
        &frame(5.5, 2, EventFlags::empty()),
        &[ControlEvent::LaunchClip { layer: 0, scene: 1 }],
    );
    assert_eq!(ids(&plan), vec![None], "launch applied before the bar line");

    for ppq in [6.0, 6.5, 7.0, 7.5] {
        let plan = show.update(&frame(ppq, 2, EventFlags::empty()), &[]);
        assert_eq!(ids(&plan), vec![None], "launch applied at ppq {ppq}");
    }

    // The bar line at ppq 8 is the first boundary at or after 5.5.
    let plan = show.update(&frame(8.0, 2, bar()), &[]);
    assert_eq!(ids(&plan), vec![Some(2)], "launch did not land on the bar line");
}

#[test]
fn launch_on_the_bar_line_applies_immediately() {
    let mut show = mosh_show::Show::new(project(
        ShowMode::Session,
        vec![clip(1)],
        grid(&[&[Some(1)]], LaunchQuantize::Bar),
    ));
    let plan =
        show.update(&frame(0.0, 2, bar()), &[ControlEvent::LaunchClip { layer: 0, scene: 0 }]);
    assert_eq!(ids(&plan), vec![Some(1)]);
}

#[test]
fn scene_launch_sets_all_layers() {
    let mut show = mosh_show::Show::new(project(
        ShowMode::Session,
        (1..=6).map(clip).collect(),
        grid(
            &[&[Some(1), Some(2), Some(3)], &[Some(4), Some(5), Some(6)]],
            LaunchQuantize::Bar,
        ),
    ));

    let plan = show.update(&frame(0.0, 2, bar()), &[ControlEvent::LaunchScene(1)]);
    assert_eq!(ids(&plan), vec![Some(4), Some(5), Some(6)]);

    // An out-of-range scene is ignored, and leaves the layers alone.
    let plan = show.update(&frame(4.0, 2, bar()), &[ControlEvent::LaunchScene(2)]);
    assert_eq!(ids(&plan), vec![Some(4), Some(5), Some(6)]);

    // An empty cell in a scene leaves that layer playing what it already plays.
    let mut show = mosh_show::Show::new(project(
        ShowMode::Session,
        (1..=3).map(clip).collect(),
        grid(&[&[Some(1), Some(2)], &[Some(3), None]], LaunchQuantize::Bar),
    ));
    let plan = show.update(&frame(0.0, 2, bar()), &[ControlEvent::LaunchScene(0)]);
    assert_eq!(ids(&plan), vec![Some(1), Some(2)]);
    let plan = show.update(&frame(4.0, 2, bar()), &[ControlEvent::LaunchScene(1)]);
    assert_eq!(ids(&plan), vec![Some(3), Some(2)], "an empty cell must not stop a layer");
}

#[test]
fn follow_random_never_repeats_same_clip_twice() {
    let clips = vec![
        following(1, FollowAction::Random),
        following(2, FollowAction::Random),
        following(3, FollowAction::Random),
    ];
    let mut show = mosh_show::Show::new(project(
        ShowMode::Session,
        clips,
        grid(&[&[Some(1)], &[Some(2)], &[Some(3)]], LaunchQuantize::Bar),
    ));

    let plan =
        show.update(&frame(0.0, 2, bar()), &[ControlEvent::LaunchClip { layer: 0, scene: 0 }]);
    let mut last = ids(&plan)[0];
    assert_eq!(last, Some(1));

    for b in 1..=24u32 {
        let plan = show.update(&frame(b as f64 * 4.0, 2, bar()), &[]);
        let now = ids(&plan)[0];
        assert!(now.is_some(), "bar {b}: the follow action stopped the column");
        assert_ne!(now, last, "bar {b}: clip {last:?} was repeated by the follow action");
        last = now;
    }
}

#[test]
fn follow_next_walks_the_column() {
    let clips = vec![
        following(1, FollowAction::Next),
        following(2, FollowAction::Next),
        following(3, FollowAction::Next),
    ];
    let mut show = mosh_show::Show::new(project(
        ShowMode::Session,
        clips,
        grid(&[&[Some(1)], &[Some(2)], &[Some(3)]], LaunchQuantize::Bar),
    ));
    show.update(&frame(0.0, 2, bar()), &[ControlEvent::LaunchClip { layer: 0, scene: 0 }]);
    let plan = show.update(&frame(4.0, 2, bar()), &[]);
    assert_eq!(ids(&plan), vec![Some(2)]);
    let plan = show.update(&frame(8.0, 2, bar()), &[]);
    assert_eq!(ids(&plan), vec![Some(3)]);
    let plan = show.update(&frame(12.0, 2, bar()), &[]);
    assert_eq!(ids(&plan), vec![Some(1)], "next wraps around the column");
}

#[test]
fn procedural_is_reproducible_from_seed() {
    fn run(seed: u64) -> (u64, [f32; 8]) {
        let clips = vec![
            following(1, FollowAction::RandomProcedural),
            following(2, FollowAction::RandomProcedural),
        ];
        let mut show = mosh_show::Show::with_seed(
            project(
                ShowMode::Session,
                clips,
                grid(&[&[Some(1)], &[Some(2)]], LaunchQuantize::Bar),
            ),
            seed,
        );
        show.update(&frame(0.0, 2, bar()), &[ControlEvent::LaunchClip { layer: 0, scene: 0 }]);
        // The follow fires on the next bar line and draws a fresh look.
        let plan = show.update(&frame(4.0, 2, bar()), &[]);
        let layer = &plan.layers[0];
        assert_eq!(layer.clip, Some(2));
        (layer.seed, layer.params.p)
    }

    let (seed_a, params_a) = run(7);
    let (seed_b, params_b) = run(7);
    assert_eq!(seed_a, seed_b, "the same seed must draw the same clip seed");
    assert_eq!(params_a, params_b, "the same seed must draw the same parameters");

    let (_, params_c) = run(8);
    assert_ne!(params_a, params_c, "a different seed must draw a different look");

    // The seed recorded in LayerState reproduces the look, so it can be saved as a preset.
    assert_eq!(params_a, mosh_show::procedural::positions(&clip(2), 2, seed_a));
}

#[test]
fn build_holds_follow_until_drop() {
    let clips = vec![following(1, FollowAction::Next), clip(2)];
    let mut show = mosh_show::Show::new(project(
        ShowMode::Session,
        clips,
        grid(&[&[Some(1)], &[Some(2)]], LaunchQuantize::Bar),
    ));
    show.update(&frame(0.0, 2, bar()), &[ControlEvent::LaunchClip { layer: 0, scene: 0 }]);

    // Bar 1: a build starts, which holds the follow action back.
    let plan = show.update(&frame(4.0, 2, bar() | EventFlags::BUILD_START), &[]);
    assert_eq!(ids(&plan), vec![Some(1)]);

    // Bar 2 would be the follow's due bar; the build holds it.
    let plan = show.update(&frame(8.0, 2, bar()), &[]);
    assert_eq!(ids(&plan), vec![Some(1)], "the follow action ran during a build");

    // Bar 3: the drop releases it onto the downbeat.
    let plan = show.update(&frame(12.0, 2, bar() | EventFlags::DROP), &[]);
    assert_eq!(ids(&plan), vec![Some(2)], "the drop did not release the held follow action");
}

#[test]
fn build_releases_after_eight_bars_without_drop() {
    let clips = vec![following(1, FollowAction::Next), clip(2)];
    let mut show = mosh_show::Show::new(project(
        ShowMode::Session,
        clips,
        grid(&[&[Some(1)], &[Some(2)]], LaunchQuantize::Bar),
    ));
    show.update(&frame(0.0, 2, bar()), &[ControlEvent::LaunchClip { layer: 0, scene: 0 }]);
    show.update(&frame(4.0, 2, bar() | EventFlags::BUILD_START), &[]);

    // Bars 2..=8 of the build are held; the ninth boundary releases it.
    for b in 2..=8u32 {
        let plan = show.update(&frame(b as f64 * 4.0, 2, bar()), &[]);
        assert_eq!(ids(&plan), vec![Some(1)], "bar {b} of the build released the follow action");
    }
    let plan = show.update(&frame(36.0, 2, bar()), &[]);
    assert_eq!(ids(&plan), vec![Some(2)], "an 8-bar build never released the follow action");
}

#[test]
fn lock_blocks_automation_not_manual() {
    let clips = vec![following(1, FollowAction::Next), following(2, FollowAction::Next)];
    let mut show = mosh_show::Show::new(project(
        ShowMode::Session,
        clips,
        grid(&[&[Some(1)], &[Some(2)]], LaunchQuantize::Bar),
    ));
    show.update(&frame(0.0, 2, bar()), &[ControlEvent::LaunchClip { layer: 0, scene: 0 }]);

    show.update(&frame(4.0, 2, bar()), &[ControlEvent::Lock(true)]);
    for ppq in [8.0, 12.0, 16.0] {
        let plan = show.update(&frame(ppq, 2, bar()), &[]);
        assert_eq!(ids(&plan), vec![Some(1)], "lock let automation change the clip at {ppq}");
    }

    // A manual parameter move still wins under lock.
    let plan = show.update(
        &frame(20.0, 2, bar()),
        &[ControlEvent::SetParam { layer: 0, index: 3, value: 0.25 }],
    );
    assert_eq!(ids(&plan), vec![Some(1)]);
    assert_eq!(plan.layers[0].params.p[3], 0.25);

    // A manual launch still wins under lock, on the bar line.
    let plan =
        show.update(&frame(24.0, 2, bar()), &[ControlEvent::LaunchClip { layer: 0, scene: 1 }]);
    assert_eq!(ids(&plan), vec![Some(2)], "lock blocked a manual launch");
    assert_eq!(plan.layers[0].params.p[3], 0.5, "a new clip kept the old manual parameter");

    // Unlocking lets automation run again: the follow action that was due fires.
    let plan = show.update(&frame(28.0, 2, bar()), &[ControlEvent::Lock(false)]);
    assert_eq!(ids(&plan), vec![Some(1)], "unlock did not resume the follow action");
}

#[test]
fn autopilot_launches_on_drop_without_input() {
    let clips = vec![clip(1), clip(2)];
    let mut show = mosh_show::Show::new(project(
        ShowMode::Autopilot,
        clips,
        grid(&[&[Some(1)], &[Some(2)]], LaunchQuantize::Bar),
    ));
    assert_eq!(show.mode(), ShowMode::Autopilot);

    // Hands-free: autopilot starts the show without any ControlEvent.
    let plan = show.update(&frame(0.0, 2, bar()), &[]);
    assert_eq!(ids(&plan), vec![Some(1)]);

    // A drop off the downbeat waits for the next downbeat.
    let plan = show.update(&frame(6.5, 2, EventFlags::DROP), &[]);
    assert_eq!(ids(&plan), vec![Some(1)], "a mid-bar drop launched immediately");
    let plan = show.update(&frame(7.9, 2, EventFlags::empty()), &[]);
    assert_eq!(ids(&plan), vec![Some(1)], "a mid-bar drop launched before the downbeat");
    let plan = show.update(&frame(8.0, 2, bar()), &[]);
    assert_eq!(ids(&plan), vec![Some(2)], "the drop did not launch on the next downbeat");

    // A level up advances the scene too, at the next downbeat.
    let plan = show.update(&frame(13.0, 3, EventFlags::LEVEL_UP), &[]);
    assert_eq!(ids(&plan), vec![Some(2)], "a mid-bar level up launched immediately");
    let plan = show.update(&frame(16.0, 3, bar()), &[]);
    assert_eq!(ids(&plan), vec![Some(1)]);
}

#[test]
fn autopilot_respects_lock() {
    let clips = vec![clip(1), clip(2)];
    let mut show = mosh_show::Show::new(project(
        ShowMode::Autopilot,
        clips,
        grid(&[&[Some(1)], &[Some(2)]], LaunchQuantize::Bar),
    ));
    show.update(&frame(0.0, 2, bar()), &[ControlEvent::Lock(true)]);
    let plan = show.update(&frame(4.0, 3, bar() | EventFlags::DROP), &[]);
    assert_eq!(ids(&plan), vec![None], "autopilot launched while locked");
}

#[test]
fn select_song_launches_its_scene_in_session_mode() {
    let mut show = mosh_show::Show::new(project(
        ShowMode::Session,
        vec![clip(1), clip(2)],
        grid(&[&[Some(1)], &[Some(2)]], LaunchQuantize::Bar),
    ));
    let plan = show.update(&frame(0.0, 2, bar()), &[ControlEvent::SelectSong(1)]);
    assert_eq!(ids(&plan), vec![Some(2)]);
    assert_eq!(show.song(), Some(1));
}

#[test]
fn stop_layer_clears_the_layer() {
    let mut show = mosh_show::Show::new(project(
        ShowMode::Session,
        vec![clip(1)],
        grid(&[&[Some(1)]], LaunchQuantize::Bar),
    ));
    show.update(&frame(0.0, 2, bar()), &[ControlEvent::LaunchClip { layer: 0, scene: 0 }]);
    let plan = show.update(&frame(4.0, 2, bar()), &[ControlEvent::StopLayer(0)]);
    assert_eq!(ids(&plan), vec![None]);
    assert_eq!(plan.layers[0].visualizer, None);
}

#[test]
fn bias_picks_the_level_when_the_analysis_has_none() {
    // Level 1 pins p0 to 0, level 4 pins it to 1; only the bias can move it here.
    let mut c = clip(1);
    c.ranges[0][0] = ParamRange { lo: 0.0, hi: 0.0 };
    c.ranges[3][0] = ParamRange { lo: 1.0, hi: 1.0 };
    let mut show = mosh_show::Show::new(project(
        ShowMode::Session,
        vec![c],
        grid(&[&[Some(1)]], LaunchQuantize::Bar),
    ));
    show.update(&frame(0.0, 0, bar()), &[ControlEvent::LaunchClip { layer: 0, scene: 0 }]);
    let plan = show.update(&frame(4.0, 0, bar()), &[ControlEvent::IntensityBias(1.0)]);
    assert_eq!(plan.layers[0].params.p[0], 1.0, "a full bias did not reach level 4");
}
