//! Arrangement behaviour: the playhead owns the picture, a Session launch takes a layer over.

mod common;

use common::*;
use mosh_core::{ControlEvent, EventFlags, LaunchQuantize, ShowMode};

fn none() -> EventFlags {
    EventFlags::empty()
}

/// Lane 0: clip 1 in bars 0..2, clip 2 in bars 2..4, nothing after ppq 16.
fn arrangement_show() -> mosh_show::Show {
    let p = with_lane0(
        project(
            ShowMode::Arrangement,
            vec![clip(1), clip(2), clip(3)],
            grid(&[&[Some(3)]], LaunchQuantize::Bar),
        ),
        &[(1, 0.0, 8.0), (2, 8.0, 16.0)],
    );
    mosh_show::Show::new(p)
}

#[test]
fn arrangement_follows_backward_jump() {
    let mut show = arrangement_show();

    let plan = show.update(&frame(1.0, 2, none()), &[]);
    assert_eq!(ids(&plan), vec![Some(1)]);

    let plan = show.update(&frame(9.0, 2, none()), &[]);
    assert_eq!(ids(&plan), vec![Some(2)]);

    // Live jumps back: the picture must change on the next frame, with no stale clip.
    let plan = show.update(&frame(2.0, 2, none()), &[]);
    assert_eq!(ids(&plan), vec![Some(1)], "a backward jump left a stale clip");

    // Scrub forward again, then past the end of every clip.
    let plan = show.update(&frame(10.0, 2, none()), &[]);
    assert_eq!(ids(&plan), vec![Some(2)]);
    let plan = show.update(&frame(20.0, 2, none()), &[]);
    assert_eq!(ids(&plan), vec![None], "a lane with no clip must be empty");
    let plan = show.update(&frame(12.0, 2, none()), &[]);
    assert_eq!(ids(&plan), vec![Some(2)], "scrubbing back into a clip did not resume it");
}

#[test]
fn arrangement_override_then_back() {
    let mut show = arrangement_show();

    let plan = show.update(&frame(2.0, 2, none()), &[]);
    assert_eq!(ids(&plan), vec![Some(1)]);

    // A Session launch on the bar line takes the layer over.
    let plan =
        show.update(&frame(8.0, 2, EventFlags::BAR), &[ControlEvent::LaunchClip { layer: 0, scene: 0 }]);
    assert_eq!(ids(&plan), vec![Some(3)], "the manual launch lost to the arrangement");

    // The arrangement keeps moving underneath; the layer stays overridden.
    for ppq in [10.0, 12.0, 14.0] {
        let plan = show.update(&frame(ppq, 2, none()), &[]);
        assert_eq!(ids(&plan), vec![Some(3)], "the arrangement took an overridden layer back at {ppq}");
    }

    // Back to arrangement: the clip under the playhead comes back on the next frame.
    let plan = show.update(&frame(13.0, 2, none()), &[ControlEvent::BackToArrangement]);
    assert_eq!(ids(&plan), vec![Some(2)], "back to arrangement did not resume the lane");
}

#[test]
fn arrangement_ignores_session_mode() {
    let p = with_lane0(
        project(
            ShowMode::Session,
            vec![clip(1)],
            grid(&[&[Some(1)]], LaunchQuantize::Bar),
        ),
        &[(1, 0.0, 64.0)],
    );
    let mut show = mosh_show::Show::new(p);
    let plan = show.update(&frame(4.0, 2, EventFlags::BAR), &[]);
    assert_eq!(ids(&plan), vec![None], "Session mode must not read the arrangement");
}

#[test]
fn arrangement_follows_overlapping_clips_by_start() {
    // Two overlapping placements: the later start wins, deterministically.
    let p = with_lane0(
        project(ShowMode::Arrangement, vec![clip(1), clip(2)], grid(&[], LaunchQuantize::Bar)),
        &[(1, 0.0, 16.0), (2, 4.0, 8.0)],
    );
    let mut show = mosh_show::Show::new(p);
    assert_eq!(ids(&show.update(&frame(1.0, 2, none()), &[])), vec![Some(1)]);
    assert_eq!(ids(&show.update(&frame(5.0, 2, none()), &[])), vec![Some(2)]);
    assert_eq!(ids(&show.update(&frame(9.0, 2, none()), &[])), vec![Some(1)]);
}

#[test]
fn arrangement_mode_has_one_layer_per_lane() {
    let mut p = project(ShowMode::Arrangement, vec![clip(1), clip(2)], grid(&[], LaunchQuantize::Bar));
    p.arrangement.lanes = vec![
        vec![mosh_core::ArrClip { clip: 1, start_ppq: 0.0, end_ppq: 8.0 }],
        vec![mosh_core::ArrClip { clip: 2, start_ppq: 0.0, end_ppq: 8.0 }],
    ];
    let mut show = mosh_show::Show::new(p);
    let plan = show.update(&frame(1.0, 2, none()), &[]);
    assert_eq!(ids(&plan), vec![Some(1), Some(2)]);
    assert_eq!(plan.layers[1].layer, 1);
}
