//! Task 1.11 acceptance: the committed fixture decodes, frame lookup is time based, and a
//! missing file is an error rather than a panic.
//!
//! `fixtures/clip.mp4` (160x90) is read-only for native lanes; these tests only open it.

use std::path::{Path, PathBuf};
use std::time::Instant;

use mosh_video::{VideoError, VideoSource};

/// `fixtures/clip.mp4`, relative to the crate root (Task 1.11).
fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../fixtures/clip.mp4")
}

#[test]
fn opens_fixture_and_reports_size() {
    let src = VideoSource::open(fixture()).expect("open fixtures/clip.mp4");
    let info = src.info();

    assert_eq!(
        (info.w, info.h),
        (160, 90),
        "fixture is 160x90, got {info:?}"
    );
    assert!(
        info.fps > 1.0 && info.fps < 240.0,
        "plausible frame rate, got {info:?}"
    );
    assert!(
        info.duration_s > 0.05 && info.duration_s < 600.0,
        "plausible duration, got {info:?}"
    );
}

#[test]
fn missing_file_is_error() {
    let err = match VideoSource::open(Path::new("fixtures/no-such-clip.mp4")) {
        Ok(_) => panic!("opening a missing file returned Ok"),
        Err(e) => e,
    };
    assert!(
        matches!(err, VideoError::NotFound(_)),
        "expected NotFound, got {err:?}"
    );
}

#[test]
fn frame_at_is_monotonic_and_loops() {
    let mut src = VideoSource::open(fixture()).expect("open fixture");
    let info = src.info();
    let dt = 1.0 / info.fps;

    // Walking the whole clip at one frame per step never goes backwards, and the frame
    // returned for a time is the one that covers it.
    let mut last_pts = -1.0;
    let mut steps = 0;
    let mut t = 0.0;
    while t < info.duration_s - dt {
        let frame = src.frame_at(t).expect("frame inside the clip");
        assert!(
            frame.pts_s >= last_pts - 1e-9,
            "pts went backwards at t={t}: {} after {last_pts}",
            frame.pts_s
        );
        assert!(
            (frame.pts_s - t).abs() <= 2.0 * dt + 1e-6,
            "frame at pts {} does not cover t={t} (fps {})",
            frame.pts_s,
            info.fps
        );
        last_pts = frame.pts_s;
        t += dt;
        steps += 1;
    }
    assert!(steps >= 3, "fixture too short to walk: {steps} steps");

    // Past the end the clip loops to the start instead of returning None.
    let wrapped = src
        .frame_at(info.duration_s + 0.25 * dt)
        .expect("loop frame");
    let wrapped_pts = wrapped.pts_s;
    assert!(
        wrapped_pts < last_pts,
        "expected a wrap, pts {wrapped_pts} after {last_pts}"
    );
    assert!(
        wrapped_pts <= dt + 1e-6,
        "loop starts at 0, got pts {wrapped_pts}"
    );

    // The same time twice is the same frame, forwards or backwards.
    let again = src.frame_at(0.0).expect("frame 0");
    assert_eq!(again.pts_s, wrapped_pts, "time 0 is not deterministic");
}

#[test]
fn frame_at_seeks_backwards() {
    let mut src = VideoSource::open(fixture()).expect("open fixture");
    let info = src.info();

    let late = src.frame_at(info.duration_s * 0.8).expect("late frame");
    let late_pts = late.pts_s;
    let early = src.frame_at(info.duration_s * 0.1).expect("early frame");
    assert!(
        early.pts_s < late_pts,
        "seek backwards returned pts {} after {late_pts}",
        early.pts_s
    );
    assert!(
        early.pts_s <= info.duration_s * 0.1 + 1e-6,
        "pts {} is after the requested 10%",
        early.pts_s
    );
}

/// The fixture draws a 3 px white bar that moves right as time advances (`scripts/make-fixtures.mjs`),
/// so its column is a fingerprint of both the geometry and the time->frame mapping: a wrong row
/// pitch shears the frame, a wrong flip or seek puts the bar somewhere else.
#[test]
fn decoded_pixels_match_the_fixture() {
    let mut src = VideoSource::open(fixture()).expect("open fixture");
    let info = src.info();
    let w = info.w as usize;
    let h = info.h as usize;

    for fraction in [0.0, 0.5] {
        let t = info.duration_s * fraction;
        let frame = src.frame_at(t).expect("frame");
        let pts = frame.pts_s;
        // The generator moves the bar 90 px/s at 30 fps: 3 px per frame, wrapping at the width.
        let index = (pts * info.fps).round() as usize;
        let expected = (index * 3) % w;
        assert!(expected + 2 < w, "bar runs off the frame at t={t}");

        // White bar: every row of the column is bright in all three channels.
        let bright: Vec<usize> = (0..w)
            .filter(|x| (0..h).all(|y| (0..3).all(|c| frame.bgra[(y * w + x) * 4 + c] > 120)))
            .collect();
        for x in expected..expected + 3 {
            assert!(
                bright.contains(&x),
                "expected the white bar at x={expected}..{} for pts {pts}, bright columns {bright:?}",
                expected + 3
            );
        }
        assert!(
            bright.len() <= 6,
            "frame at pts {pts} is not the fixture: {bright:?} whole columns are white"
        );
    }
}

#[test]
fn frame_buffer_is_reused_and_holds_pixels() {
    let mut src = VideoSource::open(fixture()).expect("open fixture");
    let info = src.info();
    let expected_bytes = (info.w * info.h * 4) as usize;

    let first = src.frame_at(0.0).expect("frame 0");
    let ptr = first.bgra.as_ptr();
    assert_eq!(first.bgra.len(), expected_bytes, "BGRA size for w*h");
    assert_eq!((first.w, first.h), (info.w, info.h));
    let lit = first.bgra.iter().filter(|b| **b != 0).count();
    assert!(
        lit > expected_bytes / 100,
        "decoded frame is nearly empty: {lit} of {expected_bytes} bytes non-zero"
    );

    // One buffer for the life of the source: a per-frame allocation would move it.
    for step in 1..6 {
        let t = info.duration_s * f64::from(step) / 8.0;
        let frame = src.frame_at(t).expect("later frame");
        assert_eq!(
            frame.bgra.as_ptr(),
            ptr,
            "frame buffer moved at t={t} (allocated per frame)"
        );
        assert_eq!(frame.bgra.len(), expected_bytes);
    }
}

/// Not a gate: prints the numbers the PR body quotes. Decode cost on the reference machine is
/// a coordinator measurement; this only records what the test run saw.
#[test]
fn decode_time_for_fixture() {
    let start = Instant::now();
    let mut src = VideoSource::open(fixture()).expect("open fixture");
    let open_ms = start.elapsed().as_secs_f64() * 1e3;
    let info = src.info();
    let dt = 1.0 / info.fps;

    assert!(src.frame_at(0.0).is_some(), "warm up the decoder");

    let frames = 24;
    let start = Instant::now();
    for i in 0..frames {
        let t = (i as f64) * dt;
        assert!(src.frame_at(t).is_some(), "frame at t={t}");
    }
    let per_frame_ms = start.elapsed().as_secs_f64() * 1e3 / frames as f64;

    println!(
        "mosh-video: {}x{} at {:.3} fps, {:.3} s; open {:.2} ms, sequential decode {:.3} ms/frame over {} frames",
        info.w, info.h, info.fps, info.duration_s, open_ms, per_frame_ms, frames
    );
}
