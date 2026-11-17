//! Device-backed tests for the WASAPI loopback fallback (Task 1.6).
//!
//! These open a real render endpoint. They play **nothing**: the point of
//! `captures_silence_as_zeros_not_error` is that silence keeps producing blocks, because
//! WASAPI loopback delivers no packets at all when the endpoint renders nothing. Run them on a
//! machine that is not playing audio; a playing endpoint is a failure of the test's premise,
//! not of the code.

use std::time::{Duration, Instant};

use mosh_capture::{BLOCK_FRAMES, CaptureError, Loopback};
use mosh_core::AudioBlock;

/// Poll for the whole `window`, sleeping 5 ms between polls, so a stalled capture thread shows
/// up as a timeout, not as a hang.
fn collect_for(lb: &mut Loopback, window: Duration) -> (Vec<AudioBlock>, Duration) {
    let mut blocks = Vec::new();
    let start = Instant::now();
    while start.elapsed() < window {
        lb.poll(&mut blocks);
        std::thread::sleep(Duration::from_millis(5));
    }
    (blocks, start.elapsed())
}

#[test]
fn lists_at_least_one_device() {
    let devices = mosh_capture::list_render_devices();
    for (i, name) in devices.iter().enumerate() {
        println!("render device {i}: {name}");
    }
    assert!(
        !devices.is_empty(),
        "no render endpoints: list_render_devices() returned {:?}",
        devices
    );
    assert!(
        devices.iter().all(|name| !name.trim().is_empty()),
        "an endpoint reported an empty friendly name: {:?}",
        devices
    );
}

#[test]
fn captures_silence_as_zeros_not_error() {
    let mut lb = Loopback::start(None).expect("open the default render endpoint for loopback");
    let rate = lb.sample_rate();
    let channels = lb.channels();
    println!("capturing at {rate} Hz, {channels} ch");
    assert!(rate > 0 && channels > 0);

    // 1 s of blocks at the device rate, one block per 256 frames.
    let (blocks, elapsed) = collect_for(&mut lb, Duration::from_secs(1));
    assert_eq!(lb.error(), None, "capture reported an error during silence");

    let frames: usize = blocks.iter().map(AudioBlock::frames).sum();
    let synthetic = lb.synthetic_blocks();
    let packet_frames = lb.packet_frames();
    println!(
        "silence: {} blocks, {} frames in {:.3} s wall ({:.1} blocks/s); endpoint delivered {} \
         frames in packets, clock synthesized {} blocks, {} dropped",
        blocks.len(),
        frames,
        elapsed.as_secs_f64(),
        blocks.len() as f64 / elapsed.as_secs_f64(),
        packet_frames,
        synthetic,
        lb.dropped_blocks()
    );

    // The block stream must not stop when the endpoint has nothing to say. The band is wide
    // enough for a slow first poll and scheduler jitter, tight enough that a dead clock fails.
    let expected = rate as f64 * elapsed.as_secs_f64();
    assert!(
        frames as f64 >= expected * 0.8,
        "only {} frames in {:.3} s at {} Hz ({:.2} of the block cadence): the stream stalled",
        frames,
        elapsed.as_secs_f64(),
        rate,
        frames as f64 / expected
    );
    assert!(
        frames as f64 <= expected * 1.25,
        "{} frames in {:.3} s at {} Hz ({:.2} of the block cadence): the stream overproduced",
        frames,
        elapsed.as_secs_f64(),
        rate,
        frames as f64 / expected
    );

    // A pure-silence endpoint sends no packets at all: every frame must then be synthetic and a
    // block of zeros. A few blocks may still be in the ring when polling stops.
    if packet_frames == 0 {
        assert!(
            synthetic > 0,
            "the endpoint delivered no packets and the clock produced no blocks: the stream would starve"
        );
        let synthetic_frames = synthetic * BLOCK_FRAMES as u64;
        assert!(
            synthetic_frames >= frames as u64
                && synthetic_frames <= frames as u64 + 4 * BLOCK_FRAMES as u64,
            "{} synthetic blocks do not account for {} frames",
            synthetic,
            frames
        );
    }

    for (i, block) in blocks.iter().enumerate() {
        assert_eq!(
            block.frames(),
            BLOCK_FRAMES,
            "block {i} is not {BLOCK_FRAMES} frames"
        );
        assert_eq!(block.channels, channels, "block {i} changed channel count");
        assert_eq!(block.sample_rate, rate, "block {i} changed sample rate");
        assert!(
            block.transport.is_none(),
            "block {i} carries a transport; loopback has none"
        );
    }

    let peak = blocks
        .iter()
        .flat_map(|block| block.samples.iter())
        .fold(0.0f32, |loudest, sample| loudest.max(sample.abs()));
    // An endpoint that streams silence can still carry dither: -84 dBFS (two 16-bit LSBs) is
    // silence for this purpose. Those samples are *not* zeroed by us — we pass through exactly
    // what WASAPI hands over, and only the clock's own blocks are exactly zero.
    const SILENT_PEAK: f32 = 1.0 / 16_384.0;
    if peak <= SILENT_PEAK {
        println!("silent endpoint, peak {peak:e} (<= {SILENT_PEAK:e})");
    } else {
        // Honest gap: "silence is zeros" needs a machine that is not playing audio, and this one
        // was. Quote this line with any run that prints it.
        println!(
            "NOT CHECKED: the default endpoint was playing audio during this run (peak {peak:e}), \
             so the silence claim was skipped; run it on a quiet machine for that claim"
        );
    }
}

#[test]
fn unknown_device_name_is_an_error() {
    let err = Loopback::start(Some("mosh-capture: no such render endpoint")).unwrap_err();
    println!("unknown device error: {err}");
    assert!(
        matches!(err, CaptureError::DeviceNotFound(_)),
        "expected DeviceNotFound, got {err:?}"
    );
}

/// The task's premise, made checkable: WASAPI loopback delivers **no packets** while an endpoint
/// renders nothing, so the capture thread has to produce those blocks itself.
///
/// It does not hold for every endpoint: a driver or APO can keep a stream alive and send silent
/// packets (the Realtek and Oculus endpoints on this machine do, so `captures_silence_as_zeros…`
/// sees packets even with nothing playing). This test looks for an endpoint that really is
/// packet-free — a virtual endpoint with no renderer is one — and checks that the clock fills the
/// stream with zeros at the block rate. With no such endpoint it says so instead of pretending.
#[test]
fn a_packet_free_endpoint_is_filled_with_zero_blocks() {
    let devices = mosh_capture::list_render_devices();
    let window = Duration::from_millis(400);
    let mut checked = None;

    for name in &devices {
        let Ok(mut lb) = Loopback::start(Some(name)) else {
            continue;
        };
        let (blocks, elapsed) = collect_for(&mut lb, window);
        let frames: usize = blocks.iter().map(AudioBlock::frames).sum();
        let synthetic = lb.synthetic_blocks();
        println!(
            "{name}: {} blocks, {} frames in {:.3} s, {} packet frames, {} synthetic blocks",
            blocks.len(),
            frames,
            elapsed.as_secs_f64(),
            lb.packet_frames(),
            synthetic
        );
        if lb.packet_frames() > 0 || synthetic == 0 {
            continue;
        }

        let rate = lb.sample_rate();
        assert_eq!(lb.error(), None, "{name}: capture failed during silence");
        assert!(
            blocks
                .iter()
                .all(|block| block.samples.iter().all(|s| *s == 0.0)),
            "{name}: a packet-free endpoint can only produce zeros"
        );
        for block in &blocks {
            assert_eq!(block.frames(), BLOCK_FRAMES, "{name}: block size");
            assert_eq!(block.channels, lb.channels(), "{name}: channel count");
            assert_eq!(block.sample_rate, rate, "{name}: sample rate");
            assert!(
                block.transport.is_none(),
                "{name}: loopback has no transport"
            );
        }
        let expected = rate as f64 * elapsed.as_secs_f64();
        assert!(
            frames as f64 >= expected * 0.8,
            "{name}: {} frames in {:.3} s at {} Hz ({:.2} of the block cadence): the clock \
             stopped filling the stream",
            frames,
            elapsed.as_secs_f64(),
            rate,
            frames as f64 / expected
        );
        checked = Some(name.clone());
        break;
    }

    match checked {
        Some(name) => println!("packet-free endpoint checked: {name}"),
        None => println!(
            "NOT CHECKED: every render endpoint is streaming packets (silent or not), so the \
             packet-free path was not exercised on this machine"
        ),
    }
}

/// Manual test: start the capture, then disable or unplug the render endpoint (or the whole
/// output device) within 60 s. The capture thread must report an error instead of panicking,
/// and polling after the failure must stay bounded.
#[test]
#[ignore = "manual: needs a human to remove the render device"]
fn device_removed_returns_error_not_panic() {
    let mut lb = Loopback::start(None).expect("open the default render endpoint for loopback");
    println!("disable or unplug the default render device now (60 s)");
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut seen = 0usize;
    while Instant::now() < deadline && lb.error().is_none() {
        let mut blocks = Vec::new();
        lb.poll(&mut blocks);
        seen += blocks.len();
        std::thread::sleep(Duration::from_millis(50));
    }
    let err = lb
        .error()
        .expect("removing the render device must surface an error");
    println!("device removal surfaced: {err} (after {seen} blocks)");

    // Polling a dead capture must not panic and must not grow without bound.
    let mut after = Vec::new();
    for _ in 0..20 {
        lb.poll(&mut after);
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        after.len() < 10_000,
        "poll returned {} blocks after failure",
        after.len()
    );
}
