//! Task 1.2, plugin behaviour that does not need a VST3 host and does not need `mosh-ipc`.
//!
//! Every test drives the same code the plugins call from `process()`: a `TapCore` /
//! `ControlCore` over a `. No nih-plug context and no shared-memory mapping is involved, so
//! this file runs with `mosh-ipc` missing (the `ipc` cargo feature is off by default) and after
//! it lands.

use mosh_core::ipc_layout::ShowRole;
use mosh_core::transport::{MidiEvent, Transport};
use mosh_tap::control::{ControlCore, ControlStats, HostEvent};
use mosh_tap::host::{HostTransport, interleave_into};
use mosh_tap::sink::{BlockSink, NullSink, PluginSink, role_for_index, role_index_of};
use mosh_tap::tap::TapCore;

/// A sink that keeps what it was given, so a test can play the reader.
#[derive(Default)]
struct RecordingSink {
    blocks: Vec<(Option<Transport>, Vec<f32>, u32)>,
    midi: Vec<MidiEvent>,
}

impl BlockSink for RecordingSink {
    fn push_block(&mut self, transport: Option<&Transport>, interleaved: &[f32], frames: u32) {
        self.blocks
            .push((transport.copied(), interleaved.to_vec(), frames));
    }

    fn push_midi(&mut self, event: &MidiEvent) {
        self.midi.push(*event);
    }
}

/// 128.5 BPM, bar 3 of 4/4 starting at ppq 8, playing, one second in.
fn tap_transport() -> HostTransport {
    HostTransport {
        tempo_bpm: 128.5,
        ppq: 12.25,
        bar_start_ppq: 8.0,
        sig_num: 4,
        sig_den: 4,
        playing: true,
        sample_pos: 48_000,
    }
}

fn bit_equal(a: &[f32], b: &[f32]) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "sample {i} changed: {x:?} ({:#010x}) vs {y:?} ({:#010x})",
            x.to_bits(),
            y.to_bits()
        );
    }
}

#[test]
fn tap_passes_audio_bit_exact() {
    const FRAMES: u32 = 256;
    const CHANNELS: usize = 2;

    // Audio that a sanitiser would love: NaN with a payload, -0.0, a denormal, infinities and
    // values outside [-1, 1] that a gain stage would clip.
    let mut input: Vec<f32> = (0..FRAMES as usize * CHANNELS)
        .map(|i| ((i as f32) * 0.001 - 0.5).sin())
        .collect();
    input[0] = f32::from_bits(0x7fc0_1234);
    input[1] = -0.0;
    input[2] = f32::from_bits(0x0000_0001);
    input[3] = f32::INFINITY;
    input[4] = f32::NEG_INFINITY;
    input[5] = 1234.5;

    let mut core = TapCore::new(RecordingSink::default(), 48_000.0, 2);
    let pushed = core.push_interleaved(Some(&tap_transport()), &input, FRAMES);

    assert_eq!(pushed, 1, "256 frames must be one block");
    let (transport, out, frames) = &core.sink().blocks[0];
    assert!(transport.is_some());
    assert_eq!(*frames, FRAMES);
    bit_equal(out, &input);
}

#[test]
fn tap_writes_transport_fields() {
    let host = tap_transport();
    let mut core = TapCore::new(RecordingSink::default(), 48_000.0, 2);
    core.push_interleaved(Some(&host), &[0.0; 512], 256);

    let got = core.sink().blocks[0].0.expect("transport must be pushed");
    assert_eq!(got, host.to_transport());
    assert_eq!(got.ppq, 12.25);
    assert_eq!(got.tempo_bpm, 128.5);
    assert_eq!(got.bar_start_ppq, 8.0);
    assert_eq!((got.sig_num, got.sig_den), (4, 4));
    assert!(got.playing);
    assert_eq!(got.sample_pos, 48_000);
}

#[test]
fn tap_without_transport_pushes_none() {
    let mut core = TapCore::new(RecordingSink::default(), 48_000.0, 2);
    core.push_interleaved(None, &[0.0; 512], 256);

    assert_eq!(core.sink().blocks[0].0, None);
}

#[test]
fn tap_works_without_reader() {
    // No mapping consumer at all: the sink discards everything and `process` keeps returning
    // normal. `PluginSink` is what `initialize()` builds; with the `ipc` feature off it is a
    // no-op sink, which is exactly the "mosh-ipc has not landed" state.
    if !cfg!(feature = "ipc") {
        assert!(
            matches!(
                PluginSink::create(ShowRole::Master, 48_000, 2),
                PluginSink::Null(_)
            ),
            "without the `ipc` feature the plugin must not create a writer"
        );
    }

    let mut core = TapCore::new(PluginSink::create(ShowRole::Master, 48_000, 2), 48_000.0, 2);
    let input = vec![0.5f32; 512];
    for _ in 0..1_000 {
        assert_eq!(
            core.push_interleaved(Some(&tap_transport()), &input, 256),
            1
        );
    }

    let mut null = TapCore::new(NullSink, 48_000.0, 2);
    assert_eq!(null.push_interleaved(None, &input, 256), 1);
}

#[test]
fn interleave_keeps_channel_data_and_order() {
    let mut left = [1.0f32, 2.0, 3.0, 4.0];
    let mut right = [10.0f32, 20.0, 30.0, 40.0];
    let left_before = left;
    let right_before = right;

    let mut dst = [0.0f32; 8];
    let written = interleave_into(&mut dst, &[&mut left[..], &mut right[..]], 0, 4);

    assert_eq!(written, 8);
    assert_eq!(dst, [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]);
    // Reading a channel must not write to it: the audio callback only ever copies.
    bit_equal(&left, &left_before);
    bit_equal(&right, &right_before);
}

#[test]
fn interleave_honours_a_frame_offset() {
    let mut left = [0.0f32, 1.0, 2.0, 3.0];
    let mut right = [0.0f32, 11.0, 12.0, 13.0];
    let mut dst = [0.0f32; 4];

    let written = interleave_into(&mut dst, &[&mut left[..], &mut right[..]], 2, 2);

    assert_eq!(written, 4);
    assert_eq!(dst, [2.0, 12.0, 3.0, 13.0]);
}

#[test]
fn interleave_refuses_to_overflow_the_scratch() {
    let mut left = [1.0f32; 4];
    let mut right = [2.0f32; 4];
    let mut dst = [0.0f32; 4];

    // 4 frames of stereo need 8 samples; a too-small destination writes nothing.
    assert_eq!(
        interleave_into(&mut dst, &[&mut left[..], &mut right[..]], 0, 4),
        0
    );
    assert_eq!(dst, [0.0; 4]);
}

#[test]
fn oversized_block_is_split_with_advanced_ppq() {
    const FRAMES: u32 = 2_048;
    let input: Vec<f32> = (0..FRAMES as usize * 2).map(|i| i as f32 * 0.25).collect();

    let mut core = TapCore::new(RecordingSink::default(), 48_000.0, 2);
    let pushed = core.push_interleaved(Some(&tap_transport()), &input, FRAMES);

    assert_eq!(pushed, 2, "2048 stereo frames are two 1024-frame blocks");
    let blocks = &core.sink().blocks;
    assert_eq!(blocks[0].2, 1_024);
    assert_eq!(blocks[1].2, 1_024);

    // The second block starts one 1024-frame chunk later: 1024/48000 s at 128.5 BPM.
    let advance = 1_024.0 / 48_000.0 * 128.5 / 60.0;
    assert_eq!(blocks[0].0.map(|t| t.ppq), Some(12.25));
    let second_ppq = blocks[1].0.expect("second chunk carries transport").ppq;
    assert!(
        (second_ppq - (12.25 + advance)).abs() < 1e-12,
        "second ppq {second_ppq}"
    );
    assert_eq!(blocks[1].0.unwrap().sample_pos, 48_000 + 1_024);

    // Together the chunks are the original block, still bit-exact.
    let mut joined = blocks[0].1.clone();
    joined.extend_from_slice(&blocks[1].1);
    bit_equal(&joined, &input);
}

#[test]
fn control_maps_note_on_and_cc() {
    let base = HostTransport {
        tempo_bpm: 120.0,
        ppq: 4.0,
        bar_start_ppq: 4.0,
        sig_num: 4,
        sig_den: 4,
        playing: true,
        sample_pos: 0,
    };
    let events = [
        HostEvent::NoteOn {
            timing: 0,
            channel: 0,
            note: 36,
            velocity: 100.0 / 127.0,
        },
        HostEvent::ControlChange {
            timing: 128,
            channel: 0,
            cc: 48,
            value: 64.0 / 127.0,
        },
    ];

    let mut core = ControlCore::new(RecordingSink::default(), 48_000.0, 2);
    let stats = core.process(Some(&base), &events, 256);

    assert_eq!(stats, ControlStats { blocks: 1, midi: 2 });
    assert_eq!(core.sink().midi.len(), 2);
    assert_eq!(
        core.sink().midi[0],
        MidiEvent {
            ppq: 4.0,
            status: 0x90,
            data1: 36,
            data2: 100
        },
        "note 36 at velocity 100"
    );
    let cc = core.sink().midi[1];
    assert_eq!(
        (cc.status, cc.data1, cc.data2),
        (0xb0, 48, 64),
        "CC 48 = 64"
    );
    // 128 samples at 120 BPM is 0.005333... beats.
    let expected = 4.0 + 128.0 / 48_000.0 * 120.0 / 60.0;
    assert!(
        (cc.ppq - expected).abs() < 1e-12,
        "cc ppq {} != {expected}",
        cc.ppq
    );

    // The controller is silent, and its block carries transport so the app sees a live writer.
    let (transport, samples, frames) = &core.sink().blocks[0];
    assert_eq!(*transport, Some(base.to_transport()));
    assert_eq!(*frames, 256);
    assert!(
        samples.iter().all(|s| *s == 0.0),
        "control output must be silent"
    );
}

#[test]
fn control_maps_note_off_and_program_change() {
    let base = HostTransport {
        tempo_bpm: 120.0,
        playing: true,
        ..Default::default()
    };
    let events = [
        HostEvent::NoteOff {
            timing: 0,
            channel: 2,
            note: 36,
        },
        HostEvent::ProgramChange {
            timing: 0,
            channel: 0,
            program: 7,
        },
    ];

    let mut core = ControlCore::new(RecordingSink::default(), 48_000.0, 2);
    core.process(Some(&base), &events, 64);

    assert_eq!(
        core.sink().midi[0],
        MidiEvent {
            ppq: 0.0,
            status: 0x82,
            data1: 36,
            data2: 0
        }
    );
    assert_eq!(
        core.sink().midi[1],
        MidiEvent {
            ppq: 0.0,
            status: 0xc0,
            data1: 7,
            data2: 0
        }
    );
}

#[test]
fn midi_velocity_is_scaled_to_the_midi_range() {
    let base = HostTransport {
        tempo_bpm: 120.0,
        playing: true,
        ..Default::default()
    };
    let events = [
        HostEvent::NoteOn {
            timing: 0,
            channel: 0,
            note: 60,
            velocity: 1.0,
        },
        HostEvent::NoteOn {
            timing: 0,
            channel: 0,
            note: 61,
            velocity: 0.0,
        },
    ];

    let mut core = ControlCore::new(RecordingSink::default(), 48_000.0, 2);
    core.process(Some(&base), &events, 32);

    // A note on at zero velocity would read as a note off in MIDI, so it clamps to 1.
    assert_eq!(core.sink().midi[0].data2, 127);
    assert_eq!(core.sink().midi[1].data2, 1);
}

#[test]
fn control_without_transport_pushes_none_and_zero_ppq() {
    let events = [HostEvent::NoteOn {
        timing: 64,
        channel: 0,
        note: 40,
        velocity: 1.0,
    }];

    let mut core = ControlCore::new(RecordingSink::default(), 48_000.0, 2);
    core.process(None, &events, 256);

    assert_eq!(core.sink().blocks[0].0, None);
    assert_eq!(core.sink().midi.len(), 1);
    assert_eq!(core.sink().midi[0].ppq, 0.0);
}

#[test]
fn role_parameter_covers_the_five_source_roles() {
    assert_eq!(role_for_index(0), ShowRole::Master);
    assert_eq!(role_for_index(1), ShowRole::Drums);
    assert_eq!(role_for_index(2), ShowRole::Bass);
    assert_eq!(role_for_index(3), ShowRole::Vocal);
    assert_eq!(role_for_index(4), ShowRole::Other);
    // Out of range clamps instead of panicking on the audio thread.
    assert_eq!(role_for_index(-5), ShowRole::Master);
    assert_eq!(role_for_index(99), ShowRole::Other);
    // The control role is not selectable: only the instrument writes it.
    assert_eq!(role_index_of(ShowRole::Master), Some(0));
    assert_eq!(role_index_of(ShowRole::Other), Some(4));
    assert_eq!(role_index_of(ShowRole::Control), None);
}

#[test]
fn tap_callback_stays_under_the_callback_budget() {
    // Design budget: < 20 us per 256-sample block for the plugin's audio callback. This measures
    // the part this crate owns (interleave into a preallocated scratch, then push), with the sink
    // discarding the samples, so it is a floor for the real number: the shared-memory write and
    // `mosh-ipc` are not in this test.
    //
    // The budget is a release-build number, because that is what Live loads. A debug build is
    // several times slower and is only made to report its number here.
    const BLOCKS: usize = 2_000;
    let mut left = vec![0.1f32; 256];
    let mut right = vec![0.2f32; 256];
    let mut scratch = [0.0f32; 512];
    let transport = tap_transport();
    let mut core = TapCore::new(NullSink, 48_000.0, 2);

    let mut times = Vec::with_capacity(BLOCKS);
    for _ in 0..BLOCKS {
        let start = std::time::Instant::now();
        let written = interleave_into(&mut scratch, &[&mut left[..], &mut right[..]], 0, 256);
        core.push_interleaved(Some(&transport), &scratch[..written], 256);
        times.push(start.elapsed().as_nanos() as u64);
    }
    times.sort_unstable();
    let median_ns = times[BLOCKS / 2];
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    println!("median process path: {median_ns} ns per 256-frame block ({profile} build)");

    if !cfg!(debug_assertions) {
        assert!(
            median_ns < 20_000,
            "median {median_ns} ns per 256-frame block is over the 20 us budget"
        );
    }
}
