//! Tests for the offline harness (Task 1.5): decoding, blocks, labels, scoring, tempo hints.
//!
//! Every fixture is generated here, in a temp directory, in a few hundred bytes: no audio is
//! committed. The WAV bytes are written by hand rather than by a writer crate, so the decoder is
//! checked against the RIFF spec and not against another encoder's idea of it.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use mosh_analysis::offline::{self, KindScore, LabelError, Labels, Mark, MarkKind, OfflineError};
use mosh_core::{EventFlags, Transport};

// ---------------------------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------------------------

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(tag: &str) -> TempDir {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "moshloom-offline-{tag}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create temp dir");
        TempDir { path }
    }

    fn file(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

const WAVE_FORMAT_PCM: u16 = 1;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;

/// A complete RIFF/WAVE file: canonical 16-byte `fmt ` chunk, one `data` chunk.
fn wav_bytes(rate: u32, channels: u16, bits: u16, format_tag: u16, data: &[u8]) -> Vec<u8> {
    let block_align = channels * (bits / 8);
    let mut out = Vec::with_capacity(44 + data.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&format_tag.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * u32::from(block_align)).to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(data);
    out
}

/// Stereo, 44.1 kHz, 512 frames, PCM 16-bit: left = i*60, right = -(i*60).
fn write_pcm16_fixture(dir: &TempDir) -> PathBuf {
    let mut data = Vec::new();
    for frame in 0..512i16 {
        data.extend_from_slice(&(frame * 60).to_le_bytes());
        data.extend_from_slice(&(-(frame * 60)).to_le_bytes());
    }
    let path = dir.file("pcm16.wav");
    fs::write(&path, wav_bytes(44_100, 2, 16, WAVE_FORMAT_PCM, &data)).expect("write fixture");
    path
}

/// Stereo, 48 kHz, 256 frames, IEEE float 32-bit: the same container shape as the tuning set's
/// "yeezus shit.wav", which is the file that would otherwise go untested.
fn write_float32_fixture(dir: &TempDir) -> PathBuf {
    let values: [f32; 4] = [0.123_456_79, -0.5, 0.999_999_94, 0.0];
    let mut data = Vec::new();
    for _ in 0..256usize {
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
    }
    let path = dir.file("float32.wav");
    fs::write(
        &path,
        wav_bytes(48_000, 4, 32, WAVE_FORMAT_IEEE_FLOAT, &data),
    )
    .expect("write fixture");
    path
}

// ---------------------------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------------------------

#[test]
fn reads_wav_16bit() {
    let dir = TempDir::new("pcm16");
    let path = write_pcm16_fixture(&dir);

    let (samples, rate, channels) = offline::read_audio(&path).expect("decode pcm16");

    assert_eq!(rate, 44_100);
    assert_eq!(channels, 2);
    assert_eq!(
        samples.len(),
        512 * 2,
        "every frame, both channels, interleaved"
    );
    for frame in [0usize, 1, 17, 511] {
        let left = samples[frame * 2];
        let right = samples[frame * 2 + 1];
        let want = (frame as i16 * 60) as f32 / 32_768.0;
        assert!(
            (left - want).abs() < 1e-6,
            "frame {frame} left: decodes to {left}, want {want}"
        );
        assert!(
            (right + want).abs() < 1e-6,
            "frame {frame} right: decodes to {right}, want {}",
            -want
        );
    }
}

#[test]
fn reads_wav_float32() {
    let dir = TempDir::new("float32");
    let path = write_float32_fixture(&dir);

    let (samples, rate, channels) = offline::read_audio(&path).expect("decode float32");

    assert_eq!(rate, 48_000);
    assert_eq!(channels, 4);
    assert_eq!(samples.len(), 256 * 4);
    let values = [0.123_456_79f32, -0.5, 0.999_999_94, 0.0];
    for frame in [0usize, 255] {
        for (channel, want) in values.iter().enumerate() {
            let got = samples[frame * 4 + channel];
            assert_eq!(
                got, *want,
                "frame {frame} channel {channel}: IEEE float must survive the read untouched"
            );
        }
    }
}

#[test]
fn missing_file_is_a_clean_error() {
    let dir = TempDir::new("missing");
    let path = dir.file("nothing-here.wav");

    let error = offline::read_audio(&path).expect_err("a missing file is an error");

    assert!(matches!(error, OfflineError::Open { .. }), "got {error:?}");
    assert!(
        error.to_string().contains("nothing-here.wav"),
        "message names the file: {error}"
    );
}

#[test]
fn garbage_is_a_clean_error() {
    let dir = TempDir::new("garbage");
    let path = dir.file("broken.mp3");
    fs::write(&path, b"this is not audio, not even slightly an MP3 frame").expect("write");
    let error = offline::read_audio(&path).expect_err("garbage is an error");
    assert!(
        matches!(
            error,
            OfflineError::Unsupported(_) | OfflineError::Decode(_)
        ),
        "got {error:?}"
    );
    assert!(
        error.to_string().contains("broken.mp3"),
        "message names the file: {error}"
    );

    let empty = dir.file("empty.wav");
    fs::write(&empty, b"").expect("write");
    let error = offline::read_audio(&empty).expect_err("an empty file is an error");
    assert!(
        matches!(
            error,
            OfflineError::Unsupported(_) | OfflineError::Decode(_)
        ),
        "got {error:?}"
    );
}

// ---------------------------------------------------------------------------------------------
// Blocks and the synthetic transport
// ---------------------------------------------------------------------------------------------

fn ramp(frames: usize, channels: u16) -> Vec<f32> {
    (0..frames * usize::from(channels))
        .map(|i| i as f32)
        .collect()
}

#[test]
fn blocks_carry_a_constant_tempo_transport() {
    let samples = ramp(1024, 2);
    let blocks: Vec<_> = offline::blocks(&samples, 48_000, 2, 256, 120.0, 0.0).collect();

    assert_eq!(blocks.len(), 4, "1024 frames / 256");
    assert_eq!(blocks[0].sample_rate, 48_000);
    assert_eq!(blocks[0].channels, 2);
    assert_eq!(blocks[0].samples.len(), 512, "256 frames * 2 channels");
    assert_eq!(blocks[0].samples[0], 0.0);
    assert_eq!(
        blocks[1].samples[0], 512.0,
        "the second block starts 512 samples in"
    );

    // 256 frames at 48 kHz is 1/187.5 s, which at 120 bpm is 2/187.5 beats.
    let expected_step = 256.0 / 48_000.0 * 2.0;
    for (index, block) in blocks.iter().enumerate() {
        let transport: Transport = block.transport.expect("a synthetic transport");
        assert_eq!(transport.tempo_bpm, 120.0);
        assert_eq!(transport.sig_num, 4);
        assert_eq!(transport.sig_den, 4);
        assert!(transport.playing);
        assert_eq!(transport.sample_pos, index as i64 * 256);
        assert!(
            (transport.ppq - index as f64 * expected_step).abs() < 1e-12,
            "block {index}: ppq {}",
            transport.ppq
        );
        assert!(transport.ppq >= transport.bar_start_ppq);
        assert!(transport.ppq - transport.bar_start_ppq < 4.0);
    }
    // Four blocks are still bar 0 at 120 bpm (a bar is 2 s, the run is 21 ms).
    assert_eq!(blocks[3].transport.expect("transport").bar_start_ppq, 0.0);
}

#[test]
fn blocks_follow_the_bar_grid_and_the_offset() {
    // At 120 bpm a bar is 2 s: 96 000 frames at 48 kHz, exactly one bar per block.
    let samples = ramp(96_000 * 4, 1);
    let blocks: Vec<_> = offline::blocks(&samples, 48_000, 1, 96_000, 120.0, 0.0).collect();

    assert_eq!(blocks.len(), 4);
    let bars: Vec<f64> = blocks
        .iter()
        .map(|block| block.transport.expect("transport").bar_start_ppq)
        .collect();
    assert_eq!(
        bars,
        vec![0.0, 4.0, 8.0, 12.0],
        "one bar per block at 120 bpm"
    );

    // A one-second offset puts the first block before the downbeat.
    let offset: Vec<_> = offline::blocks(&samples, 48_000, 1, 96_000, 120.0, 1.0).collect();
    let first = offset[0].transport.expect("transport");
    assert!(
        (first.ppq - -2.0).abs() < 1e-12,
        "ppq {} at t=0 with offset 1 s",
        first.ppq
    );
    assert_eq!(first.bar_start_ppq, -4.0, "the bar before bar 0 is -4..0");
}

#[test]
fn blocks_drop_the_partial_tail_and_refuse_nonsense() {
    let samples = ramp(1000, 2);
    let blocks: Vec<_> = offline::blocks(&samples, 44_100, 2, 256, 93.0, 0.0).collect();
    assert_eq!(blocks.len(), 3, "1000 frames keep 3 whole 256-frame blocks");
    assert!(blocks.iter().all(|block| block.samples.len() == 512));

    assert_eq!(
        offline::blocks(&samples, 44_100, 0, 256, 93.0, 0.0).count(),
        0
    );
    assert_eq!(offline::blocks(&samples, 0, 2, 256, 93.0, 0.0).count(), 0);
    assert_eq!(
        offline::blocks(&samples, 44_100, 2, 0, 93.0, 0.0).count(),
        0
    );
}

// ---------------------------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------------------------

fn four_marks() -> Vec<Mark> {
    vec![
        Mark {
            bar: 16,
            kind: MarkKind::Build,
        },
        Mark {
            bar: 32,
            kind: MarkKind::Drop,
        },
        Mark {
            bar: 48,
            kind: MarkKind::Breakdown,
        },
        Mark {
            bar: 64,
            kind: MarkKind::Section,
        },
    ]
}

#[test]
fn labels_round_trip_through_toml() {
    let dir = TempDir::new("labels");
    let path = dir.file("track.labels.toml");
    let labels = Labels {
        bpm: Some(93.0),
        offset_s: Some(0.25),
        marks: four_marks(),
    };

    labels.save(&path).expect("save labels");
    let text = fs::read_to_string(&path).expect("read back");

    assert!(text.contains("[[mark]]"), "the documented shape:\n{text}");
    assert!(
        text.contains("kind = \"build\""),
        "kinds are lower-case names:\n{text}"
    );
    assert!(text.contains("bar = 16"), "bars are integers:\n{text}");
    assert_eq!(Labels::load(&path).expect("load labels"), labels);

    // A file with no marks and no metadata is still a valid, empty label file.
    let bare = dir.file("bare.toml");
    fs::write(&bare, "# nothing marked yet\n").expect("write");
    assert_eq!(Labels::load(&bare).expect("load bare"), Labels::default());
}

#[test]
fn bad_label_file_names_the_problem() {
    let dir = TempDir::new("badlabels");
    let path = dir.file("wrong.toml");
    fs::write(&path, "[[mark]]\nbar = 8\nkind = \"chorus\"\n").expect("write");
    let error = Labels::load(&path).expect_err("an unknown kind is an error");
    assert!(matches!(error, LabelError::Parse(_)), "got {error:?}");
    let message = error.to_string();
    assert!(message.contains("wrong.toml"), "{message}");
    assert!(
        message.contains("chorus"),
        "the message names the bad value: {message}"
    );

    let missing = Labels::load(dir.file("absent.toml")).expect_err("missing file is an error");
    assert!(matches!(missing, LabelError::Io { .. }), "got {missing:?}");
}

#[test]
fn labels_path_for_sits_next_to_the_audio() {
    assert_eq!(
        Labels::path_for(Path::new("C:/tracks/reza pedia dmin 93.wav")),
        PathBuf::from("C:/tracks/reza pedia dmin 93.labels.toml")
    );
    assert_eq!(
        Labels::path_for(Path::new("loyal remix v3.mp3")),
        PathBuf::from("loyal remix v3.labels.toml")
    );
}

#[test]
fn propose_maps_events_to_marks_once_per_bar() {
    let events = vec![
        (0, EventFlags::SONG_START),
        (8, EventFlags::BEAT | EventFlags::BAR),
        (8, EventFlags::BUILD_START),
        (8, EventFlags::BUILD_START),
        (16, EventFlags::DROP),
        (24, EventFlags::BREAKDOWN | EventFlags::PHRASE),
    ];

    let labels = offline::propose_from_events(events);

    assert_eq!(
        labels.marks,
        vec![
            Mark {
                bar: 0,
                kind: MarkKind::Section
            },
            Mark {
                bar: 8,
                kind: MarkKind::Build
            },
            Mark {
                bar: 16,
                kind: MarkKind::Drop
            },
            Mark {
                bar: 24,
                kind: MarkKind::Breakdown
            },
        ]
    );
    assert_eq!(labels.bpm, None, "a proposal does not invent a tempo");
    assert_eq!(
        labels.counts(),
        vec![
            (MarkKind::Build, 1),
            (MarkKind::Drop, 1),
            (MarkKind::Breakdown, 1),
            (MarkKind::Section, 1),
        ]
    );
}

#[test]
fn propose_writes_labels_file() {
    let dir = TempDir::new("propose");
    let audio = dir.file("vampira v2_edit 137bpm - metamor.wav");
    fs::write(&audio, b"").expect("placeholder");

    let proposed = offline::propose_from_events(vec![
        (0, EventFlags::SONG_START),
        (32, EventFlags::BUILD_START),
        (48, EventFlags::DROP),
    ]);
    let out = Labels::path_for(&audio);
    assert_eq!(
        out.file_name().and_then(|n| n.to_str()),
        Some("vampira v2_edit 137bpm - metamor.labels.toml")
    );

    proposed.save(&out).expect("write proposal");
    let reread = Labels::load(&out).expect("read proposal back");
    assert_eq!(reread.marks, proposed.marks);
    assert_eq!(reread.bars_of(MarkKind::Drop), vec![48]);
}

// ---------------------------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------------------------

fn kind_of(score: &offline::Score, kind: MarkKind) -> KindScore {
    *score.kind(kind).expect("every kind is reported")
}

#[test]
fn score_counts_hits_within_one_bar() {
    let labels = Labels::new(vec![
        Mark {
            bar: 8,
            kind: MarkKind::Build,
        },
        Mark {
            bar: 16,
            kind: MarkKind::Drop,
        },
        Mark {
            bar: 32,
            kind: MarkKind::Breakdown,
        },
    ]);
    let detected = Labels::new(vec![
        Mark {
            bar: 9,
            kind: MarkKind::Build,
        },
        Mark {
            bar: 18,
            kind: MarkKind::Drop,
        },
        Mark {
            bar: 40,
            kind: MarkKind::Breakdown,
        },
        Mark {
            bar: 56,
            kind: MarkKind::Section,
        },
    ]);

    let score = offline::score(&labels, &detected);

    let build = kind_of(&score, MarkKind::Build);
    assert_eq!(
        (build.labelled, build.detected, build.hits, build.misses),
        (1, 1, 1, 0)
    );
    assert_eq!(
        build.mean_error_bars,
        Some(1.0),
        "one bar late is still a hit"
    );
    assert_eq!(build.mean_abs_error_bars, Some(1.0));

    let drop = kind_of(&score, MarkKind::Drop);
    assert_eq!(
        (
            drop.labelled,
            drop.detected,
            drop.hits,
            drop.misses,
            drop.false_positives
        ),
        (1, 1, 0, 1, 1),
        "two bars late is a miss and a false positive"
    );
    assert_eq!(
        drop.mean_error_bars, None,
        "no hits, no timing error to report"
    );

    let breakdown = kind_of(&score, MarkKind::Breakdown);
    assert_eq!(
        (breakdown.hits, breakdown.misses),
        (0, 1),
        "eight bars late is a miss"
    );

    let section = kind_of(&score, MarkKind::Section);
    assert_eq!(
        (section.labelled, section.detected, section.false_positives),
        (0, 1, 1)
    );

    assert_eq!(
        (
            score.labelled,
            score.detected,
            score.hits,
            score.misses,
            score.false_positives
        ),
        (3, 4, 1, 2, 3)
    );
}

#[test]
fn scoring_never_uses_one_detection_for_two_labels() {
    let labels = Labels::new(vec![
        Mark {
            bar: 16,
            kind: MarkKind::Drop,
        },
        Mark {
            bar: 17,
            kind: MarkKind::Drop,
        },
    ]);
    let detected = Labels::new(vec![Mark {
        bar: 16,
        kind: MarkKind::Drop,
    }]);

    let score = offline::score(&labels, &detected);
    let drop = kind_of(&score, MarkKind::Drop);

    assert_eq!(
        (drop.hits, drop.misses),
        (1, 1),
        "one detection answers one label only"
    );
    assert_eq!(drop.false_positives, 0);
}

#[test]
fn report_has_numbers_per_kind() {
    let labels = Labels::new(vec![Mark {
        bar: 16,
        kind: MarkKind::Drop,
    }]);
    let detected = Labels::new(vec![
        Mark {
            bar: 16,
            kind: MarkKind::Drop,
        },
        Mark {
            bar: 33,
            kind: MarkKind::Breakdown,
        },
    ]);

    let report = offline::score(&labels, &detected).report();

    assert!(
        report.contains("1 labelled, 2 detected: 1 hits, 0 misses, 1 false positives"),
        "{report}"
    );
    assert!(
        report.contains("drop") && report.contains("1 labelled, 1 hits, 0 misses, 0 extra"),
        "{report}"
    );
    assert!(report.contains("mean error +0.00 bars"), "{report}");
    assert!(report.contains("breakdown"), "{report}");
}

// ---------------------------------------------------------------------------------------------
// Tempo hints
// ---------------------------------------------------------------------------------------------

#[test]
fn tempo_from_the_file_name() {
    let cases: &[(&str, Option<f64>)] = &[
        ("reza pedia dmin 93.wav", Some(93.0)),
        ("biolsong 140ebmaj.wav", Some(140.0)),
        ("la suerte v3 live 87 amin.wav", Some(87.0)),
        ("vampira v2_edit 137bpm - metamor.wav", Some(137.0)),
        (
            "C:/Users/ilanf/moshloom-tracks/reza pedia dmin 93.wav",
            Some(93.0),
        ),
        ("track 128 bpm.wav", Some(128.0)),
        ("track 128_bpm.wav", Some(128.0)),
        ("slow thing 72.5.wav", Some(72.5)),
        ("yeezus shit.wav", None),
        ("dreamz.wav", None),
        ("move not dance.wav", None),
        ("peak sampling v4.wav", None),
        ("FUCK2.wav", None),
        ("todo pasa todos hacen basta v4.wav", None),
        ("t4t final.wav", None),
        ("dame de comer en la boca como un Berro v3.wav", None),
        ("2026.wav", None),
        ("too fast 240.wav", None),
        ("too slow 40.wav", None),
        ("", None),
    ];

    for (name, want) in cases {
        assert_eq!(
            offline::tempo_from_name(name),
            *want,
            "tempo_from_name({name:?})"
        );
    }
}

#[test]
fn a_marked_tempo_beats_an_earlier_number() {
    // "_120" is the year-ish number the user meant as a version; 137bpm is the tempo.
    assert_eq!(offline::tempo_from_name("edit 2 137bpm.wav"), Some(137.0));
    assert_eq!(offline::tempo_from_name("140bpm 93.wav"), Some(140.0));
}

// ---------------------------------------------------------------------------------------------
// Analysis wiring
// ---------------------------------------------------------------------------------------------

#[test]
fn decoded_audio_feeds_the_block_iterator() {
    let dir = TempDir::new("wiring");
    let audio = write_pcm16_fixture(&dir);
    let (samples, rate, channels) = offline::read_audio(&audio).expect("decode");
    let blocks: Vec<_> = offline::blocks(&samples, rate, channels, 256, 93.0, 0.0).collect();

    assert_eq!(blocks.len(), 2, "512 frames give two whole blocks");
    let first = blocks[0].transport.expect("transport");
    assert_eq!(first.tempo_bpm, 93.0);
    assert!(
        blocks[1].transport.expect("transport").ppq > first.ppq,
        "the grid advances"
    );
}
