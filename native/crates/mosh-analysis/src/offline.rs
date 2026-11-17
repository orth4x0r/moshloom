//! Offline harness for the tuning tracks: audio decoding, the label file format, and scoring.
//!
//! Task 1.5 of `docs/superpowers/plans/2026-11-17-native-port.md`.
//!
//! The live path (`Analyzer`, Task 1.3, and `IntensityModel`, Task 1.4) is not on main yet, so
//! everything here is either pure or behind the `live-analysis` cargo feature (off by default):
//!
//! - pure, always available and tested by `tests/offline.rs`: [`read_audio`], [`blocks`],
//!   [`Labels`], [`score`], [`tempo_from_name`], [`propose_from_events`];
//! - feature gated: [`analyze`], which plays a file through `Analyzer` and `IntensityModel`.
//!
//! Nothing here writes audio, and nothing here reads the presented canvas of anything else: a
//! file in, samples out.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use mosh_core::{AudioBlock, EventFlags, Transport};
use serde::{Deserialize, Serialize};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Block size the live path uses (the plugin's callback size). The harness uses the same one so
/// an offline run sees the same feature cadence as a live set.
pub const BLOCK_FRAMES: usize = 256;

/// Bars in the synthetic offline transport. 4/4 only: the tuning tracks are all 4/4.
pub const SYNTH_SIG_NUM: u16 = 4;
pub const SYNTH_SIG_DEN: u16 = 4;

/// A file name number outside this range is not a tempo (`v3`, `v4`, `2026`).
pub const TEMPO_RANGE: std::ops::RangeInclusive<f64> = 60.0..=200.0;

/// A labelled mark is a hit when a detected mark of the same kind lands within this many bars.
pub const MATCH_WINDOW_BARS: i64 = 1;

// ---------------------------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------------------------

/// Why a file could not be turned into interleaved `f32` samples.
#[derive(Debug)]
pub enum OfflineError {
    /// The path could not be opened (missing, permission, a directory).
    Open { path: String, source: io::Error },
    /// The bytes are not audio this build can read (no probe, no track, no decoder).
    Unsupported(String),
    /// The container was recognised but a packet failed to decode.
    Decode(String),
    /// The file decoded to nothing.
    Empty(String),
}

impl fmt::Display for OfflineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OfflineError::Open { path, source } => write!(f, "cannot open {path}: {source}"),
            OfflineError::Unsupported(what) => write!(f, "unsupported audio: {what}"),
            OfflineError::Decode(what) => write!(f, "decode failed: {what}"),
            OfflineError::Empty(what) => write!(f, "no audio samples: {what}"),
        }
    }
}

impl std::error::Error for OfflineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OfflineError::Open { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Read a WAV (PCM 8/16/24/32-bit or IEEE float 32/64-bit) or MP3 file into interleaved `f32`.
///
/// Returns `(samples, sample_rate, channels)`; `samples.len() == frames * channels`.
/// Float input is passed through untouched, so the tuning set's IEEE-float WAV keeps its values.
pub fn read_audio(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32, u16), OfflineError> {
    let path = path.as_ref();
    let name = path.display().to_string();
    let file = fs::File::open(path).map_err(|source| OfflineError::Open {
        path: name.clone(),
        source,
    })?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| OfflineError::Unsupported(format!("{name}: {e}")))?
        .format;

    let track = format
        .default_track()
        .ok_or_else(|| OfflineError::Unsupported(format!("{name}: no audio track")))?;
    let track_id = track.id;
    let params = track.codec_params.clone();

    let mut decoder = symphonia::default::get_codecs()
        .make(&params, &DecoderOptions::default())
        .map_err(|e| OfflineError::Unsupported(format!("{name}: {e}")))?;

    let mut rate = params.sample_rate.unwrap_or(0);
    let mut channels = 0u16;
    let mut samples: Vec<f32> = Vec::new();
    let mut buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            // Symphonia reports a normal end of stream as an UnexpectedEof io error.
            Err(SymphoniaError::IoError(e)) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(OfflineError::Decode(format!("{name}: {e}"))),
        };
        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                let frames = decoded.frames();
                if channels == 0 {
                    channels = u16::try_from(spec.channels.count()).unwrap_or(u16::MAX);
                }
                if rate == 0 {
                    rate = spec.rate;
                }
                // Reuse the conversion buffer; replace it only when a packet is bigger than the last.
                if buf.as_ref().is_none_or(|buffer| buffer.capacity() < frames) {
                    buf = Some(SampleBuffer::<f32>::new(frames as u64, spec));
                }
                if let Some(buffer) = buf.as_mut() {
                    buffer.copy_interleaved_ref(decoded);
                    samples.extend_from_slice(buffer.samples());
                }
            }
            // One bad frame in a long file is not a reason to abandon the whole track.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::IoError(e)) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(OfflineError::Decode(format!("{name}: {e}"))),
        }
    }

    if samples.is_empty() {
        return Err(OfflineError::Empty(name));
    }
    if rate == 0 || channels == 0 {
        return Err(OfflineError::Unsupported(format!(
            "{name}: decoded but the sample rate or channel count is unknown"
        )));
    }
    Ok((samples, rate, channels))
}

// ---------------------------------------------------------------------------------------------
// Blocks and the synthetic transport
// ---------------------------------------------------------------------------------------------

/// Quarter-note position at a wall-clock time, for a constant tempo with `offset_s` at ppq 0.
pub fn ppq_at(time_s: f64, bpm: f64, offset_s: f64) -> f64 {
    (time_s - offset_s) * bpm / 60.0
}

/// Start of the bar that contains `ppq`, from the time signature (never `ppq / 4` in the app,
/// but this harness is 4/4-only by construction).
pub fn bar_start_ppq(ppq: f64, sig_num: u16) -> f64 {
    let beats = f64::from(sig_num.max(1));
    (ppq / beats).floor() * beats
}

/// Split interleaved samples into [`AudioBlock`]s with a synthetic constant-tempo 4/4 transport.
///
/// The final partial block is dropped, so every block is exactly `block_frames` frames and the
/// feature stage sees the same block size it sees live. A `block_frames` or `rate` of 0, or a
/// `channels` of 0, yields no blocks instead of dividing by zero.
pub fn blocks<'a>(
    samples: &'a [f32],
    rate: u32,
    channels: u16,
    block_frames: usize,
    bpm: f64,
    offset_s: f64,
) -> impl Iterator<Item = AudioBlock> + 'a {
    let frames = if channels == 0 {
        0
    } else {
        samples.len() / usize::from(channels)
    };
    let usable = if block_frames == 0 || rate == 0 {
        0
    } else {
        frames / block_frames
    };
    let stride = usize::from(channels);
    (0..usable).map(move |index| {
        let start = index * block_frames;
        let time_s = start as f64 / f64::from(rate);
        let ppq = ppq_at(time_s, bpm, offset_s);
        let from = start * stride;
        let to = from + block_frames * stride;
        AudioBlock {
            transport: Some(Transport {
                tempo_bpm: bpm,
                ppq,
                bar_start_ppq: bar_start_ppq(ppq, SYNTH_SIG_NUM),
                sig_num: SYNTH_SIG_NUM,
                sig_den: SYNTH_SIG_DEN,
                playing: true,
                sample_pos: start as i64,
            }),
            sample_rate: rate,
            channels,
            samples: samples[from..to].to_vec(),
        }
    })
}

// ---------------------------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------------------------

/// What a mark means. The four kinds `--propose` can write.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MarkKind {
    Build,
    Drop,
    Breakdown,
    Section,
}

impl MarkKind {
    /// Every kind, in reporting order.
    pub const ALL: [MarkKind; 4] = [
        MarkKind::Build,
        MarkKind::Drop,
        MarkKind::Breakdown,
        MarkKind::Section,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            MarkKind::Build => "build",
            MarkKind::Drop => "drop",
            MarkKind::Breakdown => "breakdown",
            MarkKind::Section => "section",
        }
    }

    /// The event that proposes this kind of mark, when the intensity model fires it.
    pub fn event(self) -> EventFlags {
        match self {
            MarkKind::Build => EventFlags::BUILD_START,
            MarkKind::Drop => EventFlags::DROP,
            MarkKind::Breakdown => EventFlags::BREAKDOWN,
            MarkKind::Section => EventFlags::SONG_START,
        }
    }

    pub fn from_name(name: &str) -> Option<MarkKind> {
        MarkKind::ALL.into_iter().find(|kind| kind.as_str() == name)
    }
}

/// One hand-corrected (or proposed) moment in a track, in bars from the start of the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Mark {
    pub bar: i64,
    pub kind: MarkKind,
}

/// A label file: `[[mark]]` entries, plus the optional `bpm` / `offset_s` this file was made
/// with, so a tuning run needs no extra flags to reproduce it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Labels {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bpm: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset_s: Option<f64>,
    #[serde(default, rename = "mark")]
    pub marks: Vec<Mark>,
}

impl Labels {
    /// Labels with no tempo metadata.
    pub fn new(marks: Vec<Mark>) -> Labels {
        Labels {
            bpm: None,
            offset_s: None,
            marks,
        }
    }

    pub fn parse(text: &str) -> Result<Labels, LabelError> {
        toml::from_str(text).map_err(|e| LabelError::Parse(e.to_string()))
    }

    pub fn to_toml(&self) -> Result<String, LabelError> {
        toml::to_string(self).map_err(|e| LabelError::Encode(e.to_string()))
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Labels, LabelError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|source| LabelError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Labels::parse(&text).map_err(|e| e.with_path(path))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), LabelError> {
        let path = path.as_ref();
        let text = self.to_toml()?;
        fs::write(path, text).map_err(|source| LabelError::Io {
            path: path.display().to_string(),
            source,
        })
    }

    /// Bars carrying `kind`, ascending.
    pub fn bars_of(&self, kind: MarkKind) -> Vec<i64> {
        let mut bars: Vec<i64> = self
            .marks
            .iter()
            .filter(|mark| mark.kind == kind)
            .map(|mark| mark.bar)
            .collect();
        bars.sort_unstable();
        bars
    }

    /// How many marks of each kind, in [`MarkKind::ALL`] order.
    pub fn counts(&self) -> Vec<(MarkKind, usize)> {
        MarkKind::ALL
            .into_iter()
            .map(|kind| (kind, self.bars_of(kind).len()))
            .collect()
    }

    /// `"<audio stem>.labels.toml"` next to the audio file: where `--propose` writes.
    pub fn path_for(audio: impl AsRef<Path>) -> PathBuf {
        audio.as_ref().with_extension("labels.toml")
    }
}

/// Why a label file could not be read, parsed or written.
#[derive(Debug)]
pub enum LabelError {
    Io { path: String, source: io::Error },
    Parse(String),
    Encode(String),
}

impl LabelError {
    fn with_path(self, path: &Path) -> LabelError {
        match self {
            LabelError::Parse(message) => {
                LabelError::Parse(format!("{}: {message}", path.display()))
            }
            other => other,
        }
    }
}

impl fmt::Display for LabelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LabelError::Io { path, source } => write!(f, "cannot read {path}: {source}"),
            LabelError::Parse(message) => write!(f, "bad label file: {message}"),
            LabelError::Encode(message) => write!(f, "cannot encode labels: {message}"),
        }
    }
}

impl std::error::Error for LabelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LabelError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Turn `(bar_index, events)` pairs into the label file `--propose` writes. One mark per kind per
/// bar, in bar order. Pure, so the mapping is testable without the intensity model.
pub fn propose_from_events(events: impl IntoIterator<Item = (i64, EventFlags)>) -> Labels {
    let mut marks: Vec<Mark> = Vec::new();
    for (bar, flags) in events {
        for kind in MarkKind::ALL {
            let mark = Mark { bar, kind };
            if flags.contains(kind.event()) && !marks.contains(&mark) {
                marks.push(mark);
            }
        }
    }
    marks.sort_unstable();
    Labels::new(marks)
}

// ---------------------------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------------------------

/// How one kind of mark scored: hits within [`MATCH_WINDOW_BARS`], the rest misses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KindScore {
    pub kind: MarkKind,
    pub labelled: usize,
    pub detected: usize,
    pub hits: usize,
    pub misses: usize,
    /// Detected marks that matched no label.
    pub false_positives: usize,
    /// Mean of `detected - labelled` in bars over the hits; `None` when there were no hits.
    pub mean_error_bars: Option<f64>,
    /// Mean of `|detected - labelled|` in bars over the hits; `None` when there were no hits.
    pub mean_abs_error_bars: Option<f64>,
}

/// Ticks per kind plus the totals.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Score {
    pub kinds: Vec<KindScore>,
    pub labelled: usize,
    pub detected: usize,
    pub hits: usize,
    pub misses: usize,
    pub false_positives: usize,
}

impl Score {
    pub fn kind(&self, kind: MarkKind) -> Option<&KindScore> {
        self.kinds.iter().find(|score| score.kind == kind)
    }

    /// Per-kind report, one line per kind that has anything to say, then nothing else. Numbers,
    /// not adjectives: hits and misses are counts, timing error is in bars.
    pub fn report(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "{} labelled, {} detected: {} hits, {} misses, {} false positives",
            self.labelled, self.detected, self.hits, self.misses, self.false_positives
        );
        for kind in &self.kinds {
            if kind.labelled == 0 && kind.detected == 0 {
                continue;
            }
            let timing = match (kind.mean_error_bars, kind.mean_abs_error_bars) {
                (Some(signed), Some(abs)) => {
                    format!(", mean error {signed:+.2} bars, abs {abs:.2}")
                }
                _ => ", no hits".to_string(),
            };
            let _ = writeln!(
                out,
                "  {:<9} {} labelled, {} hits, {} misses, {} extra{timing}",
                kind.kind.as_str(),
                kind.labelled,
                kind.hits,
                kind.misses,
                kind.false_positives
            );
        }
        out
    }
}

/// Score the detected marks against the hand-corrected ones, per kind, within one bar.
///
/// A label takes the nearest still-unused detection of the same kind within
/// [`MATCH_WINDOW_BARS`], so one detection can only answer for one label.
pub fn score(labels: &Labels, detected: &Labels) -> Score {
    let kinds: Vec<KindScore> = MarkKind::ALL
        .into_iter()
        .map(|kind| score_kind(labels, detected, kind))
        .collect();
    Score {
        labelled: kinds.iter().map(|k| k.labelled).sum(),
        detected: kinds.iter().map(|k| k.detected).sum(),
        hits: kinds.iter().map(|k| k.hits).sum(),
        misses: kinds.iter().map(|k| k.misses).sum(),
        false_positives: kinds.iter().map(|k| k.false_positives).sum(),
        kinds,
    }
}

fn score_kind(labels: &Labels, detected: &Labels, kind: MarkKind) -> KindScore {
    let wanted = labels.bars_of(kind);
    let found = detected.bars_of(kind);
    let mut used = vec![false; found.len()];
    let mut hits = 0usize;
    let mut signed_error = 0i64;
    let mut abs_error = 0i64;

    for bar in &wanted {
        let mut best: Option<(usize, i64)> = None;
        for (index, candidate) in found.iter().enumerate() {
            if used[index] {
                continue;
            }
            let delta = candidate - bar;
            if delta.abs() > MATCH_WINDOW_BARS {
                continue;
            }
            if best.is_none_or(|(_, best_delta)| delta.abs() < best_delta.abs()) {
                best = Some((index, delta));
            }
        }
        if let Some((index, delta)) = best {
            used[index] = true;
            hits += 1;
            signed_error += delta;
            abs_error += delta.abs();
        }
    }

    let hits_f = if hits == 0 { None } else { Some(hits as f64) };
    KindScore {
        kind,
        labelled: wanted.len(),
        detected: found.len(),
        hits,
        misses: wanted.len() - hits,
        false_positives: used.iter().filter(|used| !**used).count(),
        mean_error_bars: hits_f.map(|hits| signed_error as f64 / hits),
        mean_abs_error_bars: hits_f.map(|hits| abs_error as f64 / hits),
    }
}

// ---------------------------------------------------------------------------------------------
// Tempo from a file name
// ---------------------------------------------------------------------------------------------

/// A tempo written into a file name, if there is one.
///
/// `"reza pedia dmin 93.wav"` -> 93, `"biolsong 140ebmaj.wav"` -> 140,
/// `"vampira v2_edit 137bpm - metamor.wav"` -> 137, `"dreamz.wav"` -> None. A number followed by
/// `bpm` wins; otherwise the first number in [`TEMPO_RANGE`] is taken, and numbers outside it
/// (`v3`, `v4`, a year) are never a tempo.
pub fn tempo_from_name(name: &str) -> Option<f64> {
    let stem = name
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(name)
        .to_ascii_lowercase();
    let bytes = stem.as_bytes();
    let mut marked: Option<f64> = None;
    let mut plain: Option<f64> = None;
    let mut index = 0usize;

    while index < bytes.len() {
        if !bytes[index].is_ascii_digit() {
            index += 1;
            continue;
        }
        let start = index;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
        // One decimal point, and only when a digit follows it: "72.5.wav" is 72.5, "93.wav" is 93.
        if index + 1 < bytes.len() && bytes[index] == b'.' && bytes[index + 1].is_ascii_digit() {
            index += 1;
            while index < bytes.len() && bytes[index].is_ascii_digit() {
                index += 1;
            }
        }
        let Ok(value) = stem[start..index].parse::<f64>() else {
            continue;
        };
        if !TEMPO_RANGE.contains(&value) {
            continue;
        }
        let tail = stem[index..].trim_start_matches([' ', '_', '-']);
        if tail.starts_with("bpm") {
            marked = Some(value);
            break;
        }
        plain.get_or_insert(value);
    }

    marked.or(plain)
}

// ---------------------------------------------------------------------------------------------
// The whole run, once the live analysis is on main
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "live-analysis")]
pub use live::{AnalysisRun, analyze};

#[cfg(feature = "live-analysis")]
mod live {
    use super::*;
    // `intensity` lives in its own module (Task 1.4, already on main). `Analyzer` and
    // `AnalysisConfig` are Task 1.3, which owns `lib.rs` and re-exports them from the crate root.
    use crate::intensity::IntensityModel;
    use crate::{AnalysisConfig, Analyzer};

    /// What one file produced.
    #[derive(Clone, Debug, Default)]
    pub struct AnalysisRun {
        /// Proposed marks, ready to be saved and corrected by hand.
        pub labels: Labels,
        /// Completed bars, i.e. how many `BarStats` the intensity model saw.
        pub bars: usize,
        /// Audio actually played through the analyzer (`floor(frames / BLOCK_FRAMES)` blocks).
        pub duration_s: f64,
    }

    /// Play interleaved samples through `Analyzer` and `IntensityModel` at a constant tempo.
    pub fn analyze(
        samples: &[f32],
        rate: u32,
        channels: u16,
        bpm: f64,
        offset_s: f64,
    ) -> AnalysisRun {
        let mut analyzer = Analyzer::new(AnalysisConfig {
            fft_size: 2048,
            phrase_bars: 8,
            history_bars: 16,
        });
        let mut model = IntensityModel::new(8, 16);
        let mut events: Vec<(i64, EventFlags)> = Vec::new();
        let mut bars = 0usize;
        let mut played = 0usize;

        for block in blocks(samples, rate, channels, BLOCK_FRAMES, bpm, offset_s) {
            played += block.frames();
            analyzer.push(&block);
            let frame = analyzer.take_frame();
            events.push((frame.bar_index, frame.events));
            for bar in analyzer.drain_bars() {
                let update = model.push_bar(&bar);
                events.push((bar.bar_index, update.events));
                bars += 1;
            }
        }

        AnalysisRun {
            labels: Labels::new(propose_from_events(events).marks),
            bars,
            duration_s: if rate == 0 {
                0.0
            } else {
                played as f64 / f64::from(rate)
            },
        }
    }
}
