//! mosh-analysis: see docs/superpowers/plans/2026-11-17-native-port.md, Task 1.3, 1.4, 1.5, 2.5.
//!
//! Task 1.3 (this module, `features`, `grid`, `loudness`): causal live features on Ableton's beat
//! grid. One [`Analyzer`] per show; push audio blocks as they arrive and take one
//! [`AnalysisFrame`](mosh_core::AnalysisFrame) per video frame. Nothing here looks ahead:
//! every value is computed from audio that has already been pushed.
//!
//! The intensity model (Task 1.4, `intensity.rs`) runs inside the analyzer: every closed bar is
//! pushed to it, and the frame carries its `intensity`, `level` and events (PHRASE, BUILD_START,
//! DROP, BREAKDOWN, LEVEL_UP, LEVEL_DOWN). Call [`Analyzer::song_start`] at a song marker.
//!
//! `BarStats` carries `loudness`, `low` and `brightness` normalised to 0..1 by [`RollingRange`]
//! (a fixed typical range blended with the bar's rank over the last `history_bars` bars) and
//! `onset_density` as raw onset peaks per beat, which the intensity model normalises itself,
//! because onset density is per beat and the model is the only place that knows the window it
//! should be read against.
//!
//! Loopback mode (Task 2.5, `tempo.rs`) has audio but no transport. A block whose
//! [`AudioBlock::transport`] is `None` feeds the [`tempo::TempoTracker`], and once it locks the
//! analyzer synthesises a 4/4 transport from it: the frame's `source` becomes
//! [`TransportSource::Tracked`], `playing` turns true, and beats, bars and phrases keep firing on
//! the tracked grid. With a transport present nothing about the tracker runs.

pub mod features;
pub mod grid;
pub mod intensity;
pub mod loudness;
pub mod offline;
pub mod tempo;

use mosh_core::analysis::{AnalysisFrame, BarStats, EventFlags, TransportSource};
use mosh_core::transport::{AudioBlock, Transport};

use features::{FeatureExtractor, FeatureValues, ONSET_PEAK_FLOOR};
use grid::{BeatGrid, CompletedBar};
use loudness::LoudnessMeter;
use tempo::TempoTracker;

/// A hanning-windowed 2048-point FFT is 43 ms of history at 48 kHz: enough for 43 Hz resolution
/// without smoothing away the transients the visuals react to.
pub const DEFAULT_FFT_SIZE: usize = 2048;
/// A phrase is eight bars.
pub const DEFAULT_PHRASE_BARS: u32 = 8;
/// Normalisation looks back this many bars: the top of the design's 8-32 range, because a
/// shorter window lets one deep bar (a one-bar gap, the last bar of an intro) set the range for
/// a whole section, and every bar after it then clips to 1.0.
pub const DEFAULT_HISTORY_BARS: u32 = 32;

/// How the analyzer is set up. The plan's defaults are [`Default`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalysisConfig {
    pub fft_size: usize,
    pub phrase_bars: u32,
    pub history_bars: u32,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            fft_size: DEFAULT_FFT_SIZE,
            phrase_bars: DEFAULT_PHRASE_BARS,
            history_bars: DEFAULT_HISTORY_BARS,
        }
    }
}

/// Largest history the causal normaliser keeps. The plan asks for 8-32 bars.
const HISTORY_CAP: usize = 32;
/// Bars buffered between `drain_bars` calls. One per video frame is the intended use.
const BAR_CAPACITY: usize = 8;
/// Typical ranges for produced music, the absolute half of each [`RollingRange`] map. They are
/// used before the rolling window has a rank to give (the first bars of a show, and after a
/// reset) and they anchor the rank afterwards, so a long quiet section still reads quiet.
const DEFAULT_LOUDNESS_DB: (f32, f32) = (-45.0, -8.0);
const DEFAULT_LOW_DB: (f32, f32) = (-60.0, -18.0);
const DEFAULT_BRIGHTNESS: (f32, f32) = (0.02, 0.45);
/// Smallest 10th-90th percentile spread the loudness map is fully confident in, in dB. Music
/// whose last 32 bars fit inside a 3 dB range has no dynamics to read a bar against, so that
/// bar stays near the middle of the scale instead of being ranked by noise.
const LOUDNESS_FULL_SPAN_DB: f32 = 3.0;
/// The same for a band's mean power, which moves less than the summed level.
const BAND_FULL_SPAN_DB: f32 = 4.0;
/// The same for brightness, which is a 0..1 spectral centroid, not a dB.
const BRIGHTNESS_FULL_SPAN: f32 = 0.08;
/// Bars before the relative rank is trusted: with fewer, the window is a handful of bars and its
/// top is not "the loudest in the last phrase", it is "the loudest in the last four bars".
const RANK_WARMUP_BARS: usize = 8;
/// `BarStats::onset_density` counts onset peaks per beat. The clamp only stops a broken
/// detector from handing the model an absurd number; typical material is 0.5-4.
const ONSETS_PER_BEAT_MAX: f32 = 8.0;
/// Loopback mode has no transport, so the tracker's grid is synthesised as 4/4, the time signature
/// the web engine assumes. A transport from the plugin always wins when one is present.
const TRACKED_SIG_NUM: u16 = 4;
const TRACKED_SIG_DEN: u16 = 4;
/// The tracked playhead follows the tracker's beat phase with this gain per block: a phase-locked
/// loop, so a re-estimate never jumps the playhead and drops a partial bar every half second.
const TRACKED_PHASE_GAIN: f64 = 0.05;

/// A causal map from a raw feature to 0..1: a fixed typical range blended with a rank over the
/// last `history_bars` bars.
///
/// Two things keep the map honest, and the model needs both:
///
/// - the **relative** half is a rank, not a min-max position: after a step up (an intro into a
///   drop) the new bars spread across the top of the window instead of all clipping to 1.0
///   together, which is what kept the old 5th-95th percentile map at "everything is maximal"
///   for a whole window. Ties (a perfectly steady window) rank in the middle, not at the top,
///   and a window with no spread at all is pulled to 0.5 rather than ranked by noise.
/// - the **absolute** half is the fixed typical range, so a long quiet section still reads
///   quiet when it is all the last bars contain, and a long loud one still reads loud.
///
/// The window never contains audio the show has not heard, so a build is not judged against a
/// later drop.
#[derive(Clone, Debug)]
pub struct RollingRange {
    buf: [f32; HISTORY_CAP],
    len: usize,
    pos: usize,
    cap: usize,
    fallback: (f32, f32),
    full_span: f32,
}

impl RollingRange {
    /// `fallback` covers the first bars of a show (before there is a window to rank against);
    /// `full_span` is the 10th-90th percentile spread at which the rank is trusted completely.
    pub fn new(fallback: (f32, f32), history_bars: u32, full_span: f32) -> Self {
        Self {
            buf: [0.0; HISTORY_CAP],
            len: 0,
            pos: 0,
            cap: (history_bars as usize).clamp(2, HISTORY_CAP),
            fallback,
            full_span: full_span.max(1e-6),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Record one finished bar. Allocates nothing.
    pub fn push(&mut self, value: f32) {
        if !value.is_finite() {
            return;
        }
        self.buf[self.pos] = value;
        self.pos = (self.pos + 1) % self.cap;
        if self.len < self.cap {
            self.len += 1;
        }
    }

    pub fn clear(&mut self) {
        self.len = 0;
        self.pos = 0;
    }

    /// Map `value` to 0..1: half from the fixed typical range for this feature, half from the
    /// bar's rank among the recorded bars and itself.
    ///
    /// The two halves answer two different questions, and the model needs both:
    ///
    /// - the **absolute** half says how loud, busy or bright this bar is for produced music at
    ///   all. Without it a long quiet section (a breakdown, a quiet intro that the song then
    ///   never beats) has nothing to be quiet *against* and settles in the middle of the scale;
    /// - the **relative** half says whether this bar moves, against the last `history_bars`
    ///   bars. Without it a track mastered 6 dB louder than the last one would sit a level higher
    ///   all night, and a per-bar change of 0.3 dB would move the level as much as a drop.
    pub fn normalize(&self, value: f32) -> f32 {
        if !value.is_finite() {
            return 0.5;
        }
        let absolute = self.fallback_map(value);
        let relative = self.relative_rank(value);
        (0.5 * absolute + 0.5 * relative).clamp(0.0, 1.0)
    }

    /// The rank of `value` among the recorded bars and itself, blended toward 0.5 as the
    /// window's spread falls below `full_span`, so a window with no dynamics at all reads the
    /// middle instead of ranking by noise.
    ///
    /// A rank needs a handful of bars to mean anything: over three bars the loudest is the top
    /// of the scale and the show opens at level 4. Until there are [`RANK_WARMUP_BARS`] bars the
    /// relative half reads the middle, so the opening bars are judged by the fixed typical range
    /// alone.
    fn relative_rank(&self, value: f32) -> f32 {
        if self.len < RANK_WARMUP_BARS {
            return 0.5;
        }
        let (sorted, n) = self.window_with(value);
        let s = &sorted[..n];
        let n = n as f32;
        let below = s.partition_point(|&other| other < value) as f32;
        let equal = s.partition_point(|&other| other <= value) as f32 - below;
        // Ties take the middle of their block, so a steady window ranks at 0.5.
        let rank = (below + 0.5 * equal) / n;
        // Full confidence when the window itself has dynamics, or when this bar is a clear
        // departure from the window's middle (a lone drop under a steady run, which the
        // percentiles alone cannot see).
        let median = percentile(s, 0.5);
        let span = percentile(s, 0.9) - percentile(s, 0.1);
        let deviation = (value - median).abs();
        let confidence = (span.max(deviation) / self.full_span).clamp(0.0, 1.0);
        (0.5 + (rank - 0.5) * confidence).clamp(0.0, 1.0)
    }

    /// Where `value` sits between the 5th and 95th percentile of the recorded bars and itself:
    /// 0.0 at the quiet edge, 0.5 in the middle, 1.0 at the loud edge. `None` when the window is
    /// still too short or its percentile range is narrower than `min_span`, in which case there
    /// is no range to place anything in and the caller should read the middle.
    pub fn position(&self, value: f32, min_span: f32) -> Option<f32> {
        if !value.is_finite() || self.len < 2 {
            return None;
        }
        let (sorted, n) = self.window_with(value);
        let s = &sorted[..n];
        let lo = percentile(s, 0.05);
        let hi = percentile(s, 0.95);
        let span = hi - lo;
        if span < min_span.max(1e-6) {
            return None;
        }
        Some(((value - lo) / span).clamp(0.0, 1.0))
    }

    fn window_with(&self, value: f32) -> ([f32; HISTORY_CAP + 1], usize) {
        // Insertion sort over at most 33 floats: no allocation, and the ring is small enough
        // that a heap sort would cost more than it saves. Every value is finite.
        let mut sorted = [0.0f32; HISTORY_CAP + 1];
        sorted[..self.len].copy_from_slice(&self.buf[..self.len]);
        sorted[self.len] = value;
        let n = self.len + 1;
        for i in 1..n {
            let mut j = i;
            while j > 0 && sorted[j - 1] > sorted[j] {
                sorted.swap(j - 1, j);
                j -= 1;
            }
        }
        (sorted, n)
    }

    fn fallback_map(&self, value: f32) -> f32 {
        let (lo, hi) = self.fallback;
        if hi - lo <= 1e-6 {
            return 0.5;
        }
        ((value - lo) / (hi - lo)).clamp(0.0, 1.0)
    }
}

fn percentile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((q * (sorted.len() as f32 - 1.0)).round() as usize).min(sorted.len() - 1);
    sorted[idx]
}

/// Everything the current bar has summed up so far.
#[derive(Clone, Copy, Debug, Default)]
struct BarAccumulator {
    blocks: u32,
    /// Summed linear power per band, averaged to dB when the bar closes.
    power: [f64; 3],
    /// Summed dB and brightness, averaged when the bar closes.
    loud_db: f64,
    brightness: f64,
    onsets: u32,
    prev_onset: f32,
    prev_prev_onset: f32,
}

impl BarAccumulator {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn add_block(&mut self, fv: &FeatureValues, loudness_db: f32) {
        self.blocks += 1;
        for b in 0..3 {
            self.power[b] += fv.power[b] as f64;
        }
        self.loud_db += loudness_db as f64;
        self.brightness += fv.brightness as f64;
        // A peak is a local maximum of the onset envelope above the detection floor. One block
        // of look-ahead, which is why the count for a bar is final one block after its end.
        if self.prev_onset > ONSET_PEAK_FLOOR
            && self.prev_onset > self.prev_prev_onset
            && self.prev_onset >= fv.onset
        {
            self.onsets += 1;
        }
        self.prev_prev_onset = self.prev_onset;
        self.prev_onset = fv.onset;
    }
}

/// The causal feature stage: one audio block in, one frame (and sometimes a bar) out.
pub struct Analyzer {
    config: AnalysisConfig,
    features: FeatureExtractor,
    loudness: LoudnessMeter,
    grid: BeatGrid,
    /// Mono downmix of the current block, reused every push.
    mono: Vec<f32>,
    low: RollingRange,
    mid: RollingRange,
    high: RollingRange,
    loud: RollingRange,
    bright: RollingRange,
    acc: BarAccumulator,
    bars: Vec<BarStats>,
    events: EventFlags,
    intensity: intensity::IntensityModel,
    level: u8,
    intensity_value: f32,
    frame: AnalysisFrame,
    sample_rate: u32,
    sample_pos: i64,
    /// Loopback mode: the causal tempo and beat-phase tracker. Fed only when a block arrives
    /// without a transport; with the plugin nothing here runs.
    tempo: TempoTracker,
    /// The synthesised musical position (ppq) of the tracked grid, and whether it is anchored.
    tracked_ppq: f64,
    tracked_started: bool,
}

impl Analyzer {
    pub fn new(config: AnalysisConfig) -> Self {
        let history = config.history_bars;
        Self {
            config,
            features: FeatureExtractor::new(config.fft_size),
            loudness: LoudnessMeter::new(),
            grid: BeatGrid::new(config.phrase_bars),
            mono: Vec::with_capacity(config.fft_size.max(1024)),
            low: RollingRange::new(DEFAULT_LOW_DB, history, BAND_FULL_SPAN_DB),
            mid: RollingRange::new(DEFAULT_LOW_DB, history, BAND_FULL_SPAN_DB),
            high: RollingRange::new(DEFAULT_LOW_DB, history, BAND_FULL_SPAN_DB),
            loud: RollingRange::new(DEFAULT_LOUDNESS_DB, history, LOUDNESS_FULL_SPAN_DB),
            bright: RollingRange::new(DEFAULT_BRIGHTNESS, history, BRIGHTNESS_FULL_SPAN),
            acc: BarAccumulator::default(),
            bars: Vec::with_capacity(BAR_CAPACITY),
            events: EventFlags::empty(),
            intensity: intensity::IntensityModel::new(config.phrase_bars, config.history_bars),
            level: 1,
            intensity_value: 0.0,
            frame: AnalysisFrame {
                // A consumer that reads the first frame before any bar has closed must see the
                // model's resting state, not the struct default of level 0.
                level: 1,
                ..AnalysisFrame::default()
            },
            sample_rate: 48_000,
            sample_pos: 0,
            tempo: TempoTracker::new(48_000),
            tracked_ppq: 0.0,
            tracked_started: false,
        }
    }

    pub fn config(&self) -> &AnalysisConfig {
        &self.config
    }

    /// Analyse one block. Allocates nothing in the steady state: the FFT plan, the block buffer,
    /// the loudness window, the normalisation rings and (loopback mode) the tempo tracker's window
    /// are created in [`Analyzer::new`] or on the first push that needs them.
    pub fn push(&mut self, b: &AudioBlock) {
        let frames = b.frames();
        if frames == 0 || b.sample_rate == 0 {
            return;
        }
        self.sample_rate = b.sample_rate;
        self.downmix(b, frames);
        let fv = self.features.process(&self.mono, b.sample_rate);
        let loudness_db = self.loudness.push(&self.mono, b.sample_rate);

        if let Some(t) = b.transport {
            self.tracked_started = false;
            self.advance(&t, TransportSource::Plugin);
        } else {
            // No transport (loopback capture, or the plugin is not running): the tracker estimates
            // the tempo and beat phase from the onset envelope it was just given. Until it locks,
            // the frame says the position is unknown; the last grid keeps the visuals running.
            self.tempo.set_sample_rate(b.sample_rate);
            self.tempo.push_onset(fv.onset, frames as u32);
            let dt = frames as f64 / b.sample_rate as f64;
            match self.tracked_transport(dt, frames) {
                Some(t) => self.advance(&t, TransportSource::Tracked),
                None => {
                    self.frame.source = TransportSource::None;
                    self.frame.playing = false;
                }
            }
        }

        self.acc.add_block(&fv, loudness_db);
        self.update_frame(&fv, loudness_db, frames);
    }

    /// The grid part of a block: phase, indices, and the beat/bar events a transport implies.
    fn advance(&mut self, t: &Transport, source: TransportSource) {
        let tick = self.grid.update(t);
        self.frame.ppq = t.ppq;
        self.frame.tempo_bpm = tick.bpm;
        self.frame.playing = t.playing;
        self.frame.source = source;
        self.frame.beat_phase = tick.beat_phase;
        self.frame.bar_phase = tick.bar_phase;
        self.frame.beat_index = tick.beat_index;
        self.frame.bar_index = tick.bar_index;

        if tick.jumped {
            // The playhead moved: the partial bar and the onset run belong to audio that is no
            // longer before us, so they are dropped instead of being reported.
            self.acc.reset();
        }
        if let Some(completed) = tick.completed {
            self.close_bar(completed);
        }
        if tick.beat_started {
            self.events |= EventFlags::BEAT;
        }
        if tick.bar_started {
            self.events |= EventFlags::BAR;
            if self.grid.is_phrase_boundary(tick.bar_index) {
                self.events |= EventFlags::PHRASE;
            }
        }
    }

    /// Synthesise a transport from the tempo tracker: 4/4, `playing` once locked, and a ppq that
    /// follows the tracker's phase through a phase-locked loop. `None` until the tracker locks.
    fn tracked_transport(&mut self, dt: f64, frames: usize) -> Option<Transport> {
        let est = self.tempo.estimate()?;
        if self.tracked_started {
            self.tracked_ppq += dt * est.bpm / 60.0;
            let phase = self.tracked_ppq.rem_euclid(1.0);
            let mut error = est.beat_phase - phase;
            if error > 0.5 {
                error -= 1.0;
            } else if error < -0.5 {
                error += 1.0;
            }
            self.tracked_ppq += error * TRACKED_PHASE_GAIN;
        } else {
            // First locked block: start on the tracker's phase, so beats land on the clicks.
            self.tracked_ppq = est.beat_phase;
            self.tracked_started = true;
        }
        let ppq = self.tracked_ppq.max(0.0);
        let bar_start_ppq = (ppq / f64::from(TRACKED_SIG_NUM)).floor() * f64::from(TRACKED_SIG_NUM);
        Some(Transport {
            tempo_bpm: est.bpm,
            ppq,
            bar_start_ppq,
            sig_num: TRACKED_SIG_NUM,
            sig_den: TRACKED_SIG_DEN,
            playing: true,
            sample_pos: self.sample_pos + frames as i64,
        })
    }

    /// The latest frame. `events` carries everything OR-ed since the previous take.
    pub fn take_frame(&mut self) -> AnalysisFrame {
        let mut f = self.frame;
        f.events = self.events;
        self.events = EventFlags::empty();
        f
    }

    /// A song marker was crossed: the intensity history restarts and the next frame carries
    /// SONG_START, so a quiet intro is judged against itself, not the previous song's drop.
    pub fn song_start(&mut self) {
        self.intensity.reset();
        self.level = 1;
        self.intensity_value = 0.0;
        self.frame.level = 1;
        self.frame.intensity = 0.0;
        self.events |= EventFlags::SONG_START;
    }

    /// Bars that closed since the previous call (for the offline harness and the stats overlay;
    /// the analyzer already feeds them to its own intensity model).
    pub fn drain_bars(&mut self) -> Vec<BarStats> {
        let mut out = Vec::with_capacity(self.bars.len().max(BAR_CAPACITY));
        out.extend_from_slice(&self.bars);
        self.bars.clear();
        out
    }

    /// Forget the loudness window and the bar in progress (a song marker), keeping the grid.
    pub fn reset(&mut self) {
        self.loudness.reset();
        self.acc.reset();
        self.low.clear();
        self.mid.clear();
        self.high.clear();
        self.loud.clear();
        self.bright.clear();
    }

    fn downmix(&mut self, b: &AudioBlock, frames: usize) {
        self.mono.clear();
        if self.mono.capacity() < frames {
            self.mono.reserve(frames - self.mono.len());
        }
        let channels = b.channels.max(1) as usize;
        if channels == 1 {
            self.mono.extend_from_slice(&b.samples[..frames]);
        } else {
            let scale = 1.0 / channels as f32;
            for f in 0..frames {
                let mut sum = 0.0;
                for c in 0..channels {
                    sum += b.samples[f * channels + c];
                }
                self.mono.push(sum * scale);
            }
        }
    }

    fn update_frame(&mut self, fv: &FeatureValues, loudness_db: f32, frames: usize) {
        self.sample_pos += frames as i64;
        let time_s = self.sample_pos as f64 / self.sample_rate as f64;

        let bands = [
            self.low.normalize(power_db(fv.power[0])),
            self.mid.normalize(power_db(fv.power[1])),
            self.high.normalize(power_db(fv.power[2])),
        ];
        let loudness = self.loud.normalize(loudness_db);
        let brightness = self.bright.normalize(fv.brightness);

        let f = &mut self.frame;
        f.time_s = time_s;
        f.bands = bands;
        f.mel = fv.mel;
        f.onset = fv.onset;
        f.loudness_db = loudness_db;
        f.loudness = loudness;
        f.brightness = brightness;
        // Intensity and level change only when a bar closes (see `close_bar`).
        f.intensity = self.intensity_value;
        f.level = self.level;
    }

    /// Turn the bar that just finished into [`BarStats`], normalised against the bars before it.
    fn close_bar(&mut self, completed: CompletedBar) {
        let blocks = self.acc.blocks.max(1);
        let inv = 1.0 / blocks as f64;
        let low_db = power_db((self.acc.power[0] * inv) as f32);
        let mid_db = power_db((self.acc.power[1] * inv) as f32);
        let high_db = power_db((self.acc.power[2] * inv) as f32);
        let loudness_db = (self.acc.loud_db * inv) as f32;
        let brightness = (self.acc.brightness * inv) as f32;
        let beats = completed.beats.max(1) as f32;
        // Onset peaks per beat, raw: the intensity model normalises it against its own window,
        // and dividing by four here would cap the term at 1.0 for 16th-note material and make
        // that normalisation dead code.
        let onset_density = (self.acc.onsets as f32 / beats).clamp(0.0, ONSETS_PER_BEAT_MAX);

        // Normalise against the bars before this one, then let this bar into the window.
        let bar = BarStats {
            bar_index: completed.bar_index,
            start_ppq: completed.start_ppq,
            beats: completed.beats,
            loudness_db,
            loudness: self.loud.normalize(loudness_db),
            low: self.low.normalize(low_db),
            onset_density,
            brightness: self.bright.normalize(brightness),
        };

        self.low.push(low_db);
        self.mid.push(mid_db);
        self.high.push(high_db);
        self.loud.push(loudness_db);
        self.bright.push(brightness);
        self.acc.reset();

        // The intensity model sees every bar. It fires PHRASE itself; the grid's PHRASE is the
        // same boundary, so OR-ing both is harmless.
        let update = self.intensity.push_bar(&bar);
        self.intensity_value = update.intensity;
        self.level = update.level;
        self.events |= update.events;
        self.frame.intensity = update.intensity;
        self.frame.level = update.level;

        if self.bars.len() < self.bars.capacity() {
            self.bars.push(bar);
        } else {
            // Never expected (drain_bars runs per video frame); keep the newest bar rather than
            // growing the buffer inside push.
            if let Some(last) = self.bars.last_mut() {
                *last = bar;
            }
        }
    }
}

/// Linear power to dBFS, floored at -120 dB so silence has a finite number.
fn power_db(power: f32) -> f32 {
    10.0 * (power.max(1e-12)).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FALLBACK: (f32, f32) = (-45.0, -8.0);
    const FULL_SPAN: f32 = 3.0;

    fn range() -> RollingRange {
        RollingRange::new(FALLBACK, 32, FULL_SPAN)
    }

    /// The first bars of a show have no window to rank against, so they read the fixed typical
    /// range blended with the middle of a rank they have not earned yet.
    #[test]
    fn the_first_bars_use_the_typical_range() {
        let mut r = range();
        assert!((r.normalize(-8.0) - 0.75).abs() < 1e-6);
        assert!((r.normalize(-45.0) - 0.25).abs() < 1e-6);
        assert!((r.normalize(-26.5) - 0.5).abs() < 1e-6);
        r.push(-26.5);
        // Still inside the warm-up: no rank yet, so the same value reads the same.
        assert!((r.normalize(-26.5) - 0.5).abs() < 1e-6);
        // The loud end of the typical range is still clearly above the quiet end.
        assert!(r.normalize(-8.0) > r.normalize(-26.5));
    }

    /// A window with no spread has no rank to give, so a perfectly steady bar only reads its
    /// absolute position: a steady loop at -20 dB reads 0.59, not the top of the scale.
    #[test]
    fn a_steady_window_reads_its_absolute_position() {
        let mut r = range();
        for _ in 0..32 {
            r.push(-20.0);
        }
        let mapped = r.normalize(-20.0);
        assert!(
            (0.55..0.65).contains(&mapped),
            "a steady -20 dB window mapped to {mapped}"
        );
    }

    /// After a step up the new bars spread across the top of the window instead of clipping to
    /// 1.0 together, which is what the old percentile map did for a whole window.
    #[test]
    fn a_step_up_spreads_the_ranks() {
        let mut r = range();
        for _ in 0..16 {
            r.push(-30.0);
        }
        let first = r.normalize(-10.0);
        r.push(-10.0);
        for _ in 0..5 {
            r.push(-10.0);
        }
        let sixth = r.normalize(-10.0);
        assert!(
            first > sixth,
            "the bar that steps up ({first}) must read higher than the settled one ({sixth})"
        );
        assert!(
            (0.6..0.9).contains(&sixth),
            "the settled loud bar mapped to {sixth}"
        );
        // A quiet bar still reads low: its rank is the bottom of a window that is mostly quiet.
        assert!(r.normalize(-30.0) < 0.45);
    }

    /// The absolute half is what keeps a long steady section where it belongs: the same window
    /// shape at two listening levels maps 0.25 apart, so a breakdown is quiet relative to the
    /// drop even once it is all the window contains.
    #[test]
    fn the_absolute_half_separates_quiet_from_loud() {
        let mut quiet = range();
        let mut loud = range();
        for _ in 0..32 {
            quiet.push(-30.0);
            loud.push(-10.0);
        }
        let q = quiet.normalize(-30.0);
        let l = loud.normalize(-10.0);
        assert!(
            l - q > 0.25,
            "a steady -10 dB window mapped {l}, a steady -30 dB one {q}"
        );
        assert!(q < 0.5 && l > 0.65, "{q} and {l}");
    }

    /// A lone deep bar (a one-bar gap, a filtered bar) does not pin the bars after it at 1.0:
    /// the ranks stay spread.
    #[test]
    fn one_outlier_does_not_pin_the_following_bars() {
        let mut r = range();
        for _ in 0..10 {
            r.push(-20.0);
        }
        r.push(-40.0);
        let top = r.normalize(-20.0);
        let below = r.normalize(-21.0);
        assert!(top < 0.99, "the steady level ranked {top}");
        assert!(
            below < top && below > 0.3,
            "-21 dB below a -20 dB window ranked {below}, top {top}"
        );
    }

    /// Values that are not finite never enter the window and never rank.
    #[test]
    fn non_finite_values_are_ignored() {
        let mut r = range();
        r.push(f32::NAN);
        assert!(r.is_empty());
        r.push(-20.0);
        assert_eq!(r.len(), 1);
        assert_eq!(r.normalize(f32::NAN), 0.5);
    }
}
