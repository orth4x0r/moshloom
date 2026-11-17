//! mosh-analysis: see docs/superpowers/plans/2026-11-17-native-port.md, Task 1.3, 1.4, 1.5.
//!
//! Task 1.3 (this module, `features`, `grid`, `loudness`): causal live features on Ableton's beat
//! grid. One [`Analyzer`] per show; push audio blocks as they arrive and take one
//! [`AnalysisFrame`](mosh_core::AnalysisFrame) per video frame. Nothing here looks ahead:
//! every value is computed from audio that has already been pushed.
//!
//! The intensity model (Task 1.4, `intensity.rs`) turns the per-bar [`BarStats`] this stage
//! produces into `intensity`, `level` and level/section events; until it is wired in the frame
//! carries the neutral `intensity = 0.0, level = 1`.

pub mod features;
pub mod grid;
pub mod intensity;
pub mod loudness;
pub mod offline;

use mosh_core::analysis::{AnalysisFrame, BarStats, EventFlags, TransportSource};
use mosh_core::transport::AudioBlock;

use features::{FeatureValues, FeatureExtractor, ONSET_PEAK_FLOOR};
use grid::{BeatGrid, CompletedBar};
use loudness::LoudnessMeter;

/// A hanning-windowed 2048-point FFT is 43 ms of history at 48 kHz: enough for 43 Hz resolution
/// without smoothing away the transients the visuals react to.
pub const DEFAULT_FFT_SIZE: usize = 2048;
/// A phrase is eight bars.
pub const DEFAULT_PHRASE_BARS: u32 = 8;
/// Normalisation looks back this many bars (a phrase's worth and then some).
pub const DEFAULT_HISTORY_BARS: u32 = 16;

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
/// Default normalisation ranges, used until the rolling window has two bars of history (the
/// first two seconds of a show, and after a reset). They are typical ranges for the raw values,
/// not tuned constants: once history exists, the 5th-95th percentile of the last bars wins.
const DEFAULT_LOUDNESS_DB: (f32, f32) = (-45.0, -8.0);
const DEFAULT_LOW_DB: (f32, f32) = (-60.0, -18.0);
const DEFAULT_BRIGHTNESS: (f32, f32) = (0.02, 0.45);
/// `BarStats::onset_density` counts onset peaks per beat, saturating at four (16th notes).
const ONSETS_PER_BEAT_MAX: f32 = 4.0;

/// A causal percentile range over the last `history_bars` bars.
///
/// `normalize` maps a raw value through the 5th and 95th percentile of the bars that have already
/// finished: the range is never widened by data the show has not heard yet, so a build does not
/// look quiet because of a loud drop later in the song.
#[derive(Clone, Debug)]
pub struct RollingRange {
    buf: [f32; HISTORY_CAP],
    len: usize,
    pos: usize,
    cap: usize,
    fallback: (f32, f32),
}

impl RollingRange {
    pub fn new(fallback: (f32, f32), history_bars: u32) -> Self {
        Self {
            buf: [0.0; HISTORY_CAP],
            len: 0,
            pos: 0,
            cap: (history_bars as usize).clamp(2, HISTORY_CAP),
            fallback,
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

    /// Map `value` to 0..1 against the 5th-95th percentile of the recorded bars.
    pub fn normalize(&self, value: f32) -> f32 {
        if self.len >= 2 {
            // Insertion sort over at most 32 floats: no allocation, and the ring is small enough
            // that a heap sort would cost more than it saves.
            let mut sorted = [0.0f32; HISTORY_CAP];
            sorted[..self.len].copy_from_slice(&self.buf[..self.len]);
            let s = &mut sorted[..self.len];
            for i in 1..s.len() {
                let mut j = i;
                while j > 0 && s[j - 1] > s[j] {
                    s.swap(j - 1, j);
                    j -= 1;
                }
            }
            let lo = percentile(s, 0.05);
            let hi = percentile(s, 0.95);
            if hi - lo > 1e-4 {
                return ((value - lo) / (hi - lo)).clamp(0.0, 1.0);
            }
        }
        self.fallback_map(value)
    }

    fn fallback_map(&self, value: f32) -> f32 {
        let (lo, hi) = self.fallback;
        if hi - lo <= 1e-6 {
            return 0.0;
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
    frame: AnalysisFrame,
    sample_rate: u32,
    sample_pos: i64,
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
            low: RollingRange::new(DEFAULT_LOW_DB, history),
            mid: RollingRange::new(DEFAULT_LOW_DB, history),
            high: RollingRange::new(DEFAULT_LOW_DB, history),
            loud: RollingRange::new(DEFAULT_LOUDNESS_DB, history),
            bright: RollingRange::new(DEFAULT_BRIGHTNESS, history),
            acc: BarAccumulator::default(),
            bars: Vec::with_capacity(BAR_CAPACITY),
            events: EventFlags::empty(),
            frame: AnalysisFrame::default(),
            sample_rate: 48_000,
            sample_pos: 0,
        }
    }

    pub fn config(&self) -> &AnalysisConfig {
        &self.config
    }

    /// Analyse one block. Allocates nothing: the FFT plan, the block buffer, the loudness window
    /// and the normalisation rings are all created in [`Analyzer::new`] or on the first push
    /// (when the sample rate is first known).
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
            let tick = self.grid.update(&t);
            self.frame.ppq = t.ppq;
            self.frame.tempo_bpm = tick.bpm;
            self.frame.playing = t.playing;
            self.frame.source = TransportSource::Plugin;
            self.frame.beat_phase = tick.beat_phase;
            self.frame.bar_phase = tick.bar_phase;
            self.frame.beat_index = tick.beat_index;
            self.frame.bar_index = tick.bar_index;

            if tick.jumped {
                // The playhead moved: the partial bar and the onset run belong to audio that is
                // no longer before us, so they are dropped instead of being reported.
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
        } else {
            // No transport (loopback, or the plugin is not running): keep the last grid so the
            // visuals hold their tempo, but say where the position came from.
            self.frame.source = TransportSource::None;
            self.frame.playing = false;
        }

        self.acc.add_block(&fv, loudness_db);
        self.update_frame(&fv, loudness_db, frames);
    }

    /// The latest frame. `events` carries everything OR-ed since the previous take.
    pub fn take_frame(&mut self) -> AnalysisFrame {
        let mut f = self.frame;
        f.events = self.events;
        self.events = EventFlags::empty();
        f
    }

    /// Bars that closed since the previous call. Consumed by the intensity model (Task 1.4).
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
        // Task 1.4 owns the intensity model; until the app wires it in, the frame says "level 1,
        // no intensity", which every visual interprets as the neutral state.
        f.intensity = 0.0;
        f.level = 1;
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
        let onset_density =
            (self.acc.onsets as f32 / beats / ONSETS_PER_BEAT_MAX).clamp(0.0, 1.0);

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
