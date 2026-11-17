//! Causal tempo and beat-phase tracking for loopback mode (plan Task 2.5).
//!
//! Loopback capture carries audio but no transport: no tempo, no bar position, no play state. The
//! show still needs a beat grid, so this tracker estimates one from the onset envelope the feature
//! stage already produces ([`crate::features::FeatureValues::onset`]). It is the web engine's
//! comb-filter idea (`comb()` and the BPM loop in `engine.c`) made causal: the same 70-180 BPM
//! search, the same log-Gaussian prior around 118 BPM, the same "score a grid, then refine the
//! phase", but over a **rolling 12 s window** instead of a whole track, so it runs during a set.
//!
//! Nothing here looks ahead. [`TempoTracker::push_onset`] only ever sees the block that just
//! arrived, and the search runs on the trailing window; [`TempoTracker::estimate`] reports the
//! phase at the last block that was pushed. The grid is allowed to be wrong for the first few
//! seconds: [`estimate`](TempoTracker::estimate) returns `None` until a candidate has won three
//! analyses in a row, and the analyzer keeps its last grid until then.
//!
//! The score for a candidate `(bpm, phase)` is coverage times precision:
//!
//! - **coverage**: the onset energy that falls on the grid, divided by all the onset energy in the
//!   window. A grid at half the true tempo covers half the energy; a grid that is not a multiple of
//!   the beat covers almost none.
//! - **precision**: the fraction of grid positions that carry a peak. A grid at twice the tempo
//!   covers every onset but leaves every other position empty.
//!
//! Noise has neither: its energy is spread over every block, so coverage lands near `1 / period`,
//! tens of times below a click track. That is what keeps `no_lock_on_noise` unlocked.
//!
//! The tempo prior is the engine's: `exp(-0.5 * (log2(bpm / 118) / 0.8)^2)`. It only matters
//! between octaves that the search range allows, which is why 140 BPM is reported as 140 and not
//! 70: a half-tempo grid has half the coverage, and the prior prefers 118 anyway.

use std::cmp::Ordering;

/// Seconds of onset history the search sees (the plan's rolling window).
pub const WINDOW_S: f64 = 12.0;
/// Shortest history that may produce an estimate. 8 s is the lock budget, so the first decision
/// has to be possible well inside it.
pub const MIN_WINDOW_S: f64 = 3.0;
/// The engine's search range, in BPM.
pub const MIN_BPM: f64 = 70.0;
pub const MAX_BPM: f64 = 180.0;
/// The engine's tempo prior: a log-Gaussian around this tempo, `PRIOR_SIGMA_OCTAVES` wide.
pub const PRIOR_BPM: f64 = 118.0;
pub const PRIOR_SIGMA_OCTAVES: f64 = 0.8;

/// How often the window is searched. Between searches the phase is extrapolated from the last
/// grid, so this is only the tempo/phase correction rate, not the frame rate.
const ANALYSIS_PERIOD_S: f64 = 0.5;
/// Coarse tempo scan step and phase count. The fine stage refines the top few candidates.
const COARSE_BPM_STEP: f64 = 0.5;
const COARSE_PHASES: usize = 32;
/// Fine tempo scan: `FINE_BPM_STEP` over the coarse winner +- `FINE_BPM_SPAN`.
const FINE_BPM_STEP: f64 = 0.02;
const FINE_BPM_SPAN: f64 = 0.7;
const FINE_PHASES: usize = 16;
/// Coarse candidates that go through the fine stage (in score order).
const TOP_CANDIDATES: usize = 4;
/// The final phase scan runs on the raw envelope at this step, in blocks: 1.3 ms at 48 kHz / 256.
const FINAL_PHASE_STEP_BLOCKS: f64 = 0.25;
/// A grid position "carries a peak" when its onset is this fraction of the window's p95.
const HIT_FRACTION: f32 = 0.25;
/// Lock when a candidate's score is at least this, `LOCK_CONFIRMATIONS` times in a row. Below it a
/// locked tracker holds the grid it has: a breakdown or an outro does not throw the tempo away, and
/// only a window with no onsets at all (`UNLOCK_ANALYSES` of them) does.
const LOCK_SCORE: f64 = 0.06;
const LOCK_CONFIRMATIONS: u32 = 3;
const UNLOCK_ANALYSES: u32 = 4;
/// A new analysis is "the same grid" when the tempo is within this fraction (or 2 BPM) and the
/// phase within this many beats.
const LOCK_BPM_TOLERANCE: f64 = 0.03;
const LOCK_PHASE_TOLERANCE: f64 = 0.125;
/// When locked, the phase is pulled this far towards each new fit, per analysis.
const PHASE_CORRECTION: f64 = 0.25;
/// Score that maps to confidence 1.0.
const CONFIDENCE_SCORE: f64 = 0.4;

/// What the tracker believes right now, for the analyzer to synthesise a [`mosh_core::Transport`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TempoEstimate {
    /// Quarter-note tempo.
    pub bpm: f64,
    /// 0..1 within the beat: 0 on the beat (a click), rising to just under 1 before the next.
    pub beat_phase: f64,
    /// 0..1: how strongly the onset energy supports this grid (see the module docs).
    pub confidence: f32,
}

/// A tempo and the absolute time of a beat on that grid.
#[derive(Clone, Copy, Debug, Default)]
struct Candidate {
    bpm: f64,
    /// Absolute time, in seconds, of a grid point (the beat phase is the fractional part).
    origin_s: f64,
    score: f64,
}

/// A causal tempo and beat-phase tracker over a rolling onset window.
///
/// One per capture stream. Push the onset value of every block; ask for the estimate as often as
/// needed (it is cheap: the search runs inside `push_onset`, [`TempoTracker::estimate`] only
/// extrapolates the phase).
pub struct TempoTracker {
    sample_rate: u32,
    /// Block hop in seconds; also the onset envelope's sample period.
    hop_s: f64,
    /// Onset envelope, `cap` entries, circular in arrival order.
    ring: Vec<f32>,
    cap: usize,
    /// Next write slot in `ring`.
    head: usize,
    /// Entries in `ring` (up to `cap`).
    count: usize,
    /// The newest `count` entries in age order, then a 3-tap smoothing of them, then a sort buffer.
    /// Allocated with `ring` so that [`TempoTracker::push_onset`] never allocates after warm-up.
    window: Vec<f32>,
    smooth: Vec<f32>,
    scratch: Vec<f32>,
    /// Time at the end of the newest block, in seconds since the window was created.
    time_s: f64,
    /// Time since the last window search.
    since_analysis_s: f64,

    /// Consecutive analyses that agreed on the candidate below.
    agree: u32,
    cand_bpm: f64,
    cand_origin_s: f64,

    locked: bool,
    bpm: f64,
    /// Absolute time of a beat on the locked grid.
    beat_origin_s: f64,
    confidence: f32,
    /// Consecutive analyses that scored below `UNLOCK_SCORE`.
    weak: u32,
}

impl TempoTracker {
    /// A tracker for a stream at `sample_rate`. The window itself is allocated on the first push,
    /// once the block size (the hop) is known.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            hop_s: 0.0,
            ring: Vec::new(),
            cap: 0,
            head: 0,
            count: 0,
            window: Vec::new(),
            smooth: Vec::new(),
            scratch: Vec::new(),
            time_s: 0.0,
            since_analysis_s: 0.0,
            agree: 0,
            cand_bpm: 0.0,
            cand_origin_s: 0.0,
            locked: false,
            bpm: 0.0,
            beat_origin_s: 0.0,
            confidence: 0.0,
            weak: 0,
        }
    }

    /// Follow the stream's sample rate. A real change starts a fresh window: the history is in
    /// blocks of the old rate, so it cannot be reused.
    pub fn set_sample_rate(&mut self, sample_rate: u32) {
        if sample_rate == 0 || sample_rate == self.sample_rate {
            return;
        }
        self.sample_rate = sample_rate;
        // Force the next push to allocate a window for the new hop.
        self.hop_s = 0.0;
        self.count = 0;
        self.head = 0;
        self.time_s = 0.0;
        self.since_analysis_s = 0.0;
        self.lose_lock();
    }

    /// Add one block's onset value. `block_frames` is the hop, so the tracker knows the envelope's
    /// sample rate. Allocation-free once the window exists.
    pub fn push_onset(&mut self, onset: f32, block_frames: u32) {
        if self.sample_rate == 0 || block_frames == 0 {
            return;
        }
        let hop_s = f64::from(block_frames) / f64::from(self.sample_rate);
        if (hop_s - self.hop_s).abs() > 1e-9 {
            self.allocate(hop_s);
        }
        let value = if onset.is_finite() {
            onset.clamp(0.0, 1.0e3)
        } else {
            0.0
        };
        self.ring[self.head] = value;
        self.head = (self.head + 1) % self.cap;
        if self.count < self.cap {
            self.count += 1;
        }
        self.time_s += hop_s;
        self.since_analysis_s += hop_s;
        if self.since_analysis_s >= ANALYSIS_PERIOD_S {
            self.since_analysis_s -= ANALYSIS_PERIOD_S;
            if self.count as f64 * hop_s >= MIN_WINDOW_S {
                self.analyse();
            }
        }
    }

    /// The current estimate, or `None` until the tracker has locked. The phase is reported at the
    /// last pushed block, so it advances between searches.
    pub fn estimate(&self) -> Option<TempoEstimate> {
        if !self.locked || self.bpm <= 0.0 {
            return None;
        }
        let period = 60.0 / self.bpm;
        let beats = (self.time_s - self.beat_origin_s) / period;
        Some(TempoEstimate {
            bpm: self.bpm,
            beat_phase: beats.rem_euclid(1.0),
            confidence: self.confidence,
        })
    }

    /// Allocate the window for a hop of `hop_s` seconds. Runs on the first push and after a sample
    /// rate or block size change; never in the steady state.
    fn allocate(&mut self, hop_s: f64) {
        self.hop_s = hop_s;
        self.cap = (WINDOW_S / hop_s).ceil() as usize + 4;
        self.ring = vec![0.0; self.cap];
        self.window = vec![0.0; self.cap];
        self.smooth = vec![0.0; self.cap];
        self.scratch = vec![0.0; self.cap];
        self.head = 0;
        self.count = 0;
        self.time_s = 0.0;
        self.since_analysis_s = 0.0;
        self.lose_lock();
    }

    fn lose_lock(&mut self) {
        self.locked = false;
        self.agree = 0;
        self.weak = 0;
        self.bpm = 0.0;
        self.beat_origin_s = 0.0;
        self.confidence = 0.0;
    }

    /// Search the window: coarse tempo and phase, refine the best few, then place the beat on the
    /// raw envelope. Runs every `ANALYSIS_PERIOD_S`.
    fn analyse(&mut self) {
        let n = self.count;
        let hop = self.hop_s;
        if n < 8 || hop <= 0.0 {
            return;
        }
        let span = (n as f64 - 1.0) * hop;
        let start = (self.head + self.cap - n) % self.cap;
        let mut total = 0.0f64;
        for k in 0..n {
            let v = self.ring[(start + k) % self.cap];
            self.window[k] = v;
            total += f64::from(v);
        }
        for k in 0..n {
            let a = self.window[k.saturating_sub(1)];
            let b = self.window[k];
            let c = self.window[(k + 1).min(n - 1)];
            self.smooth[k] = 0.25 * a + 0.5 * b + 0.25 * c;
        }
        // A silent window has no grid: let the lock decay, do not search.
        if total <= 1e-9 {
            self.decay();
            return;
        }

        let p95_raw = percentile(&self.window[..n], &mut self.scratch[..n], 0.95);
        let p95_smooth = percentile(&self.smooth[..n], &mut self.scratch[..n], 0.95);
        let threshold = HIT_FRACTION * p95_smooth;
        let mut total_smooth = 0.0f64;
        for k in 0..n {
            total_smooth += f64::from(self.smooth[k]);
        }

        // Coarse search over the engine's range, keeping the best few tempos for the fine stage.
        let mut top = [Candidate::default(); TOP_CANDIDATES];
        let mut bpm = MIN_BPM;
        while bpm <= MAX_BPM + 1e-9 {
            let period = 60.0 / bpm;
            for j in 0..COARSE_PHASES {
                let phase = period * j as f64 / COARSE_PHASES as f64;
                let score = score_grid(
                    &self.smooth[..n],
                    hop,
                    span,
                    bpm,
                    phase,
                    total_smooth,
                    threshold,
                );
                if score > 0.0 {
                    keep_candidate(
                        &mut top,
                        Candidate {
                            bpm,
                            origin_s: phase,
                            score,
                        },
                    );
                }
            }
            bpm += COARSE_BPM_STEP;
        }

        // Fine search around each coarse winner: the coarse phase grid is one period in 32, the fine
        // stage is one period in 512, and the tempo step is 0.02 BPM.
        let mut best = Candidate::default();
        for coarse in top.iter().filter(|c| c.score > 0.0) {
            let lo = (coarse.bpm - FINE_BPM_SPAN).max(MIN_BPM);
            let hi = (coarse.bpm + FINE_BPM_SPAN).min(MAX_BPM);
            let coarse_period = 60.0 / coarse.bpm;
            let mut candidate_bpm = lo;
            while candidate_bpm <= hi + 1e-9 {
                let period = 60.0 / candidate_bpm;
                for k in 0..FINE_PHASES {
                    let offset = (k as f64 - FINE_PHASES as f64 / 2.0) * coarse_period
                        / (FINE_PHASES as f64 * 2.0);
                    let phase = (coarse.origin_s + offset).rem_euclid(period);
                    let score = score_grid(
                        &self.smooth[..n],
                        hop,
                        span,
                        candidate_bpm,
                        phase,
                        total_smooth,
                        threshold,
                    );
                    if score > best.score {
                        best = Candidate {
                            bpm: candidate_bpm,
                            origin_s: phase,
                            score,
                        };
                    }
                }
                candidate_bpm += FINE_BPM_STEP;
            }
        }
        if best.score <= 0.0 {
            self.decay();
            return;
        }

        // Phase on the raw envelope at sub-block resolution: the smoothed envelope found the tempo,
        // the raw one says where the transient actually is.
        let period = 60.0 / best.bpm;
        let steps = ((period / (FINAL_PHASE_STEP_BLOCKS * hop)).round() as usize).max(8);
        let raw_threshold = HIT_FRACTION * p95_raw;
        let mut phase = best.origin_s;
        let mut phase_score = f64::NEG_INFINITY;
        for k in 0..steps {
            let p = (best.origin_s + k as f64 * period / steps as f64).rem_euclid(period);
            let score = score_grid(
                &self.window[..n],
                hop,
                span,
                best.bpm,
                p,
                total,
                raw_threshold,
            );
            if score > phase_score {
                phase_score = score;
                phase = p;
            }
        }

        // The grid's absolute anchor: the window's oldest block is at `time_s - span`.
        let origin_s = self.time_s - span + phase;
        let score = best.score;

        if self.locked {
            if score < LOCK_SCORE {
                // A sparse or ambiguous window (a breakdown, an outro, a spoken section): keep the
                // grid instead of chasing a fit the onsets do not support, and let the confidence
                // fade. Only a window with no onsets at all takes the lock away (`decay`).
                self.confidence *= 0.95;
                return;
            }
            self.weak = 0;
            let close = (best.bpm - self.bpm).abs() <= (self.bpm * LOCK_BPM_TOLERANCE).max(2.0);
            if close {
                // Follow the estimate slowly: the visual grid must not jitter with every search.
                self.bpm = 0.5 * (self.bpm + best.bpm);
                let period = 60.0 / self.bpm;
                let error = wrap_beats(origin_s - self.beat_origin_s, period);
                self.beat_origin_s += error * PHASE_CORRECTION * period;
                self.confidence = confidence_from(score);
            } else {
                // A different tempo owns the window now (a new song, or a section that clearly
                // halves or doubles): start the confirmation over rather than dragging the grid.
                self.lose_lock();
                self.agree = 1;
                self.cand_bpm = best.bpm;
                self.cand_origin_s = origin_s;
            }
            return;
        }

        if score < LOCK_SCORE {
            self.agree = 0;
            return;
        }
        let same = self.agree > 0
            && (best.bpm - self.cand_bpm).abs() <= (best.bpm * LOCK_BPM_TOLERANCE).max(2.0)
            && wrap_beats(origin_s - self.cand_origin_s, 60.0 / best.bpm).abs()
                <= LOCK_PHASE_TOLERANCE;
        if same {
            self.agree += 1;
            self.cand_bpm = 0.5 * (self.cand_bpm + best.bpm);
            self.cand_origin_s = origin_s;
        } else {
            self.agree = 1;
            self.cand_bpm = best.bpm;
            self.cand_origin_s = origin_s;
        }
        if self.agree >= LOCK_CONFIRMATIONS {
            self.locked = true;
            self.bpm = self.cand_bpm;
            self.beat_origin_s = self.cand_origin_s;
            self.confidence = confidence_from(score);
            self.weak = 0;
        }
    }

    /// One analysis with nothing to score: count towards the unlock, do not create a lock.
    fn decay(&mut self) {
        if self.locked {
            self.weak += 1;
            if self.weak >= UNLOCK_ANALYSES {
                self.lose_lock();
            }
        } else {
            self.agree = 0;
        }
    }
}

/// Score one grid: the fraction of the window's onset energy that falls on it (coverage), times
/// the fraction of its positions that carry a peak (precision), times the tempo prior.
fn score_grid(
    data: &[f32],
    hop: f64,
    span: f64,
    bpm: f64,
    phase_s: f64,
    total_energy: f64,
    threshold: f32,
) -> f64 {
    let period = 60.0 / bpm;
    if !(period.is_finite() && period > 0.0) || span <= 0.0 || total_energy <= 1e-9 {
        return 0.0;
    }
    let mut t = phase_s.rem_euclid(period);
    let mut on_grid = 0.0f64;
    let mut hits = 0u32;
    let mut positions = 0u32;
    while t <= span {
        let v = f64::from(sample_at(data, t / hop));
        on_grid += v;
        if v >= f64::from(threshold) {
            hits += 1;
        }
        positions += 1;
        t += period;
    }
    if positions == 0 {
        return 0.0;
    }
    let coverage = (on_grid / total_energy).max(0.0);
    let precision = f64::from(hits) / f64::from(positions);
    coverage * precision * prior(bpm)
}

/// The engine's log-Gaussian tempo prior around [`PRIOR_BPM`].
fn prior(bpm: f64) -> f64 {
    if bpm <= 0.0 {
        return 0.0;
    }
    let octaves = (bpm / PRIOR_BPM).ln() / std::f64::consts::LN_2 / PRIOR_SIGMA_OCTAVES;
    (-0.5 * octaves * octaves).exp()
}

/// The envelope value at a fractional block position (0 = oldest sample), linearly interpolated.
fn sample_at(data: &[f32], pos: f64) -> f32 {
    if !pos.is_finite() || pos < 0.0 {
        return 0.0;
    }
    let last = data.len() as f64 - 1.0;
    if pos > last {
        return 0.0;
    }
    let i = pos.floor() as usize;
    let frac = (pos - i as f64) as f32;
    let a = data[i];
    if frac <= 0.0 {
        return a;
    }
    match data.get(i + 1) {
        Some(&b) => a + (b - a) * frac,
        None => a,
    }
}

/// `diff_s` in beats, wrapped to `-0.5..=0.5` of a beat of length `period` seconds.
fn wrap_beats(diff_s: f64, period: f64) -> f64 {
    let beats = diff_s / period;
    beats - beats.round()
}

fn confidence_from(score: f64) -> f32 {
    (score / CONFIDENCE_SCORE).clamp(0.0, 1.0) as f32
}

/// A percentile of `values`, sorted into `scratch` (which must be at least as long). Allocation-free.
fn percentile(values: &[f32], scratch: &mut [f32], q: f64) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len().min(scratch.len());
    scratch[..n].copy_from_slice(&values[..n]);
    let s = &mut scratch[..n];
    s.sort_unstable_by(|a, b| a.total_cmp(b));
    let idx = ((q * (n as f64 - 1.0)).round() as usize).min(n - 1);
    s[idx]
}

/// Keep the best `TOP_CANDIDATES` tempos in `top`, descending, dropping near-duplicate tempos (the
/// same peak found at a neighbouring phase step).
fn keep_candidate(top: &mut [Candidate; TOP_CANDIDATES], candidate: Candidate) {
    for slot in top.iter_mut() {
        if slot.score > 0.0 && (slot.bpm - candidate.bpm).abs() < 1.0 {
            if candidate.score > slot.score {
                *slot = candidate;
            }
            return;
        }
    }
    let mut worst = 0;
    for i in 1..TOP_CANDIDATES {
        if top[i].score < top[worst].score {
            worst = i;
        }
    }
    if top[worst].score < candidate.score {
        top[worst] = candidate;
        top.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    }
}
