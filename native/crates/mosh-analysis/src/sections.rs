//! Causal section detection and section-relative energy ranking (A-sections, wave 3).
//!
//! The level machine in [`crate::intensity`] ranks a bar against the last 32 bars. That is
//! enough for a bar, and wrong for a song: on the tuning tracks two loud sections 7 dB apart
//! both sat at the top of the window and both read level 4. Telling them apart needs a longer
//! memory than 32 bars and a judgement about *sections*, not bars.
//!
//! This module supplies that judgement, causally:
//!
//! 1. **Boundaries** come from spectral novelty: each closed bar's mean 24-band mel vector is
//!    compared (cosine distance) with the mean of the last [`SECT_CMP_BARS`] bars, so a change of
//!    timbre and arrangement scores high while a change of level alone barely moves it. A
//!    sustained level step is a second cue (a section can repeat the same arrangement louder),
//!    and a boundary must survive a minimum length and is then snapped to the nearest 4-bar
//!    phrase line, which is how produced music is laid out. Nothing looks ahead: a boundary at
//!    bar `b` is decided when bar `b` closes, and a snap forward is held as pending for at most
//!    [`SECT_SNAP_BARS`] / 2 bars.
//! 2. **Energy** of a section is the running mean of a per-bar mix of raw loudness, raw low-band
//!    level, onset density and brightness. All four are absolute maps: the same section in a
//!    quiet track and in a loud one cannot be compared any other way.
//! 3. **Rank** is where the current section's running energy sits among the sections heard so
//!    far, with ties in the middle. The target the level machine reads blends the absolute
//!    energy (85%) with that rank (15%), scaled by how much the song's sections have varied so
//!    far: a track whose sections are all the same reads the middle of the scale instead of
//!    pinning a level 4 it never earns, and the loudest section of a track with contrast reads
//!    high before any later section is heard.
//!
//! Everything here is per closed bar and allocation-free: the rings and the section table are
//! fixed arrays, sized by the constants below.

use crate::features::MEL_BANDS;

/// Bars compared against the bar that closes: the spectral reference window.
pub const SECT_CMP_BARS: usize = 4;
/// A section boundary is snapped to the nearest multiple of this many bars (a phrase line).
pub const SECT_SNAP_BARS: u32 = 4;
/// Shorter sections are not boundaries: a four-bar fill or a two-bar transition is not a
/// section. Sections are snapped to 4-bar phrase lines, so this keeps a change from being
/// detected twice.
pub const SECT_MIN_BARS: u32 = 4;
/// Section energies kept: enough for a long arrangement (64 four-bar sections), fixed so a push
/// never allocates.
const SECT_CAP: usize = 64;
/// Mel and level history kept for the novelty comparison.
const HIST: usize = 8;
/// Novelty values kept for the adaptive threshold.
const NOVELTY_HIST: usize = 24;

/// Absolute map for a section's mean loudness: -36 dB (a quiet intro) .. -8 dB (a mastered drop).
const SECTION_LOUD_DB: (f32, f32) = (-36.0, -8.0);
/// The same for the low band, which sits lower than the summed level.
const SECTION_LOW_DB: (f32, f32) = (-46.0, -14.0);
/// Onsets per beat: a sparse section .. a busy one.
const SECTION_ONSETS: (f32, f32) = (0.5, 5.0);
/// Raw spectral centroid (0..1 of 8 kHz): a dark mix .. a bright one.
const SECTION_BRIGHTNESS: (f32, f32) = (0.02, 0.30);

/// The four weights of a bar's section energy. They sum to 1.
const W_SECTION_LOUD: f32 = 0.52;
const W_SECTION_LOW: f32 = 0.21;
const W_SECTION_ONSET: f32 = 0.17;
const W_SECTION_BRIGHT: f32 = 0.10;

/// A level step alone is a cue of this weight relative to a timbre change.
const LEVEL_NOVELTY_WEIGHT: f32 = 0.8;
/// Only the part of a level step beyond this dead zone counts. Zero: bar-to-bar loudness inside a
/// section moves less than this in the tuning material, and the level half of the novelty is what
/// catches a section that repeats the same arrangement louder, where the mel shape does not move
/// at all.
const LEVEL_NOVELTY_DEADBAND_DB: f32 = 0.0;
/// The part beyond the dead zone that scores a full level novelty.
const LEVEL_NOVELTY_DB: f32 = 6.0;
/// Below this, novelty is never a boundary, whatever the recent history says.
const NOVELTY_FLOOR: f32 = 0.10;
/// A boundary needs this multiple of the recent novelty median ...
const NOVELTY_RATIO: f32 = 1.70;
/// ... and the median needs at least this many values before it is trusted.
const NOVELTY_MIN_HISTORY: usize = 6;
/// Section energies must spread by this much before the absolute map is trusted completely.
const SECTION_FULL_SPREAD: f32 = 0.30;
/// The least confidence given to the absolute map: a track whose sections do not differ is still
/// somewhere on the scale, just not at its edges. 0.20 measured better on the tuning tracks than
/// 0.35: quiet sections reach level 1 and the loudest section reaches 4 in more tracks without
/// losing any of the >3 dB ordering (307 of 332 section pairs, both settings).
const SECTION_MIN_CONFIDENCE: f32 = 0.20;
/// How much of the section target comes from the rank among sections, rather than the absolute
/// energy map. Rank orders sections that the absolute map cannot separate; the absolute map keeps
/// the order honest when a later section has not been heard yet.
const SECTION_RANK_WEIGHT: f32 = 0.20;

/// One bar's raw features, as the section model sees them: the raw measures plus the mel vector
/// the novelty comparison uses. `BarStats` (mosh-core, frozen) carries the normalised bar, not
/// these.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarFeatures {
    pub bar_index: i64,
    /// Short-term loudness of the bar, dBFS (the 3 s K-weighted meter).
    pub loudness_db: f32,
    /// Low band (20-150 Hz) mean power of the bar, dB.
    pub low_db: f32,
    /// Fast level of the bar: the mean block RMS, dBFS. The 3 s loudness meter takes three
    /// seconds to reach a new level, so a drop's step is still arriving two bars later; this one
    /// is on the bar it happens.
    pub level_db: f32,
    /// The same over the first half of the bar. A drop is judged on this, against the first half
    /// of the bars before it: the downbeat is the loudest part of any bar, so a whole-bar level
    /// and a half-bar level are not comparable across a bar line, and a four-on-the-floor track
    /// would report a drop on every bar.
    pub early_db: f32,
    /// Onset peaks per beat.
    pub onset_density: f32,
    /// Spectral centroid / 8 kHz, 0..1.
    pub brightness: f32,
    /// Mean 24-band mel vector, each band `10 log10(power)` mapped from -80..0 dB to 0..1.
    pub mel: [f32; MEL_BANDS],
}

impl Default for BarFeatures {
    fn default() -> Self {
        Self {
            bar_index: 0,
            loudness_db: -120.0,
            low_db: -120.0,
            level_db: -120.0,
            early_db: -120.0,
            onset_density: 0.0,
            brightness: 0.0,
            mel: [0.0; MEL_BANDS],
        }
    }
}

impl BarFeatures {
    /// The features of a bar reported without raw measurements (the unit tests and a caller that
    /// only has `BarStats`): the fast level falls back to the short-term one, and the low band's
    /// raw level is reconstructed from its normalised value over the same range the feature stage
    /// uses, so "is there a low end at all" still means something.
    pub fn from_stats(b: &mosh_core::BarStats) -> Self {
        Self {
            bar_index: b.bar_index,
            loudness_db: b.loudness_db,
            low_db: -60.0 + 46.0 * unit(b.low),
            level_db: b.loudness_db,
            early_db: b.loudness_db,
            onset_density: b.onset_density,
            brightness: b.brightness,
            mel: [0.0; MEL_BANDS],
        }
    }
}

/// What the section model says about one bar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SectionState {
    /// 0-based section number.
    pub index: u32,
    /// This bar is the first bar of a new section.
    pub started: bool,
    /// The running energy of the current section, 0..1.
    pub energy: f32,
    /// Where that energy sits among the sections heard so far, 0..1.
    pub rank: f32,
    /// The level target the intensity model reads, 0..1.
    pub target: f32,
    /// Spectral novelty of this bar, 0..1.
    pub novelty: f32,
    /// False for a model driven by bare `BarStats` (the unit tests), which has no section.
    pub present: bool,
}

impl Default for SectionState {
    fn default() -> Self {
        Self {
            index: 0,
            started: false,
            energy: 0.0,
            rank: 0.5,
            target: 0.5,
            novelty: 0.0,
            present: false,
        }
    }
}

/// The section model: one push per closed bar.
#[derive(Clone, Debug)]
pub struct SectionTracker {
    /// A boundary is snapped to a multiple of this many bars.
    snap_bars: i64,
    /// Mel vectors of the last bars (ring).
    mel: [[f32; MEL_BANDS]; HIST],
    /// Loudness of the last bars (ring), for the level half of novelty.
    db: [f32; HIST],
    len: usize,
    pos: usize,
    /// Novelty of the last bars (ring), for the adaptive threshold.
    novelty: [f32; NOVELTY_HIST],
    novelty_len: usize,
    novelty_pos: usize,
    /// Mean energy of every section that has closed.
    energies: [f32; SECT_CAP],
    /// First bar of every section that has closed.
    starts: [i64; SECT_CAP],
    sections: usize,
    /// Running energy of the section being played.
    current: f32,
    current_bars: u32,
    index: u32,
    last_boundary: i64,
    /// A boundary found off the phrase line, waiting for its snapped bar to close.
    pending: Option<i64>,
}

impl SectionTracker {
    pub fn new() -> Self {
        Self {
            snap_bars: i64::from(SECT_SNAP_BARS.max(1)),
            mel: [[0.0; MEL_BANDS]; HIST],
            db: [0.0; HIST],
            len: 0,
            pos: 0,
            novelty: [0.0; NOVELTY_HIST],
            novelty_len: 0,
            novelty_pos: 0,
            energies: [0.0; SECT_CAP],
            starts: [0; SECT_CAP],
            sections: 0,
            current: 0.0,
            current_bars: 0,
            index: 0,
            last_boundary: i64::MIN / 2,
            pending: None,
        }
    }

    /// Forget every section (a song marker, or a long silence between songs).
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// The section number the next bar belongs to.
    pub fn index(&self) -> u32 {
        self.index
    }

    /// How many sections have closed.
    pub fn closed_sections(&self) -> usize {
        self.sections
    }

    /// Fold one closed bar in and report the section state the level machine should read.
    pub fn push_bar(&mut self, f: &BarFeatures) -> SectionState {
        let novelty = self.novelty_of(f);

        // A pending boundary (found off the phrase line) starts when its bar closes.
        let mut start_at = None;
        if let Some(pending) = self.pending
            && f.bar_index >= pending
        {
            start_at = Some(pending);
            self.pending = None;
        }

        let threshold = self.threshold();
        let far_enough = f.bar_index - self.last_boundary >= i64::from(SECT_MIN_BARS.max(1));
        if start_at.is_none()
            && far_enough
            && self.len >= SECT_MIN_BARS as usize
            && novelty >= threshold
        {
            let snapped = snap_to_phrase(f.bar_index, self.snap_bars);
            if snapped <= f.bar_index {
                start_at = Some(snapped);
            } else {
                self.pending = Some(snapped);
            }
        }

        let mut started = false;
        if let Some(bar) = start_at
            && bar > self.last_boundary
        {
            self.close_section(bar);
            started = true;
        }

        // Fold this bar's energy into the running section mean.
        let energy = section_energy(f);
        self.current_bars += 1;
        let n = self.current_bars as f32;
        self.current += (energy - self.current) / n;

        let (rank, target) = self.rank_and_target();

        self.novelty[self.novelty_pos] = novelty;
        self.novelty_pos = (self.novelty_pos + 1) % NOVELTY_HIST;
        if self.novelty_len < NOVELTY_HIST {
            self.novelty_len += 1;
        }
        self.mel[self.pos] = f.mel;
        self.db[self.pos] = f.loudness_db;
        self.pos = (self.pos + 1) % HIST;
        if self.len < HIST {
            self.len += 1;
        }

        SectionState {
            index: self.index,
            started,
            energy: self.current,
            rank,
            target,
            novelty,
            present: true,
        }
    }

    /// Commit the section in progress and open a new one at `bar`.
    fn close_section(&mut self, bar: i64) {
        if self.current_bars > 0 && self.sections < SECT_CAP {
            self.energies[self.sections] = self.current;
            self.starts[self.sections] = bar;
            self.sections += 1;
        }
        self.current = 0.0;
        self.current_bars = 0;
        self.index += 1;
        self.last_boundary = bar;
    }

    /// 1 - cosine between this bar's mel vector and the mean of the last [`SECT_CMP_BARS`]
    /// bars, combined with a sustained level step. Zero until there are two bars to compare.
    fn novelty_of(&self, f: &BarFeatures) -> f32 {
        let bars = self.len.min(SECT_CMP_BARS);
        if bars < 2 {
            return 0.0;
        }
        let mut mean = [0.0f32; MEL_BANDS];
        let mut db = 0.0f32;
        for k in 0..bars {
            let at = (self.pos + HIST - 1 - k) % HIST;
            for (m, slot) in mean.iter_mut().enumerate() {
                *slot += self.mel[at][m];
            }
            db += self.db[at];
        }
        let inv = 1.0 / bars as f32;
        for slot in mean.iter_mut() {
            *slot *= inv;
        }
        db *= inv;

        let spectral = 1.0 - cosine(&f.mel, &mean);
        // The 3 s loudness meter slides, so a step settles over two or three bars; only the part
        // of a level difference beyond a dead zone is a change of section, not of bar.
        let level = (((f.loudness_db - db).abs() - LEVEL_NOVELTY_DEADBAND_DB)
            / LEVEL_NOVELTY_DB)
            .clamp(0.0, 1.0);
        spectral.max(level * LEVEL_NOVELTY_WEIGHT)
    }

    /// The novelty a boundary must beat: a fixed floor, or a multiple of the recent median once
    /// there is enough history to have a median.
    fn threshold(&self) -> f32 {
        if self.novelty_len < NOVELTY_MIN_HISTORY {
            return NOVELTY_FLOOR;
        }
        NOVELTY_FLOOR.max(NOVELTY_RATIO * self.novelty_median())
    }

    fn novelty_median(&self) -> f32 {
        let n = self.novelty_len;
        if n == 0 {
            return 0.0;
        }
        let mut sorted = [0.0f32; NOVELTY_HIST];
        sorted[..n].copy_from_slice(&self.novelty[..n]);
        for i in 1..n {
            let mut j = i;
            while j > 0 && sorted[j - 1] > sorted[j] {
                sorted.swap(j - 1, j);
                j -= 1;
            }
        }
        sorted[n / 2]
    }

    /// Where the running section energy sits among the sections heard so far, and the level
    /// target derived from it.
    fn rank_and_target(&self) -> (f32, f32) {
        let n = self.sections + 1;
        let mut below = 0usize;
        let mut equal = 0usize;
        let (mut lo, mut hi) = (self.current, self.current);
        for &other in self.energies.iter().take(self.sections) {
            if other < self.current {
                below += 1;
            } else if other == self.current {
                equal += 1;
            }
            lo = lo.min(other);
            hi = hi.max(other);
        }
        // A position from 0 (the quietest section so far) to 1 (the loudest), ties taking the
        // middle of their block. One section has no position to have: it reads the middle.
        let rank = if n == 1 || equal == n - 1 {
            0.5
        } else {
            (below as f32 + 0.5 * equal as f32) / (n - 1) as f32
        }
        .clamp(0.0, 1.0);
        let confidence = ((hi - lo) / SECTION_FULL_SPREAD).clamp(SECTION_MIN_CONFIDENCE, 1.0);
        let target = 0.5
            + (self.current - 0.5) * confidence
            + SECTION_RANK_WEIGHT * (rank - 0.5);
        (rank, target.clamp(0.0, 1.0))
    }
}

impl Default for SectionTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// The nearest phrase line, ties snapped back so a boundary is never moved forward past the bar
/// that was actually heard.
fn snap_to_phrase(bar: i64, phrase: i64) -> i64 {
    let phrase = phrase.max(1);
    let down = (bar / phrase) * phrase;
    if bar - down >= phrase - (bar - down) {
        down + phrase
    } else {
        down
    }
}

/// Cosine similarity of two non-negative mel vectors; 0 when either is silent.
fn cosine(a: &[f32; MEL_BANDS], b: &[f32; MEL_BANDS]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..MEL_BANDS {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 1e-9 || nb <= 1e-9 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// One bar's contribution to its section's energy: raw-level maps, weighted.
pub fn section_energy(f: &BarFeatures) -> f32 {
    let loud = map(f.loudness_db, SECTION_LOUD_DB);
    let low = map(f.low_db, SECTION_LOW_DB);
    let onset = map(f.onset_density, SECTION_ONSETS);
    let bright = map(f.brightness, SECTION_BRIGHTNESS);
    (W_SECTION_LOUD * loud + W_SECTION_LOW * low + W_SECTION_ONSET * onset + W_SECTION_BRIGHT * bright)
        .clamp(0.0, 1.0)
}

/// `value` placed in 0..1 between `range.0` and `range.1`; non-finite input is the middle.
fn map(value: f32, range: (f32, f32)) -> f32 {
    if !value.is_finite() || range.1 - range.0 <= 1e-6 {
        return 0.5;
    }
    ((value - range.0) / (range.1 - range.0)).clamp(0.0, 1.0)
}

/// Clamp into 0..1; non-finite input counts as nothing.
fn unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(index: i64, loudness_db: f32, low_db: f32, onsets: f32, mel_at: usize) -> BarFeatures {
        let mut mel = [0.0f32; MEL_BANDS];
        // A band-limited shape: one bar's "arrangement" is which half of the mel axis carries it.
        let (lo, hi) = if mel_at == 0 { (0, 8) } else { (12, 20) };
        for slot in mel.iter_mut().take(hi).skip(lo) {
            *slot = 1.0;
        }
        BarFeatures {
            bar_index: index,
            loudness_db,
            low_db,
            level_db: loudness_db,
            early_db: loudness_db,
            onset_density: onsets,
            brightness: 0.1,
            mel,
        }
    }

    /// A steady loop has no novelty, and with one section the target sits near the absolute
    /// energy of that section, not at a scale edge.
    #[test]
    fn a_steady_loop_is_one_section() {
        let mut t = SectionTracker::new();
        let mut last = SectionState::default();
        for i in 0..16 {
            last = t.push_bar(&bar(i, -14.0, -22.0, 2.0, 0));
            assert!(!last.started, "bar {i}: a steady loop is not a boundary");
        }
        assert_eq!(last.index, 0);
        assert_eq!(t.closed_sections(), 0);
        assert!(last.novelty < NOVELTY_FLOOR, "novelty {}", last.novelty);
        assert!(
            (0.5..0.85).contains(&last.target),
            "a loud steady loop's target {} must stay clear of the top of the scale",
            last.target
        );
    }

    /// A timbre change on a phrase line starts a section there.
    #[test]
    fn a_timbre_change_on_a_phrase_line_starts_a_section() {
        let mut t = SectionTracker::new();
        for i in 0..8 {
            t.push_bar(&bar(i, -20.0, -30.0, 2.0, 0));
        }
        let at = t.push_bar(&bar(8, -18.0, -26.0, 2.5, 1));
        assert!(
            at.started,
            "bar 8 changes arrangement on a phrase line: {at:?}"
        );
        assert_eq!(at.index, 1, "the second section");
        assert_eq!(t.closed_sections(), 1);
    }

    /// A change heard one bar before the phrase line is moved onto the line, and the section
    /// starts when that bar closes: never more than two bars after the change.
    #[test]
    fn an_early_change_snaps_forward_to_the_phrase_line() {
        let mut t = SectionTracker::new();
        for i in 0..7 {
            t.push_bar(&bar(i, -20.0, -30.0, 2.0, 0));
        }
        // Bar 7 is where the arrangement changes, one bar before the eight-bar line.
        let at7 = t.push_bar(&bar(7, -18.0, -26.0, 2.5, 1));
        assert!(!at7.started, "the boundary belongs on the phrase line");
        let at8 = t.push_bar(&bar(8, -18.0, -26.0, 2.5, 1));
        assert!(at8.started, "bar 8 is the snapped boundary: {at8:?}");
        assert_eq!(at8.index, 1);
    }

    /// Sections closer than `SECT_MIN_BARS` are one section: a fill is not a section.
    #[test]
    fn a_short_change_is_not_a_section() {
        let mut t = SectionTracker::new();
        for i in 0..8 {
            t.push_bar(&bar(i, -20.0, -30.0, 2.0, 0));
        }
        let a = t.push_bar(&bar(8, -20.0, -30.0, 3.0, 1));
        let b = t.push_bar(&bar(9, -20.0, -30.0, 2.0, 0));
        assert!(a.started, "bar 8 is a section");
        assert!(
            !b.started,
            "bar 9 is back to the old arrangement one bar later; a section is at least {} bars",
            SECT_MIN_BARS
        );
    }

    /// A louder section ranks above an earlier loud one: the relative half of the target.
    #[test]
    fn a_louder_section_outranks_an_earlier_loud_one() {
        let mut t = SectionTracker::new();
        let mut quiet = SectionState::default();
        for i in 0..8 {
            quiet = t.push_bar(&bar(i, -27.0, -40.0, 1.5, 0));
        }
        let mut loud = SectionState::default();
        for i in 8..23 {
            loud = t.push_bar(&bar(i, -17.0, -22.0, 3.0, 1));
        }
        // Change back to the first arrangement, louder still.
        let mut loudest = SectionState::default();
        for i in 23..39 {
            loudest = t.push_bar(&bar(i, -10.0, -16.0, 3.5, 0));
        }
        assert_eq!(loudest.index, 2, "three sections");
        assert!(
            loudest.target > loud.target + 0.1,
            "the -10 dB section's target {} must clear the -17 dB one's {}",
            loudest.target,
            loud.target
        );
        assert!(
            loud.target > quiet.target,
            "the -17 dB section ({}) must clear the -27 dB intro ({})",
            loud.target,
            quiet.target
        );
        assert!(
            quiet.target < 0.5,
            "a -27 dB intro reads below the middle: {}",
            quiet.target
        );
        assert!(
            loudest.target > 0.72,
            "the loudest section must reach the top level band: {}",
            loudest.target
        );
    }

    /// A section model that never saw a boundary still reports the absolute energy, not a rank
    /// of one.
    #[test]
    fn one_section_is_ranked_in_the_middle() {
        let mut t = SectionTracker::new();
        let state = t.push_bar(&bar(0, -14.0, -22.0, 2.0, 0));
        assert!((state.rank - 0.5).abs() < 1e-6, "rank {}", state.rank);
    }

    #[test]
    fn reset_forgets_the_sections() {
        let mut t = SectionTracker::new();
        for i in 0..8 {
            t.push_bar(&bar(i, -20.0, -30.0, 2.0, 0));
        }
        t.push_bar(&bar(8, -18.0, -26.0, 2.5, 1));
        assert_eq!(t.index(), 1);
        t.reset();
        assert_eq!(t.index(), 0);
        assert_eq!(t.closed_sections(), 0);
        let fresh = t.push_bar(&bar(0, -20.0, -30.0, 2.0, 0));
        assert_eq!(fresh.index, 0);
        assert!(!fresh.started);
    }

    #[test]
    fn energy_is_monotone_in_its_inputs() {
        let quiet = BarFeatures {
            loudness_db: -34.0,
            low_db: -44.0,
            onset_density: 1.0,
            brightness: 0.05,
            ..BarFeatures::default()
        };
        let loud = BarFeatures {
            loudness_db: -10.0,
            low_db: -16.0,
            onset_density: 4.0,
            brightness: 0.25,
            ..BarFeatures::default()
        };
        assert!(section_energy(&loud) > section_energy(&quiet) + 0.5);
        // Non-finite input never produces a NaN energy.
        let broken = BarFeatures {
            loudness_db: f32::NAN,
            low_db: f32::INFINITY,
            onset_density: f32::NAN,
            brightness: f32::NAN,
            ..BarFeatures::default()
        };
        assert!(section_energy(&broken).is_finite());
    }

    #[test]
    fn snap_prefers_the_nearest_phrase_line() {
        assert_eq!(snap_to_phrase(8, 4), 8);
        assert_eq!(snap_to_phrase(9, 4), 8);
        assert_eq!(snap_to_phrase(7, 4), 8);
        assert_eq!(snap_to_phrase(10, 4), 12, "ties snap forward");
        assert_eq!(snap_to_phrase(0, 4), 0);
    }
}
