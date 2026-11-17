//! The intensity model: per-bar features to a continuous intensity, a 1..=4 level
//! with hysteresis, and the phrase / build / drop / breakdown events the show uses.
//!
//! Task 1.4 of `docs/superpowers/plans/2026-11-17-native-port.md`, revised in wave 3 by
//! A-sections: the level is read against sections as well as bars, and builds and drops are
//! detected from the cues a build and a drop actually have.
//!
//! Three causal stages, all against what has already been heard:
//!
//! 1. **Intensity** is the weighted mean of the four terms (loudness 0.45, low 0.25, onset 0.18,
//!    brightness 0.12; the plan's starting point was 0.35 / 0.30 / 0.20 / 0.15 and the change is
//!    explained at `W_LOUDNESS`). `loudness`, `low` and `brightness` arrive already normalised to
//!    0..1 by the feature stage, which blends a fixed typical range with the bar's rank over the
//!    last `history_bars` bars; `onset_density` arrives raw, as onsets per beat, and this model
//!    normalises it the same way. `loudness_db` is the raw short-term level and decides silence,
//!    because a normalised feature of a noise floor can be anything.
//! 2. **Level** is where that intensity sits in the recent dynamic range, blended with where the
//!    *section* sits among the sections heard so far (see [`crate::sections`]). The section half
//!    is 0.85 of the reading: a 32-bar window cannot tell two loud sections 7 dB apart (both sit
//!    at the top of the window and both read level 4, which is what the tuning tracks showed), so
//!    the section target, not the bar window, decides the level of a section. The four
//!    thresholds (0.25 / 0.5 / 0.75), the hysteresis band and the one-step-per-bar rule still apply
//!    to the reading, so a level never moves faster than a bar and a collapse walks down instead
//!    of snapping.
//! 3. **Events.** BUILD_START needs a rising onset density (snare rolls), a rising high end
//!    (filter or noise sweeps) or a rising intensity, with the low end not rising — those are the
//!    cues a build has, and any of them starts one. DROP needs a big level jump or a low-end
//!    return on the bar it happens, with a build or a gap behind it; it is evaluated once mid-bar
//!    as well, so it lands on the downbeat instead of a bar later at the next bar line. BREAKDOWN
//!    is the opposite: low end and onsets falling away for a full bar.
//!
//! Everything here is causal and per bar: `push_bar` is the only way in, so a level
//! never changes in the middle of a bar, never moves more than one step per bar, and the same
//! bar sequence always produces the same events. `reset` is for song markers.

use mosh_core::{BarStats, EventFlags};

use crate::sections::{BarFeatures, SectionState};
use crate::RollingRange;

/// Weights of the four normalised terms. They sum to 1, so the intensity is 0..1.
///
/// The plan's starting point was 0.35 / 0.30 / 0.20 / 0.15. Loudness carries more of it after
/// measurement on the 16 tuning tracks: brightness and onset density move with timbre and
/// arrangement within a bar, so at equal weight they could outvote the level of the music, and a
/// sparse bar in the loudest part of a track read lower than a busy bar in a quiet one. Loudness
/// is the term the four levels are about; the others colour it.
const W_LOUDNESS: f32 = 0.45;
const W_LOW: f32 = 0.25;
const W_ONSET: f32 = 0.18;
const W_BRIGHTNESS: f32 = 0.12;

/// How much of the level reading comes from the section target rather than the bar's position in
/// the last `history_bars` bars. The bar half cannot separate two loud sections, and the section
/// half cannot follow a one-bar change inside a section; 0.85 / 0.15 keeps both jobs.
const SECTION_WEIGHT: f32 = 0.85;

/// Level boundaries: below 0.25 is level 1, at or above 0.75 is level 4.
const LEVEL_THRESHOLDS: [f32; 3] = [0.25, 0.5, 0.75];
/// Total width of the hysteresis band, split evenly around each threshold: a level
/// rises 0.04 above the boundary and falls only 0.04 below it.
const LEVEL_HYSTERESIS: f32 = 0.08;
const LEVEL_BAND: f32 = LEVEL_HYSTERESIS * 0.5;

/// The level reads where the bar's intensity sits in the recent dynamic range: its 5th-95th
/// percentile over the last `history_bars` bars. A range narrower than this has nothing to place a
/// bar in, so the reading is the middle: a steady loop settles at level 2 or 3 instead of being
/// ranked by its own noise.
const LEVEL_MIN_SPAN: f32 = 0.1;

/// A breakdown needs a level this high first.
const BREAKDOWN_LEVEL: u8 = 3;
/// Breakdown: normalised low and onsets both below this for a full bar.
const BREAKDOWN_QUIET: f32 = 0.25;

/// The low band is "there" above this normalised level. Below it the bar has no low end: a gap,
/// a low-pass, a breakdown.
const LOW_PRESENT: f32 = 0.30;
/// ... and above this raw level, dB on the bar's 20-150 Hz mean power. The normalised value is
/// relative, so a track with no low end at all still maps one bar to 0.5; a drop's low end is a
/// kick and a bass, not a normalised noise floor.
const LOW_DB_PRESENT: f32 = -35.0;
/// ... and it "returns" when it rises this far above the previous bar's low band, staying above
/// [`LOW_PRESENT`]. A return is a jump in the low band, not a fixed level: the same arrangement
/// in a quieter section normalises lower on every bar.
const LOW_RETURN_JUMP: f32 = 0.20;
/// A bar with low below this counts as "no low end" for the gap counter.
const LOW_GAP: f32 = 0.2;
/// A drop needs a level jump of this many dB over the loudest of the last few bars ...
const DROP_JUMP_DB: f32 = 4.5;
/// ... measured over this many bars. The first half of a bar is compared with the first half of
/// the bars before it, and against their *loudest*, not their average: a four-on-the-floor track
/// alternates a kick-heavy first half with a sparse one every other bar, and an average makes
/// every second bar look like a drop. A drop is above everything recent, which is what this says.
const DROP_LOOKBACK_BARS: usize = 4;
/// A bar this far below the loudest recent bar is a gap (a low-pass, a filter sweep's silence,
/// a drop-out) even when the low band is still there: a drop comes out of one.
const GAP_LEVEL_DB: f32 = 5.0;
/// No second drop this soon: a drop is a structural event, and a re-trigger on the next bar is
/// the same drop heard twice.
const DROP_COOLDOWN_BARS: i64 = 2;

/// Silence: short-term loudness below this many dB.
const SILENCE_DB: f32 = -50.0;
/// More than this many silent bars in a row (a gap between songs) forgets the level
/// and the history, so the next song starts from nothing.
const SILENCE_BARS_TO_RESET: u32 = 2;

/// Build: the cue is measured over the last this many bars ...
const BUILD_BARS: usize = 3;
/// ... and needs this much of a rise.
const BUILD_RISE: f32 = 0.10;
/// A snare roll: onsets per beat rising this much over [`BUILD_BARS`].
const BUILD_ONSET_RISE: f32 = 1.0;
/// A filter or noise sweep: normalised brightness rising this much over [`BUILD_BARS`].
const BUILD_BRIGHT_RISE: f32 = 0.12;
/// A build that adds low end is a drop arriving, not a build: the low band may rise by at most
/// this much while a build is confirmed.
const BUILD_LOW_ALLOWANCE: f32 = 0.12;
/// A build stays open for at most this many bars when no drop arrives.
const BUILD_MAX_BARS: u32 = 8;

/// Onset densities are ranked against their own causal window, as onsets per beat: the fallback
/// covers the first bars of a show, and a window whose 10th-90th percentile spread is under
/// `ONSET_FULL_SPAN` (nothing is changing) keeps the term near the middle instead of ranking by
/// noise.
const ONSET_FALLBACK: (f32, f32) = (1.0, 4.0);
const ONSET_FULL_SPAN: f32 = 1.5;

/// The four normalised terms of one bar, as the model used them. Diagnostic: `mosh-analyze
/// --dump` prints them, the show reads only `intensity`, `level` and `events`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IntensityTerms {
    pub loudness: f32,
    pub low: f32,
    pub onset: f32,
    pub brightness: f32,
}

/// What one bar did to the model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntensityUpdate {
    /// Continuous intensity 0..1: the weighted mean of the four normalised terms.
    pub intensity: f32,
    /// The reading the level machine saw: the section target blended with the bar's position in
    /// the recent range (0..1).
    pub reading: f32,
    /// 1..=4, changing only here, on a bar boundary.
    pub level: u8,
    /// PHRASE / BUILD_START / DROP / BREAKDOWN / LEVEL_UP / LEVEL_DOWN for this bar.
    /// BEAT, BAR and SONG_START belong to the grid and app layers, not to this model.
    pub events: EventFlags,
    /// The normalised inputs this bar's intensity was built from.
    pub terms: IntensityTerms,
}

/// Turns the per-bar features of a song into intensity, level and events.
#[derive(Debug)]
pub struct IntensityModel {
    phrase_bars: u32,
    level: u8,
    intensity: f32,
    phrase_pos: u32,
    /// Onset densities of the last `history_bars` bars.
    onsets: RollingRange,
    /// Intensities of the last `history_bars` bars: the range the level reads a bar against.
    scale: RollingRange,
    /// The first-half fast level of the last [`DROP_LOOKBACK_BARS`] bars, oldest first: a drop is
    /// above the loudest of them.
    early_recent: [f32; DROP_LOOKBACK_BARS],
    early_len: usize,
    /// Intensities, onset densities and brightness of the last [`BUILD_BARS`] bars.
    recent_intensity: [f32; BUILD_BARS],
    recent_onset: [f32; BUILD_BARS],
    recent_bright: [f32; BUILD_BARS],
    recent_len: usize,
    /// BUILD_START fired and no DROP or timeout has closed the build yet.
    build_active: bool,
    /// Bars left before an open build times out.
    build_bars_left: u32,
    /// Consecutive bars with low below `LOW_GAP` (or with the level collapsed), before this bar.
    gap_bars: u32,
    /// A level of at least `BREAKDOWN_LEVEL` was seen since the last breakdown.
    breakdown_armed: bool,
    /// Low band of the previous bar, for the "the low end returns" crossing.
    last_low: Option<f32>,
    /// The bar a DROP already fired on, from the mid-bar pass: the bar close must not fire it
    /// again.
    drop_bar: Option<i64>,
    /// The bar of the last DROP of any kind, for the cooldown.
    last_drop_bar: Option<i64>,
    /// Consecutive silent bars.
    silent_bars: u32,
}

impl IntensityModel {
    /// `phrase_bars` is the PHRASE period (clamped to at least 1), `history_bars` the
    /// length of the onset window (clamped to at least 2).
    pub fn new(phrase_bars: u32, history_bars: u32) -> Self {
        IntensityModel {
            phrase_bars: phrase_bars.max(1),
            level: 1,
            intensity: 0.0,
            phrase_pos: 0,
            scale: RollingRange::new((0.0, 1.0), history_bars, LEVEL_MIN_SPAN),
            onsets: RollingRange::new(ONSET_FALLBACK, history_bars, ONSET_FULL_SPAN),
            early_recent: [0.0; DROP_LOOKBACK_BARS],
            early_len: 0,
            recent_intensity: [0.0; BUILD_BARS],
            recent_onset: [0.0; BUILD_BARS],
            recent_bright: [0.0; BUILD_BARS],
            recent_len: 0,
            build_active: false,
            build_bars_left: 0,
            gap_bars: 0,
            breakdown_armed: false,
            last_low: None,
            drop_bar: None,
            last_drop_bar: None,
            silent_bars: 0,
        }
    }

    /// The level of the last bar (1..=4).
    pub fn level(&self) -> u8 {
        self.level
    }

    /// The intensity of the last bar (0..1).
    pub fn intensity(&self) -> f32 {
        self.intensity
    }

    /// Forget everything, for a song marker: level 1, no intensity, empty history,
    /// phrase counter restarted. A silence gap longer than two bars does the same.
    pub fn reset(&mut self) {
        self.clear_state();
        self.silent_bars = 0;
    }

    /// Fold one completed bar in and report what it did. No raw features: the fast level falls
    /// back to the short-term one and no section is known, which is the pre-sections behaviour and
    /// what the unit tests drive.
    pub fn push_bar(&mut self, b: &BarStats) -> IntensityUpdate {
        let raw = BarFeatures::from_stats(b);
        self.push_bar_with(b, &raw, &SectionState::default())
    }

    /// Fold one completed bar in, with its raw features and the section state of that bar.
    pub fn push_bar_with(
        &mut self,
        b: &BarStats,
        raw: &BarFeatures,
        section: &SectionState,
    ) -> IntensityUpdate {
        // Silence is decided by the raw level: a noise floor normalises to whatever the
        // window makes of it, so the normalised features cannot be trusted there.
        // A NaN level counts as silent rather than poisoning the bar.
        let silent = b.loudness_db <= SILENCE_DB || b.loudness_db.is_nan();
        let (loudness, low, brightness, onset_density) = if silent {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            (
                unit(b.loudness),
                unit(b.low),
                unit(b.brightness),
                onset_density(b.onset_density),
            )
        };

        if silent {
            self.silent_bars += 1;
            // A silent bar is the extreme of a gap: the low end is gone for a whole bar.
            self.gap_bars = self.gap_bars.saturating_add(1);
        } else {
            self.silent_bars = 0;
        }
        if self.silent_bars > SILENCE_BARS_TO_RESET {
            self.clear_state();
        }

        // The window includes this bar: everything used is already known. `normalize` folds the
        // value into its own window, so the push after it is the same set either way.
        let onset = self.onsets.normalize(onset_density);
        self.onsets.push(onset_density);

        // A silent bar is not normalised at all: its terms are zero because that is what the
        // intensity used, so `terms` always explains `intensity`.
        let terms = if silent {
            IntensityTerms::default()
        } else {
            IntensityTerms {
                loudness,
                low,
                onset,
                brightness,
            }
        };

        // The level reads the intensity itself: the four terms already mix an absolute range
        // with a local rank, so a quiet section stays low and a loud one high even when either
        // lasts longer than the window.
        let intensity = if silent {
            0.0
        } else {
            (W_LOUDNESS * terms.loudness
                + W_LOW * terms.low
                + W_ONSET * terms.onset
                + W_BRIGHTNESS * terms.brightness)
                .clamp(0.0, 1.0)
        };

        // The bar's position in the recent dynamic range, and the section target that replaces
        // most of it (see `SECTION_WEIGHT`). `position` includes this bar.
        let bar_position = if silent {
            0.0
        } else {
            self.scale
                .position(intensity, LEVEL_MIN_SPAN)
                .unwrap_or(0.5)
        };
        self.scale.push(intensity);
        let reading = if silent {
            0.0
        } else if section.present {
            SECTION_WEIGHT * section.target + (1.0 - SECTION_WEIGHT) * bar_position
        } else {
            bar_position
        };

        let previous_level = self.level;
        let mut events = EventFlags::empty();

        if !silent {
            // BUILD_START before DROP: a build that ends on this bar is still a build.
            self.track_build(intensity, low, onset_density, brightness, &mut events);
            let already = self.drop_bar == Some(b.bar_index);
            if !already && self.drop_ready(b, raw) {
                events |= EventFlags::DROP;
                self.close_build();
                self.gap_bars = 0;
                self.last_drop_bar = Some(b.bar_index);
                self.gap_bars = 0;
            }
            self.track_breakdown(previous_level, low, onset, &mut events);
            // The build and the gap bookkeeping see the bar after the drop test: a drop ends
            // both, and a bar with no low end (a gap, a low-pass) starts one.
            if !events.contains(EventFlags::DROP) {
                // A gap is a bar with no low end, or one this far below the loudest recent bar: a
                // low-pass, a filter sweep's silence, a gap. Both are what a drop comes out of.
                let dropped_out = self.early_len > 0
                    && raw.early_db + GAP_LEVEL_DB
                        < self.early_recent[..self.early_len]
                            .iter()
                            .copied()
                            .fold(f32::NEG_INFINITY, f32::max);
                if low < LOW_GAP || dropped_out {
                    self.gap_bars += 1;
                } else {
                    self.gap_bars = 0;
                }
            }
        }

        self.level = next_level(previous_level, reading);
        if self.level > previous_level {
            events |= EventFlags::LEVEL_UP;
        } else if self.level < previous_level {
            events |= EventFlags::LEVEL_DOWN;
        }

        self.phrase_pos += 1;
        if self.phrase_pos >= self.phrase_bars {
            self.phrase_pos = 0;
            events |= EventFlags::PHRASE;
        }
        // A new section is a section change: the show gets it as a phrase boundary, together
        // with the level events this bar may carry.
        if section.present && section.started {
            events |= EventFlags::PHRASE;
        }

        self.intensity = intensity;
        self.last_low = if silent { None } else { Some(low) };
        self.push_recent(intensity, onset_density, brightness);

        if silent {
            self.early_len = 0;
        } else {
            if self.early_len < DROP_LOOKBACK_BARS {
                self.early_recent[self.early_len] = raw.early_db;
                self.early_len += 1;
            } else {
                self.early_recent.rotate_left(1);
                self.early_recent[DROP_LOOKBACK_BARS - 1] = raw.early_db;
            }
        }

        IntensityUpdate {
            intensity,
            reading,
            level: self.level,
            events,
            terms,
        }
    }

    /// Whether this bar is a drop, from its own raw features and what the last bars did. Pure:
    /// the mid-bar pass (`Analyzer::push`) calls it before the bar closes, and the bar close calls
    /// it again in case the mid-bar pass saw a partial bar that did not qualify.
    ///
    /// The mid-bar pass needs the level jump: a partial bar's low band is the first half's, which
    /// is the kick-heavy part of every bar, so a low-band return seen there is not yet evidence.
    pub fn is_drop(&self, b: &BarStats, raw: &BarFeatures) -> bool {
        self.drop_ready(b, raw) && self.level_jump(raw) >= DROP_JUMP_DB
    }

    /// A DROP already fired on this bar from the mid-bar pass.
    pub fn note_drop(&mut self, bar_index: i64) {
        self.drop_bar = Some(bar_index);
        self.last_drop_bar = Some(bar_index);
        self.close_build();
        self.gap_bars = 0;
    }

    /// The bar a DROP already fired on, if any (for a consumer that watches the frame events).
    pub fn drop_bar(&self) -> Option<i64> {
        self.drop_bar
    }

    fn drop_ready(&self, b: &BarStats, raw: &BarFeatures) -> bool {
        if !b.loudness_db.is_finite() || b.loudness_db <= SILENCE_DB || self.early_len == 0 {
            // The first bar of a song has nothing to drop from.
            return false;
        }
        if self
            .last_drop_bar
            .is_some_and(|last| b.bar_index - last < DROP_COOLDOWN_BARS)
        {
            return false;
        }
        let low = unit(b.low);
        let low_audible = raw.low_db.is_finite() && raw.low_db > LOW_DB_PRESENT;
        let previous_low = self.last_low.unwrap_or(0.0);
        let low_returns =
            low_audible && low > LOW_PRESENT && low - previous_low >= LOW_RETURN_JUMP;
        let level_jump = self.level_jump(raw);
        // A drop is a big energy jump. The low-end return and the build are what make it a drop
        // rather than a crash; they are not a substitute for the jump. A low band that comes back
        // under a bar of the same level is an arrangement change, not a drop.
        let jump = level_jump >= DROP_JUMP_DB;
        // After a build, after a bar whose low end disappeared (a gap, a low-pass) or when the
        // low band itself comes back.
        let context = self.build_active || self.gap_bars >= 1 || low_returns;
        // The low end has to be there. A big fast-level jump without it is a crash, a fill or a
        // one-off transient on a quiet bar, which is what the intro of the reza track is made of.
        jump && context && low_audible && low > LOW_PRESENT
    }

    /// How far this bar's first half is above the loudest first half of the last
    /// [`DROP_LOOKBACK_BARS`] bars, in dB.
    fn level_jump(&self, raw: &BarFeatures) -> f32 {
        if self.early_len == 0 {
            return 0.0;
        }
        let recent = self.early_recent[..self.early_len]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        raw.early_db - recent
    }

    fn close_build(&mut self) {
        self.build_active = false;
        self.build_bars_left = 0;
    }

    /// BUILD_START: a build's cues, any one of them. Rising intensity (a riser), rising onsets (a
    /// snare roll) or a rising high end (a sweep), with the low end not rising: once the low end
    /// arrives, the build is over and the drop is next.
    fn track_build(
        &mut self,
        intensity: f32,
        low: f32,
        onset_density: f32,
        brightness: f32,
        events: &mut EventFlags,
    ) {
        if self.build_active {
            if self.build_bars_left == 0 {
                self.build_active = false;
            } else {
                self.build_bars_left -= 1;
            }
        }
        if self.build_active || self.recent_len < BUILD_BARS {
            return;
        }

        let previous_low = self.last_low.unwrap_or(low);
        let low_flat = low <= previous_low + BUILD_LOW_ALLOWANCE;
        // The ring holds the previous [`BUILD_BARS`] bars; this bar is the next one. A cue needs
        // two consecutive rising steps, not one: a single bright or busy bar is a crash or a fill.
        let last = BUILD_BARS - 1;
        let intensity_rising = intensity > self.recent_intensity[last]
            && self.recent_intensity[last] > self.recent_intensity[last - 1]
            && intensity - self.recent_intensity[0] >= BUILD_RISE;
        let onsets_rising = onset_density > self.recent_onset[last]
            && self.recent_onset[last] > self.recent_onset[last - 1]
            && onset_density - self.recent_onset[0] >= BUILD_ONSET_RISE;
        let bright_rising = brightness > self.recent_bright[last]
            && self.recent_bright[last] > self.recent_bright[last - 1]
            && brightness - self.recent_bright[0] >= BUILD_BRIGHT_RISE;

        if ((intensity_rising || onsets_rising) && low_flat) || bright_rising {
            self.build_active = true;
            self.build_bars_left = BUILD_MAX_BARS;
            *events |= EventFlags::BUILD_START;
        }
    }

    /// BREAKDOWN: low and normalised onsets both under 0.25 for a full bar, once the
    /// level had reached 3. Silence is a song gap, not a breakdown, and never gets here.
    ///
    /// Re-arming needs a bar that is *not* quiet as well as a level of 3: with the level taking
    /// one step per bar, a collapse spends two bars above level 3, and without this guard a
    /// breakdown would fire on every one of them.
    fn track_breakdown(
        &mut self,
        previous_level: u8,
        low: f32,
        onset: f32,
        events: &mut EventFlags,
    ) {
        let quiet = low < BREAKDOWN_QUIET && onset < BREAKDOWN_QUIET;
        if previous_level >= BREAKDOWN_LEVEL && !quiet {
            self.breakdown_armed = true;
        }
        if self.breakdown_armed && quiet {
            self.breakdown_armed = false;
            *events |= EventFlags::BREAKDOWN;
        }
    }

    fn push_recent(&mut self, intensity: f32, onset: f32, bright: f32) {
        if self.recent_len < BUILD_BARS {
            self.recent_intensity[self.recent_len] = intensity;
            self.recent_onset[self.recent_len] = onset;
            self.recent_bright[self.recent_len] = bright;
            self.recent_len += 1;
        } else {
            self.recent_intensity.rotate_left(1);
            self.recent_onset.rotate_left(1);
            self.recent_bright.rotate_left(1);
            self.recent_intensity[BUILD_BARS - 1] = intensity;
            self.recent_onset[BUILD_BARS - 1] = onset;
            self.recent_bright[BUILD_BARS - 1] = bright;
        }
    }

    fn clear_state(&mut self) {
        self.level = 1;
        self.intensity = 0.0;
        self.phrase_pos = 0;
        self.scale.clear();
        self.onsets.clear();
        self.recent_len = 0;
        self.early_len = 0;
        self.build_active = false;
        self.build_bars_left = 0;
        self.gap_bars = 0;
        self.breakdown_armed = false;
        self.last_low = None;
        self.drop_bar = None;
        self.last_drop_bar = None;
    }
}

/// The level a bar's intensity falls in, ignoring hysteresis.
fn level_from_intensity(intensity: f32) -> u8 {
    let mut level = 1u8;
    for threshold in LEVEL_THRESHOLDS {
        if intensity >= threshold {
            level += 1;
        }
    }
    level
}

/// Move at most one level per bar, and only when the hysteresis band allows it: crossing a
/// boundary by `LEVEL_BAND` is needed to leave a level, in either direction. A collapse from
/// level 4 to level 1 therefore takes three bars, which is what keeps a breakdown from
/// snapping the show's parameters in a single bar.
fn next_level(level: u8, intensity: f32) -> u8 {
    let candidate = level_from_intensity(intensity);
    let target = if candidate > level {
        let boundary = LEVEL_THRESHOLDS[usize::from(level).saturating_sub(1)];
        if intensity >= boundary + LEVEL_BAND {
            candidate
        } else {
            level
        }
    } else if candidate < level {
        let boundary = LEVEL_THRESHOLDS[usize::from(level).saturating_sub(2)];
        if intensity < boundary - LEVEL_BAND {
            candidate
        } else {
            level
        }
    } else {
        level
    };
    match target.cmp(&level) {
        std::cmp::Ordering::Greater => level + 1,
        std::cmp::Ordering::Less => level - 1,
        std::cmp::Ordering::Equal => level,
    }
}

/// Clamp a normalised feature into 0..1; non-finite input counts as nothing.
fn unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Onset density as a non-negative, finite number.
fn onset_density(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published thresholds and the hysteresis band, as the plan states them.
    #[test]
    fn level_from_intensity_uses_the_published_thresholds() {
        let cases = [
            (0.0, 1),
            (0.249, 1),
            (0.25, 2),
            (0.499, 2),
            (0.5, 3),
            (0.749, 3),
            (0.75, 4),
            (1.0, 4),
        ];
        for (intensity, level) in cases {
            assert_eq!(
                level_from_intensity(intensity),
                level,
                "intensity {intensity}"
            );
        }
    }

    /// Entering a level needs `threshold + LEVEL_BAND`, leaving it needs `threshold - LEVEL_BAND`,
    /// and the level never moves more than one step per bar.
    #[test]
    fn next_level_needs_the_band_and_moves_one_step() {
        // From level 1: 0.28 is above 0.25 but inside the band, so nothing happens.
        assert_eq!(next_level(1, 0.28), 1);
        assert_eq!(next_level(1, 0.29), 2);
        // From level 2: down needs 0.25 - 0.04, up needs 0.5 + 0.04.
        assert_eq!(next_level(2, 0.22), 2);
        assert_eq!(next_level(2, 0.21), 1);
        assert_eq!(next_level(2, 0.53), 2);
        assert_eq!(next_level(2, 0.54), 3);
        // A jump of three levels is taken one step at a time.
        assert_eq!(next_level(1, 0.99), 2);
        assert_eq!(next_level(4, 0.01), 3);
        // Staying put.
        assert_eq!(next_level(3, 0.5), 3);
        assert_eq!(next_level(4, 0.75), 4);
        assert_eq!(next_level(4, 0.7), 3);
    }

    /// The band is the plan's, and the boundary table is what the two functions above assume.
    #[test]
    fn the_band_and_the_thresholds_are_the_published_ones() {
        assert_eq!(LEVEL_THRESHOLDS, [0.25, 0.5, 0.75]);
        assert!((LEVEL_HYSTERESIS - 0.08).abs() < 1e-6);
        assert!((LEVEL_BAND - 0.04).abs() < 1e-6);
    }

    fn bar(index: i64, loudness_db: f32, low: f32, onsets: f32) -> BarStats {
        BarStats {
            bar_index: index,
            start_ppq: index as f64 * 4.0,
            beats: 4,
            loudness_db,
            loudness: 0.5,
            low,
            onset_density: onsets,
            brightness: 0.5,
        }
    }

    fn raw(loudness_db: f32, level_db: f32) -> BarFeatures {
        BarFeatures {
            loudness_db,
            low_db: -20.0,
            level_db,
            early_db: level_db,
            ..BarFeatures::from_stats(&BarStats::default())
        }
    }

    /// The mid-bar pass and the bar close agree on what a drop is: a big fast-level jump with the
    /// low end present, after a gap.
    #[test]
    fn a_level_jump_after_a_gap_is_a_drop() {
        let mut model = IntensityModel::new(8, 32);
        // One bar with no low end: a gap.
        model.push_bar_with(&bar(0, -24.0, 0.1, 2.0), &raw(-24.0, -24.0), &SectionState::default());
        // The next bar jumps 10 dB and brings the low end back.
        let drop_bar = bar(1, -12.0, 0.85, 3.0);
        let drop_raw = raw(-12.0, -12.0);
        assert!(
            model.is_drop(&drop_bar, &drop_raw),
            "a 12 dB jump with the low end back is a drop"
        );
        let u = model.push_bar_with(&drop_bar, &drop_raw, &SectionState::default());
        assert!(u.events.contains(EventFlags::DROP), "events {:?}", u.events);
    }

    /// A drop fired mid-bar is not fired again when the bar closes.
    #[test]
    fn a_drop_fires_once() {
        let mut model = IntensityModel::new(8, 32);
        model.push_bar_with(&bar(0, -24.0, 0.1, 2.0), &raw(-24.0, -24.0), &SectionState::default());
        let drop_bar = bar(1, -12.0, 0.85, 3.0);
        model.note_drop(1);
        let u = model.push_bar_with(&drop_bar, &raw(-12.0, -12.0), &SectionState::default());
        assert!(
            !u.events.contains(EventFlags::DROP),
            "the bar close must not repeat the mid-bar drop: {:?}",
            u.events
        );
    }

    /// A steady loop with no build, no gap and no jump is never a drop, however the low band is
    /// normalised: the old rule fired on the low band alone.
    #[test]
    fn a_steady_low_end_is_not_a_drop() {
        let mut model = IntensityModel::new(8, 32);
        for i in 0..16 {
            let b = bar(i, -16.0, 0.8, 2.0);
            let u = model.push_bar_with(&b, &raw(-16.0, -16.0), &SectionState::default());
            assert!(
                !u.events.contains(EventFlags::DROP),
                "bar {i}: a steady low end is not a drop"
            );
        }
    }

    /// A snare roll raises the onset density without the low end: that is a build.
    #[test]
    fn rising_onsets_start_a_build() {
        let mut model = IntensityModel::new(8, 32);
        let mut build_bars = Vec::new();
        for i in 0..8i64 {
            let u = model.push_bar_with(
                &bar(i, -20.0, 0.2, 1.0),
                &raw(-20.0, -20.0),
                &SectionState::default(),
            );
            assert!(!u.events.contains(EventFlags::BUILD_START));
        }
        for (i, onsets) in [1.4f32, 2.2, 3.0, 3.8].into_iter().enumerate() {
            let index = 8 + i as i64;
            let u = model.push_bar_with(
                &bar(index, -19.0, 0.2, onsets),
                &raw(-19.0, -19.0),
                &SectionState::default(),
            );
            if u.events.contains(EventFlags::BUILD_START) {
                build_bars.push(index);
            }
        }
        assert_eq!(
            build_bars,
            vec![9],
            "BUILD_START on the second bar of the roll, once"
        );
    }

    /// The section target is most of the level reading.
    #[test]
    fn the_section_target_drives_the_reading() {
        let mut with = IntensityModel::new(8, 32);
        for i in 0..8i64 {
            with.push_bar_with(
                &bar(i, -18.0, 0.6, 2.0),
                &raw(-18.0, -18.0),
                &SectionState::default(),
            );
        }
        let section = SectionState {
            target: 0.9,
            present: true,
            ..SectionState::default()
        };
        let u = with.push_bar_with(
            &bar(8, -18.0, 0.6, 2.0),
            &raw(-18.0, -18.0),
            &section,
        );
        assert!(
            u.reading > 0.80,
            "a section target of 0.9 must carry the reading: {}",
            u.reading
        );
    }
}

