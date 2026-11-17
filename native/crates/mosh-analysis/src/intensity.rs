//! The intensity model: per-bar features to a continuous intensity, a 1..=4 level
//! with hysteresis, and the phrase / build / drop / breakdown events the show uses.
//!
//! Task 1.4 of `docs/superpowers/plans/2026-11-17-native-port.md`.
//!
//! Two causal stages, both against recent bars:
//!
//! 1. **Intensity** is the weighted mean of the four terms (loudness 0.45, low 0.25, onset 0.18,
//!    brightness 0.12; the plan's starting point was 0.35 / 0.30 / 0.20 / 0.15 and the change is
//!    explained at `W_LOUDNESS`). `loudness`, `low` and `brightness` arrive already normalised to
//!    0..1 by the feature stage, which blends a fixed typical range with the bar's rank over the
//!    last `history_bars` bars; `onset_density` arrives raw, as onsets per beat, and this model
//!    normalises it the same way. `loudness_db` is the raw short-term level and decides silence,
//!    because a normalised feature of a noise floor can be anything.
//! 2. **Level** is where that intensity sits in the recent dynamic range (its 5th-95th percentile
//!    over the last `history_bars` bars), through the plan's thresholds (0.25 / 0.5 / 0.75) with
//!    its hysteresis band and at most one step per bar. Reading a position in the range is what
//!    makes the levels usable: the weighted mean of four correlated terms sits in a narrow band,
//!    so thresholds on the mean itself put almost every bar in one or two levels. A section that
//!    does not change reads the middle; a drop reads the top; a breakdown reads the bottom.
//!
//! Everything here is causal and per bar: `push_bar` is the only way in, so a level
//! never changes in the middle of a bar, never moves more than one step per bar, and the same
//! bar sequence always produces the same events. `reset` is for song markers.

use std::collections::VecDeque;

use mosh_core::{BarStats, EventFlags};

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

/// Drop: the low band returns above this, having been at or below it.
const DROP_LOW: f32 = 0.6;
/// A bar with low below this counts as "no low end" for the drop detection.
const LOW_GAP: f32 = 0.2;

/// Silence: short-term loudness below this many dB.
const SILENCE_DB: f32 = -50.0;
/// More than this many silent bars in a row (a gap between songs) forgets the level
/// and the history, so the next song starts from nothing.
const SILENCE_BARS_TO_RESET: u32 = 2;

/// BUILD_START: intensity rose for this many consecutive bars ...
const BUILD_BARS: usize = 4;
/// ... by more than this in total.
const BUILD_RISE: f32 = 0.15;
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
    /// The intensity through the level machine's calibration (0..1). Diagnostic: this is what
    /// the thresholds saw.
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
    /// Intensities of the last \history_bars\ bars: the range the level places a bar in.
    scale: RollingRange,
    /// Intensities of the last `BUILD_BARS` bars, current bar excluded, oldest first.
    recent: VecDeque<f32>,
    /// Consecutive bars with a strictly rising intensity.
    rising_bars: usize,
    /// BUILD_START fired and no DROP or timeout has closed the build yet.
    build_active: bool,
    /// Bars left before an open build times out.
    build_bars_left: u32,
    /// Consecutive bars with low below `LOW_GAP`, before the current bar.
    low_gap_bars: u32,
    /// A level of at least `BREAKDOWN_LEVEL` was seen since the last breakdown.
    breakdown_armed: bool,
    /// Low band of the previous bar, for the "the low end returns" crossing.
    last_low: Option<f32>,
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
            recent: VecDeque::new(),
            rising_bars: 0,
            build_active: false,
            build_bars_left: 0,
            low_gap_bars: 0,
            breakdown_armed: false,
            last_low: None,
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

    /// Fold one completed bar in and report what it did.
    pub fn push_bar(&mut self, b: &BarStats) -> IntensityUpdate {
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
            // A rise interrupted by silence is not consecutive.
            self.rising_bars = 0;
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

        // The level places this bar in the recent dynamic range (see `LEVEL_MIN_SPAN`), which is
        // what makes the four levels cover a section instead of hugging the middle of a weighted
        // mean. The window includes this bar; `position` folds it in. `intensity` stays the plain
        // weighted mean: the show modulates with it.
        let reading = if silent {
            0.0
        } else {
            self.scale
                .position(intensity, LEVEL_MIN_SPAN)
                .unwrap_or(0.5)
        };
        self.scale.push(intensity);

        let previous_level = self.level;
        let mut events = EventFlags::empty();

        if !silent {
            self.track_build(intensity, &mut events);
            self.track_drop(low, &mut events);
            self.track_breakdown(previous_level, low, onset, &mut events);
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

        self.intensity = intensity;
        self.last_low = if silent { None } else { Some(low) };
        self.recent.push_back(intensity);
        if self.recent.len() > BUILD_BARS {
            self.recent.pop_front();
        }

        IntensityUpdate {
            intensity,
            reading,
            level: self.level,
            events,
            terms,
        }
    }

    /// BUILD_START: four consecutive rising bars that together rose by > 0.15.
    fn track_build(&mut self, intensity: f32, events: &mut EventFlags) {
        match self.recent.back().copied() {
            Some(previous) if intensity > previous => self.rising_bars += 1,
            Some(_) => self.rising_bars = 0,
            None => self.rising_bars = 0,
        }

        if self.build_active {
            if self.build_bars_left == 0 {
                self.build_active = false;
            } else {
                self.build_bars_left -= 1;
            }
        }

        if self.build_active || self.rising_bars < BUILD_BARS || self.recent.len() != BUILD_BARS {
            return;
        }
        if let Some(&base) = self.recent.front()
            && intensity - base > BUILD_RISE
        {
            self.build_active = true;
            self.build_bars_left = BUILD_MAX_BARS;
            *events |= EventFlags::BUILD_START;
        }
    }

    /// DROP: the low band crosses back above `DROP_LOW` after a build or after at
    /// least one bar with no low end.
    fn track_drop(&mut self, low: f32, events: &mut EventFlags) {
        let returned = low > DROP_LOW && self.last_low.is_none_or(|previous| previous <= DROP_LOW);
        if returned && (self.build_active || self.low_gap_bars >= 1) {
            *events |= EventFlags::DROP;
            self.build_active = false;
            self.build_bars_left = 0;
            self.low_gap_bars = 0;
        } else if low < LOW_GAP {
            self.low_gap_bars += 1;
        } else {
            self.low_gap_bars = 0;
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

    fn clear_state(&mut self) {
        self.level = 1;
        self.intensity = 0.0;
        self.phrase_pos = 0;
        self.scale.clear();
        self.onsets.clear();
        self.recent.clear();
        self.rising_bars = 0;
        self.build_active = false;
        self.build_bars_left = 0;
        self.low_gap_bars = 0;
        self.breakdown_armed = false;
        self.last_low = None;
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
}
