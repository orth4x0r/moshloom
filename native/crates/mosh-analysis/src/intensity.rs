//! The intensity model: per-bar features to a continuous intensity, a 1..=4 level
//! with hysteresis, and the phrase / build / drop / breakdown events the show uses.
//!
//! Task 1.4 of `docs/superpowers/plans/2026-11-17-native-port.md`.
//!
//! Input contract. `BarStats` comes from the feature stage (Task 1.3), which normalises
//! `loudness`, `low` and `brightness` to 0..1 against a causal window of the last
//! `history_bars` bars. `onset_density` is *not* normalised there: it is spectral-flux
//! peaks per beat, so this model divides it by the largest value in its own window of
//! the last `history_bars` bars, floored at one onset per beat. `loudness_db` is the raw
//! short-term level and decides silence, because a normalised feature of a noise floor
//! can be anything.
//!
//! Everything here is causal and per bar: `push_bar` is the only way in, so a level
//! never changes in the middle of a bar and the same bar sequence always produces the
//! same events. `reset` is for song markers.

use std::collections::VecDeque;

use mosh_core::{BarStats, EventFlags};

/// Weights of the four normalised terms. They sum to 1, so the intensity is 0..1.
const W_LOUDNESS: f32 = 0.35;
const W_LOW: f32 = 0.30;
const W_ONSET: f32 = 0.20;
const W_BRIGHTNESS: f32 = 0.15;

/// Level boundaries: below 0.25 is level 1, at or above 0.75 is level 4.
const LEVEL_THRESHOLDS: [f32; 3] = [0.25, 0.5, 0.75];
/// Total width of the hysteresis band, split evenly around each threshold: a level
/// rises 0.04 above the boundary and falls only 0.04 below it.
const LEVEL_HYSTERESIS: f32 = 0.08;
const LEVEL_BAND: f32 = LEVEL_HYSTERESIS * 0.5;

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

/// Onset densities are read against the window maximum, but never against less than
/// one onset per beat, so a lone onset in a quiet bar cannot look maximal.
const ONSET_REF_FLOOR: f32 = 1.0;

/// What one bar did to the model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntensityUpdate {
    /// Continuous intensity 0..1.
    pub intensity: f32,
    /// 1..=4, changing only here, on a bar boundary.
    pub level: u8,
    /// PHRASE / BUILD_START / DROP / BREAKDOWN / LEVEL_UP / LEVEL_DOWN for this bar.
    /// BEAT, BAR and SONG_START belong to the grid and app layers, not to this model.
    pub events: EventFlags,
}

/// Turns the per-bar features of a song into intensity, level and events.
#[derive(Debug)]
pub struct IntensityModel {
    phrase_bars: u32,
    history_bars: usize,
    level: u8,
    intensity: f32,
    phrase_pos: u32,
    /// Onset densities of the last `history_bars` bars, current bar included.
    onsets: VecDeque<f32>,
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
    /// length of the onset window (clamped to at least 1).
    pub fn new(phrase_bars: u32, history_bars: u32) -> Self {
        IntensityModel {
            phrase_bars: phrase_bars.max(1),
            history_bars: history_bars.max(1) as usize,
            level: 1,
            intensity: 0.0,
            phrase_pos: 0,
            onsets: VecDeque::new(),
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

        // The window includes this bar: everything used is already known.
        self.onsets.push_back(onset_density);
        while self.onsets.len() > self.history_bars {
            self.onsets.pop_front();
        }
        let onset_ref = self.onsets.iter().copied().fold(ONSET_REF_FLOOR, f32::max);
        let onset = (onset_density / onset_ref).clamp(0.0, 1.0);

        let intensity = if silent {
            0.0
        } else {
            (W_LOUDNESS * loudness + W_LOW * low + W_ONSET * onset + W_BRIGHTNESS * brightness)
                .clamp(0.0, 1.0)
        };

        let previous_level = self.level;
        let mut events = EventFlags::empty();

        if !silent {
            self.track_build(intensity, &mut events);
            self.track_drop(low, &mut events);
            self.track_breakdown(previous_level, low, onset, &mut events);
        }

        self.level = next_level(previous_level, intensity);
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
            level: self.level,
            events,
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
    fn track_breakdown(
        &mut self,
        previous_level: u8,
        low: f32,
        onset: f32,
        events: &mut EventFlags,
    ) {
        if previous_level >= BREAKDOWN_LEVEL {
            self.breakdown_armed = true;
        }
        if self.breakdown_armed && low < BREAKDOWN_QUIET && onset < BREAKDOWN_QUIET {
            self.breakdown_armed = false;
            *events |= EventFlags::BREAKDOWN;
        }
    }

    fn clear_state(&mut self) {
        self.level = 1;
        self.intensity = 0.0;
        self.phrase_pos = 0;
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

/// Move at most as far as the hysteresis band allows: crossing a boundary by
/// `LEVEL_BAND` is needed to leave a level, in either direction.
fn next_level(level: u8, intensity: f32) -> u8 {
    let candidate = level_from_intensity(intensity);
    if candidate > level {
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
