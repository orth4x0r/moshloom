//! Task 1.4: the intensity model and its events.
//!
//! Three rules are tested separately, because they are separate stages:
//!
//! - the composition: intensity is the weighted mean of the four normalised terms (loudness
//!   0.35, low 0.30, onset 0.20, brightness 0.15), and `IntensityUpdate::terms` reports exactly
//!   what went in. `push` checks that identity on every bar of every test;
//! - the level machine: thresholds 0.25 / 0.5 / 0.75, a 0.08 hysteresis band, at most one level
//!   per bar. Its exact numbers are unit-tested in `src/intensity.rs`; here it is exercised
//!   through whole bar sequences. Silence is loudness below -50 dB.
//!
//! `BarStats::loudness`, `low` and `brightness` arrive already normalised to 0..1 from the
//! feature stage, so a test can hand the model the term it wants. `onset_density` is raw onsets
//! per beat and the model normalises it: `ONSET_MID` is the value whose term is exactly 0.5 both
//! under the first-bars fallback and under a settled window.

use mosh_analysis::intensity::{IntensityModel, IntensityUpdate};
use mosh_core::{BarStats, EventFlags};

const PHRASE_BARS: u32 = 8;
const HISTORY_BARS: u32 = 32;

/// Weights from the plan. They sum to 1.
const W_LOUDNESS: f32 = 0.45;
const W_LOW: f32 = 0.25;
const W_ONSET: f32 = 0.18;
const W_BRIGHTNESS: f32 = 0.12;

/// Onset density, in raw onsets per beat, whose normalised term is exactly 0.5: the fallback
/// (1..4 per beat) and a settled window both map it to the middle.
const ONSET_MID: f32 = 2.5;

fn bar(
    bar_index: i64,
    loudness_db: f32,
    loudness: f32,
    low: f32,
    onset_density: f32,
    brightness: f32,
) -> BarStats {
    BarStats {
        bar_index,
        start_ppq: (bar_index as f64) * 4.0,
        beats: 4,
        loudness_db,
        loudness,
        low,
        onset_density,
        brightness,
    }
}

/// A normalised bar that is audible (so the silence gate does not apply).
fn terms(bar_index: i64, loudness: f32, low: f32, onset_density: f32, brightness: f32) -> BarStats {
    bar(bar_index, -20.0, loudness, low, onset_density, brightness)
}

/// A bar of a steady loop, at the given strength in every non-onset term.
fn steady(bar_index: i64, x: f32) -> BarStats {
    terms(bar_index, x, x, ONSET_MID, x)
}

/// A bar whose intensity is `0.8 * x + 0.1` (the onset term sits at 0.5).
fn at_intensity(bar_index: i64, x: f32) -> BarStats {
    steady(bar_index, x)
}

/// A bar of digital silence.
fn silent(bar_index: i64) -> BarStats {
    bar(bar_index, -60.0, 0.0, 0.0, 0.0, 0.0)
}

/// The weighted mean the model is specified to compute.
fn weighted(loudness: f32, low: f32, onset: f32, brightness: f32) -> f32 {
    (W_LOUDNESS * loudness + W_LOW * low + W_ONSET * onset + W_BRIGHTNESS * brightness)
        .clamp(0.0, 1.0)
}

/// Push one bar and check the identity every bar must satisfy: the update carries the model's
/// state, the terms are in 0..1, and the intensity is their weighted mean.
fn push(model: &mut IntensityModel, b: BarStats) -> IntensityUpdate {
    let u = model.push_bar(&b);
    assert_eq!(
        u.level,
        model.level(),
        "the update must carry the model's level"
    );
    assert!(
        (u.intensity - model.intensity()).abs() < 1e-6,
        "the update must carry the model's intensity"
    );
    let t = u.terms;
    for (name, value) in [
        ("loudness", t.loudness),
        ("low", t.low),
        ("onset", t.onset),
        ("brightness", t.brightness),
    ] {
        assert!(
            (0.0..=1.0).contains(&value),
            "bar {}: {name} {value} is outside 0..1",
            b.bar_index
        );
    }
    let want = weighted(t.loudness, t.low, t.onset, t.brightness);
    assert!(
        (u.intensity - want).abs() < 1e-6,
        "bar {}: intensity {} is not the weighted mean {want} of {t:?}",
        b.bar_index,
        u.intensity
    );
    u
}

/// Push a run of bars and return the levels.
fn levels_of(model: &mut IntensityModel, bars: impl IntoIterator<Item = BarStats>) -> Vec<u8> {
    bars.into_iter()
        .map(|b| push(model, b).level)
        .collect::<Vec<u8>>()
}

#[test]
fn weights_sum_to_one() {
    let sum = W_LOUDNESS + W_LOW + W_ONSET + W_BRIGHTNESS;
    assert!((sum - 1.0).abs() < 1e-6, "weights sum to {sum}");
}

#[test]
fn steady_loop_settles_to_one_level() {
    let mut model = IntensityModel::new(PHRASE_BARS, HISTORY_BARS);
    // x = 0.6 in the three normalised terms and 0.5 for onsets:
    // 0.45 * 0.6 + 0.25 * 0.6 + 0.18 * 0.5 + 0.12 * 0.6 = 0.582.
    let expected = weighted(0.6, 0.6, 0.5, 0.6);
    assert!((expected - 0.582).abs() < 1e-5, "{expected}");

    let mut levels = Vec::new();
    let mut late_events = EventFlags::empty();
    for i in 0..32i64 {
        let u = push(&mut model, steady(i, 0.6));
        assert!(
            (u.intensity - expected).abs() < 1e-5,
            "bar {i}: intensity {} != {expected}",
            u.intensity
        );
        assert!(
            (u.terms.onset - 0.5).abs() < 1e-5,
            "bar {i}: a steady onset density reads {}, not the middle",
            u.terms.onset
        );
        // Nothing in a flat loop is a build: the low band never *returns*, there is no gap
        // below 0.2, and intensity never rises.
        assert!(!u.events.contains(EventFlags::DROP), "bar {i}: spurious DROP");
        assert!(
            !u.events.contains(EventFlags::BUILD_START),
            "bar {i}: a flat loop is not a build"
        );
        assert!(
            !u.events.contains(EventFlags::BREAKDOWN),
            "bar {i}: spurious BREAKDOWN"
        );
        levels.push(u.level);
        if i >= 8 {
            late_events |= u.events;
        }
    }

    assert_eq!(levels[7], 2, "a window with no range reads the middle");
    for (i, level) in levels.iter().enumerate().skip(8) {
        assert_eq!(*level, 2, "bar {i}: level moved after it settled");
    }
    assert!(
        !late_events.intersects(EventFlags::LEVEL_UP | EventFlags::LEVEL_DOWN),
        "a steady loop keeps its level"
    );
}

#[test]
fn a_steady_loud_loop_settles_below_level_four() {
    // The saturation case, in the model's own terms: every normalised term at the ceiling for 32
    // bars. The level reads the intensity's position in the recent range, and a window with no
    // range reads the middle, so the loop settles at 2 or 3 and the top level stays reserved for
    // the bar where something actually lands. `tests/intensity_levels.rs` runs the same case
    // through real audio.
    let mut model = IntensityModel::new(PHRASE_BARS, HISTORY_BARS);
    let levels = levels_of(&mut model, (0..32i64).map(|i| steady(i, 1.0)));

    assert!(
        !levels.contains(&4),
        "a loop with no dynamics must never reach the top level: {levels:?}"
    );
    assert!(
        levels[3..].iter().all(|&level| level == levels[3]),
        "a steady loop settles, it does not oscillate: {levels:?}"
    );
    assert!((2..=3).contains(&levels[3]), "settled at {}", levels[3]);
}

#[test]
fn a_quiet_passage_reaches_the_bottom_and_a_loud_one_the_top() {
    // Criterion (a) at model level: over a track-shaped sequence the four levels all appear.
    let mut model = IntensityModel::new(PHRASE_BARS, HISTORY_BARS);
    let mut levels = Vec::new();

    // 8 quiet bars from a cold start settle at 2: the reading is the raw intensity while there
    // is no window, then the middle of a window that has no spread.
    for i in 0..8i64 {
        levels.push(push(&mut model, at_intensity(i, 0.125)).level); // 0.2
    }
    // 8 loud bars, each the loudest thing in the window: the level climbs to the top.
    for i in 8..16i64 {
        levels.push(push(&mut model, at_intensity(i, 0.9)).level); // 0.82
    }
    // 8 bars louder still.
    for i in 16..24i64 {
        levels.push(push(&mut model, at_intensity(i, 1.0)).level); // 0.9
    }
    // 8 quiet bars again: now they are the bottom of the window, and they fall to level 1.
    for i in 24..32i64 {
        levels.push(push(&mut model, at_intensity(i, 0.125)).level); // 0.2
    }

    for want in 1..=4u8 {
        assert!(
            levels.contains(&want),
            "level {want} never appeared: {levels:?}"
        );
    }
    assert_eq!(
        &levels[8..16],
        &[3, 4, 4, 4, 4, 4, 4, 4],
        "the loud passage climbs and then settles: {levels:?}"
    );
    assert_eq!(
        &levels[24..],
        &[3, 2, 1, 1, 1, 1, 1, 1],
        "the quiet passage falls one step per bar: {levels:?}"
    );
}

#[test]
fn a_collapse_moves_one_level_per_bar() {
    let mut model = IntensityModel::new(PHRASE_BARS, HISTORY_BARS);

    // Climb to the top on rising bars ...
    for i in 0..8i64 {
        push(&mut model, at_intensity(i, 0.55 + 0.05 * i as f32));
    }
    assert_eq!(model.level(), 4, "a six-bar rise reaches the top");

    // ... then pull the floor out. The level comes down one step per bar: 3, 2, 1.
    let steps = levels_of(&mut model, (8..11i64).map(|i| at_intensity(i, 0.0)));
    assert_eq!(steps, vec![3, 2, 1], "one level per bar: {steps:?}");
}

#[test]
fn build_then_drop_detected() {
    let mut model = IntensityModel::new(PHRASE_BARS, HISTORY_BARS);

    let mut build_bars = Vec::new();
    let mut drop_bars = Vec::new();

    // Bars 0..8 (1-based 1..8): a low, quiet section. No low end, no brightness.
    for i in 0..8i64 {
        let u = push(&mut model, terms(i, 0.4, 0.1, 0.8, 0.2));
        assert!(
            !u.events.contains(EventFlags::BUILD_START),
            "bar {i}: no build yet"
        );
        assert!(!u.events.contains(EventFlags::DROP), "bar {i}: no drop yet");
    }

    // Bars 8..12 (1-based 9..12): four bars rising by more than 0.15 in total,
    // low end still absent.
    for (i, (l, o, br)) in [
        (0.5, 1.4, 0.4),
        (0.6, 2.0, 0.55),
        (0.7, 2.6, 0.7),
        (0.78, 3.2, 0.8),
    ]
    .into_iter()
    .enumerate()
    {
        let index = 8 + i as i64;
        let low = if i == 3 { 0.15 } else { 0.1 };
        let u = push(&mut model, terms(index, l, low, o, br));
        if u.events.contains(EventFlags::BUILD_START) {
            build_bars.push(index);
        }
        assert!(
            !u.events.contains(EventFlags::DROP),
            "bar {index}: no drop before the low end"
        );
    }

    // Bar 12 (1-based 13): the low end returns on a full bar.
    let u = push(&mut model, bar(12, -8.0, 0.9, 0.85, 3.4, 0.9));
    if u.events.contains(EventFlags::BUILD_START) {
        build_bars.push(12);
    }
    if u.events.contains(EventFlags::DROP) {
        drop_bars.push(12);
    }
    let level_at_drop = u.level;

    // Bar 13 (1-based 14): the low end stays, so there is nothing left to drop.
    let u = push(&mut model, bar(13, -8.0, 0.9, 0.85, 3.4, 0.9));
    if u.events.contains(EventFlags::DROP) {
        drop_bars.push(13);
    }

    assert_eq!(
        build_bars,
        vec![9],
        "BUILD_START once the sweep has risen for two bars, and once only"
    );
    assert_eq!(drop_bars, vec![12], "DROP when the low end returns");
    assert_eq!(level_at_drop, 4, "the drop bar is a loud bar");
}

#[test]
fn breakdown_detected() {
    let mut model = IntensityModel::new(PHRASE_BARS, HISTORY_BARS);

    let mut breakdown_bars = Vec::new();
    let mut levels = Vec::new();
    for i in 0..8i64 {
        // A loud, busy section rising over its eight bars, so the level climbs into 3 and 4
        // instead of settling at the middle of a window with no range.
        let loudness = 0.55 + 0.055 * i as f32;
        let u = push(&mut model, terms(i, loudness, 0.8, 1.5, 0.7));
        assert!(
            !u.events.contains(EventFlags::BREAKDOWN),
            "bar {i}: nothing broke down yet"
        );
        assert!(
            u.level >= 2,
            "bar {i}: a loud section is not at level 1 ({})",
            u.level
        );
        levels.push(u.level);
    }
    assert!(
        levels.iter().any(|&level| level >= 3),
        "the section must arm the breakdown detector: {levels:?}"
    );

    // Low end and onsets both fall away: a breakdown for a full bar, and the floor sliding out
    // from under it over the next two bars.
    for (i, (l, lo, br)) in [(0.35, 0.10, 0.30), (0.32, 0.08, 0.25), (0.30, 0.06, 0.20)]
        .into_iter()
        .enumerate()
    {
        let index = 8 + i as i64;
        let u = push(&mut model, terms(index, l, lo, 0.0, br));
        assert!(
            u.terms.onset < 0.25 && u.terms.low < 0.25,
            "bar {index}: the breakdown bar is quiet in low and onsets: {:?}",
            u.terms
        );
        if u.events.contains(EventFlags::BREAKDOWN) {
            breakdown_bars.push(index);
        }
        levels.push(u.level);
    }

    assert_eq!(
        breakdown_bars,
        vec![8],
        "BREAKDOWN fires once, on the first bar after level >= 3"
    );
    assert_eq!(
        &levels[8..],
        &[3, 2, 1],
        "a collapse takes one level per bar, not a jump: {levels:?}"
    );
}

#[test]
fn silence_never_raises_level() {
    let mut model = IntensityModel::new(PHRASE_BARS, HISTORY_BARS);

    // Digital silence in dB, but with feature values high enough that ignoring the dB gate
    // would read as a loud bar (0.35*0.9 + 0.3*0.9 + 0.2*0.5 + 0.15*0.9 = 0.82).
    for i in 0..12i64 {
        let u = push(&mut model, bar(i, -60.0, 0.9, 0.9, 2.0, 0.9));
        assert_eq!(u.intensity, 0.0, "bar {i}: silence has no intensity");
        assert_eq!(u.level, 1, "bar {i}: silence never raises the level");
        assert!(
            !u.events.intersects(
                EventFlags::LEVEL_UP
                    | EventFlags::BUILD_START
                    | EventFlags::DROP
                    | EventFlags::BREAKDOWN
            ),
            "bar {i}: silence is not a musical event"
        );
    }

    // A noise floor just above the gate stays at level 1 too.
    for i in 12..20i64 {
        let u = push(&mut model, bar(i, -55.0, 0.05, 0.05, 0.05, 0.05));
        assert_eq!(u.level, 1, "bar {i}: a noise floor is not a level");
    }
}

#[test]
fn history_resets_after_silence() {
    let mut model = IntensityModel::new(PHRASE_BARS, HISTORY_BARS);

    // A quiet opening, then a loud section: the level climbs to the top on the step up, so a
    // stale history would still be loud afterwards.
    for i in 0..4i64 {
        push(&mut model, bar(i, -30.0, 0.3, 0.3, 2.5, 0.3));
    }
    let mut loud_top = 0u8;
    for i in 4..12i64 {
        let u = push(&mut model, bar(i, -8.0, 0.95, 0.9, 2.5, 0.9));
        assert!(u.level >= 2, "bar {i}: the loud section is not quiet");
        loud_top = loud_top.max(u.level);
    }
    assert_eq!(loud_top, 4, "the step up reaches level 4");
    assert!(
        model.level() >= 3,
        "and the section settles at 3 or 4, not {}",
        model.level()
    );

    // > 2 bars of silence is a gap between songs: history and level start over.
    for i in 8..11i64 {
        push(&mut model, silent(i));
    }
    assert_eq!(model.level(), 1, "the gap reset the level");
    assert_eq!(model.intensity(), 0.0, "the gap reset the intensity");

    // The first bar of the next song is quiet: it must not arrive at a high level, and its
    // onset density must be read against a clean history: the cold-start fallback blended with
    // the middle (2 onsets per beat maps to 1/3, so the term is 0.5 * 1/3 + 0.25), not a window
    // of the previous song's 2.5 onsets per beat (which would read 0.5).
    let u = push(&mut model, bar(11, -45.0, 0.4, 0.3, 2.0, 0.3));
    let cold = 0.5 * (1.0 / 3.0) + 0.25;
    assert!(
        (u.terms.onset - cold).abs() < 1e-5,
        "onset term {} is not the cold-start value {cold}: the history did not restart",
        u.terms.onset
    );
    let quiet = weighted(0.4, 0.3, cold, 0.3);
    assert!(
        (u.intensity - quiet).abs() < 1e-4,
        "intensity {} != {quiet}",
        u.intensity
    );
    assert!(
        u.level <= 2,
        "a quiet bar after a gap is level {}, not 4",
        u.level
    );
    assert!(
        !u.events.contains(EventFlags::LEVEL_DOWN),
        "the quiet bar does not fall from a stale level"
    );
    assert!(
        !u.events.intersects(EventFlags::BREAKDOWN | EventFlags::DROP),
        "a song's first quiet bar is not a breakdown or a drop"
    );
}

#[test]
fn phrase_event_every_phrase_bars() {
    let mut model = IntensityModel::new(4, HISTORY_BARS);

    let mut phrase_bars = Vec::new();
    for i in 0..9i64 {
        let u = push(&mut model, steady(i, 0.6));
        if u.events.contains(EventFlags::PHRASE) {
            phrase_bars.push(i);
        }
    }
    assert_eq!(
        phrase_bars,
        vec![3, 7],
        "PHRASE every four bars, from the model's start"
    );

    // A song marker restarts the phrase counter.
    model.reset();
    let mut after_reset = Vec::new();
    for i in 0..4i64 {
        let u = push(&mut model, steady(20 + i, 0.6));
        if u.events.contains(EventFlags::PHRASE) {
            after_reset.push(i);
        }
    }
    assert_eq!(
        after_reset,
        vec![3],
        "the phrase counter restarts at reset()"
    );
}

#[test]
fn reset_clears_level_and_history() {
    let mut model = IntensityModel::new(PHRASE_BARS, HISTORY_BARS);

    for i in 0..4i64 {
        push(&mut model, bar(i, -30.0, 0.3, 0.3, 2.5, 0.3));
    }
    let mut top = 0u8;
    for i in 4..12i64 {
        top = top.max(push(&mut model, bar(i, -8.0, 0.95, 0.9, 2.5, 0.9)).level);
    }
    assert_eq!(top, 4);

    model.reset();
    assert_eq!(model.level(), 1, "reset() drops back to level 1");
    assert_eq!(model.intensity(), 0.0, "reset() forgets the last intensity");

    // A quiet first bar of the next song starts low, from a clean history.
    let u = push(&mut model, bar(8, -45.0, 0.4, 0.3, 0.5, 0.3));
    assert!(
        u.level <= 2,
        "after reset a quiet bar is level {}, not 4",
        u.level
    );
    assert!(u.intensity < 0.5);
}

#[test]
fn zero_configuration_is_clamped() {
    // The app may build the model from a config with empty values; it must not panic
    // or divide by a zero-length window.
    let mut model = IntensityModel::new(0, 0);
    for i in 0..5i64 {
        let u = push(&mut model, steady(i, 0.6));
        assert!((1..=4).contains(&u.level));
        assert!(
            u.events.contains(EventFlags::PHRASE),
            "bar {i}: phrase_bars 0 means every bar"
        );
    }
}
