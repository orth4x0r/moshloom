//! Task 1.4: the intensity model and its events.
//!
//! The numbers are worked out from the rule in the plan: intensity is the weighted
//! mean of normalised loudness 0.35, low 0.30, onset density 0.20, brightness 0.15;
//! levels come from thresholds 0.25 / 0.5 / 0.75 with a 0.08 hysteresis band; silence
//! is loudness below -50 dB.

use mosh_analysis::intensity::IntensityModel;
use mosh_core::{BarStats, EventFlags};

const PHRASE_BARS: u32 = 8;
const HISTORY_BARS: u32 = 16;

/// The weighted mean the model is specified to compute, for bars whose onset density
/// stays at or below one onset per beat (so the history reference does not rescale it).
fn weighted(loudness: f32, low: f32, onset_density: f32, brightness: f32) -> f32 {
    0.35 * loudness + 0.30 * low + 0.20 * onset_density + 0.15 * brightness
}

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

/// A bar of digital silence.
fn silent(bar_index: i64) -> BarStats {
    bar(bar_index, -60.0, 0.0, 0.0, 0.0, 0.0)
}

fn push(model: &mut IntensityModel, b: BarStats) -> (f32, u8, EventFlags) {
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
    (u.intensity, u.level, u.events)
}

#[test]
fn steady_loop_settles_to_one_level() {
    let mut model = IntensityModel::new(PHRASE_BARS, HISTORY_BARS);
    let expected = weighted(0.6, 0.65, 0.5, 0.5);

    let mut levels = Vec::new();
    let mut late_events = EventFlags::empty();
    for i in 0..32i64 {
        let (intensity, level, events) = push(&mut model, terms(i, 0.6, 0.65, 0.5, 0.5));
        assert!(
            (intensity - expected).abs() < 1e-5,
            "bar {i}: intensity {intensity} != {expected}"
        );
        // The low band is above 0.6 for the whole loop, but it never *returns*:
        // there is no build and no gap below 0.2, so nothing is a drop.
        assert!(!events.contains(EventFlags::DROP), "bar {i}: spurious DROP");
        assert!(
            !events.contains(EventFlags::BUILD_START),
            "bar {i}: a flat loop is not a build"
        );
        assert!(
            !events.contains(EventFlags::BREAKDOWN),
            "bar {i}: spurious BREAKDOWN"
        );
        levels.push(level);
        if i >= 8 {
            late_events |= events;
        }
    }

    assert_eq!(levels[7], 3, "0.58 is above 0.5, so level 3");
    for (i, level) in levels.iter().enumerate().skip(8) {
        assert_eq!(*level, levels[7], "bar {i}: level moved after it settled");
    }
    assert!(
        !late_events.intersects(EventFlags::LEVEL_UP | EventFlags::LEVEL_DOWN),
        "a steady loop keeps its level"
    );
}

#[test]
fn build_then_drop_detected() {
    let mut model = IntensityModel::new(PHRASE_BARS, HISTORY_BARS);

    let mut build_bars = Vec::new();
    let mut drop_bars = Vec::new();

    // Bars 0..8 (1-based 1..8): a low, quiet section. No low end, no brightness.
    for i in 0..8i64 {
        let (_, _, events) = push(&mut model, terms(i, 0.4, 0.1, 0.2, 0.2));
        assert!(
            !events.contains(EventFlags::BUILD_START),
            "bar {i}: no build yet"
        );
        assert!(!events.contains(EventFlags::DROP), "bar {i}: no drop yet");
    }

    // Bars 8..12 (1-based 9..12): four bars rising by more than 0.15 in total,
    // low end still absent.
    for (i, (l, o, br)) in [
        (0.5, 0.4, 0.4),
        (0.6, 0.7, 0.55),
        (0.7, 1.0, 0.7),
        (0.78, 1.2, 0.8),
    ]
    .into_iter()
    .enumerate()
    {
        let index = 8 + i as i64;
        let low = if i == 3 { 0.15 } else { 0.1 };
        let (_, _, events) = push(&mut model, terms(index, l, low, o, br));
        if events.contains(EventFlags::BUILD_START) {
            build_bars.push(index);
        }
        assert!(
            !events.contains(EventFlags::DROP),
            "bar {index}: no drop before the low end"
        );
    }

    // Bar 12 (1-based 13): the low end returns on a full bar.
    let (_, level_at_drop, events) = push(&mut model, bar(12, -8.0, 0.9, 0.85, 1.0, 0.9));
    if events.contains(EventFlags::BUILD_START) {
        build_bars.push(12);
    }
    if events.contains(EventFlags::DROP) {
        drop_bars.push(12);
    }

    // Bar 13 (1-based 14): the low end stays, so there is nothing left to drop.
    let (_, _, events) = push(&mut model, bar(13, -8.0, 0.9, 0.85, 1.0, 0.9));
    if events.contains(EventFlags::DROP) {
        drop_bars.push(13);
    }

    assert_eq!(
        build_bars,
        vec![11],
        "BUILD_START on the fourth rising bar, and once only"
    );
    assert_eq!(drop_bars, vec![12], "DROP when the low end returns");
    assert_eq!(level_at_drop, 4, "the drop bar is a loud bar");
}

#[test]
fn breakdown_detected() {
    let mut model = IntensityModel::new(PHRASE_BARS, HISTORY_BARS);

    let mut breakdown_bars = Vec::new();
    for i in 0..8i64 {
        let (_, level, events) = push(&mut model, terms(i, 0.8, 0.8, 1.5, 0.7));
        assert!(
            !events.contains(EventFlags::BREAKDOWN),
            "bar {i}: nothing broke down yet"
        );
        assert_eq!(level, 4, "bar {i}: 0.825 is level 4");
    }

    // Low end and onsets both fall away: a breakdown for a full bar.
    for i in 8..10i64 {
        let (_, level, events) = push(&mut model, terms(i, 0.35, 0.1, 0.0, 0.3));
        assert_eq!(level, 1, "bar {i}: the collapse is immediate");
        if events.contains(EventFlags::BREAKDOWN) {
            breakdown_bars.push(i);
        }
    }

    assert_eq!(
        breakdown_bars,
        vec![8],
        "BREAKDOWN fires once, on the first bar after level >= 3"
    );
}

#[test]
fn silence_never_raises_level() {
    let mut model = IntensityModel::new(PHRASE_BARS, HISTORY_BARS);

    // Digital silence in dB, but with feature values high enough that ignoring the
    // dB gate would read as a loud bar (0.35*0.9 + 0.3*0.9 + 0.2*1.0 + 0.15*0.9 = 0.90).
    for i in 0..12i64 {
        let (intensity, level, events) = push(&mut model, bar(i, -60.0, 0.9, 0.9, 2.0, 0.9));
        assert_eq!(intensity, 0.0, "bar {i}: silence has no intensity");
        assert_eq!(level, 1, "bar {i}: silence never raises the level");
        assert!(
            !events.intersects(
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
        let (_, level, _) = push(&mut model, bar(i, -55.0, 0.05, 0.05, 0.05, 0.05));
        assert_eq!(level, 1, "bar {i}: a noise floor is not a level");
    }
}

#[test]
fn history_resets_after_silence() {
    let mut model = IntensityModel::new(PHRASE_BARS, HISTORY_BARS);

    // A loud section first, so a stale history would still be loud afterwards.
    for i in 0..8i64 {
        let (_, level, _) = push(&mut model, bar(i, -8.0, 0.9, 0.85, 2.0, 0.9));
        assert_eq!(level, 4, "bar {i}: the loud section is level 4");
    }

    // > 2 bars of silence is a gap between songs: history and level start over.
    for i in 8..11i64 {
        push(&mut model, silent(i));
    }
    assert_eq!(model.level(), 1, "the gap reset the level");
    assert_eq!(model.intensity(), 0.0, "the gap reset the intensity");

    // The first bar of the next song is quiet: it must not arrive at a high level,
    // and its onset density must be read against a clean (floored) history.
    let (intensity, level, events) = push(&mut model, bar(11, -45.0, 0.4, 0.3, 0.5, 0.3));
    let quiet = weighted(0.4, 0.3, 0.5, 0.3);
    assert!(
        (intensity - quiet).abs() < 1e-3,
        "intensity {intensity} != {quiet}: the onset history did not restart"
    );
    assert!(level <= 2, "a quiet bar after a gap is not level {level}");
    assert!(intensity < 0.5, "a quiet bar after a gap is not loud");
    assert!(
        events.contains(EventFlags::LEVEL_UP) && !events.contains(EventFlags::LEVEL_DOWN),
        "the quiet bar lifts the reset level from 1, it does not fall from a stale one"
    );
    assert!(
        !events.intersects(EventFlags::BREAKDOWN | EventFlags::DROP),
        "a song's first quiet bar is not a breakdown or a drop"
    );
}

#[test]
fn level_changes_only_on_bars() {
    let mut model = IntensityModel::new(PHRASE_BARS, HISTORY_BARS);

    // Inside the hysteresis band: 0.28 is below 0.25 + 0.04, so it stays level 1.
    let (_, level, events) = push(&mut model, terms(0, 0.3, 0.3, 0.2, 0.3));
    assert_eq!(level, 1);
    assert!(
        events.is_empty(),
        "no event without a level change: {events:?}"
    );

    // 0.315 crosses 0.25 + 0.04: up one level, on that bar's boundary only.
    let (_, level, events) = push(&mut model, terms(1, 0.4, 0.3, 0.2, 0.3));
    assert_eq!(level, 2);
    assert!(events.contains(EventFlags::LEVEL_UP));
    assert!(!events.contains(EventFlags::LEVEL_DOWN));

    // 0.48 stays inside the band around 0.5: no change in either direction.
    for i in 2..5i64 {
        let (_, level, events) = push(&mut model, terms(i, 0.5, 0.5, 0.4, 0.5));
        assert_eq!(level, 2, "bar {i}: 0.48 does not reach 0.5 + 0.04");
        assert!(
            !events.intersects(EventFlags::LEVEL_UP | EventFlags::LEVEL_DOWN),
            "bar {i}: no level event inside the hysteresis band"
        );
    }

    // 0.2975 is still above 0.25 - 0.04: still level 2.
    let (_, level, _) = push(&mut model, terms(5, 0.35, 0.3, 0.2, 0.3));
    assert_eq!(level, 2, "the level does not fall inside the band");

    // 0.20 is below 0.25 - 0.04: back to level 1, again on a bar boundary.
    let (_, level, events) = push(&mut model, terms(6, 0.2, 0.2, 0.2, 0.2));
    assert_eq!(level, 1);
    assert!(events.contains(EventFlags::LEVEL_DOWN));
    assert!(!events.contains(EventFlags::LEVEL_UP));
}

#[test]
fn phrase_event_every_phrase_bars() {
    let mut model = IntensityModel::new(4, HISTORY_BARS);

    let mut phrase_bars = Vec::new();
    for i in 0..9i64 {
        let (_, _, events) = push(&mut model, terms(i, 0.6, 0.65, 0.5, 0.5));
        if events.contains(EventFlags::PHRASE) {
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
        let (_, _, events) = push(&mut model, terms(20 + i, 0.6, 0.65, 0.5, 0.5));
        if events.contains(EventFlags::PHRASE) {
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

    for i in 0..8i64 {
        push(&mut model, bar(i, -8.0, 0.9, 0.85, 2.0, 0.9));
    }
    assert_eq!(model.level(), 4);

    model.reset();
    assert_eq!(model.level(), 1, "reset() drops back to level 1");
    assert_eq!(model.intensity(), 0.0, "reset() forgets the last intensity");

    // A quiet first bar of the next song starts low, from a clean history.
    let (intensity, level, _) = push(&mut model, bar(8, -45.0, 0.4, 0.3, 0.5, 0.3));
    assert!(
        level <= 2,
        "after reset a quiet bar is level {level}, not 4"
    );
    assert!(intensity < 0.5);
}

#[test]
fn zero_configuration_is_clamped() {
    // The app may build the model from a config with empty values; it must not panic
    // or divide by a zero-length window.
    let mut model = IntensityModel::new(0, 0);
    for i in 0..5i64 {
        let (_, level, events) = push(&mut model, terms(i, 0.6, 0.65, 0.5, 0.5));
        assert!((1..=4).contains(&level));
        assert!(
            events.contains(EventFlags::PHRASE),
            "bar {i}: phrase_bars 0 means every bar"
        );
    }
}
