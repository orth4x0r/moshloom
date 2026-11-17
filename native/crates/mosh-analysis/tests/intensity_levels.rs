//! The intensity levels as a real (synthetic) track drives them: the acceptance properties from
//! the tuning brief, measured through `Analyzer`, not through hand-built `BarStats`.
//!
//! The audio is generated here (no fixtures): band-limited noise plus a 50 Hz kick every beat at
//! 120 BPM, 4/4. "Loud" is a full-scale mix, "quiet" is the same shape 30 dB down and without the
//! kick, so a section changes level and timbre together the way a real arrangement does.

use mosh_analysis::{AnalysisConfig, Analyzer};
use mosh_core::{AudioBlock, EventFlags, Transport};

const RATE: u32 = 48_000;
const BLOCK: usize = 256;
const BPM: f64 = 120.0;

/// What one bar of the run produced.
#[derive(Clone, Copy, Debug)]
struct Bar {
    level: u8,
    loudness_db: f32,
}

/// Which section to play.
#[derive(Clone, Copy, PartialEq)]
enum Section {
    Loud,
    Quiet,
}

/// Play `bars` bars starting at `start_bar`, appending one entry per closed bar.
fn play(
    a: &mut Analyzer,
    start_bar: u32,
    bars: u32,
    section: Section,
    seed: &mut u32,
    out: &mut Vec<Bar>,
) {
    let samples_per_beat = RATE as f64 * 60.0 / BPM;
    let total = (bars as f64 * 4.0 * samples_per_beat) as usize;
    let first = (start_bar as f64 * 4.0 * samples_per_beat) as i64;
    let mut n = 0usize;
    while n < total {
        let pos = first + n as i64;
        let ppq = pos as f64 / samples_per_beat;
        let mut samples = Vec::with_capacity(BLOCK * 2);
        for i in 0..BLOCK {
            *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (*seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
            let t = (pos + i as i64) as f32;
            let beat_frac = (t / samples_per_beat as f32).fract();
            let v = match section {
                Section::Loud => {
                    let kick = if beat_frac < 0.15 {
                        (t / RATE as f32 * 50.0 * std::f32::consts::TAU).sin() * 0.9
                    } else {
                        0.0
                    };
                    noise * 0.3 + kick
                }
                Section::Quiet => noise * 0.05,
            };
            samples.push(v);
            samples.push(v);
        }
        let bar_start = (ppq / 4.0).floor() * 4.0;
        a.push(&AudioBlock {
            transport: Some(Transport {
                tempo_bpm: BPM,
                ppq,
                bar_start_ppq: bar_start,
                sig_num: 4,
                sig_den: 4,
                playing: true,
                sample_pos: pos,
            }),
            sample_rate: RATE,
            channels: 2,
            samples,
        });
        let frame = a.take_frame();
        for bar in a.drain_bars() {
            out.push(Bar {
                // The analyzer closes the bar and updates its level inside `push`, so the frame
                // taken right after it carries the level of the bar that has just closed.
                level: frame.level,
                loudness_db: bar.loudness_db,
            });
        }
        n += BLOCK;
    }
}

#[test]
fn a_steady_loud_loop_settles_below_level_four() {
    // The reported bug: a loop that simply stays loud must not pin the model at 4, or there is
    // nothing left to say at the drop. Once the window has filled, a loop with no dynamics reads
    // the middle of the scale.
    let mut a = Analyzer::new(AnalysisConfig::default());
    let mut seed = 7u32;
    let mut bars = Vec::new();
    play(&mut a, 0, 32, Section::Loud, &mut seed, &mut bars);
    assert!(bars.len() >= 30, "32 bars of audio gave {}", bars.len());

    let settled: Vec<u8> = bars[8..].iter().map(|b| b.level).collect();
    assert!(
        settled.iter().all(|&level| (2..=3).contains(&level)),
        "a steady loud loop settled at {settled:?}"
    );
    assert!(
        bars.iter().all(|b| b.level != 4),
        "a loop with no dynamics reached level 4: {:?}",
        bars.iter().map(|b| b.level).collect::<Vec<_>>()
    );
}

#[test]
fn a_quiet_section_then_a_drop_ends_at_level_three_or_four() {
    let mut a = Analyzer::new(AnalysisConfig::default());
    let mut seed = 11u32;
    let mut bars = Vec::new();
    play(&mut a, 0, 16, Section::Quiet, &mut seed, &mut bars);
    play(&mut a, 16, 16, Section::Loud, &mut seed, &mut bars);
    assert!(bars.len() >= 30, "32 bars of audio gave {}", bars.len());

    // The quiet section is a *sustained* level, and the model reads the level relative to the
    // recent average: a section that simply stays quiet settles in the middle rather than at 1.
    // What the level must show is the drop itself.
    let quiet: Vec<u8> = bars[..12].iter().map(|b| b.level).collect();
    assert!(
        quiet.iter().all(|&level| level <= 3),
        "a quiet section reached {quiet:?}"
    );

    let tail: Vec<u8> = bars[bars.len() - 4..].iter().map(|b| b.level).collect();
    assert!(
        tail.iter().all(|&level| level >= 3),
        "the drop ended at {tail:?}"
    );
    assert!(
        bars.iter().any(|b| b.level == 4),
        "the drop never reached level 4: {:?}",
        bars.iter().map(|b| b.level).collect::<Vec<_>>()
    );
}

/// A short arrangement: quiet, loud, quiet, loud, each eight bars.
fn arrangement() -> Vec<Bar> {
    let mut a = Analyzer::new(AnalysisConfig::default());
    let mut seed = 21u32;
    let mut bars = Vec::new();
    for (index, section) in [
        Section::Quiet,
        Section::Loud,
        Section::Quiet,
        Section::Loud,
    ]
    .into_iter()
    .enumerate()
    {
        play(
            &mut a,
            (index * 8) as u32,
            8,
            section,
            &mut seed,
            &mut bars,
        );
    }
    bars
}

#[test]
fn the_level_moves_with_the_music() {
    let bars = arrangement();
    assert!(bars.len() >= 30, "32 bars of audio gave {}", bars.len());

    // (a) at least three of the four levels are used.
    let mut used: Vec<u8> = bars.iter().map(|b| b.level).collect();
    used.sort_unstable();
    used.dedup();
    assert!(
        used.len() >= 3,
        "only {} levels used: {used:?}",
        used.len()
    );

    // (b) by absolute loudness: the quietest quarter is levels 1-2, the loudest 3-4.
    let mut sorted = bars.to_vec();
    sorted.sort_by(|a, b| a.loudness_db.total_cmp(&b.loudness_db));
    let quarter = sorted.len() / 4;
    let quiet = &sorted[..quarter];
    let loud = &sorted[sorted.len() - quarter..];
    let quiet_low = quiet.iter().filter(|b| b.level <= 2).count() as f32 / quarter as f32;
    let loud_high = loud.iter().filter(|b| b.level >= 3).count() as f32 / quarter as f32;
    assert!(quiet_low >= 0.7, "quietest quarter is {quiet_low} levels 1-2");
    assert!(loud_high >= 0.7, "loudest quarter is {loud_high} levels 3-4");
}

#[test]
fn the_level_never_moves_more_than_one_step_per_bar() {
    let bars = arrangement();
    for (index, pair) in bars.windows(2).enumerate() {
        let jump = pair[1].level.abs_diff(pair[0].level);
        assert!(
            jump <= 1,
            "bar {index}: level {} -> {} in one bar",
            pair[0].level,
            pair[1].level
        );
    }
}

#[test]
fn level_changes_carry_an_event() {
    // A frame-level consumer (the show, the stats overlay) can rely on LEVEL_UP / LEVEL_DOWN
    // matching the change it sees in `frame.level`, so nothing has to diff levels itself.
    let mut a = Analyzer::new(AnalysisConfig::default());
    let mut seed = 31u32;
    let mut previous = a.take_frame().level;
    let mut changes = 0usize;
    for (index, section) in [Section::Quiet, Section::Loud, Section::Quiet, Section::Loud]
        .into_iter()
        .enumerate()
    {
        let samples_per_beat = RATE as f64 * 60.0 / BPM;
        let total = (8.0 * 4.0 * samples_per_beat) as usize;
        let first = (index as f64 * 8.0 * 4.0 * samples_per_beat) as i64;
        let mut n = 0usize;
        while n < total {
            let pos = first + n as i64;
            let ppq = pos as f64 / samples_per_beat;
            let mut samples = Vec::with_capacity(BLOCK * 2);
            for _ in 0..BLOCK {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
                let v = match section {
                    Section::Loud => noise * 0.3,
                    Section::Quiet => noise * 0.05,
                };
                samples.push(v);
                samples.push(v);
            }
            a.push(&AudioBlock {
                transport: Some(Transport {
                    tempo_bpm: BPM,
                    ppq,
                    bar_start_ppq: (ppq / 4.0).floor() * 4.0,
                    sig_num: 4,
                    sig_den: 4,
                    playing: true,
                    sample_pos: pos,
                }),
                sample_rate: RATE,
                channels: 2,
                samples,
            });
            let frame = a.take_frame();
            if frame.level != previous {
                changes += 1;
                let up = frame.events.contains(EventFlags::LEVEL_UP);
                let down = frame.events.contains(EventFlags::LEVEL_DOWN);
                assert!(
                    up || down,
                    "level moved {previous} -> {} with no LEVEL_UP/LEVEL_DOWN",
                    frame.level
                );
                assert_eq!(
                    frame.level.abs_diff(previous),
                    1,
                    "level moved {previous} -> {} in one frame",
                    frame.level
                );
                assert_eq!(
                    frame.level > previous,
                    up,
                    "the event does not match the direction"
                );
                previous = frame.level;
            }
            n += BLOCK;
            let _ = a.drain_bars();
        }
    }
    assert!(changes > 0, "the arrangement never changed level");
}
