//! The analyzer feeds its own intensity model: frames carry level, intensity and its events.

use mosh_analysis::{AnalysisConfig, Analyzer};
use mosh_core::{AudioBlock, EventFlags, Transport};

const RATE: u32 = 48_000;
const BLOCK: usize = 256;
const BPM: f64 = 120.0;

/// Plays `bars` bars of 4/4 at 120 BPM starting at `start_bar`; `loud` adds full-range noise and a
/// 50 Hz kick on every beat. Returns the highest level seen and every event seen.
fn play(
    a: &mut Analyzer,
    start_bar: u32,
    bars: u32,
    loud: bool,
    seed: &mut u32,
) -> (u8, EventFlags) {
    let samples_per_beat = RATE as f64 * 60.0 / BPM;
    let total = (bars as f64 * 4.0 * samples_per_beat) as usize;
    let first = (start_bar as f64 * 4.0 * samples_per_beat) as i64;
    let (mut max_level, mut seen) = (0u8, EventFlags::empty());
    let mut n = 0usize;
    while n < total {
        let pos = first + n as i64;
        let ppq = pos as f64 / samples_per_beat;
        let mut samples = Vec::with_capacity(BLOCK * 2);
        for i in 0..BLOCK {
            *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (*seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
            let t = (pos + i as i64) as f64;
            let beat_frac = (t / samples_per_beat).fract();
            let v = if loud {
                let kick = if beat_frac < 0.15 {
                    (t / RATE as f64 * 50.0 * std::f64::consts::TAU).sin() as f32 * 0.9
                } else {
                    0.0
                };
                noise * 0.3 + kick
            } else {
                noise * 0.003
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
        let f = a.take_frame();
        max_level = max_level.max(f.level);
        seen |= f.events;
        n += BLOCK;
    }
    (max_level, seen)
}

#[test]
fn frames_carry_the_intensity_model() {
    let mut a = Analyzer::new(AnalysisConfig::default());
    let mut seed = 7u32;
    let (quiet_level, _) = play(&mut a, 0, 16, false, &mut seed);
    let (loud_level, events) = play(&mut a, 16, 16, true, &mut seed);
    assert!(
        quiet_level <= 2,
        "quiet section reached level {quiet_level}"
    );
    assert!(
        loud_level >= 3,
        "loud section only reached level {loud_level}"
    );
    assert!(
        events.contains(EventFlags::LEVEL_UP),
        "no LEVEL_UP in {events:?}"
    );
}

#[test]
fn song_start_resets_level_and_flags_the_frame() {
    let mut a = Analyzer::new(AnalysisConfig::default());
    let mut seed = 11u32;
    play(&mut a, 0, 16, true, &mut seed);
    a.song_start();
    let f = a.take_frame();
    assert!(f.events.contains(EventFlags::SONG_START));
    assert_eq!(f.level, 1);
    assert_eq!(f.intensity, 0.0);
}
