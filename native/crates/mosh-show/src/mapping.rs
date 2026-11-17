//! How a level and the rhythm turn a clip's parameters into the values a frame draws.
//!
//! The design's rule is `p[i] = lerp(range.lo, range.hi, base + pulse_depth[level] * beat_pulse)`:
//! the level picks the range, and the beat moves the value inside it.

use mosh_core::Clip;

/// The web app's beat pulse (`fPulse` in `engine.c`): `exp(-4 * beat_phase)`, 1 on the beat and
/// down to `exp(-4) = 0.018` just before the next one.
pub fn beat_pulse(beat_phase: f32) -> f32 {
    (-4.0 * beat_phase.clamp(0.0, 1.0)).exp()
}

/// The intensity level 1..=4 for a continuous intensity, used when the analysis reports none
/// (`AnalysisFrame::level == 0`). The analysis owns the level otherwise; it has the hysteresis.
pub fn level_from_intensity(intensity: f32) -> u8 {
    let i = intensity.clamp(0.0, 1.0);
    if i < 0.25 {
        1
    } else if i < 0.5 {
        2
    } else if i < 0.75 {
        3
    } else {
        4
    }
}

/// Index 0..=3 for a level 1..=4, clamped so a bad level cannot index out of `ranges`.
pub fn level_index(level: u8) -> usize {
    (level.clamp(1, 4) - 1) as usize
}

/// The eight parameter values for this clip at `level`, from positions in 0..1 and the beat pulse.
pub fn resolve_params(clip: &Clip, level: u8, pulse: f32, base: &[f32; 8]) -> [f32; 8] {
    let level_index = level_index(level);
    let depth = clip.pulse_depth[level_index];
    let mut out = [0.0f32; 8];
    for (i, slot) in out.iter_mut().enumerate() {
        let position = (base[i] + depth * pulse).clamp(0.0, 1.0);
        let range = clip.ranges[level_index][i];
        *slot = range.lo + (range.hi - range.lo) * position;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use mosh_core::{ClipParams, ParamRange};

    #[test]
    fn pulse_is_one_on_the_beat_and_decays() {
        assert_eq!(beat_pulse(0.0), 1.0);
        assert!((beat_pulse(0.5) - (-2.0f32).exp()).abs() < 1e-6);
        assert!(beat_pulse(0.99) < 0.02);
        // Out-of-range phases are clamped rather than amplified.
        assert_eq!(beat_pulse(-1.0), 1.0);
        assert_eq!(beat_pulse(2.0), beat_pulse(1.0));
    }

    #[test]
    fn levels_follow_intensity_thresholds() {
        assert_eq!(level_from_intensity(0.0), 1);
        assert_eq!(level_from_intensity(0.24), 1);
        assert_eq!(level_from_intensity(0.25), 2);
        assert_eq!(level_from_intensity(0.6), 3);
        assert_eq!(level_from_intensity(0.75), 4);
        assert_eq!(level_from_intensity(4.0), 4);
        assert_eq!(level_from_intensity(-1.0), 1);
    }

    #[test]
    fn level_index_never_leaves_the_table() {
        assert_eq!(level_index(0), 0);
        assert_eq!(level_index(1), 0);
        assert_eq!(level_index(4), 3);
        assert_eq!(level_index(9), 3);
    }

    #[test]
    fn ranges_and_pulse_combine() {
        let mut ranges = [[ParamRange::default(); 8]; 4];
        ranges[3][0] = ParamRange { lo: 10.0, hi: 20.0 };
        let clip = Clip {
            id: 1,
            name: "c".to_string(),
            visualizer: "tunnel".to_string(),
            params: ClipParams { p: [0.5; 8], palette: 0, seed: 0 },
            ranges,
            pulse_depth: [0.25; 4],
            effects: Vec::new(),
            transition_in: mosh_core::Transition::default(),
            follow: mosh_core::FollowAction::None,
            follow_after_bars: 1,
            react_to_events: false,
            video: None,
        };
        let base = [0.5f32; 8];
        let flat = resolve_params(&clip, 4, 0.0, &base);
        assert!((flat[0] - 15.0).abs() < 1e-6);
        let pulsed = resolve_params(&clip, 4, 1.0, &base);
        assert!((pulsed[0] - 17.5).abs() < 1e-6);
    }
}
