//! Seeded parameter draws for `FollowAction::RandomProcedural`.
//!
//! A procedural launch draws a fresh `u64` from the show's `Pcg64`, records it in
//! `LayerState::seed`, and derives every parameter from that seed alone — so the look on screen
//! can be recomputed, saved as a preset, and a reviewer can replay a set exactly.

use mosh_core::{Clip, ClipParams};
use rand_pcg::Pcg64;
use rand_pcg::rand_core::{Rng, SeedableRng};

use crate::mapping;

/// The seed a `Show` starts from when the caller does not choose one.
pub const DEFAULT_SEED: u64 = 0x5EED_0000_0001;

/// The positions (0..1 in the clip's ranges) that `seed` draws for `clip` at `level`.
///
/// A uniform draw from `[lo, hi]` is a uniform position in `0..1`, so the draw is the same, but it
/// is expressed through the range the rule names (a degenerate range draws its own value).
pub fn positions(clip: &Clip, level: u8, seed: u64) -> [f32; 8] {
    let mut rng = Pcg64::seed_from_u64(seed);
    let index = mapping::level_index(level);
    let mut out = [0.0f32; 8];
    for (i, slot) in out.iter_mut().enumerate() {
        let range = clip.ranges[index][i];
        let drawn = range.lo + (range.hi - range.lo) * unit(&mut rng);
        if range.hi > range.lo {
            *slot = (drawn - range.lo) / (range.hi - range.lo);
        } else {
            *slot = 0.0;
        }
    }
    out
}

/// The parameters `seed` produces, ready to save as a preset (`p` resolved at pulse 0).
pub fn params(clip: &Clip, level: u8, seed: u64) -> ClipParams {
    let base = positions(clip, level, seed);
    ClipParams {
        p: mapping::resolve_params(clip, level, 0.0, &base),
        palette: clip.params.palette,
        seed,
    }
}

/// Uniform in `[0, 1)` from the top 24 bits of a draw: every step lands on an f32.
fn unit(rng: &mut Pcg64) -> f32 {
    ((rng.next_u64() >> 40) as f32) / (1u32 << 24) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use mosh_core::{FollowAction, ParamRange, Transition};

    fn clip() -> Clip {
        let mut ranges = [[ParamRange::default(); 8]; 4];
        ranges[1][0] = ParamRange { lo: 10.0, hi: 30.0 };
        Clip {
            id: 1,
            name: "c".to_string(),
            visualizer: "tunnel".to_string(),
            params: ClipParams { p: [0.5; 8], palette: 2, seed: 0 },
            ranges,
            pulse_depth: [0.5; 4],
            effects: Vec::new(),
            transition_in: Transition::default(),
            follow: FollowAction::RandomProcedural,
            follow_after_bars: 4,
            react_to_events: false,
            video: None,
        }
    }

    #[test]
    fn the_same_seed_draws_the_same_positions() {
        let c = clip();
        let a = positions(&c, 2, 12345);
        let b = positions(&c, 2, 12345);
        assert_eq!(a, b);
        assert_ne!(a, positions(&c, 2, 12346));
    }

    #[test]
    fn positions_stay_in_range() {
        let c = clip();
        for seed in 0..64u64 {
            for p in positions(&c, 2, seed) {
                assert!((0.0..=1.0).contains(&p));
            }
        }
    }

    #[test]
    fn params_are_inside_the_level_range() {
        let c = clip();
        for seed in 0..64u64 {
            let p = params(&c, 2, seed);
            assert!((10.0..=30.0).contains(&p.p[0]), "p0 was {}", p.p[0]);
            assert_eq!(p.seed, seed);
            assert_eq!(p.palette, 2);
        }
    }
}
