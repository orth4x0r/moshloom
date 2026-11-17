//! The effects lane: Fold, Mirror, Streak, Crush, Zoom, Spin, Echo, Negative, applied in ONE
//! pass (Mosh is not here; see `mosh`). Owner: V-fx (Task 2.3). Signatures are frozen.
//!
//! Reference: `gpu.js` (read-only) spreads these eight effects over three passes — `WG_PRE`
//! (mirror, fold, crush, streak), `WG_FB` (zoom, spin, echo, as transforms of a per-layer feedback
//! buffer) and `WG_OUT` (negative) — because the web app keeps that buffer next to its scene. The
//! native chain draws a layer as scene -> mosh -> effects -> transition with one target per stage
//! (`lib.rs`), so the whole lane is the single fragment pass in
//! `mosh-shaders/wgsl/fx_effects.wgsl`.
//!
//! Three deliberate differences, all consequences of "one pass":
//!
//! - **Zoom and Spin are geometric**, a transform of the sample point about the centre, instead of
//!   the web app's feedback zoom/spin (which reads its buffer displaced and accumulates). The
//!   picture is sampled once, so there is no infinite-zoom trail.
//! - **Echo reads the effects target's own previous frame** — the `dst` ping-pong pair, which is
//!   per layer — offset 1.2% of the width to the left. The web app blends its already-displaced
//!   feedback buffer at the same `uv`, which on its own would change nothing.
//! - **No vignette, grain or fade**: those live in the web app's output pass (`WG_OUT`), which is
//!   the compositor's job here, not the layer's.
//!
//! Amounts are clamped to 0..1 and 0 is the exact identity for every effect, so an inactive slot
//! cannot move a pixel. The pass is a pure function of (`src`, previous `dst`); `frame` and `t`
//! are accepted for the frozen signature and deliberately unused.

use mosh_core::{AnalysisFrame, EffectKind, EffectSlot};
use mosh_render::{FrameRecorder, Gpu, LayerTarget, Pipeline, RenderError};

/// Bytes of the effect pass's uniform block, which is the size of
/// `struct EU` in `fx_effects.wgsl`.
const EFFECT_UNIFORM_BYTES: usize = 80;

/// One draw of every non-Mosh slot, in the order the layer's effect list names them.
pub struct EffectChain {
    pipeline: Pipeline,
}

impl EffectChain {
    pub fn new(gpu: &Gpu) -> Result<EffectChain, RenderError> {
        let spirv = mosh_shaders::spirv("fx_effects").ok_or(RenderError::EmptyShader)?;
        let pipeline =
            Pipeline::fullscreen(gpu, spirv, EFFECT_UNIFORM_BYTES, 2)?.with_name("fx_effects");
        Ok(EffectChain { pipeline })
    }

    /// Apply every non-Mosh slot with amount > 0 to `src`, writing `dst`. Returns `Ok(false)`
    /// (and writes nothing) when there is nothing to apply.
    pub fn apply(
        &mut self,
        rec: &mut FrameRecorder<'_>,
        src: &LayerTarget,
        dst: &LayerTarget,
        effects: &[EffectSlot],
        _frame: &AnalysisFrame,
        _t: f32,
    ) -> Result<bool, RenderError> {
        let uniforms = EffectUniforms::build([src.width() as f32, src.height() as f32], effects);
        if !uniforms.any() {
            return Ok(false);
        }
        // `dst` is also the echo history: its front image is last frame's effects output.
        rec.draw(&self.pipeline, dst, &uniforms.as_bytes(), &[src, dst])?;
        Ok(true)
    }
}

/// The pass's uniform block. Field order and padding must match `struct EU` in
/// `fx_effects.wgsl`; the unit test below asserts the size against the reflected shader struct.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct EffectUniforms {
    res: [f32; 2],
    echo_decay: f32,
    pad0: f32,
    /// fold, mirror, zoom, spin
    fxa: [f32; 4],
    /// streak, crush, echo, negative
    fxb: [f32; 4],
    /// fold segments, mirror y, zoom out, spin back
    fpa: [f32; 4],
    /// streak bands, crush size, echo decay, invert tint
    fpb: [f32; 4],
}

impl EffectUniforms {
    /// One slot per kind: the amounts of duplicate kinds are merged by `max`, later slots win the
    /// parameter. Mosh slots belong to `mosh`, and are ignored here.
    fn build(res: [f32; 2], effects: &[EffectSlot]) -> EffectUniforms {
        let mut uniforms = EffectUniforms {
            res,
            ..EffectUniforms::default()
        };
        for slot in effects {
            let amount = slot.amount.clamp(0.0, 1.0);
            if amount <= 0.0 {
                continue;
            }
            let param = slot.param.clamp(0.0, 1.0);
            let (amount_slot, param_slot) = match slot.kind {
                EffectKind::Fold => (&mut uniforms.fxa[0], &mut uniforms.fpa[0]),
                EffectKind::Mirror => (&mut uniforms.fxa[1], &mut uniforms.fpa[1]),
                EffectKind::Zoom => (&mut uniforms.fxa[2], &mut uniforms.fpa[2]),
                EffectKind::Spin => (&mut uniforms.fxa[3], &mut uniforms.fpa[3]),
                EffectKind::Streak => (&mut uniforms.fxb[0], &mut uniforms.fpb[0]),
                EffectKind::Crush => (&mut uniforms.fxb[1], &mut uniforms.fpb[1]),
                EffectKind::Echo => {
                    uniforms.echo_decay = 0.72 + 0.22 * param;
                    (&mut uniforms.fxb[2], &mut uniforms.fpb[2])
                }
                EffectKind::Negative => (&mut uniforms.fxb[3], &mut uniforms.fpb[3]),
                EffectKind::Mosh => continue,
            };
            if amount >= *amount_slot {
                *amount_slot = amount;
                *param_slot = param;
            }
        }
        uniforms
    }

    /// True when at least one effect is active.
    fn any(&self) -> bool {
        self.fxa.iter().chain(&self.fxb).any(|value| *value > 0.0)
    }

    fn as_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(EFFECT_UNIFORM_BYTES);
        let fields = [self.res[0], self.res[1], self.echo_decay, self.pad0]
            .into_iter()
            .chain(self.fxa)
            .chain(self.fxb)
            .chain(self.fpa)
            .chain(self.fpb);
        for value in fields {
            out.extend_from_slice(&value.to_le_bytes());
        }
        debug_assert_eq!(out.len(), EFFECT_UNIFORM_BYTES);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    fn slot(kind: EffectKind, amount: f32, param: f32) -> EffectSlot {
        EffectSlot {
            kind,
            amount,
            param,
        }
    }

    #[test]
    fn uniform_block_matches_the_shader() {
        assert_eq!(size_of::<EffectUniforms>(), EFFECT_UNIFORM_BYTES);
        assert_eq!(
            mosh_shaders::uniform_bytes("fx_effects"),
            Some(EFFECT_UNIFORM_BYTES as u32),
            "fx_effects.wgsl struct EU and EffectUniforms disagree"
        );
        let empty = EffectUniforms::build([320.0, 180.0], &[]);
        assert_eq!(empty.as_bytes().len(), EFFECT_UNIFORM_BYTES);
        // The first four floats are res, echo_decay, pad0.
        let bytes = empty.as_bytes();
        let first = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(first, 320.0);
    }

    #[test]
    fn only_active_slots_reach_the_uniform() {
        let zero = EffectUniforms::build(
            [64.0, 36.0],
            &[
                slot(EffectKind::Fold, 0.0, 0.5),
                slot(EffectKind::Mosh, 1.0, 0.5),
            ],
        );
        assert!(
            !zero.any(),
            "a zero Fold and a Mosh slot are nothing to apply"
        );

        let one = EffectUniforms::build(
            [64.0, 36.0],
            &[
                slot(EffectKind::Fold, 0.5, 0.3),
                slot(EffectKind::Fold, 0.75, 0.6),
            ],
        );
        assert_eq!(one.fxa[0], 0.75, "duplicate kinds merge by max");
        assert_eq!(one.fpa[0], 0.6, "the winning amount carries its parameter");
        assert!(one.any());

        // Every kind lands in its own slot and nothing else moves.
        for (kind, index) in [
            (EffectKind::Fold, (0, 0)),
            (EffectKind::Mirror, (0, 1)),
            (EffectKind::Zoom, (0, 2)),
            (EffectKind::Spin, (0, 3)),
            (EffectKind::Streak, (1, 0)),
            (EffectKind::Crush, (1, 1)),
            (EffectKind::Echo, (1, 2)),
            (EffectKind::Negative, (1, 3)),
        ] {
            let uniforms = EffectUniforms::build([8.0, 8.0], &[slot(kind, 0.4, 0.9)]);
            let amounts = if index.0 == 0 {
                uniforms.fxa
            } else {
                uniforms.fxb
            };
            let params = if index.0 == 0 {
                uniforms.fpa
            } else {
                uniforms.fpb
            };
            assert_eq!(amounts[index.1], 0.4, "{kind:?} amount");
            assert_eq!(params[index.1], 0.9, "{kind:?} param");
            let others: usize = amounts
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != index.1)
                .map(|(_, v)| (*v > 0.0) as usize)
                .sum::<usize>()
                + (if index.0 == 0 {
                    uniforms.fxb
                } else {
                    uniforms.fxa
                })
                .iter()
                .map(|v| (*v > 0.0) as usize)
                .sum::<usize>();
            assert_eq!(others, 0, "{kind:?} moved another slot");
        }
        // Echo's decay is the web app's 0.72 + 0.22 * param.
        let echo = EffectUniforms::build([8.0, 8.0], &[slot(EffectKind::Echo, 1.0, 0.5)]);
        assert!((echo.echo_decay - 0.83).abs() < 1e-6, "{}", echo.echo_decay);
    }
}
