//! Per-layer transitions: Cut, Crossfade, Datamosh, Melt, Strobe. Every full-frame flash goes
//! through the shared `FlashLimiter` (400 ms, no opt-out). Owner: V-fx (Task 2.3).
//! Signatures are frozen.
//!
//! `Visuals::draw_layer` calls [`TransitionPass::begin`] on the frame a layer's clip changes, with
//! last frame's picture, and [`TransitionPass::apply`] every frame the layer has an
//! `ActiveTransition`. `begin` keeps its own copy of that picture (a draw into a `LayerTarget`
//! this pass owns, one per layer), because `from` is only guaranteed to be readable for the frame
//! that hands it over.
//!
//! The blend itself is one fragment pass, `mosh-shaders/wgsl/tr_mix.wgsl`: from, to and a mode.
//! Cut is a second pass, `tr_copy.wgsl`, so a cut is exact rather than a blend at progress 0.
//!
//! Reference: `gpu.js` (read-only) crossfades with `xf` in `WG_MIX` and `engine.c`'s `decide()`
//! (read-only) drives melt (freeze the old picture) and strobe (invert, then a tint frame).
//!
//! **Datamosh is not a blend and never reaches `tr_mix`**: [`TransitionPass::apply`] returns
//! without drawing for it. The web app's datamosh cut is the *feedback buffer* carried across the
//! cut - `decide()` sets `skipres` for the entering frame and no I-frame, and the new clip's block
//! motion pushes the old picture (`WG_FB`). That is `MoshPass`, driven by `Visuals::draw_layer`
//! (`MoshPass::carry_over` plus a forced mosh amount for the transition's duration), which is
//! exactly where it belongs: one implementation, not a second one here. The old luminance-gradient
//! displacement this pass used to draw is gone; it was a crossfade with a wobble.
//!
//! One approximation is documented where it is made: **Melt** sags the old picture by a per-column
//! drip while the new one fades in, instead of the web app's hard freeze of the whole buffer.

//!
//! ## The flash path
//!
//! Strobe is the only transition that produces a full-frame flash, and it cannot produce one
//! without `FlashLimiter::allow`: the limiter is passed in from `Visuals::flash`, shared by every
//! layer and every other flash source, and it is the only thing that decides whether a flash
//! starts. A requested flash that the limiter refuses draws no flash at all — the transition
//! simply keeps crossfading. There is deliberately no parameter that raises the floor or bypasses
//! the limiter.

use std::time::{Duration, Instant};

use mosh_core::{ActiveTransition, AnalysisFrame, TransitionKind};
use mosh_render::{FlashLimiter, FrameRecorder, Gpu, LayerTarget, Pipeline, RenderError};

/// Bytes of the blend pass's uniform block, which is the size of `struct TU` in `tr_mix.wgsl`.
const TRANSITION_UNIFORM_BYTES: usize = 32;

/// Per-layer state this pass keeps. A layer index at or above this is drawn without a transition
/// rather than growing a vector without bound: the show has a handful of layers, and a bogus
/// index must not allocate a texture per frame.
const MAX_LAYERS: usize = 32;

/// How often a strobe asks the limiter for a flash: 16 flashes per second, the rate the task and
/// the design's flash rule are written against. The limiter still has the last word.
const STROBE_PERIOD: Duration = Duration::from_micros(62_500);

/// Per-layer transition state: the saved "from" picture, and the strobe's progress.
pub struct TransitionPass {
    copy: Pipeline,
    blend: Pipeline,
    /// One saved picture per layer, sized to that layer's targets.
    saved: Vec<Option<SavedLayer>>,
    /// One strobe cycle per layer.
    strobe: Vec<StrobeRun>,
}

struct SavedLayer {
    target: LayerTarget,
}

/// Where one layer's strobe is: when it last asked, and how far into an allowed flash it is.
#[derive(Clone, Copy, Debug, Default)]
struct StrobeRun {
    last_request: Option<Instant>,
    /// 0: not flashing. 1: the inverted frame. 2: the tint frame that belongs to it.
    age: u8,
}

impl TransitionPass {
    pub fn new(gpu: &Gpu) -> Result<TransitionPass, RenderError> {
        let copy_shader = mosh_shaders::spirv("tr_copy").ok_or(RenderError::EmptyShader)?;
        let blend_shader = mosh_shaders::spirv("tr_mix").ok_or(RenderError::EmptyShader)?;
        let copy = Pipeline::fullscreen(gpu, copy_shader, 0, 1)?.with_name("tr_copy");
        let blend = Pipeline::fullscreen(gpu, blend_shader, TRANSITION_UNIFORM_BYTES, 2)?
            .with_name("tr_mix");
        Ok(TransitionPass {
            copy,
            blend,
            saved: Vec::new(),
            strobe: Vec::new(),
        })
    }

    /// A clip change on `layer`: `from` holds last frame's picture (sample it as an input this
    /// frame; keep your own copy if you need it after this frame).
    ///
    /// The copy is taken for every kind, including Cut, because the kind is not known here (the
    /// frozen signature has none): one extra full-frame pass per clip change is cheaper than a
    /// second code path, and a cut ignores the copy.
    pub fn begin(
        &mut self,
        rec: &mut FrameRecorder<'_>,
        layer: usize,
        from: &LayerTarget,
    ) -> Result<(), RenderError> {
        if layer >= MAX_LAYERS {
            return Ok(());
        }
        if self.saved.len() <= layer {
            self.saved.resize_with(layer + 1, || None);
        }
        let stale = match &self.saved[layer] {
            None => true,
            Some(saved) => {
                saved.target.width() != from.width() || saved.target.height() != from.height()
            }
        };
        if stale {
            self.saved[layer] = Some(SavedLayer {
                target: LayerTarget::new_single(rec.gpu(), from.width(), from.height()),
            });
        }
        let saved = self.saved[layer].as_ref().expect("saved above");
        rec.draw(&self.copy, &saved.target, &[], &[from])?;
        Ok(())
    }

    /// Blend from the saved picture to `src` by `transition.progress`, writing `dst`.
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &mut self,
        rec: &mut FrameRecorder<'_>,
        layer: usize,
        src: &LayerTarget,
        dst: &LayerTarget,
        transition: &ActiveTransition,
        _frame: &AnalysisFrame,
        flash: &mut FlashLimiter,
    ) -> Result<bool, RenderError> {
        // A cut is not a blend: the new picture replaces the old one on this frame.
        if transition.kind == TransitionKind::Cut {
            rec.draw(&self.copy, dst, &[], &[src])?;
            return Ok(true);
        }
        // Datamosh is not a blend either: the layer's mosh feedback buffer was carried across the
        // cut by `Visuals::draw_layer`, so `src` already *is* the datamosh. Drawing here would put
        // a second, unrelated picture over it (see the module docs).
        if transition.kind == TransitionKind::Datamosh {
            return Ok(false);
        }
        // Without a saved picture (no `begin`, or a layer index past MAX_LAYERS) there is nothing
        // to blend from, and the layer keeps showing this frame's picture.
        if !self.saved.get(layer).is_some_and(Option::is_some) {
            return Ok(false);
        }
        let progress = transition.progress.clamp(0.0, 1.0);
        if progress >= 1.0 {
            // Finished: the layer shows `src` itself, so there is nothing left to draw.
            return Ok(false);
        }

        let flash_mode = if transition.kind == TransitionKind::Strobe {
            self.strobe_flash(layer, flash)
        } else {
            0.0
        };
        let mode = match transition.kind {
            TransitionKind::Crossfade | TransitionKind::Strobe => 0.0,
            TransitionKind::Melt => 2.0,
            TransitionKind::Cut => unreachable!("cut returned above"),
            TransitionKind::Datamosh => unreachable!("datamosh returned above"),
        };
        let uniforms = TransitionUniforms {
            res: [src.width() as f32, src.height() as f32],
            progress,
            mode,
            flash: flash_mode,
            param: 0.0,
            pad0: 0.0,
            pad1: 0.0,
        };
        let saved = self.saved[layer].as_ref().expect("checked above");
        rec.draw(
            &self.blend,
            dst,
            &uniforms.as_bytes(),
            &[&saved.target, src],
        )?;
        Ok(true)
    }

    /// Advances one layer's strobe by a frame and returns the flash mode for it: 0 for none, 1 to
    /// invert the whole frame, 2 for the red tint frame that belongs to the flash just started.
    ///
    /// The strobe *asks* at 16 Hz; only [`FlashLimiter::allow`] decides whether a flash starts, and
    /// its 400 ms floor is what limits this to at most three flashes in any second.
    fn strobe_flash(&mut self, layer: usize, flash: &mut FlashLimiter) -> f32 {
        if self.strobe.len() <= layer {
            self.strobe.resize_with(layer + 1, StrobeRun::default);
        }
        let run = &mut self.strobe[layer];
        let now = Instant::now();
        // A flash is the inverted frame plus the tint frame after it (`decide()` in engine.c).
        if run.age > 0 {
            run.age += 1;
            if run.age > 2 {
                run.age = 0;
            }
        }
        let due = match run.last_request {
            None => true,
            Some(last) => now.saturating_duration_since(last) >= STROBE_PERIOD,
        };
        if due && run.age == 0 {
            run.last_request = Some(now);
            if flash.allow(now) {
                run.age = 1;
            }
        }
        match run.age {
            1 => 1.0,
            2 => 2.0,
            _ => 0.0,
        }
    }
}

/// The blend pass's uniform block. Field order must match `struct TU` in `tr_mix.wgsl`; the unit
/// test below asserts the size against the reflected shader struct.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct TransitionUniforms {
    res: [f32; 2],
    /// 0..1, smoothstepped by the shader.
    progress: f32,
    /// 0 crossfade, 2 melt. 1 is retired: Datamosh is the mosh pass's carry-over and never reaches
    /// this shader.
    mode: f32,
    /// 0 none, 1 invert, 2 tint.
    flash: f32,
    /// Reserved for a per-kind parameter (the melt drip).
    param: f32,
    pad0: f32,
    pad1: f32,
}

impl TransitionUniforms {
    fn as_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(TRANSITION_UNIFORM_BYTES);
        let fields = [
            self.res[0],
            self.res[1],
            self.progress,
            self.mode,
            self.flash,
            self.param,
            self.pad0,
            self.pad1,
        ];
        for value in fields {
            out.extend_from_slice(&value.to_le_bytes());
        }
        debug_assert_eq!(out.len(), TRANSITION_UNIFORM_BYTES);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn uniform_block_matches_the_shader() {
        assert_eq!(size_of::<TransitionUniforms>(), TRANSITION_UNIFORM_BYTES);
        assert_eq!(
            mosh_shaders::uniform_bytes("tr_mix"),
            Some(TRANSITION_UNIFORM_BYTES as u32),
            "tr_mix.wgsl struct TU and TransitionUniforms disagree"
        );
        assert_eq!(TransitionUniforms::default().as_bytes().len(), 32);
        // The copy pass declares no uniform at all.
        assert_eq!(mosh_shaders::uniform_bytes("tr_copy"), None);
    }

    #[test]
    fn fields_are_laid_out_in_shader_order() {
        let uniforms = TransitionUniforms {
            res: [128.0, 72.0],
            progress: 0.25,
            mode: 1.0,
            flash: 2.0,
            param: 0.0,
            pad0: 0.0,
            pad1: 0.0,
        };
        let bytes = uniforms.as_bytes();
        let at = |index: usize| {
            f32::from_le_bytes([
                bytes[index * 4],
                bytes[index * 4 + 1],
                bytes[index * 4 + 2],
                bytes[index * 4 + 3],
            ])
        };
        assert_eq!(at(0), 128.0);
        assert_eq!(at(1), 72.0);
        assert_eq!(at(2), 0.25);
        assert_eq!(at(3), 1.0);
        assert_eq!(at(4), 2.0);
    }

    #[test]
    fn strobe_period_is_sixteen_flashes_per_second() {
        assert_eq!(STROBE_PERIOD.as_micros(), 62_500);
        // A one-second strobe at that rate asks 16 times; the 400 ms floor allows three starts.
        let mut limiter = FlashLimiter::new();
        let start = Instant::now();
        let mut starts = 0;
        for step in 0..16u32 {
            let asked = start + STROBE_PERIOD * step;
            if limiter.allow(asked) {
                starts += 1;
            }
        }
        assert_eq!(starts, 3, "16 asks in one second through a 400 ms floor");
    }
}
