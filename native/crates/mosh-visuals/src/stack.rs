//! The whole layer stack: consecutive picture layers in one composite pass, with `Adjustment`
//! and `Mask` layers as the seams between segments. Owner: L-stack (wave 3).
//!
//! A frame is composed by [`Stack::compose`], which is what the app calls instead of
//! `Compositor::composite(recorder.layers(), out)`:
//!
//! ```text
//! layers below the first Adjustment   -> one composite pass (unchanged: one write of the output)
//! Adjustment:  below -> its clip's mosh, then its other effects -> the layers above drawn over it
//! Mask:        its picture's luminance becomes the opacity of the layer directly above it
//! ```
//!
//! # The two layer kinds
//!
//! `LayerKind::Adjustment` draws nothing of its own. Its clip's effects (Fold, Mirror, Streak,
//! Crush, Zoom, Spin, Echo, Negative and Mosh) are applied to the composite of every layer below
//! it, like an effect on a group in Ableton, and its opacity blends the effected composite over
//! the un-effected one (0 = the adjustment does nothing, 1 = fully effected). The Mosh slot runs
//! first and is keyed by the adjustment's layer index; the rest are the single effects pass.
//!
//! An adjustment with no picture layer below it (it is at the bottom, or only `Mask` layers are
//! below it) has nothing to adjust: it is skipped without a pass.
//!
//! `LayerKind::Mask` draws nothing of its own either: its picture's luminance is the opacity of
//! the layer directly above it, multiplied with that layer's own opacity. Black hides the layer,
//! white leaves it alone, and the compositor samples it in the same pass as the layer it masks.
//! A mask with no picture layer directly above it (top of the stack, or above an adjustment)
//! masks nothing and is not drawn. Its own blend mode and opacity are ignored: only its luminance
//! is read, from the picture its own layer chain left in `LayerSlot::output()` (so a mask clip's
//! effects are part of the mask).
//!
//! # Cost
//!
//! A stack with no `Adjustment` and no `Mask` layer is exactly today's frame: one pass, no
//! intermediate target, no extra submission, and no allocation until the first adjustment. Each
//! adjustment adds a segment: the composite below it (one pass into a half-float target), the
//! effects pass (one submission of its own, because `EffectChain` records into a `FrameRecorder`),
//! and the adjustment's own blend pass when its opacity is below 1. An adjustment as the top layer
//! writes its result straight into the output, so "a group effect on the whole stack" costs one
//! composite pass, the effects pass and one final pass.
//!
//! `mosh-budget` measures all three cases (plain, with a mask, with an adjustment) at 1080p.

use mosh_core::{AnalysisFrame, BlendMode, EffectKind, LayerKind, LayerState};
use mosh_render::{
    CompositeTarget, Compositor, FrameRecorder, Gpu, LayerTarget, RenderError, StackInput,
};

use crate::effects::EffectChain;
use crate::mosh::MoshPass;
use crate::visuals::LayerSlot;

/// One layer of the stack, in draw order (bottom first).
#[derive(Clone, Copy)]
pub struct StackLayer<'a> {
    /// The layer's slot: its picture for a `Visualizer` or `Overlay` layer, the mask's picture for
    /// a `Mask` layer, unused for an `Adjustment` layer.
    pub slot: &'a LayerSlot,
    pub kind: LayerKind,
    pub blend: BlendMode,
    pub opacity: f32,
    /// This frame's clip state: its effects are what an `Adjustment` layer applies.
    pub state: &'a LayerState,
}

impl<'a> StackLayer<'a> {
    pub fn new(
        slot: &'a LayerSlot,
        kind: LayerKind,
        blend: BlendMode,
        opacity: f32,
        state: &'a LayerState,
    ) -> StackLayer<'a> {
        StackLayer {
            slot,
            kind,
            blend,
            opacity,
            state,
        }
    }
}

/// Which intermediate holds "everything below the current position".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Base {
    /// Nothing below: transparent black.
    Nothing,
    /// The composite of the layers below the current adjustment (also where an adjustment's
    /// result is blended over it).
    Below,
    /// The adjustment clip's mosh output.
    Mosh,
    /// The adjustment clip's effects output.
    Effected,
}

/// The whole layer stack composited for one frame, with the GPU state an `Adjustment` layer needs.
///
/// Everything is allocated on first use: a stack with no `Adjustment` layer allocates nothing
/// beyond the compositor it needs anyway, and a stack with one allocates its intermediates at the
/// current output size.
pub struct Stack {
    /// The size of the intermediates, which is the output's size.
    extent: (u32, u32),
    /// The composite of the layers below the current adjustment. A ping-pong pair: the pass that
    /// blends an adjustment's result over it samples one image and writes the other.
    below: Option<LayerTarget>,
    /// The adjustment clip's mosh output. A ping-pong pair, and needed: the mosh pass reads its
    /// own previous frame as the feedback buffer.
    mosh_out: Option<LayerTarget>,
    /// The adjustment clip's effects output. A ping-pong pair: its previous frame is Echo's
    /// history.
    effected: Option<LayerTarget>,
    mosh: Option<MoshPass>,
    effects: Option<EffectChain>,
    /// The compositor for the output's colour format, rebuilt when that format changes.
    compositor: Option<Compositor>,
    /// The clip drawn last frame, per layer index: an adjustment whose clip changed takes an
    /// I-frame, exactly as `Visuals::draw_layer` does for a picture layer.
    last_clip: Vec<Option<u32>>,
}

impl Stack {
    /// A stack with nothing allocated yet.
    pub fn new() -> Stack {
        Stack {
            extent: (0, 0),
            below: None,
            mosh_out: None,
            effected: None,
            mosh: None,
            effects: None,
            compositor: None,
            last_clip: Vec::new(),
        }
    }

    /// Composites the whole stack and writes it to `out`.
    ///
    /// `rec` holds this frame's layer draws (every layer of `layers`, drawn with
    /// `Visuals::draw_layer`); `compose` submits and waits for it first, because a composite pass
    /// samples the layers and they must be readable. `flow` is the fluid visualizer's velocity
    /// field for an adjustment's mosh, exactly as `Visuals::draw_layer` passes it; `frame` and `t`
    /// are the analysis frame and the time the layers were drawn with.
    ///
    /// `layers` is in draw order, bottom first, one entry per layer of the show: a `Mask` entry is
    /// consumed by the entry above it, an `Adjustment` entry splits the stack.
    pub fn compose(
        &mut self,
        rec: FrameRecorder<'_>,
        flow: Option<&LayerTarget>,
        layers: &[StackLayer<'_>],
        out: CompositeTarget<'_>,
        frame: &AnalysisFrame,
        t: f32,
    ) -> Result<(), RenderError> {
        let gpu = rec.gpu();
        // The layer pictures have to be finished before anything samples them.
        gpu.wait_frame(rec.submit());
        self.prepare(gpu, &out)?;

        // Everything below the current position: starts transparent, and is replaced by each
        // applied adjustment.
        let mut acc = Base::Nothing;
        let mut start = 0usize;

        for (index, layer) in layers.iter().enumerate() {
            if layer.kind != LayerKind::Adjustment {
                continue;
            }
            // The adjustment's own opacity: 0 leaves the composite exactly as it is.
            if layer.opacity <= 0.0 {
                continue;
            }
            // The picture layers of the segment, with their masks, over everything below.
            if layers[start..index]
                .iter()
                .any(|layer| is_picture(layer.kind))
            {
                let inputs = resolve_inputs(layers, start, index);
                self.fold_below(gpu, acc, &inputs)?;
                acc = Base::Below;
            }
            start = index + 1;
            if acc == Base::Nothing {
                // Nothing below to adjust, so the layers above start from the same place.
                continue;
            }

            let mut effected = self.apply_adjustment(gpu, index, layer, acc, flow, frame, t)?;
            if effected == acc {
                // The clip has no effect to apply: the adjustment is a no-op this frame.
                continue;
            }
            if index + 1 == layers.len() {
                // The adjustment is the top layer: its result is the frame. One pass, and no
                // intermediate copy of the adjusted composite.
                let inputs = [picture(
                    &self.below,
                    &self.mosh_out,
                    &self.effected,
                    effected,
                    layer.opacity,
                )];
                let base = resolve(&self.below, &self.mosh_out, &self.effected, acc);
                compositor(&mut self.compositor)?.composite_over(base, &inputs, out)?;
                return Ok(());
            }

            if layer.opacity < 1.0 {
                // Blend the effected composite over the un-effected one: the adjustment's
                // slider. `below` is a pair, so the pass can sample its front image (the
                // un-effected composite) and write its back one.
                self.ensure_below(gpu);
                let inputs = [picture(
                    &self.below,
                    &self.mosh_out,
                    &self.effected,
                    effected,
                    layer.opacity,
                )];
                let below = self.below.as_ref().expect("below");
                let base = resolve(&self.below, &self.mosh_out, &self.effected, Base::Below);
                compositor(&mut self.compositor)?.composite_into(base, &inputs, below)?;
                effected = Base::Below;
            } else if overwrites_effected(layers, start) {
                // The effect targets are shared by every adjustment in the stack. When no picture
                // layer stands between this adjustment and the next one, copy the result into
                // `below` before the next adjustment overwrites it.
                self.ensure_below(gpu);
                let below = self.below.as_ref().expect("below");
                let base = resolve(&self.below, &self.mosh_out, &self.effected, effected);
                compositor(&mut self.compositor)?.composite_into(base, &[], below)?;
                effected = Base::Below;
            }
            acc = effected;
        }

        // The layers above the last adjustment, over everything below them.
        let inputs = resolve_inputs(layers, start, layers.len());
        let base = resolve(&self.below, &self.mosh_out, &self.effected, acc);
        compositor(&mut self.compositor)?.composite_over(base, &inputs, out)?;
        Ok(())
    }

    /// Runs an adjustment on the composite below it and returns where the result ended up: its
    /// clip's mosh, then its other effects, both recorded into one submission of their own.
    #[allow(clippy::too_many_arguments)]
    fn apply_adjustment(
        &mut self,
        gpu: &Gpu,
        index: usize,
        layer: &StackLayer<'_>,
        acc: Base,
        flow: Option<&LayerTarget>,
        frame: &AnalysisFrame,
        t: f32,
    ) -> Result<Base, RenderError> {
        debug_assert!(
            !matches!(acc, Base::Mosh | Base::Effected),
            "an adjustment's input is never one of its own outputs: they are copied into `below` first"
        );
        let state = layer.state;
        // A clip change takes an I-frame, as it does for a picture layer.
        if self.last_clip.len() <= index {
            self.last_clip.resize(index + 1, None);
        }
        if self.last_clip[index] != state.clip
            && let Some(mosh) = self.mosh.as_mut()
        {
            mosh.invalidate(index);
        }
        self.last_clip[index] = state.clip;

        let (amount, bloom) = state
            .effects
            .iter()
            .filter(|slot| slot.kind == EffectKind::Mosh)
            .fold((0.0f32, 0.0f32), |(max_amt, max_bloom), slot| {
                let next_bloom = if slot.amount > 0.0 {
                    max_bloom.max(slot.param)
                } else {
                    max_bloom
                };
                (max_amt.max(slot.amount), next_bloom)
            });
        // No mosh slot this frame is an I-frame, exactly as in `Visuals::draw_layer`: the web app
        // does `if (m <= 0.001) iframe = 1`, so the buffer is forgotten and the next mosh frame
        // starts from a fresh picture instead of carrying a stale one.
        if amount <= 0.0
            && let Some(mosh) = self.mosh.as_mut()
        {
            mosh.invalidate(index);
        }
        let has_effects = state
            .effects
            .iter()
            .any(|slot| slot.kind != EffectKind::Mosh && slot.amount > 0.0);
        if amount <= 0.0 && !has_effects {
            return Ok(acc);
        }

        let (width, height) = self.extent;
        if amount > 0.0 && self.mosh.is_none() {
            self.mosh = Some(MoshPass::new(gpu, width, height)?);
        }
        if amount > 0.0 && self.mosh_out.is_none() {
            self.mosh_out = Some(LayerTarget::new(gpu, width, height));
        }
        if self.effected.is_none() {
            self.effected = Some(LayerTarget::new(gpu, width, height));
        }
        if has_effects && self.effects.is_none() {
            self.effects = Some(EffectChain::new(gpu)?);
        }

        let mut recorder = FrameRecorder::new(gpu);
        let mut current = acc;
        if amount > 0.0 {
            let mosh = self.mosh.as_mut().expect("mosh pass");
            let source = resolve(&self.below, &self.mosh_out, &self.effected, acc)
                .expect("an applied adjustment has a composite below it");
            let destination = self.mosh_out.as_ref().expect("mosh output");
            mosh.set_freeze(index, MoshPass::bloom_freeze(bloom, frame.bar_phase));
            if mosh.apply(&mut recorder, index, source, destination, amount, frame, flow)? {
                current = Base::Mosh;
            }
        }
        if has_effects {
            let effects = self.effects.as_mut().expect("effects pass");
            let source = resolve(&self.below, &self.mosh_out, &self.effected, current)
                .expect("the mosh output or the composite below it");
            let destination = self.effected.as_ref().expect("effects output");
            if effects.apply(
                &mut recorder,
                source,
                destination,
                &state.effects,
                frame,
                t,
            )? {
                current = Base::Effected;
            }
        }
        gpu.wait_frame(recorder.submit());
        Ok(current)
    }

    /// Composites `inputs` over `base` into `below`, or copies `base` into `below` when there are
    /// no inputs.
    fn fold_below(
        &mut self,
        gpu: &Gpu,
        base: Base,
        inputs: &[StackInput<'_>],
    ) -> Result<(), RenderError> {
        self.ensure_below(gpu);
        let below = self.below.as_ref().expect("below");
        let base = resolve(&self.below, &self.mosh_out, &self.effected, base);
        compositor(&mut self.compositor)?.composite_into(base, inputs, below)
    }

    /// The composite-of-the-below target, created at the output size on first use.
    fn ensure_below(&mut self, gpu: &Gpu) {
        if self.below.is_none() {
            let (width, height) = self.extent;
            self.below = Some(LayerTarget::new(gpu, width, height));
        }
    }

    /// Sizes the intermediates and the compositor for this frame's output.
    fn prepare(&mut self, gpu: &Gpu, out: &CompositeTarget<'_>) -> Result<(), RenderError> {
        let extent = out.extent();
        if self.extent != (extent.width, extent.height) {
            self.below = None;
            self.mosh_out = None;
            self.effected = None;
            self.mosh = None;
            self.extent = (extent.width, extent.height);
        }
        if self.compositor.as_ref().map(Compositor::format) != Some(out.format()) {
            self.compositor = Some(Compositor::new_for_format(gpu, out.format())?);
        }
        Ok(())
    }
}

impl Default for Stack {
    fn default() -> Self {
        Stack::new()
    }
}

/// The compositor [`Stack::prepare`] built.
fn compositor(compositor: &mut Option<Compositor>) -> Result<&mut Compositor, RenderError> {
    compositor
        .as_mut()
        .ok_or(RenderError::MissingFeature("compositor"))
}

/// The composite a `Base` names, if it exists yet.
fn resolve<'a>(
    below: &'a Option<LayerTarget>,
    mosh_out: &'a Option<LayerTarget>,
    effected: &'a Option<LayerTarget>,
    base: Base,
) -> Option<&'a LayerTarget> {
    match base {
        Base::Nothing => None,
        Base::Below => below.as_ref(),
        Base::Mosh => mosh_out.as_ref(),
        Base::Effected => effected.as_ref(),
    }
}

/// One layer: the composite `base` names, blended normally at `opacity`.
fn picture<'a>(
    below: &'a Option<LayerTarget>,
    mosh_out: &'a Option<LayerTarget>,
    effected: &'a Option<LayerTarget>,
    base: Base,
    opacity: f32,
) -> StackInput<'a> {
    let target = resolve(below, mosh_out, effected, base)
        .expect("an applied adjustment has a result to blend");
    StackInput::new(target, BlendMode::Normal, opacity)
}

/// The picture layers of `start..end`, each with the mask of a `Mask` layer directly below it.
///
/// Only `Visualizer` and `Overlay` layers are picture layers. A `Mask` layer is consumed as the
/// opacity of the layer above it, and an `Adjustment` layer draws nothing of its own, so neither
/// is ever a composite input.
fn resolve_inputs<'a>(layers: &[StackLayer<'a>], start: usize, end: usize) -> Vec<StackInput<'a>> {
    let mut inputs = Vec::with_capacity(end - start);
    for index in start..end {
        let layer = &layers[index];
        if !is_picture(layer.kind) {
            continue;
        }
        let mask = index
            .checked_sub(1)
            .map(|below| &layers[below])
            .filter(|below| below.kind == LayerKind::Mask)
            .map(|below| below.slot.output());
        inputs.push(match mask {
            Some(mask) => StackInput::masked(
                layer.slot.output(),
                layer.blend,
                layer.opacity,
                mask,
            ),
            None => StackInput::new(layer.slot.output(), layer.blend, layer.opacity),
        });
    }
    inputs
}

/// True for the layer kinds that draw a picture into the composite.
fn is_picture(kind: LayerKind) -> bool {
    matches!(kind, LayerKind::Visualizer | LayerKind::Overlay)
}

/// True when the next `Adjustment` layer comes before any picture layer, so it would overwrite the
/// shared effect targets this adjustment's result lives in.
fn overwrites_effected(layers: &[StackLayer<'_>], start: usize) -> bool {
    for layer in &layers[start..] {
        match layer.kind {
            LayerKind::Adjustment => return true,
            LayerKind::Visualizer | LayerKind::Overlay => return false,
            LayerKind::Mask => {}
        }
    }
    false
}
