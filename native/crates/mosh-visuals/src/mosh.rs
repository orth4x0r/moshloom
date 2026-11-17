//! Datamosh: block-matching motion estimation and the feedback pass, per layer.
//! Owner: V-mosh (Task 2.2). Stub: does nothing. Signatures are frozen.

use mosh_core::AnalysisFrame;
use mosh_render::{FrameRecorder, Gpu, LayerTarget, RenderError};

pub struct MoshPass {}

impl MoshPass {
    /// `width`/`height`: the layers' internal size. Per-layer state (feedback, motion field) is
    /// keyed by the layer index passed to `apply`.
    pub fn new(_gpu: &Gpu, _width: u32, _height: u32) -> Result<MoshPass, RenderError> {
        Ok(MoshPass {})
    }

    /// Mosh `src` into `dst` with `amount` 0..1. `extra_flow` is the fluid visualizer's velocity
    /// field (RG = velocity in pixels/frame) when a fluid layer is running; add it to the motion.
    /// Returns `Ok(true)` when `dst` was written.
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &mut self,
        _rec: &mut FrameRecorder<'_>,
        _layer: usize,
        _src: &LayerTarget,
        _dst: &LayerTarget,
        _amount: f32,
        _frame: &AnalysisFrame,
        _extra_flow: Option<&LayerTarget>,
    ) -> Result<bool, RenderError> {
        Ok(false)
    }
}
