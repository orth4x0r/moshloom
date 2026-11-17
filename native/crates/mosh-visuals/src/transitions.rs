//! Per-layer transitions: Cut, Crossfade, Datamosh, Melt, Strobe. Every full-frame flash goes
//! through the shared `FlashLimiter` (400 ms, no opt-out). Owner: V-fx (Task 2.3).
//! Stub: does nothing. Signatures are frozen.

use mosh_core::{ActiveTransition, AnalysisFrame};
use mosh_render::{FlashLimiter, FrameRecorder, Gpu, LayerTarget, RenderError};

pub struct TransitionPass {}

impl TransitionPass {
    pub fn new(_gpu: &Gpu) -> Result<TransitionPass, RenderError> {
        Ok(TransitionPass {})
    }

    /// A clip change on `layer`: `from` holds last frame's picture (sample it as an input this
    /// frame; keep your own copy if you need it after this frame).
    pub fn begin(
        &mut self,
        _rec: &mut FrameRecorder<'_>,
        _layer: usize,
        _from: &LayerTarget,
    ) -> Result<(), RenderError> {
        Ok(())
    }

    /// Blend from the saved picture to `src` by `transition.progress`, writing `dst`.
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &mut self,
        _rec: &mut FrameRecorder<'_>,
        _layer: usize,
        _src: &LayerTarget,
        _dst: &LayerTarget,
        _transition: &ActiveTransition,
        _frame: &AnalysisFrame,
        _flash: &mut FlashLimiter,
    ) -> Result<bool, RenderError> {
        Ok(false)
    }
}
