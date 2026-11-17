//! The effects lane: Fold, Mirror, Streak, Crush, Zoom, Spin, Echo, Negative, applied in ONE
//! pass (Mosh is not here; see `mosh`). Owner: V-fx (Task 2.3). Stub: does nothing.
//! Signatures are frozen.

use mosh_core::{AnalysisFrame, EffectSlot};
use mosh_render::{FrameRecorder, Gpu, LayerTarget, RenderError};

pub struct EffectChain {}

impl EffectChain {
    pub fn new(_gpu: &Gpu) -> Result<EffectChain, RenderError> {
        Ok(EffectChain {})
    }

    /// Apply every non-Mosh slot with amount > 0 to `src`, writing `dst`. Returns `Ok(false)`
    /// (and writes nothing) when there is nothing to apply.
    pub fn apply(
        &mut self,
        _rec: &mut FrameRecorder<'_>,
        _src: &LayerTarget,
        _dst: &LayerTarget,
        _effects: &[EffectSlot],
        _frame: &AnalysisFrame,
        _t: f32,
    ) -> Result<bool, RenderError> {
        Ok(false)
    }
}
