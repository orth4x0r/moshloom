//! Fluid as its own visualizer: stable fluids on a 256x144 grid, driven by the music.
//! Owner: V-fluid (Task 2.4). Stub: draws nothing. Signatures are frozen.

use mosh_core::{AnalysisFrame, ClipParams};
use mosh_render::{FrameRecorder, Gpu, LayerTarget, RenderError};

pub struct FluidVisualizer {}

impl FluidVisualizer {
    /// `width`/`height`: the layers' internal size (the dye is drawn at this size).
    pub fn new(_gpu: &Gpu, _width: u32, _height: u32) -> Result<FluidVisualizer, RenderError> {
        Ok(FluidVisualizer {})
    }

    /// Step the simulation once and draw the dye into `dst`.
    pub fn draw(
        &mut self,
        _rec: &mut FrameRecorder<'_>,
        _dst: &LayerTarget,
        _frame: &AnalysisFrame,
        _params: &ClipParams,
        _t: f32,
    ) -> Result<(), RenderError> {
        Ok(())
    }

    /// The velocity field (RG, pixels per frame at the layers' internal size) after the last
    /// `draw`, for a mosh layer to stir with. `None` until the simulation has run.
    pub fn velocity(&self) -> Option<&LayerTarget> {
        None
    }
}
