//! The 16 web scenes plus custom WGSL scenes. Owner: V-scenes (Task 2.1).
//! Stub: draws nothing. Signatures are frozen.

use mosh_core::{AnalysisFrame, ClipParams};
use mosh_render::{FrameRecorder, Gpu, LayerTarget, RenderError};

pub struct Scenes {}

impl Scenes {
    pub fn new(_gpu: &Gpu) -> Result<Scenes, RenderError> {
        Ok(Scenes {})
    }

    /// Draw scene `visualizer` ("tunnel", ..., or "custom:<name>") into `dst`.
    pub fn draw(
        &mut self,
        _rec: &mut FrameRecorder<'_>,
        _dst: &LayerTarget,
        _visualizer: &str,
        _frame: &AnalysisFrame,
        _params: &ClipParams,
        _t: f32,
    ) -> Result<(), RenderError> {
        Ok(())
    }

    /// Compile and register a custom scene (`custom:<name>`) from WGSL written against
    /// `mosh_shaders::web_prelude()`. On error the previous version (if any) stays active.
    pub fn set_custom(&mut self, _gpu: &Gpu, _name: &str, _wgsl: &str) -> Result<(), String> {
        Err("custom scenes are not implemented yet".into())
    }
}
