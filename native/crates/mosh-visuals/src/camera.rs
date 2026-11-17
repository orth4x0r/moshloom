//! Camera layer: a live webcam picture (e.g. the performer singing), so it can be moshed and
//! effected like any other layer. Owner: L-camera. Stub: draws black. Signatures are frozen.

use mosh_render::{FrameRecorder, Gpu, LayerTarget, RenderError};

/// What `p0..p7` mean for `camera` (shown in the clip editor).
pub const PARAMS: [&str; 8] = [
    "zoom",
    "pan x",
    "pan y",
    "mirror",
    "exposure",
    "contrast",
    "tint",
    "posterize",
];

pub struct CameraLayer {}

impl CameraLayer {
    pub fn new(_gpu: &Gpu) -> Result<CameraLayer, RenderError> {
        Ok(CameraLayer {})
    }

    /// Which camera `layer` shows: `None` = the default camera. Called by the app when a camera
    /// clip starts; opening a device never blocks the render thread (capture runs on its own
    /// thread and the newest frame is taken each draw).
    pub fn set_source(&mut self, _layer: usize, _device: Option<&str>) {}

    /// The camera names Windows reports, for the clip editor.
    pub fn devices() -> Vec<String> {
        Vec::new()
    }

    /// Draw `layer`'s newest camera frame into `dst` (cover-scaled, `params` applied). Black
    /// until a frame has arrived; never waits for one.
    pub fn draw(
        &mut self,
        _rec: &mut FrameRecorder<'_>,
        _layer: usize,
        _dst: &LayerTarget,
        _params: &mosh_core::ClipParams,
    ) -> Result<(), RenderError> {
        Ok(())
    }
}
