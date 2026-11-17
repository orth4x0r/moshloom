//! Overlay video layers: decoded BGRA frames (mosh-video) uploaded and drawn into a layer.
//! Owner: V-overlay (Task 2.8). Stub: draws nothing. Signatures are frozen.

use mosh_render::{FrameRecorder, Gpu, LayerTarget, RenderError};

pub struct OverlayLayer {}

impl OverlayLayer {
    pub fn new(_gpu: &Gpu) -> Result<OverlayLayer, RenderError> {
        Ok(OverlayLayer {})
    }

    /// Hand over this frame's video picture for `layer` (BGRA8, `width * height * 4` bytes,
    /// rows tightly packed). Called by the app before `Visuals::draw_layer`.
    pub fn set_frame(
        &mut self,
        _gpu: &Gpu,
        _layer: usize,
        _width: u32,
        _height: u32,
        _bgra: &[u8],
    ) -> Result<(), RenderError> {
        Ok(())
    }

    /// Draw `layer`'s latest picture into `dst`, scaled to cover it (aspect kept, centred).
    pub fn draw(
        &mut self,
        _rec: &mut FrameRecorder<'_>,
        _layer: usize,
        _dst: &LayerTarget,
    ) -> Result<(), RenderError> {
        Ok(())
    }
}
