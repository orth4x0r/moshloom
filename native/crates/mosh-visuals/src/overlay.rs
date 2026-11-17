//! Overlay video layers: decoded BGRA frames (mosh-video) uploaded and drawn into a layer.
//! Owner: V-overlay (Task 2.8). Signatures are frozen.
//!
//! The app decodes a frame with `mosh_video`, hands it over with [`OverlayLayer::set_frame`] (a
//! memcpy into a mapped staging buffer), and the layer pass copies it to the GPU and draws it into
//! the layer's target with aspect-preserving cover scaling when the facade reaches it. The GPU
//! half lives in `mosh_render::upload`, next to the device it needs.
//!
//! Both halves of the overlay — the staging copy and the cover draw — are recorded into the
//! `FrameRecorder`'s command buffer, so an overlay layer adds no submission and no fence wait to
//! the frame (it used to be 1.7-2.6 ms of CPU wait per layer per frame in debug builds). The
//! staging buffer it writes belongs to the frame slot the recorder will claim, so the CPU never
//! overwrites bytes a queued copy is still reading.
//!
//! One texture per layer index, reused for every frame of that size; a clip change or a resize
//! reallocates once. A layer with no frame draws transparent black, so a stopped overlay never
//! shows a stale picture.

use mosh_render::upload::{CoverPass, VideoTexture};
use mosh_render::{FrameRecorder, Gpu, LayerTarget, RenderError};

/// The shader module this pass needs (`mosh-shaders/wgsl/overlay_video.wgsl`).
const SHADER: &str = "overlay_video";

/// A show has far fewer layers than this; the cap only stops a bogus index from growing the
/// per-layer vector without bound.
const MAX_LAYERS: usize = 32;

pub struct OverlayLayer {
    /// One cover pass for every layer: it only holds the pipeline, which is per-draw state, not
    /// per-layer state.
    cover: CoverPass,
    /// One video texture per layer index, created by that layer's first `set_frame`.
    layers: Vec<Option<VideoTexture>>,
}

impl OverlayLayer {
    pub fn new(gpu: &Gpu) -> Result<OverlayLayer, RenderError> {
        let spirv = mosh_shaders::spirv(SHADER).ok_or(RenderError::EmptyShader)?;
        Ok(OverlayLayer {
            cover: CoverPass::new(gpu, spirv)?,
            layers: Vec::new(),
        })
    }

    /// Hand over this frame's video picture for `layer` (BGRA8, `width * height * 4` bytes,
    /// rows tightly packed). Called by the app before `Visuals::draw_layer`.
    ///
    /// Nothing is submitted here: the copy to the GPU is recorded into the next [`Self::draw`],
    /// and a frame of the same size reuses the image, the staging buffers and the pipeline.
    pub fn set_frame(
        &mut self,
        gpu: &Gpu,
        layer: usize,
        width: u32,
        height: u32,
        bgra: &[u8],
    ) -> Result<(), RenderError> {
        if layer >= MAX_LAYERS {
            return Err(RenderError::TooManyLayers {
                max: MAX_LAYERS,
                got: layer + 1,
            });
        }
        if self.layers.len() <= layer {
            self.layers.resize_with(layer + 1, || None);
        }
        if self.layers[layer].is_none() {
            self.layers[layer] = Some(VideoTexture::new(gpu, width, height)?);
        }
        let video = self.layers[layer]
            .as_mut()
            .expect("the texture was created just above");
        video.upload(gpu, width, height, bgra)
    }

    /// Draw `layer`'s latest picture into `dst`, scaled to cover it (aspect kept, centred), inside
    /// `rec`'s frame: one recorded copy, one recorded pass, no submission and no CPU wait.
    ///
    /// A layer that never received a frame is filled with transparent black, also inside the frame.
    pub fn draw(
        &mut self,
        rec: &mut FrameRecorder<'_>,
        layer: usize,
        dst: &LayerTarget,
    ) -> Result<(), RenderError> {
        match self.layers.get_mut(layer).and_then(Option::as_mut) {
            Some(video) => self.cover.record(rec, video, dst),
            None => rec.clear(dst),
        }
    }
}
