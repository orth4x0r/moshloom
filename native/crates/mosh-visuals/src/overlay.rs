//! Overlay video layers: decoded frames (mosh-video) uploaded and drawn into a layer. Owner:
//! V-overlay (Task 2.8); the HAP/transparency path is O-hap's. Existing signatures are frozen.
//!
//! Two kinds of frame arrive here from `mosh_video::VideoSource`, one per decoder:
//!
//! * a Media Foundation frame: BGRA, handed over with [`OverlayLayer::set_frame`], uploaded into a
//!   `B8G8R8A8_UNORM` texture and drawn with cover scaling (opaque, as v1 always was);
//! * a HAP frame: BC1/BC3/BC4 blocks, handed over with [`OverlayLayer::set_hap_frame`] and copied
//!   straight into a block-compressed texture — no CPU pixel decode — then drawn by the HAP cover
//!   pass, which applies HAP Q's scaled YCoCg and HAP's alpha.
//!
//! # Alpha
//!
//! HAP stores **straight** (un-premultiplied) colour with alpha: `Hap5`'s BC3 alpha channel, or
//! `HapM` (HAP Q Alpha)'s separate BC4 plane. The HAP pass premultiplies in the shader, once, so
//! the layer target holds a premultiplied picture. That is what the compositor's
//! `BlendMode::Normal` expects (`src + dst * (1 - alpha)`), so a transparent part of a HAP overlay
//! shows the layers underneath instead of black.
//!
//! # Cost
//!
//! Both halves of an overlay — the staging copy and the cover draw — are recorded into the
//! `FrameRecorder`'s command buffer, so an overlay layer adds no submission and no fence wait to
//! the frame (it used to be 1.7-2.6 ms of CPU wait per layer per frame in debug builds). The
//! staging buffer it writes belongs to the frame slot the recorder will claim, so the CPU never
//! overwrites bytes a queued copy is still reading.
//!
//! One texture set per layer index, reused for every frame of that size; a clip change or a resize
//! reallocates once. A layer with no frame draws transparent black, so a stopped overlay never
//! shows a stale picture.

use mosh_render::upload::{BlockFormat, CoverPass, HapCoverPass, VideoTexture};
use mosh_render::{FrameRecorder, Gpu, LayerTarget, RenderError};

/// The shader module this pass needs (`mosh-shaders/wgsl/overlay_video.wgsl`).
const SHADER: &str = "overlay_video";

/// A show has far fewer layers than this; the cap only stops a bogus index from growing the
/// per-layer vector without bound.
const MAX_LAYERS: usize = 32;

/// What one layer index is currently showing.
///
/// Boxed: a `VideoTexture` is a large value (its staging ring is inline) and a HAP layer holds two
/// of them, so the enum would otherwise be three textures wide.
enum LayerTexture {
    /// A decoded BGRA frame (Media Foundation).
    Bgra(Box<VideoTexture>),
    /// A HAP frame: the colour plane plus, for HAP Q Alpha, a separate BC4 alpha plane.
    Hap(Box<HapTexture>),
}

struct HapTexture {
    color: VideoTexture,
    alpha: Option<VideoTexture>,
    /// The clip's display size; the textures are padded up to 4x4 blocks from it.
    display: (u32, u32),
}

pub struct OverlayLayer {
    /// One cover pass for every BGRA layer: it only holds the pipeline, which is per-draw state,
    /// not per-layer state.
    cover: CoverPass,
    /// The HAP cover pass, when the device can sample block-compressed textures at all. `None`
    /// leaves BGRA overlays working on a device without `textureCompressionBC`; a HAP frame then
    /// returns an error instead of drawing.
    hap: Option<HapCoverPass>,
    /// One texture set per layer index, created by that layer's first frame.
    layers: Vec<Option<LayerTexture>>,
}

impl OverlayLayer {
    pub fn new(gpu: &Gpu) -> Result<OverlayLayer, RenderError> {
        let spirv = mosh_shaders::spirv(SHADER).ok_or(RenderError::EmptyShader)?;
        Ok(OverlayLayer {
            cover: CoverPass::new(gpu, spirv)?,
            // A device without block compression is not an error for ordinary overlays.
            hap: HapCoverPass::new(gpu).ok(),
            layers: Vec::new(),
        })
    }

    /// True when this device can draw HAP overlays (block-compressed textures).
    pub fn hap_supported(&self) -> bool {
        self.hap.is_some()
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
        if !matches!(self.layers[layer], Some(LayerTexture::Bgra(_))) {
            self.layers[layer] = Some(LayerTexture::Bgra(Box::new(VideoTexture::new(gpu, width, height)?)));
        }
        let Some(LayerTexture::Bgra(video)) = self.layers[layer].as_mut() else {
            unreachable!("the texture was created just above");
        };
        video.upload(gpu, width, height, bgra)
    }

    /// Hand over this frame's HAP picture for `layer`: `data` is the colour plane's BCn block
    /// payload, `alpha` the separate BC4 plane of a `HapM` / HAP Q Alpha frame (or `None`).
    /// `width` and `height` are the clip's **display** size, which is what cover scaling and the
    /// 4x4 padding are computed against.
    ///
    /// `format` comes from `mosh_video::hap::HapFrame::format`:
    ///
    /// | `HapTextureFormat` | [`BlockFormat`] |
    /// | --- | --- |
    /// | `Bc1` | `Bc1` |
    /// | `Bc3` | `Bc3` |
    /// | `Bc3YCoCg` | `Bc3YCoCg` |
    /// | `Bc4Alpha` | `Bc4` |
    ///
    /// Nothing is submitted; the copy joins the next [`Self::draw`]'s frame, exactly like
    /// [`Self::set_frame`].
    #[allow(clippy::too_many_arguments)]
    pub fn set_hap_frame(
        &mut self,
        gpu: &Gpu,
        layer: usize,
        width: u32,
        height: u32,
        format: BlockFormat,
        data: &[u8],
        alpha: Option<&[u8]>,
    ) -> Result<(), RenderError> {
        if layer >= MAX_LAYERS {
            return Err(RenderError::TooManyLayers {
                max: MAX_LAYERS,
                got: layer + 1,
            });
        }
        if self.hap.is_none() {
            return Err(RenderError::MissingFeature("textureCompressionBC"));
        }
        if self.layers.len() <= layer {
            self.layers.resize_with(layer + 1, || None);
        }
        let same = match self.layers[layer].as_ref() {
            Some(LayerTexture::Hap(hap)) => {
                hap.display == (width, height)
                    && hap.color.plane_format() == Some(format)
                    && hap.alpha.is_some() == alpha.is_some()
            }
            _ => false,
        };
        if !same {
            let color = VideoTexture::new_plane(gpu, format, width, height)?;
            let alpha_texture = match alpha {
                Some(_) => Some(VideoTexture::new_plane(gpu, BlockFormat::Bc4, width, height)?),
                None => None,
            };
            self.layers[layer] = Some(LayerTexture::Hap(Box::new(HapTexture {
                color,
                alpha: alpha_texture,
                display: (width, height),
            })));
        }
        let Some(LayerTexture::Hap(hap)) = self.layers[layer].as_mut() else {
            unreachable!("the textures were created just above");
        };
        hap.color.upload_plane(gpu, format, width, height, data)?;
        match (alpha, hap.alpha.as_mut()) {
            (Some(data), Some(texture)) => {
                texture.upload_plane(gpu, BlockFormat::Bc4, width, height, data)?;
            }
            (None, None) => {}
            _ => unreachable!("`same` checked that the alpha plane's presence matches"),
        }
        Ok(())
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
            Some(LayerTexture::Bgra(video)) => self.cover.record(rec, video.as_mut(), dst),
            Some(LayerTexture::Hap(hap)) => {
                let Some(pass) = self.hap.as_mut() else {
                    return Err(RenderError::MissingFeature("textureCompressionBC"));
                };
                pass.record(rec, &mut hap.color, hap.alpha.as_mut(), dst, hap.display)
            }
            None => rec.clear(dst),
        }
    }
}
