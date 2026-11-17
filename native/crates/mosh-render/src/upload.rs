//! Video textures: a decoded frame on the GPU, and the pass that draws it into a layer.
//!
//! Owner: V-overlay (Task 2.8), with the block-compressed (HAP) path added by O-hap. `mosh-visuals`'
//! `overlay` module uses this module; `mosh-render` itself does not need a video texture, so it is
//! the only file here that knows about one.
//!
//! Two kinds of picture land in a [`VideoTexture`]:
//!
//! * a Media Foundation frame: BGRA8, copied into a `B8G8R8A8_UNORM` image and drawn by
//!   [`CoverPass`] (`shaders/overlay_video.wgsl`).
//! * a HAP frame: BC1/BC3/BC4 blocks, copied **as blocks** into a block-compressed image, so no
//!   CPU pixel decode ever happens here. [`HapCoverPass`] draws it with
//!   `shaders/overlay_hap.frag.wgsl`, which converts HAP Q's scaled YCoCg and applies HAP's
//!   straight alpha (see [`BlockFormat`]). HAP Q Alpha's separate BC4 plane is a second texture,
//!   sampled in the same pass.
//!
//! # Why the frame lands in its own image
//!
//! `FrameRecorder::draw` takes `&LayerTarget` inputs, and a `LayerTarget` is RGBA16F: BGRA8 bytes
//! cannot be copied into it, and a transfer cannot convert between the two formats. So the decoded
//! frame lands in its own image (a format the sampler reads as RGB, with the byte order Windows
//! gives us, or a BCn format the GPU decodes itself) and a fullscreen pass with cover scaling
//! writes it into the layer target.
//!
//! # The copy is part of the frame
//!
//! [`FrameRecorder::upload_video`] records the staging -> image copy into the frame's own command
//! buffer and [`FrameRecorder::draw_video`] samples the image later in the same frame, so an
//! overlay layer costs no fence wait of its own: the frame's single submit and its timeline wait
//! cover both. Before that, the copy and the cover pass were a one-shot submission that the CPU
//! waited on, which measured 1.7-2.6 ms per overlay layer per frame in debug builds.
//!
//! # One staging buffer per frame in flight
//!
//! [`VideoTexture::upload`] writes the staging buffer of the frame slot the next
//! [`FrameRecorder::new`] will claim, after waiting for that slot's last submission — the same
//! frames-in-flight wait the recorder does a moment later. Two staging buffers, one per frame in
//! flight, mean the CPU never writes bytes while a queued copy is still reading them.
//!
//! # What is allocated when
//!
//! `VideoTexture::new` allocates the image and two staging buffers; `CoverPass::new` and
//! `HapCoverPass::new` allocate their pipeline. Uploading a frame of the same size after that
//! allocates nothing at all: every staging buffer is mapped once and reused, and the copy is
//! recorded per frame. [`allocation_count`] counts the GPU allocations this module makes, so a
//! test can prove that (`tests/overlay.rs::no_new_allocations_after_first`).
//!
//! # Memory
//!
//! A 1920x1080 BGRA layer costs 8.3 MB for the image and 8.3 MB for each staging buffer, about
//! 25 MB per overlay video: four overlays are ~100 MB of the design's 500 MB budget. The same
//! frame as HAP costs 1.0 MB (BC1) or 2.0 MB (BC3/BC3 YCoCg) for the image and the same per
//! staging buffer, so four HAP overlays are 6-12 MB instead.

use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk;

use crate::RenderError;
use crate::device::{FRAMES_IN_FLIGHT, Gpu, MappedBuffer};
use crate::pipeline::{FrameRecorder, Pipeline};
use crate::target::LayerTarget;

/// The format a decoder's BGRA frame is uploaded in. Windows hands over RGB32, which is
/// `B,G,R,X` in memory; sampled as `B8G8R8A8_UNORM` the red byte lands in the red channel.
pub const VIDEO_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;

/// Bytes of the cover pass's uniform block: `dst_size` and `src_size` as two `vec2<f32>`.
/// Keep in sync with `mosh-shaders/wgsl/overlay_video.wgsl`.
pub const COVER_UNIFORM_BYTES: u64 = 16;

/// Bytes of the HAP cover pass's uniform block: `dst`, `src`, `tex` as three `vec2<f32>`, then
/// the mode and one padding float. Keep in sync with
/// `mosh-render/src/shaders/overlay_hap.frag.wgsl`.
pub const HAP_COVER_UNIFORM_BYTES: u64 = 32;

/// GPU allocations this module has made (images, buffers and the memory behind them).
static GPU_ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

/// How many GPU allocations [`VideoTexture`] and the cover passes have made so far.
pub fn allocation_count() -> u64 {
    GPU_ALLOCATIONS.load(Ordering::Relaxed)
}

fn count_allocation() {
    GPU_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
}

/// A block-compressed picture format a HAP frame can arrive in: what the blocks mean, and which
/// texture format on the GPU holds them.
///
/// HAP's variants are named by their section type, so this is what a HAP sample's planes map to:
///
/// | HAP | plane format | GPU format | colour |
/// | --- | --- | --- | --- |
/// | `Hap1` | [`BlockFormat::Bc1`] | `BC1_RGBA_UNORM_BLOCK` | RGB, opaque |
/// | `Hap5` | [`BlockFormat::Bc3`] | `BC3_UNORM_BLOCK` | RGB + straight alpha |
/// | `HapY` / `HapM` colour | [`BlockFormat::Bc3YCoCg`] | `BC3_UNORM_BLOCK` | scaled YCoCg, needs the shader's conversion |
/// | `HapA` / `HapM` alpha | [`BlockFormat::Bc4`] | `BC4_UNORM_BLOCK` | one channel: an alpha mask |
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockFormat {
    /// DXT1 / BC1: 8 bytes per 4x4 block, RGB, always opaque.
    Bc1,
    /// DXT5 / BC3: 16 bytes per 4x4 block, straight alpha in the block.
    Bc3,
    /// DXT5 / BC3 carrying scaled YCoCg: luma in the alpha channel, Co/Cg in red/green, the
    /// chroma scale in blue. The cover shader converts it (`ycocg2rgba` from FFmpeg's decoder).
    Bc3YCoCg,
    /// RGTC1 / BC4: 8 bytes per 4x4 block, one channel. An alpha mask.
    Bc4,
}

impl BlockFormat {
    /// Bytes one 4x4 block occupies.
    pub fn bytes_per_block(self) -> u32 {
        match self {
            Self::Bc1 | Self::Bc4 => 8,
            Self::Bc3 | Self::Bc3YCoCg => 16,
        }
    }

    /// The Vulkan format the blocks upload into.
    pub fn vk_format(self) -> vk::Format {
        match self {
            Self::Bc1 => vk::Format::BC1_RGBA_UNORM_BLOCK,
            Self::Bc3 | Self::Bc3YCoCg => vk::Format::BC3_UNORM_BLOCK,
            Self::Bc4 => vk::Format::BC4_UNORM_BLOCK,
        }
    }

    /// True for the scaled YCoCg variant, which the shader has to convert.
    pub fn is_ycocg(self) -> bool {
        self == Self::Bc3YCoCg
    }

    /// True for a single-channel alpha mask.
    pub fn is_alpha_only(self) -> bool {
        self == Self::Bc4
    }

    /// Bytes a `tex_w` x `tex_h` plane occupies; the extent must already be a multiple of 4.
    pub fn plane_bytes(self, tex_w: u32, tex_h: u32) -> u64 {
        u64::from(tex_w.div_ceil(4))
            .saturating_mul(u64::from(tex_h.div_ceil(4)))
            .saturating_mul(u64::from(self.bytes_per_block()))
    }

    /// The `mode` bits the HAP cover shader reads (`overlay_hap.frag.wgsl`).
    fn shader_mode(self, separate_alpha: bool) -> f32 {
        const YCOCG: u32 = 1;
        const SEPARATE_ALPHA: u32 = 2;
        const TEXTURE_ALPHA: u32 = 4;
        const ALPHA_ONLY: u32 = 8;
        let bits = match self {
            Self::Bc1 => 0,
            Self::Bc3 => TEXTURE_ALPHA,
            Self::Bc3YCoCg => YCOCG,
            Self::Bc4 => ALPHA_ONLY,
        } | if separate_alpha { SEPARATE_ALPHA } else { 0 };
        bits as f32
    }
}

/// A texture extent rounded up to whole 4x4 blocks: what a HAP plane covers.
pub fn block_extent(width: u32, height: u32) -> (u32, u32) {
    (width.div_ceil(4) * 4, height.div_ceil(4) * 4)
}

/// One video picture on the GPU: an image, one mapped staging buffer per frame in flight, and the
/// layout the image is in between copies.
///
/// The image is `B8G8R8A8_UNORM` for a decoded BGRA frame ([`VideoTexture::new`]) or a BCn format
/// for a HAP plane ([`VideoTexture::new_plane`]).
pub struct VideoTexture {
    device: ash::Device,
    allocations: crate::device::Allocations,
    format: vk::Format,
    /// `Some` for a block-compressed plane, `None` for BGRA.
    plane: Option<BlockFormat>,
    width: u32,
    height: u32,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    layout: vk::ImageLayout,
    /// Indexed by frame slot: the buffer a frame's copy reads.
    staging: Vec<MappedBuffer>,
    /// The staging buffer index holding a frame no copy has taken yet.
    pending: Option<usize>,
}

impl VideoTexture {
    /// A BGRA8 video image.
    pub fn new(gpu: &Gpu, width: u32, height: u32) -> Result<VideoTexture, RenderError> {
        VideoTexture::create(gpu, VIDEO_FORMAT, None, width, height)
    }

    /// A block-compressed HAP plane. `width` and `height` are the **display** size; the texture is
    /// the same size rounded up to whole 4x4 blocks, which is what a HAP encoder writes.
    ///
    /// Fails when the device has no `textureCompressionBC`, or cannot sample/transfer the format:
    /// a HAP overlay then draws transparent black instead of a wrong picture.
    pub fn new_plane(
        gpu: &Gpu,
        format: BlockFormat,
        width: u32,
        height: u32,
    ) -> Result<VideoTexture, RenderError> {
        if !gpu.texture_compression_bc() {
            return Err(RenderError::MissingFeature("textureCompressionBC"));
        }
        let vk_format = format.vk_format();
        let features = gpu.format_properties(vk_format).optimal_tiling_features;
        let needed =
            vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::TRANSFER_DST;
        if !features.contains(needed) {
            return Err(RenderError::FormatUnsupported(vk_format));
        }
        let (tex_w, tex_h) = block_extent(width, height);
        VideoTexture::create(gpu, vk_format, Some(format), tex_w, tex_h)
    }

    fn create(
        gpu: &Gpu,
        format: vk::Format,
        plane: Option<BlockFormat>,
        width: u32,
        height: u32,
    ) -> Result<VideoTexture, RenderError> {
        if width == 0 || height == 0 {
            return Err(RenderError::ZeroSized);
        }
        let (image, memory) = gpu.create_image(
            vk::Extent2D { width, height },
            format,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )?;
        count_allocation();
        let view = gpu.create_image_view(image, format)?;
        let staging = mapped_staging(gpu, texture_bytes(format, plane, width, height))?;
        Ok(VideoTexture {
            device: gpu.dev.clone(),
            allocations: gpu.allocations(),
            format,
            plane,
            width,
            height,
            image,
            memory,
            view,
            layout: vk::ImageLayout::UNDEFINED,
            staging,
            pending: None,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// The image's Vulkan format: `B8G8R8A8_UNORM`, or the BCn format of a HAP plane.
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// The block format of a HAP plane, `None` for a BGRA image.
    pub fn plane_format(&self) -> Option<BlockFormat> {
        self.plane
    }

    /// The size the plane's blocks cover: the display size rounded up to 4x4 blocks.
    pub fn byte_len(&self) -> u64 {
        texture_bytes(self.format, self.plane, self.width, self.height)
    }

    /// True while a frame is in the staging buffer and no frame has copied it to the GPU.
    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Copies one decoded frame into the staging buffer of the frame slot the next recorder will
    /// use. `bgra` is `width * height * 4` bytes, rows tightly packed, which is what
    /// `mosh_video::VideoFrame` holds.
    ///
    /// Nothing is submitted here: the copy to the GPU is recorded into the next
    /// [`FrameRecorder`]. A size (or format) change reallocates once; every other call is a
    /// memcpy.
    pub fn upload(
        &mut self,
        gpu: &Gpu,
        width: u32,
        height: u32,
        bgra: &[u8],
    ) -> Result<(), RenderError> {
        if width == 0 || height == 0 {
            return Err(RenderError::ZeroSized);
        }
        let expected = texels(width, height) * 4;
        if bgra.len() as u64 != expected {
            // The only error variant that fits a wrong-sized input; a decoder bug, not user input.
            return Err(RenderError::InputCountMismatch {
                expected: expected.min(u64::from(u32::MAX)) as u32,
                got: bgra.len().min(u32::MAX as usize) as u32,
            });
        }
        if self.plane.is_some() || (width, height) != (self.width, self.height) {
            self.reallocate(gpu, VIDEO_FORMAT, None, width, height)?;
        }
        self.stage(gpu, bgra)
    }

    /// Copies one HAP plane into the staging buffer: `data` is the BCn block payload, 4x4 blocks
    /// row by row, exactly `BlockFormat::plane_bytes` of it for the padded height.
    ///
    /// Same handover contract as [`Self::upload`]: no submission, no wait beyond the frame slot's.
    pub fn upload_plane(
        &mut self,
        gpu: &Gpu,
        format: BlockFormat,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Result<(), RenderError> {
        if width == 0 || height == 0 {
            return Err(RenderError::ZeroSized);
        }
        let (tex_w, tex_h) = block_extent(width, height);
        let expected = format.plane_bytes(tex_w, tex_h);
        if data.len() as u64 != expected {
            return Err(RenderError::InputCountMismatch {
                expected: expected.min(u64::from(u32::MAX)) as u32,
                got: data.len().min(u32::MAX as usize) as u32,
            });
        }
        if self.plane != Some(format) || (tex_w, tex_h) != (self.width, self.height) {
            if !gpu.texture_compression_bc() {
                return Err(RenderError::MissingFeature("textureCompressionBC"));
            }
            self.reallocate(gpu, format.vk_format(), Some(format), tex_w, tex_h)?;
        }
        self.stage(gpu, data)
    }

    /// Writes `bytes` into the staging buffer of the frame slot the next recorder claims.
    fn stage(&mut self, gpu: &Gpu, bytes: &[u8]) -> Result<(), RenderError> {
        // The buffer this frame's copy will read belongs to the slot the next recorder claims. If
        // that slot's last submission is still running, it may be reading the same buffer: wait for
        // it here, which is the same wait `FrameRecorder::new` would do a moment later.
        let slot = gpu.next_frame_slot();
        crate::timing::wait_timeline(gpu, gpu.slot_last_value(slot))?;
        // Safety: the staging buffer holds `byte_len` bytes (checked by the caller for the
        // payload) and the submission that last read it has completed (waited for just above).
        unsafe { self.staging[slot].write(0, bytes) };
        self.pending = Some(slot);
        Ok(())
    }

    /// The size or the format changed. One `device_wait_idle` because the old image and buffers
    /// may still be referenced by a submission that was never waited on.
    fn reallocate(
        &mut self,
        gpu: &Gpu,
        format: vk::Format,
        plane: Option<BlockFormat>,
        width: u32,
        height: u32,
    ) -> Result<(), RenderError> {
        unsafe {
            gpu.dev.device_wait_idle()?;
            self.destroy();
        }
        let (image, memory) = gpu.create_image(
            vk::Extent2D { width, height },
            format,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )?;
        count_allocation();
        let view = gpu.create_image_view(image, format)?;
        let staging = mapped_staging(gpu, texture_bytes(format, plane, width, height))?;
        self.format = format;
        self.plane = plane;
        self.width = width;
        self.height = height;
        self.image = image;
        self.memory = memory;
        self.view = view;
        self.layout = vk::ImageLayout::UNDEFINED;
        self.staging = staging;
        self.pending = None;
        Ok(())
    }

    /// The image view the cover pass samples.
    pub(crate) fn view(&self) -> vk::ImageView {
        self.view
    }

    pub(crate) fn image(&self) -> vk::Image {
        self.image
    }

    pub(crate) fn image_layout(&self) -> vk::ImageLayout {
        self.layout
    }

    /// The staging buffer index a recorded copy must read, if any.
    pub(crate) fn pending_staging(&self) -> Option<usize> {
        self.pending
    }

    pub(crate) fn staging_buffer(&self, index: usize) -> vk::Buffer {
        self.staging[index].buffer
    }

    /// The frame this texture was waiting on is on the GPU in `SHADER_READ_ONLY_OPTIMAL`.
    pub(crate) fn mark_copied(&mut self) {
        self.pending = None;
        self.layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    }

    /// # Safety
    /// Nothing the GPU is executing may use the image, the view or the staging buffers.
    unsafe fn destroy(&mut self) {
        unsafe {
            if self.view != vk::ImageView::null() {
                self.device.destroy_image_view(self.view, None);
                self.device.destroy_image(self.image, None);
                self.device.free_memory(self.memory, None);
                self.allocations.release(self.memory);
                for staging in &self.staging {
                    self.device.unmap_memory(staging.memory);
                    self.device.destroy_buffer(staging.buffer, None);
                    self.device.free_memory(staging.memory, None);
                    self.allocations.release(staging.memory);
                }
                self.staging.clear();
                self.view = vk::ImageView::null();
            }
        }
    }
}

impl Drop for VideoTexture {
    fn drop(&mut self) {
        unsafe {
            // A drop in the middle of a frame would race a queued copy; the app drops textures
            // between frames, and a reallocation waits first. Be defensive about the former.
            let _ = self.device.device_wait_idle();
            self.destroy();
        }
    }
}

/// The fullscreen pass that draws a [`VideoTexture`] into a layer target with aspect-preserving
/// cover scaling, centred. One per `OverlayLayer`, shared by every layer index.
pub struct CoverPass {
    pipeline: Pipeline,
}

impl CoverPass {
    /// `frag_spirv` is `mosh_shaders::spirv("overlay_video")`: one uniform block, one texture,
    /// one sampler.
    pub fn new(gpu: &Gpu, frag_spirv: &[u32]) -> Result<CoverPass, RenderError> {
        let pipeline = Pipeline::fullscreen(gpu, frag_spirv, COVER_UNIFORM_BYTES as usize, 1)?
            .with_name("overlay.video");
        Ok(CoverPass { pipeline })
    }

    /// Records `video`'s pending copy and the cover draw into `rec`'s frame: no submission and no
    /// CPU wait. `dst` is readable when the frame completes, not when this returns.
    pub fn record(
        &mut self,
        rec: &mut FrameRecorder<'_>,
        video: &mut VideoTexture,
        dst: &LayerTarget,
    ) -> Result<(), RenderError> {
        rec.upload_video(video)?;
        let uniforms = cover_uniforms(dst.width(), dst.height(), video.width, video.height);
        rec.draw_video(&self.pipeline, dst, &uniforms, video)
    }

    /// [`Self::record`] in a submission of its own, waited on before returning: for a caller that
    /// has no frame to join (the tests' timing split, a still image). The app never needs it.
    pub fn draw(
        &mut self,
        gpu: &Gpu,
        video: &mut VideoTexture,
        dst: &LayerTarget,
    ) -> Result<(), RenderError> {
        let mut rec = FrameRecorder::new(gpu);
        self.record(&mut rec, video, dst)?;
        gpu.wait_frame(rec.submit());
        Ok(())
    }

    /// The pipeline this pass records, for `GpuTimings` and for callers that draw it themselves.
    pub fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }
}

/// The fullscreen pass that draws a HAP plane (or a colour plane plus its separate alpha plane)
/// into a layer target, cover-scaled, straight from the block-compressed texture.
///
/// The shader (`shaders/overlay_hap.frag.wgsl`) applies HAP's scaled-YCoCg conversion and
/// premultiplies HAP's straight alpha, so the layer target holds a premultiplied picture that the
/// compositor's `BlendMode::Normal` ("over" with premultiplied source) blends correctly.
///
/// The sampler is linear when the device reports linear filtering for every BC format this pass
/// can sample, and nearest otherwise: block-compressed formats are not required to filter
/// linearly, and a linear sampler on such a format is a validation error, not a slower picture.
pub struct HapCoverPass {
    pipeline: Pipeline,
}

impl HapCoverPass {
    pub fn new(gpu: &Gpu) -> Result<HapCoverPass, RenderError> {
        if !gpu.texture_compression_bc() {
            return Err(RenderError::MissingFeature("textureCompressionBC"));
        }
        let formats = [
            BlockFormat::Bc1,
            BlockFormat::Bc3,
            BlockFormat::Bc3YCoCg,
            BlockFormat::Bc4,
        ];
        let needed = vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::TRANSFER_DST;
        for format in formats {
            let vk_format = format.vk_format();
            if !gpu
                .format_properties(vk_format)
                .optimal_tiling_features
                .contains(needed)
            {
                return Err(RenderError::FormatUnsupported(vk_format));
            }
        }
        let linear = formats
            .iter()
            .all(|format| gpu.supports_linear_filter(format.vk_format()));
        let filter = if linear {
            vk::Filter::LINEAR
        } else {
            vk::Filter::NEAREST
        };
        let pipeline = Pipeline::fullscreen_with_filter(
            gpu,
            crate::shaders::OVERLAY_HAP_FRAG,
            HAP_COVER_UNIFORM_BYTES as usize,
            2,
            filter,
        )?
        .with_name("overlay.hap");
        Ok(HapCoverPass { pipeline })
    }

    /// Records the pending copies of `color` (and `alpha`, when the clip has a second plane) and
    /// the cover draw into `rec`'s frame. `display` is the clip's size, which the padding is
    /// cropped against.
    pub fn record(
        &mut self,
        rec: &mut FrameRecorder<'_>,
        color: &mut VideoTexture,
        mut alpha: Option<&mut VideoTexture>,
        dst: &LayerTarget,
        display: (u32, u32),
    ) -> Result<(), RenderError> {
        let Some(format) = color.plane_format() else {
            return Err(RenderError::FormatUnsupported(color.format()));
        };
        let mode = format.shader_mode(alpha.is_some());
        let uniforms = hap_uniforms(
            dst.width(),
            dst.height(),
            display.0,
            display.1,
            color.width(),
            color.height(),
            mode,
        );
        rec.upload_video(color)?;
        if let Some(alpha) = alpha.as_deref_mut() {
            rec.upload_video(alpha)?;
        }
        // With no second plane the colour texture is bound twice: the shader samples both
        // unconditionally and ignores the second unless the mode asks for it.
        let color_view: &VideoTexture = color;
        let alpha_view = alpha.as_deref().unwrap_or(color_view);
        rec.draw_video_pair(&self.pipeline, dst, &uniforms, color_view, alpha_view)
    }

    /// The pipeline this pass records, for `GpuTimings` and for callers that draw it themselves.
    pub fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }
}

/// The cover pass's uniform block: destination size and video size, both in pixels.
fn cover_uniforms(dst_w: u32, dst_h: u32, src_w: u32, src_h: u32) -> [u8; COVER_UNIFORM_BYTES as usize] {
    let mut block = [0u8; COVER_UNIFORM_BYTES as usize];
    block[0..4].copy_from_slice(&(dst_w as f32).to_le_bytes());
    block[4..8].copy_from_slice(&(dst_h as f32).to_le_bytes());
    block[8..12].copy_from_slice(&(src_w as f32).to_le_bytes());
    block[12..16].copy_from_slice(&(src_h as f32).to_le_bytes());
    block
}

/// The HAP cover pass's uniform block: destination, display size, padded texture size, mode.
#[allow(clippy::too_many_arguments)]
fn hap_uniforms(
    dst_w: u32,
    dst_h: u32,
    src_w: u32,
    src_h: u32,
    tex_w: u32,
    tex_h: u32,
    mode: f32,
) -> [u8; HAP_COVER_UNIFORM_BYTES as usize] {
    let mut block = [0u8; HAP_COVER_UNIFORM_BYTES as usize];
    let mut put = |offset: usize, value: f32| {
        block[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    };
    put(0, dst_w as f32);
    put(4, dst_h as f32);
    put(8, src_w as f32);
    put(12, src_h as f32);
    put(16, tex_w as f32);
    put(20, tex_h as f32);
    put(24, mode);
    block
}

/// Fills `dst`'s back image with transparent black and swaps it, so a layer with no video this
/// frame is black instead of last frame's picture. One submission, waited on.
///
/// [`FrameRecorder::clear`] does the same inside a frame and is what the overlay layer uses; this
/// stays for callers that have no frame in hand.
pub fn clear_target(gpu: &Gpu, dst: &LayerTarget) -> Result<(), RenderError> {
    let mut rec = FrameRecorder::new(gpu);
    rec.clear(dst)?;
    gpu.wait_frame(rec.submit());
    Ok(())
}

fn texels(width: u32, height: u32) -> u64 {
    u64::from(width) * u64::from(height)
}

/// Bytes of one image (and one staging buffer) for a format.
fn texture_bytes(
    format: vk::Format,
    plane: Option<BlockFormat>,
    width: u32,
    height: u32,
) -> u64 {
    match plane {
        Some(format) => format.plane_bytes(width, height),
        None => {
            debug_assert_eq!(format, VIDEO_FORMAT);
            texels(width, height) * 4
        }
    }
}

/// One mapped staging buffer per frame in flight.
fn mapped_staging(gpu: &Gpu, bytes: u64) -> Result<Vec<MappedBuffer>, RenderError> {
    let mut buffers = Vec::with_capacity(FRAMES_IN_FLIGHT);
    for _ in 0..FRAMES_IN_FLIGHT {
        let (buffer, memory) = gpu.new_buffer(
            bytes,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        count_allocation();
        let pointer = unsafe {
            gpu.dev
                .map_memory(memory, 0, bytes, vk::MemoryMapFlags::empty())
        }? as *mut u8;
        buffers.push(MappedBuffer {
            buffer,
            memory,
            ptr: pointer,
            len: bytes,
        });
    }
    Ok(buffers)
}
