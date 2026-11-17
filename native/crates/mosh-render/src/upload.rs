//! Video textures: a decoded BGRA8 frame on the GPU, and the pass that draws it into a layer.
//!
//! Owner: V-overlay (Task 2.8). `mosh-visuals::overlay` uses this module; `mosh-render` itself does
//! not need a video texture, so it is the only file here that knows about one.
//!
//! # Why the frame lands in its own image
//!
//! `FrameRecorder::draw` takes `&LayerTarget` inputs, and a `LayerTarget` is RGBA16F: BGRA8 bytes
//! cannot be copied into it, and a transfer cannot convert between the two formats. So the decoded
//! frame lands in its own `B8G8R8A8_UNORM` image (a format the sampler reads as RGB, with the
//! byte order Windows gives us) and a fullscreen pass with cover scaling writes it into the layer
//! target.
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
//! `VideoTexture::new` allocates the image and two staging buffers; `CoverPass::new` allocates the
//! pipeline. Uploading a frame of the same size after that allocates nothing at all: every staging
//! buffer is mapped once and reused, and the copy is recorded per frame. [`allocation_count`]
//! counts the GPU allocations this module makes, so a test can prove that
//! (`tests/overlay.rs::no_new_allocations_after_first`).
//!
//! # Memory
//!
//! A 1920x1080 layer costs 8.3 MB for the image and 8.3 MB for each staging buffer, about 25 MB
//! per overlay video: four overlays are ~100 MB of the design's 500 MB budget.

use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk;

use crate::RenderError;
use crate::device::{FRAMES_IN_FLIGHT, Gpu, MappedBuffer, create_buffer};
use crate::pipeline::{FrameRecorder, Pipeline};
use crate::target::LayerTarget;

/// The format a decoder's BGRA frame is uploaded in. Windows hands over RGB32, which is
/// `B,G,R,X` in memory; sampled as `B8G8R8A8_UNORM` the red byte lands in the red channel.
pub const VIDEO_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;

/// Bytes of the cover pass's uniform block: `dst_size` and `src_size` as two `vec2<f32>`.
/// Keep in sync with `mosh-shaders/wgsl/overlay_video.wgsl`.
pub const COVER_UNIFORM_BYTES: u64 = 16;

/// GPU allocations this module has made (images, buffers and the memory behind them).
static GPU_ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

/// How many GPU allocations [`VideoTexture`] and [`CoverPass`] have made so far.
pub fn allocation_count() -> u64 {
    GPU_ALLOCATIONS.load(Ordering::Relaxed)
}

fn count_allocation() {
    GPU_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
}

/// One video picture on the GPU: a `B8G8R8A8_UNORM` image, one mapped staging buffer per frame in
/// flight, and the layout the image is in between copies.
pub struct VideoTexture {
    device: ash::Device,
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
    pub fn new(gpu: &Gpu, width: u32, height: u32) -> Result<VideoTexture, RenderError> {
        if width == 0 || height == 0 {
            return Err(RenderError::ZeroSized);
        }
        let (image, memory) = gpu.create_image(
            vk::Extent2D { width, height },
            VIDEO_FORMAT,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )?;
        count_allocation();
        let view = gpu.create_image_view(image, VIDEO_FORMAT)?;
        let staging = mapped_staging(gpu, texels(width, height) * 4)?;
        Ok(VideoTexture {
            device: gpu.dev.clone(),
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

    /// True while a frame is in the staging buffer and no frame has copied it to the GPU.
    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Copies one decoded frame into the staging buffer of the frame slot the next recorder will
    /// use. `bgra` is `width * height * 4` bytes, rows tightly packed, which is what
    /// `mosh_video::VideoFrame` holds.
    ///
    /// Nothing is submitted here: the copy to the GPU is recorded into the next
    /// [`FrameRecorder`]. A size change reallocates once; every other call is a memcpy.
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
        if (width, height) != (self.width, self.height) {
            self.reallocate(gpu, width, height)?;
        }
        // The buffer this frame's copy will read belongs to the slot the next recorder claims. If
        // that slot's last submission is still running, it may be reading the same buffer: wait for
        // it here, which is the same wait `FrameRecorder::new` would do a moment later.
        let slot = gpu.next_frame_slot();
        crate::timing::wait_timeline(gpu, gpu.slot_last_value(slot))?;
        // Safety: the staging buffer is `width * height * 4` bytes and the submission that last
        // read it has completed (waited for just above).
        unsafe { self.staging[slot].write(0, bgra) };
        self.pending = Some(slot);
        Ok(())
    }

    /// The size changed. One `device_wait_idle` because the old image and buffers may still be
    /// referenced by a submission that was never waited on.
    fn reallocate(&mut self, gpu: &Gpu, width: u32, height: u32) -> Result<(), RenderError> {
        unsafe {
            gpu.dev.device_wait_idle()?;
            self.destroy();
        }
        let (image, memory) = gpu.create_image(
            vk::Extent2D { width, height },
            VIDEO_FORMAT,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )?;
        count_allocation();
        let view = gpu.create_image_view(image, VIDEO_FORMAT)?;
        let staging = mapped_staging(gpu, texels(width, height) * 4)?;
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
                for staging in &self.staging {
                    self.device.unmap_memory(staging.memory);
                    self.device.destroy_buffer(staging.buffer, None);
                    self.device.free_memory(staging.memory, None);
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

/// The cover pass's uniform block: destination size and video size, both in pixels.
fn cover_uniforms(dst_w: u32, dst_h: u32, src_w: u32, src_h: u32) -> [u8; COVER_UNIFORM_BYTES as usize] {
    let mut block = [0u8; COVER_UNIFORM_BYTES as usize];
    block[0..4].copy_from_slice(&(dst_w as f32).to_le_bytes());
    block[4..8].copy_from_slice(&(dst_h as f32).to_le_bytes());
    block[8..12].copy_from_slice(&(src_w as f32).to_le_bytes());
    block[12..16].copy_from_slice(&(src_h as f32).to_le_bytes());
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

/// One mapped staging buffer per frame in flight.
fn mapped_staging(gpu: &Gpu, bytes: u64) -> Result<Vec<MappedBuffer>, RenderError> {
    let mut buffers = Vec::with_capacity(FRAMES_IN_FLIGHT);
    for _ in 0..FRAMES_IN_FLIGHT {
        let (buffer, memory) = create_buffer(
            &gpu.dev,
            &gpu.memory,
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
