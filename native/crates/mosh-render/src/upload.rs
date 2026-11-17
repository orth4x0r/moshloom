//! Video textures: a decoded BGRA8 frame on the GPU, and the pass that draws it into a layer.
//!
//! Owner: V-overlay (Task 2.8). `mosh-visuals::overlay` uses this module; `mosh-render` itself does
//! not need a video texture, so it is the only file here that knows about one.
//!
//! # Why this is a small subsystem rather than one copy
//!
//! `FrameRecorder::draw` takes `&LayerTarget` inputs, and a `LayerTarget` is RGBA16F: BGRA8 bytes
//! cannot be copied into it, and a transfer cannot convert between the two formats. So the decoded
//! frame lands in its own `B8G8R8A8_UNORM` image (a format the sampler reads as RGB, with the
//! byte order Windows gives us) and a fullscreen pass with cover scaling writes it into the layer
//! target. That pass needs its own command buffer, which `FrameRecorder` does not lend out, so it
//! is recorded as a one-shot submission: the CPU waits for it, which is why it costs one fence
//! wait per overlay layer per frame instead of one draw slot.
//!
//! # What is allocated when
//!
//! `VideoTexture::new` allocates the image and the staging buffer; `CoverPass::new` allocates the
//! pipeline, one descriptor set and one uniform buffer. Uploading a frame of the same size after
//! that allocates nothing at all: the staging buffer is mapped once and reused, and the copy is
//! recorded per frame. [`allocation_count`] counts the GPU allocations this module makes, so a
//! test can prove that (`tests/overlay.rs::no_new_allocations_after_first`).
//!
//! # Memory
//!
//! A 1920x1080 layer costs 8.3 MB for the image and 8.3 MB for the staging buffer, so four
//! overlays are about 66 MB of the design's 500 MB budget with four videos.

use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk;

use crate::device::{align_up, create_buffer, Gpu, MappedBuffer};
use crate::pipeline::Pipeline;
use crate::target::{cmd_barrier, to_color_attachment, to_shader_read, Barrier, LayerTarget};
use crate::RenderError;

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

/// One video picture on the GPU: a `B8G8R8A8_UNORM` image, its mapped staging buffer, and the
/// layout the image is in between copies.
///
/// The staging buffer is written by the CPU in [`VideoTexture::upload`] and read by the GPU in
/// [`CoverPass::draw`], which waits for its own submission before returning, so a later upload
/// never overwrites bytes a copy is still reading.
pub struct VideoTexture {
    device: ash::Device,
    width: u32,
    height: u32,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    layout: vk::ImageLayout,
    staging: MappedBuffer,
    /// CPU bytes are in the staging buffer and not yet on the GPU.
    pending: bool,
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
        let (staging, staging_memory, staging_ptr) = mapped_staging(gpu, texels(width, height) * 4)?;
        count_allocation();
        Ok(VideoTexture {
            device: gpu.dev.clone(),
            width,
            height,
            image,
            memory,
            view,
            layout: vk::ImageLayout::UNDEFINED,
            staging: MappedBuffer {
                buffer: staging,
                memory: staging_memory,
                ptr: staging_ptr,
                len: texels(width, height) * 4,
            },
            pending: false,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// True while a frame is waiting for [`CoverPass::draw`] to copy it to the GPU.
    pub fn is_pending(&self) -> bool {
        self.pending
    }

    /// Copies one decoded frame into the staging buffer. `bgra` is `width * height * 4` bytes,
    /// rows tightly packed, which is what `mosh_video::VideoFrame` holds.
    ///
    /// A size change reallocates the image and the staging buffer once; every other call is a
    /// memcpy. Nothing is submitted here: the copy to the GPU is recorded into the next draw.
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
        // Safety: the staging buffer is `width * height * 4` bytes and no copy is in flight (every
        // draw waits for its own submission before returning).
        unsafe { self.staging.write(0, bgra) };
        self.pending = true;
        Ok(())
    }

    /// The size changed. One `device_wait_idle` because the old image may still be referenced by
    /// a submission that was never waited on (an upload with no draw in between).
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
        let bytes = texels(width, height) * 4;
        let (staging, staging_memory, staging_ptr) = mapped_staging(gpu, bytes)?;
        count_allocation();
        self.width = width;
        self.height = height;
        self.image = image;
        self.memory = memory;
        self.view = view;
        self.layout = vk::ImageLayout::UNDEFINED;
        self.staging = MappedBuffer {
            buffer: staging,
            memory: staging_memory,
            ptr: staging_ptr,
            len: bytes,
        };
        self.pending = false;
        Ok(())
    }

    /// # Safety
    /// Nothing the GPU is executing may use the image, the view or the staging buffer.
    unsafe fn destroy(&mut self) {
        unsafe {
            if self.view != vk::ImageView::null() {
                self.device.destroy_image_view(self.view, None);
                self.device.destroy_image(self.image, None);
                self.device.free_memory(self.memory, None);
                self.device.unmap_memory(self.staging.memory);
                self.device.destroy_buffer(self.staging.buffer, None);
                self.device.free_memory(self.staging.memory, None);
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
    device: ash::Device,
    pipeline: Pipeline,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    uniforms: MappedBuffer,
    /// The image view the descriptor set currently points at.
    bound_view: vk::ImageView,
}

impl CoverPass {
    /// `frag_spirv` is `mosh_shaders::spirv("overlay_video")`: one uniform block, one texture,
    /// one sampler.
    pub fn new(gpu: &Gpu, frag_spirv: &[u32]) -> Result<CoverPass, RenderError> {
        let pipeline = Pipeline::fullscreen(gpu, frag_spirv, COVER_UNIFORM_BYTES as usize, 1)?
            .with_name("overlay.video");

        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1),
            // Immutable in the set layout, but the pool still declares the type.
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(1),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&sizes);
        let pool = unsafe { gpu.dev.create_descriptor_pool(&pool_info, None) }?;
        count_allocation();
        let set_layouts = [pipeline.set_layout()];
        let allocate = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&set_layouts);
        let set = match unsafe { gpu.dev.allocate_descriptor_sets(&allocate) } {
            Ok(sets) => sets[0],
            Err(error) => {
                unsafe { gpu.dev.destroy_descriptor_pool(pool, None) };
                return Err(RenderError::Vulkan(error));
            }
        };

        let stride = align_up(COVER_UNIFORM_BYTES, gpu.min_uniform_offset_alignment());
        let (uniform_buffer, uniform_memory) = create_buffer(
            &gpu.dev,
            &gpu.memory,
            stride,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        count_allocation();
        let uniform_ptr = unsafe {
            gpu.dev
                .map_memory(uniform_memory, 0, stride, vk::MemoryMapFlags::empty())
        }? as *mut u8;

        Ok(CoverPass {
            device: gpu.dev.clone(),
            pipeline,
            pool,
            set,
            uniforms: MappedBuffer {
                buffer: uniform_buffer,
                memory: uniform_memory,
                ptr: uniform_ptr,
                len: stride,
            },
            bound_view: vk::ImageView::null(),
        })
    }

    /// Records the pending copy from `video`'s staging buffer to its image, then draws the image
    /// into `dst`'s back image, scaled to cover it (aspect kept, centred), and swaps `dst`'s
    /// images so the result is what the next pass samples.
    ///
    /// One submission, waited on before returning: `dst` is readable when this returns.
    pub fn draw(
        &mut self,
        gpu: &Gpu,
        video: &mut VideoTexture,
        dst: &LayerTarget,
    ) -> Result<(), RenderError> {
        let mut block = [0u8; COVER_UNIFORM_BYTES as usize];
        block[0..4].copy_from_slice(&(dst.width() as f32).to_le_bytes());
        block[4..8].copy_from_slice(&(dst.height() as f32).to_le_bytes());
        block[8..12].copy_from_slice(&(video.width as f32).to_le_bytes());
        block[12..16].copy_from_slice(&(video.height as f32).to_le_bytes());
        // Safety: the buffer is at least COVER_UNIFORM_BYTES and the previous draw, which read it,
        // finished before returning.
        unsafe { self.uniforms.write(0, &block) };
        self.bind(gpu, video.view);

        let copy = video.pending;
        let old_layout = video.layout;
        let extent = vk::Rect2D::default().extent(vk::Extent2D {
            width: dst.width(),
            height: dst.height(),
        });
        let viewport = vk::Viewport::default()
            .width(dst.width() as f32)
            .height(dst.height() as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        gpu.one_shot(|cmd| {
            if copy {
                // Staging buffer -> video image, then readable by the fragment shader below.
                cmd_barrier(
                    &gpu.dev,
                    cmd,
                    video.image,
                    Barrier {
                        old_layout,
                        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        src_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                        src_access: vk::AccessFlags::SHADER_READ,
                        dst_stage: vk::PipelineStageFlags::TRANSFER,
                        dst_access: vk::AccessFlags::TRANSFER_WRITE,
                    },
                );
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(0)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width: video.width,
                        height: video.height,
                        depth: 1,
                    });
                unsafe {
                    gpu.dev.cmd_copy_buffer_to_image(
                        cmd,
                        video.staging.buffer,
                        video.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[region],
                    );
                }
                cmd_barrier(
                    &gpu.dev,
                    cmd,
                    video.image,
                    Barrier {
                        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        src_stage: vk::PipelineStageFlags::TRANSFER,
                        src_access: vk::AccessFlags::TRANSFER_WRITE,
                        dst_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                        dst_access: vk::AccessFlags::SHADER_READ,
                    },
                );
            }

            unsafe {
                cmd_barrier(&gpu.dev, cmd, dst.back_image(), to_color_attachment());
                let clear = vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                };
                let attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(dst.back_view())
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(clear);
                let attachments = [attachment];
                let rendering = vk::RenderingInfo::default()
                    .render_area(extent)
                    .layer_count(1)
                    .color_attachments(&attachments);
                gpu.dev.cmd_begin_rendering(cmd, &rendering);
                gpu.dev
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline.raw());
                gpu.dev.cmd_set_viewport(cmd, 0, &[viewport]);
                gpu.dev.cmd_set_scissor(cmd, 0, &[extent]);
                gpu.dev.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.pipeline_layout(),
                    0,
                    &[self.set],
                    &[0],
                );
                gpu.dev.cmd_draw(cmd, 3, 1, 0, 0);
                gpu.dev.cmd_end_rendering(cmd);
                cmd_barrier(&gpu.dev, cmd, dst.back_image(), to_shader_read());
            }
        })?;

        video.pending = false;
        if copy {
            video.layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        }
        dst.set_back_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        dst.swap();
        Ok(())
    }

    /// Points the descriptor set at `view` when it changed (a layer's video texture was
    /// reallocated on a size change).
    fn bind(&mut self, gpu: &Gpu, view: vk::ImageView) {
        if self.bound_view == view {
            return;
        }
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(self.uniforms.buffer)
            .offset(0)
            .range(COVER_UNIFORM_BYTES)];
        let image_info = [vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .buffer_info(&buffer_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&image_info),
        ];
        unsafe { gpu.dev.update_descriptor_sets(&writes, &[]) };
        self.bound_view = view;
    }
}

impl Drop for CoverPass {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.unmap_memory(self.uniforms.memory);
            self.device
                .destroy_buffer(self.uniforms.buffer, None);
            self.device.free_memory(self.uniforms.memory, None);
            self.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

/// Fills `dst`'s back image with transparent black and swaps it, so a layer with no video this
/// frame is black instead of last frame's picture.
pub fn clear_target(gpu: &Gpu, dst: &LayerTarget) -> Result<(), RenderError> {
    gpu.one_shot(|cmd| {
        unsafe {
            cmd_barrier(
                &gpu.dev,
                cmd,
                dst.back_image(),
                Barrier {
                    old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    src_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                    src_access: vk::AccessFlags::SHADER_READ,
                    dst_stage: vk::PipelineStageFlags::TRANSFER,
                    dst_access: vk::AccessFlags::TRANSFER_WRITE,
                },
            );
            let clear = vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            };
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            gpu.dev.cmd_clear_color_image(
                cmd,
                dst.back_image(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear,
                &[range],
            );
            cmd_barrier(
                &gpu.dev,
                cmd,
                dst.back_image(),
                Barrier {
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    src_stage: vk::PipelineStageFlags::TRANSFER,
                    src_access: vk::AccessFlags::TRANSFER_WRITE,
                    dst_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                    dst_access: vk::AccessFlags::SHADER_READ,
                },
            );
        }
    })?;
    dst.set_back_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    dst.swap();
    Ok(())
}

fn texels(width: u32, height: u32) -> u64 {
    u64::from(width) * u64::from(height)
}

fn mapped_staging(
    gpu: &Gpu,
    bytes: u64,
) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), RenderError> {
    let (buffer, memory) = create_buffer(
        &gpu.dev,
        &gpu.memory,
        bytes,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let pointer = unsafe {
        gpu.dev
            .map_memory(memory, 0, bytes, vk::MemoryMapFlags::empty())
    }? as *mut u8;
    Ok((buffer, memory, pointer))
}
