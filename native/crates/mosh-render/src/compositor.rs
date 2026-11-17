//! The compositor: blends layer targets into the output image, in layer order, with the blend
//! mode and opacity the show model asked for.
//!
//! Blending is fixed-function where it can be (`Normal`, `Add`, `Screen`, `Multiply`) and
//! shader-side only for the luma key, which needs to discard. The fragment shader is
//! `src/shaders/compositor.frag.wgsl`; its uniform block is
//! `{ opacity: f32, luma_threshold: f32, flags: u32, pad: u32 }`.

use std::sync::Arc;

use ash::vk;
use mosh_core::show::BlendMode;

use crate::device::{create_buffer, Gpu, MappedBuffer};
use crate::pipeline::{Blend, DescriptorLayouts, Pipeline};
use crate::target::{cmd_barrier, Barrier, LayerTarget, OutputImage, OFFSCREEN_FORMAT};
use crate::RenderError;

/// Bytes in the compositor's per-layer uniform block (`COMPOSITOR_UNIFORM_SIZE` in the WGSL).
pub(crate) const COMPOSITOR_UNIFORM_SIZE: usize = 16;
/// Front door for the uniform ring when the show grows past this: layers are 3-4 today.
const MAX_LAYERS: usize = 24;
/// Luma below this fraction is keyed out; the overlay's luma key is refined in Task 2.3.
const LUMA_KEY_THRESHOLD: f32 = 0.08;

/// Blends layer targets into an `OutputImage`.
pub struct Compositor {
    device: ash::Device,
    queue: vk::Queue,
    /// Indexed by `blend_slot`.
    pipelines: Vec<Pipeline>,
    pool: vk::DescriptorPool,
    uniforms: MappedBuffer,
    stride: u64,
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
}

impl Compositor {
    pub fn new(gpu: &Gpu) -> Result<Compositor, RenderError> {
        let layouts = Arc::new(DescriptorLayouts::new(gpu, COMPOSITOR_UNIFORM_SIZE, 1)?);
        let mut pipelines = Vec::with_capacity(5);
        for mode in [
            BlendMode::Normal,
            BlendMode::Add,
            BlendMode::Screen,
            BlendMode::Multiply,
            BlendMode::LumaKey,
        ] {
            let pipeline = Pipeline::with_layouts(
                gpu,
                &layouts,
                crate::shaders::COMPOSITOR_FRAG,
                OFFSCREEN_FORMAT,
                Some(Blend::of(mode)),
                match mode {
                    BlendMode::Normal => "composite.normal",
                    BlendMode::Add => "composite.add",
                    BlendMode::Screen => "composite.screen",
                    BlendMode::Multiply => "composite.multiply",
                    BlendMode::LumaKey => "composite.lumakey",
                },
            )?;
            pipelines.push(pipeline);
        }

        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(MAX_LAYERS as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(MAX_LAYERS as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(MAX_LAYERS as u32),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(MAX_LAYERS as u32)
            .pool_sizes(&sizes);
        let pool = unsafe { gpu.dev.create_descriptor_pool(&pool_info, None) }?;

        // The per-layer uniform slices are laid out with the device's required alignment.
        let stride = crate::device::align_up(
            COMPOSITOR_UNIFORM_SIZE as u64,
            gpu.min_uniform_offset_alignment(),
        );
        let bytes = stride * MAX_LAYERS as u64;
        let (buffer, memory) = create_buffer(
            &gpu.dev,
            &gpu.memory,
            bytes,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let pointer = unsafe {
            gpu.dev
                .map_memory(memory, 0, bytes, vk::MemoryMapFlags::empty())?
        } as *mut u8;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(gpu.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let cmd_pool = unsafe { gpu.dev.create_command_pool(&pool_info, None) }?;
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { gpu.dev.allocate_command_buffers(&allocate) }?[0];
        let fence = unsafe {
            gpu.dev
                .create_fence(&vk::FenceCreateInfo::default(), None)?
        };

        Ok(Compositor {
            device: gpu.dev.clone(),
            queue: gpu.queue,
            pipelines,
            pool,
            uniforms: MappedBuffer {
                buffer,
                memory,
                ptr: pointer,
                len: bytes,
            },
            stride,
            cmd_pool,
            cmd,
            fence,
        })
    }

    /// Composites `layers` into `out`, bottom layer first. `out` is cleared first, so the same
    /// `OutputImage` can be composited into every frame.
    pub fn composite(
        &mut self,
        layers: &[(&LayerTarget, BlendMode, f32)],
        out: &OutputImage,
    ) -> Result<(), RenderError> {
        if layers.len() > MAX_LAYERS {
            return Err(RenderError::TooManyLayers {
                max: MAX_LAYERS,
                got: layers.len(),
            });
        }

        // The uniform block per layer: opacity, luma threshold, flags.
        for (index, (_, mode, opacity)) in layers.iter().enumerate() {
            let mut block = [0u8; COMPOSITOR_UNIFORM_SIZE];
            block[0..4].copy_from_slice(&opacity.clamp(0.0, 1.0).to_le_bytes());
            block[4..8].copy_from_slice(&LUMA_KEY_THRESHOLD.to_le_bytes());
            let flags: u32 = if *mode == BlendMode::LumaKey { 1 } else { 0 };
            block[8..12].copy_from_slice(&flags.to_le_bytes());
            let offset = index as u64 * self.stride;
            unsafe {
                self.uniforms.zero(offset, self.stride);
                self.uniforms.write(offset, &block);
            }
        }

        unsafe {
            self.device
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())?;
            self.device
                .reset_command_pool(self.cmd_pool, vk::CommandPoolResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(self.cmd, &begin)?;
        }

        let old_layout = out.layout();
        if old_layout != vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
            cmd_barrier(
                &self.device,
                self.cmd,
                out.image(),
                Barrier {
                    old_layout,
                    new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    src_stage: vk::PipelineStageFlags::FRAGMENT_SHADER
                        | vk::PipelineStageFlags::TRANSFER,
                    src_access: vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ,
                    dst_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    dst_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                },
            );
        }

        let extent = vk::Rect2D::default().extent(vk::Extent2D {
            width: out.width(),
            height: out.height(),
        });
        let viewport = vk::Viewport::default()
            .width(out.width() as f32)
            .height(out.height() as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            },
        };
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(out.view())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear);
        let attachments = [attachment];

        unsafe {
            let rendering = vk::RenderingInfo::default()
                .render_area(extent)
                .layer_count(1)
                .color_attachments(&attachments);
            self.device.cmd_begin_rendering(self.cmd, &rendering);
        }

        for (index, (layer, mode, _)) in layers.iter().enumerate() {
            let pipeline = &self.pipelines[blend_slot(*mode)];
            debug_assert_eq!(
                layer.front_layout(),
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                "layers must be readable when composited"
            );
            let set = unsafe {
                let set_layouts = [pipeline.set_layout()];
                let allocate = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.pool)
                    .set_layouts(&set_layouts);
                self.device.allocate_descriptor_sets(&allocate)?[0]
            };
            let offset = index as u64 * self.stride;
            let buffer_info = [vk::DescriptorBufferInfo::default()
                .buffer(self.uniforms.buffer)
                .offset(0)
                .range(COMPOSITOR_UNIFORM_SIZE as u64)];
            let image_info = [vk::DescriptorImageInfo::default()
                .image_view(layer.front_view())
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                    .buffer_info(&buffer_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&image_info),
            ];
            unsafe {
                self.device.update_descriptor_sets(&writes, &[]);
                self.device.cmd_bind_pipeline(
                    self.cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.raw(),
                );
                self.device.cmd_set_viewport(self.cmd, 0, &[viewport]);
                self.device.cmd_set_scissor(self.cmd, 0, &[extent]);
                self.device.cmd_bind_descriptor_sets(
                    self.cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.pipeline_layout(),
                    0,
                    &[set],
                    &[offset as u32],
                );
                self.device.cmd_draw(self.cmd, 3, 1, 0, 0);
            }
        }

        unsafe {
            self.device.cmd_end_rendering(self.cmd);
        }
        cmd_barrier(
            &self.device,
            self.cmd,
            out.image(),
            Barrier {
                old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                src_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_stage: vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::TRANSFER,
                dst_access: vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ,
            },
        );

        unsafe {
            self.device.end_command_buffer(self.cmd)?;
            self.device.reset_fences(&[self.fence])?;
            let command_buffers = [self.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            crate::device::count_submission();
            self.device
                .queue_submit(self.queue, &[submit], self.fence)?;
            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        out.set_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        Ok(())
    }

    /// The pipeline a blend mode uses, for callers that want to record the composite pass into
    /// their own command buffer (the stage re-uses it to blit the output to the swapchain).
    pub fn pipeline_for(&self, mode: BlendMode) -> &Pipeline {
        &self.pipelines[blend_slot(mode)]
    }
}

impl Drop for Compositor {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_descriptor_pool(self.pool, None);
            self.device.unmap_memory(self.uniforms.memory);
            self.device.destroy_buffer(self.uniforms.buffer, None);
            self.device.free_memory(self.uniforms.memory, None);
        }
    }
}

fn blend_slot(mode: BlendMode) -> usize {
    match mode {
        BlendMode::Normal => 0,
        BlendMode::Add => 1,
        BlendMode::Screen => 2,
        BlendMode::Multiply => 3,
        BlendMode::LumaKey => 4,
    }
}
