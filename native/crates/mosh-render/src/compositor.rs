//! The compositor: blends layer targets into the output image, in layer order, with the blend
//! mode and opacity the show model asked for.
//!
//! Up to [`MAX_STACK`] layers go through **one** pass (`shaders/compositor_stack.frag.wgsl`): it
//! samples every layer once and applies the whole blend chain in registers, so a four-layer frame
//! writes the output image once instead of four times. Above that the per-layer path
//! (`shaders/compositor.frag.wgsl`) keeps the old behaviour: one full-screen pass per layer with
//! the fixed-function blend unit, which is why `Pipeline`s for the five blend modes still exist.
//!
//! Both paths produce the same arithmetic; the stack pass just keeps the intermediate values in
//! f32 instead of round-tripping them through the RGBA16F output image.
//!
//! The fragment shaders' uniform blocks are
//! `{ opacity: f32, luma_threshold: f32, flags: u32, pad: u32 }` (per layer) and
//! `{ count: u32, pad: u32 x3, layers: [vec4<f32>; 8] }` (the stack).

use std::sync::Arc;

use ash::vk;
use mosh_core::show::BlendMode;

use crate::device::{Gpu, MappedBuffer};
use crate::pipeline::{Blend, DescriptorLayouts, Pipeline};
use crate::target::{cmd_barrier, Barrier, LayerTarget, OutputImage, OFFSCREEN_FORMAT};
use crate::RenderError;

/// Bytes in the compositor's per-layer uniform block (`COMPOSITOR_UNIFORM_SIZE` in the WGSL).
pub(crate) const COMPOSITOR_UNIFORM_SIZE: usize = 16;
/// Front door for the uniform ring when the show grows past this: layers are 3-4 today.
const MAX_LAYERS: usize = 24;
/// Layers the single stack pass can take (three visualizer layers plus an overlay, with room).
pub const MAX_STACK: usize = 8;
/// Bytes of the stack pass's uniform block: a header plus one `vec4<f32>` per stack layer.
const COMPOSITOR_STACK_UNIFORM_SIZE: usize =
    16 + MAX_STACK * 16;
/// Luma below this fraction is keyed out; the overlay's luma key is refined in Task 2.3.
const LUMA_KEY_THRESHOLD: f32 = 0.08;

/// Blends layer targets into an `OutputImage`.
pub struct Compositor {
    device: ash::Device,
    queue: vk::Queue,
    /// The colour format this compositor writes: the stage's swapchain format, or RGBA16F for an
    /// offscreen `OutputImage`. The stack shader blends in registers, so a format change is only
    /// a pipeline rebuild, not a different result (up to the target's precision).
    format: vk::Format,
    /// Indexed by `blend_slot`: the per-layer path (used above [`MAX_STACK`] layers).
    pipelines: Vec<Pipeline>,
    /// The whole stack in one pass (`compositor_stack.frag.wgsl`).
    stack: Pipeline,
    /// A 1x1 target bound to the stack pass's unused inputs (the shader samples at most `count`).
    empty: LayerTarget,
    pool: vk::DescriptorPool,
    uniforms: MappedBuffer,
    allocations: crate::device::Allocations,
    stride: u64,
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    /// Two timestamp queries around the layers of the last `composite`, so the frame budget shows
    /// the compositor's own GPU cost instead of hiding it in the frame's wall clock.
    query_pool: vk::QueryPool,
    timestamp_period_ns: f32,
    timings: crate::GpuTimings,
}

impl Compositor {
    /// A compositor for an offscreen RGBA16F `OutputImage` (tests, the budget bench).
    pub fn new(gpu: &Gpu) -> Result<Compositor, RenderError> {
        Compositor::new_for_format(gpu, OFFSCREEN_FORMAT)
    }

    /// A compositor that writes `format`: the stage's swapchain format, so the composite pass
    /// writes the presented image directly and no offscreen copy of the frame is needed.
    pub fn new_for_format(gpu: &Gpu, format: vk::Format) -> Result<Compositor, RenderError> {
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
                format,
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
        let stack_layouts = Arc::new(DescriptorLayouts::new(
            gpu,
            COMPOSITOR_STACK_UNIFORM_SIZE,
            MAX_STACK as u32,
        )?);
        let stack = Pipeline::with_layouts(
            gpu,
            &stack_layouts,
            crate::shaders::COMPOSITOR_STACK_FRAG,
            format,
            None,
            "composite.stack",
        )?;

        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(MAX_LAYERS as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(MAX_LAYERS as u32 + MAX_STACK as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(MAX_LAYERS as u32 + MAX_STACK as u32),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(MAX_LAYERS as u32)
            .pool_sizes(&sizes);
        let pool = unsafe { gpu.dev.create_descriptor_pool(&pool_info, None) }?;

        // The per-layer uniform slices are laid out with the device's required alignment, after
        // the stack pass's own block (which is written once per frame at the end of the buffer).
        let stride = crate::device::align_up(
            COMPOSITOR_UNIFORM_SIZE as u64,
            gpu.min_uniform_offset_alignment(),
        );
        let bytes = stride * MAX_LAYERS as u64 + COMPOSITOR_STACK_UNIFORM_SIZE as u64;
        let (buffer, memory) = gpu.new_buffer(
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
        let query_pool = unsafe {
            gpu.dev.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )?
        };

        Ok(Compositor {
            device: gpu.dev.clone(),
            queue: gpu.queue,
            format,
            pipelines,
            stack,
            empty: LayerTarget::new(gpu, 1, 1),
            pool,
            uniforms: MappedBuffer {
                buffer,
                memory,
                ptr: pointer,
                len: bytes,
            },
            allocations: gpu.allocations(),
            stride,
            cmd_pool,
            cmd,
            fence,
            query_pool,
            timestamp_period_ns: gpu.timestamp_period_ns(),
            timings: crate::GpuTimings::default(),
        })
    }

    /// GPU time of the last [`Self::composite`], as one `composite` entry in
    /// `GpuTimings::per_pass_ms` (the pass writes the whole output once per layer).
    ///
    /// Empty before the first composite and on a device without timestamp support.
    pub fn timings(&self) -> &crate::GpuTimings {
        &self.timings
    }

    /// The colour format this compositor writes.
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Records the whole stack into `view` (a swapchain image, or any colour attachment in
    /// `self.format()`), inside a command buffer the caller submits. Clears `extent` first.
    ///
    /// This is the stage's path: the composite is the last pass of the frame's present
    /// submission, so a four-layer 1080p frame writes the presented image once instead of
    /// compositing into an offscreen copy and blitting that. No timestamps and no submit here;
    /// [`Self::composite`] is the measured, self-contained variant.
    pub fn record_into(
        &self,
        cmd: vk::CommandBuffer,
        layers: &[(&LayerTarget, BlendMode, f32)],
        view: vk::ImageView,
        extent: vk::Extent2D,
    ) -> Result<(), RenderError> {
        if layers.len() > MAX_STACK {
            return Err(RenderError::TooManyLayers {
                max: MAX_STACK,
                got: layers.len(),
            });
        }
        write_stack_uniform(&self.uniforms, self.stack_offset(), layers);
        let area = vk::Rect2D::default().extent(extent);
        let viewport = vk::Viewport::default()
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            },
        };
        // The stack shader writes every pixel of the frame (no discard, no partial coverage), so
        // the attachment does not need clearing first: a clear of a 1920x1080 target is another
        // 8 MB write. `record_into` (the stage) claims the same.
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear);
        let attachments = [attachment];
        let set = unsafe {
            self.device
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())?;
            let set_layouts = [self.stack.set_layout()];
            let allocate = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.pool)
                .set_layouts(&set_layouts);
            self.device.allocate_descriptor_sets(&allocate)?[0]
        };
        bind_stack_set(self, set, layers);
        unsafe {
            let rendering = vk::RenderingInfo::default()
                .render_area(area)
                .layer_count(1)
                .color_attachments(&attachments);
            self.device.cmd_begin_rendering(cmd, &rendering);
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.stack.raw());
            self.device.cmd_set_viewport(cmd, 0, &[viewport]);
            self.device.cmd_set_scissor(cmd, 0, &[area]);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.stack.pipeline_layout(),
                0,
                &[set],
                &[self.stack_offset() as u32],
            );
            self.device.cmd_draw(cmd, 3, 1, 0, 0);
            self.device.cmd_end_rendering(cmd);
        }
        Ok(())
    }

    /// Composites `layers` into `out`, bottom layer first. `out` is cleared first, so the same
    /// `OutputImage` can be composited into every frame.
    ///
    /// Up to [`MAX_STACK`] layers are one pass (`compositor_stack.frag.wgsl`); more fall back to
    /// one pass per layer with the fixed-function blend unit, which is the same arithmetic with a
    /// round trip through `out` between layers.
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
        let stack = layers.len() <= MAX_STACK;

        // The uniform block per layer: opacity, luma threshold, flags. The stack pass reads the
        // same four values per layer, plus the count.
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
        if stack {
            write_stack_uniform(&self.uniforms, self.stack_offset(), layers);
        }

        unsafe {
            self.device
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())?;
            self.device
                .reset_command_pool(self.cmd_pool, vk::CommandPoolResetFlags::empty())?;
            self.device.reset_query_pool(self.query_pool, 0, 2);
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
            self.device.cmd_write_timestamp(
                self.cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool,
                0,
            );
            let rendering = vk::RenderingInfo::default()
                .render_area(extent)
                .layer_count(1)
                .color_attachments(&attachments);
            self.device.cmd_begin_rendering(self.cmd, &rendering);
        }

        if stack {
            self.draw_stack(layers, extent, viewport)?;
        } else {
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
        }

        unsafe {
            self.device.cmd_end_rendering(self.cmd);
            self.device.cmd_write_timestamp(
                self.cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                1,
            );
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
        self.read_timings();
        Ok(())
    }

    /// Reads the two timestamps of the composite that just finished. Empty when the device has no
    /// usable timestamp support, or when the driver refuses the read.
    fn read_timings(&mut self) {
        if self.timestamp_period_ns <= 0.0 {
            self.timings = crate::GpuTimings::default();
            return;
        }
        let mut raw = [0u64; 2];
        let result = unsafe {
            self.device.get_query_pool_results(
                self.query_pool,
                0,
                &mut raw,
                vk::QueryResultFlags::TYPE_64,
            )
        };
        if result.is_err() {
            self.timings = crate::GpuTimings::default();
            return;
        }
        let period_ms = self.timestamp_period_ns as f64 / 1_000_000.0;
        let delta = raw[1].saturating_sub(raw[0]);
        let ms = (delta as f64 * period_ms) as f32;
        self.timings = crate::GpuTimings {
            per_pass_ms: vec![("composite".to_string(), ms)],
            frame_ms: ms,
        };
    }

    /// The pipeline a blend mode uses, for callers that want to record the composite pass into
    /// their own command buffer (the stage re-uses it to blit the output to the swapchain).
    pub fn pipeline_for(&self, mode: BlendMode) -> &Pipeline {
        &self.pipelines[blend_slot(mode)]
    }

    /// Offset of the stack pass's uniform block: after the per-layer slices.
    fn stack_offset(&self) -> u64 {
        self.stride * MAX_LAYERS as u64
    }

    /// Records the whole layer stack as one pass: every layer sampled once, the blend chain in
    /// registers, one write of the output image.
    fn draw_stack(
        &self,
        layers: &[(&LayerTarget, BlendMode, f32)],
        extent: vk::Rect2D,
        viewport: vk::Viewport,
    ) -> Result<(), RenderError> {
        let set = unsafe {
            let set_layouts = [self.stack.set_layout()];
            let allocate = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.pool)
                .set_layouts(&set_layouts);
            self.device.allocate_descriptor_sets(&allocate)?[0]
        };
        bind_stack_set(self, set, layers);
        unsafe {
            self.device.cmd_bind_pipeline(
                self.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.stack.raw(),
            );
            self.device.cmd_set_viewport(self.cmd, 0, &[viewport]);
            self.device.cmd_set_scissor(self.cmd, 0, &[extent]);
            self.device.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.stack.pipeline_layout(),
                0,
                &[set],
                &[self.stack_offset() as u32],
            );
            self.device.cmd_draw(self.cmd, 3, 1, 0, 0);
        }
        Ok(())
    }
}

impl Drop for Compositor {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_query_pool(self.query_pool, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_descriptor_pool(self.pool, None);
            self.device.unmap_memory(self.uniforms.memory);
            self.device.destroy_buffer(self.uniforms.buffer, None);
            self.device.free_memory(self.uniforms.memory, None);
            self.allocations.release(self.uniforms.memory);
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

/// The blend mode as `compositor_stack.frag.wgsl` names it: `NORMAL`, `ADD`, `SCREEN`,
/// `MULTIPLY`, `LUMAKEY` (0..4).
fn blend_mode_value(mode: BlendMode) -> f32 {
    blend_slot(mode) as f32
}

/// Fills the stack pass's uniform block: the count, then (opacity, mode, luma threshold, flip)
/// per layer.
fn write_stack_uniform(
    uniforms: &MappedBuffer,
    offset: u64,
    layers: &[(&LayerTarget, BlendMode, f32)],
) {
    let mut block = [0u8; COMPOSITOR_STACK_UNIFORM_SIZE];
    block[0..4].copy_from_slice(&(layers.len() as u32).to_le_bytes());
    for (index, (_, mode, opacity)) in layers.iter().enumerate() {
        let at = 16 + index * 16;
        let params = [
            opacity.clamp(0.0, 1.0),
            blend_mode_value(*mode),
            if *mode == BlendMode::LumaKey {
                LUMA_KEY_THRESHOLD
            } else {
                0.0
            },
            0.0,
        ];
        for (slot, value) in params.iter().enumerate() {
            block[at + slot * 4..at + slot * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    unsafe {
        uniforms.zero(offset, COMPOSITOR_STACK_UNIFORM_SIZE as u64);
        uniforms.write(offset, &block);
    }
}

/// Writes the stack pass's descriptor set: its uniform slice and one image per layer, with the
/// compositor's 1x1 target standing in for the slots past `layers.len()`.
fn bind_stack_set(
    compositor: &Compositor,
    set: vk::DescriptorSet,
    layers: &[(&LayerTarget, BlendMode, f32)],
) {
    let offset = compositor.stack_offset();
    let buffer_info = [vk::DescriptorBufferInfo::default()
        .buffer(compositor.uniforms.buffer)
        .offset(0)
        .range(COMPOSITOR_STACK_UNIFORM_SIZE as u64)];
    let empty = compositor.empty.front_view();
    let mut images: Vec<vk::DescriptorImageInfo> = Vec::with_capacity(MAX_STACK);
    for index in 0..MAX_STACK {
        let view = match layers.get(index) {
            Some((layer, _, _)) => {
                debug_assert_eq!(
                    layer.front_layout(),
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    "layers must be readable when composited"
                );
                layer.front_view()
            }
            None => empty,
        };
        images.push(
            vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        );
    }
    let mut writes = Vec::with_capacity(1 + MAX_STACK);
    writes.push(
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .buffer_info(&buffer_info),
    );
    for (index, info) in images.iter().enumerate() {
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1 + index as u32 * 2)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(std::slice::from_ref(info)),
        );
    }
    let _ = offset;
    unsafe {
        compositor.device.update_descriptor_sets(&writes, &[]);
    }
}
