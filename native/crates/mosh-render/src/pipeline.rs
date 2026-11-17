//! Fullscreen fragment pipelines from SPIR-V, and the recorder that runs them.
//!
//! A `Pipeline` is one fragment shader plus the crate's own full-screen vertex shader, with a
//! fixed descriptor set layout: binding 0 is a dynamic uniform buffer, then one sampled image and
//! one immutable-sampler binding per input texture (2,3), (4,5), ... That layout is what the app
//! and mosh-shaders code against.
//!
//! `FrameRecorder` records the layer passes of one frame into one command buffer (dynamic
//! rendering, no render passes), brackets each pass with timestamp queries and submits with the
//! frame's timeline-semaphore value. Two frames may be in flight.

use std::sync::Arc;

use ash::vk;
use mosh_core::show::BlendMode;

use crate::device::{Gpu, FRAMES_IN_FLIGHT, MAX_DRAWS_PER_FRAME, MAX_UNIFORM_BYTES};
use crate::target::{cmd_barrier, to_color_attachment, to_shader_read, LayerTarget, OFFSCREEN_FORMAT};
use crate::timing::{FrameTicket, GpuTimings};
use crate::RenderError;

/// The descriptor set layout and pipeline layout a family of pipelines shares. Sharing one
/// object (instead of creating five identical ones for the compositor) keeps the descriptor
/// sets provably compatible with every pipeline that uses them.
pub(crate) struct DescriptorLayouts {
    device: ash::Device,
    pub(crate) set: vk::DescriptorSetLayout,
    pub(crate) pipeline: vk::PipelineLayout,
    pub(crate) uniform_range: u64,
    pub(crate) textures: u32,
}

impl DescriptorLayouts {
    pub(crate) fn new(
        gpu: &Gpu,
        uniform_size: usize,
        textures: u32,
    ) -> Result<DescriptorLayouts, RenderError> {
        let uniform_range = (uniform_size as u64).max(16);
        if uniform_range > MAX_UNIFORM_BYTES {
            return Err(RenderError::UniformTooLarge {
                size: uniform_range,
                max: MAX_UNIFORM_BYTES,
            });
        }
        let mut bindings = Vec::with_capacity(1 + textures as usize * 2);
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        );
        // The layer sampler is immutable, so a descriptor write only carries the image.
        let immutable_samplers = [gpu.sampler];
        for index in 0..textures {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1 + index * 2)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            );
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(2 + index * 2)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .immutable_samplers(&immutable_samplers),
            );
        }

        let set_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let set = unsafe { gpu.dev.create_descriptor_set_layout(&set_info, None) }?;
        let set_layouts = [set];
        let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        let pipeline = unsafe { gpu.dev.create_pipeline_layout(&layout_info, None) }?;
        Ok(DescriptorLayouts {
            device: gpu.dev.clone(),
            set,
            pipeline,
            uniform_range,
            textures,
        })
    }
}

impl Drop for DescriptorLayouts {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline_layout(self.pipeline, None);
            self.device.destroy_descriptor_set_layout(self.set, None);
        }
    }
}

/// How the fixed-function blend unit combines a layer with what is already in the output.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Blend {
    pub(crate) src_color: vk::BlendFactor,
    pub(crate) dst_color: vk::BlendFactor,
    pub(crate) src_alpha: vk::BlendFactor,
    pub(crate) dst_alpha: vk::BlendFactor,
}

impl Blend {
    /// Premultiplied alpha over: the layer shader already multiplied the colour by its opacity,
    /// so `src + dst * (1 - alpha)` is the correct "over".
    pub(crate) const NORMAL: Blend = Blend {
        src_color: vk::BlendFactor::ONE,
        dst_color: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        src_alpha: vk::BlendFactor::ONE,
        dst_alpha: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
    };
    pub(crate) const ADD: Blend = Blend {
        src_color: vk::BlendFactor::ONE,
        dst_color: vk::BlendFactor::ONE,
        src_alpha: vk::BlendFactor::ONE,
        dst_alpha: vk::BlendFactor::ONE,
    };
    pub(crate) const SCREEN: Blend = Blend {
        src_color: vk::BlendFactor::ONE,
        dst_color: vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        src_alpha: vk::BlendFactor::ONE,
        dst_alpha: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
    };
    /// `dst * src`, faded back towards `dst` by the layer's opacity.
    pub(crate) const MULTIPLY: Blend = Blend {
        src_color: vk::BlendFactor::DST_COLOR,
        dst_color: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        src_alpha: vk::BlendFactor::ONE,
        dst_alpha: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
    };
    pub(crate) fn of(mode: BlendMode) -> Blend {
        match mode {
            BlendMode::Normal | BlendMode::LumaKey => Blend::NORMAL,
            BlendMode::Add => Blend::ADD,
            BlendMode::Screen => Blend::SCREEN,
            BlendMode::Multiply => Blend::MULTIPLY,
        }
    }
}

/// One fullscreen fragment pipeline.
pub struct Pipeline {
    device: ash::Device,
    handle: vk::Pipeline,
    layouts: Arc<DescriptorLayouts>,
    color_format: vk::Format,
    name: String,
}

impl Pipeline {
    /// Builds a pipeline that writes into an offscreen RGBA16F target.
    ///
    /// `uniform_size` is the size of the shader's uniform block in bytes (0 when it has none);
    /// `textures` is the number of sampled inputs the shader declares.
    pub fn fullscreen(
        gpu: &Gpu,
        frag_spirv: &[u32],
        uniform_size: usize,
        textures: u32,
    ) -> Result<Pipeline, RenderError> {
        Pipeline::fullscreen_with_format(
            gpu,
            frag_spirv,
            uniform_size,
            textures,
            OFFSCREEN_FORMAT,
        )
    }

    /// `Pipeline::fullscreen` for a specific colour attachment format (a swapchain's).
    pub fn fullscreen_with_format(
        gpu: &Gpu,
        frag_spirv: &[u32],
        uniform_size: usize,
        textures: u32,
        color_format: vk::Format,
    ) -> Result<Pipeline, RenderError> {
        let layouts = Arc::new(DescriptorLayouts::new(gpu, uniform_size, textures)?);
        Pipeline::with_layouts(
            gpu,
            &layouts,
            frag_spirv,
            color_format,
            None,
            "pipeline",
        )
    }

    /// Builds a pipeline whose colour attachment has `color_format` (the stage's swapchain
    /// format) and, optionally, a blend state.
    pub(crate) fn with_layouts(
        gpu: &Gpu,
        layouts: &Arc<DescriptorLayouts>,
        frag_spirv: &[u32],
        color_format: vk::Format,
        blend: Option<Blend>,
        name: &str,
    ) -> Result<Pipeline, RenderError> {
        let vertex = shader_module(gpu, crate::shaders::FULLSCREEN_VERT)?;
        let fragment = match shader_module(gpu, frag_spirv) {
            Ok(module) => module,
            Err(error) => {
                unsafe { gpu.dev.destroy_shader_module(vertex, None) };
                return Err(error);
            }
        };
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex)
                .name(c"vs_main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment)
                .name(c"fs_main"),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample =
            vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
        let attachment = match blend {
            Some(blend) => vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .src_color_blend_factor(blend.src_color)
                .dst_color_blend_factor(blend.dst_color)
                .src_alpha_blend_factor(blend.src_alpha)
                .dst_alpha_blend_factor(blend.dst_alpha)
                .color_blend_op(vk::BlendOp::ADD)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::RGBA),
            None => vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(false)
                .color_write_mask(vk::ColorComponentFlags::RGBA),
        };
        let attachments = [attachment];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let color_formats = [color_format];
        let mut rendering =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(layouts.pipeline)
            .base_pipeline_handle(vk::Pipeline::null())
            .render_pass(vk::RenderPass::null())
            .push_next(&mut rendering);

        let created = unsafe {
            gpu.dev
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
        };
        unsafe {
            gpu.dev.destroy_shader_module(vertex, None);
            gpu.dev.destroy_shader_module(fragment, None);
        }
        let handle = match created {
            Ok(mut pipelines) => pipelines.pop().ok_or(RenderError::MissingFeature("pipeline"))?,
            Err((_, error)) => return Err(RenderError::Vulkan(error)),
        };
        Ok(Pipeline {
            device: gpu.dev.clone(),
            handle,
            layouts: Arc::clone(layouts),
            color_format,
            name: name.to_string(),
        })
    }

    /// Names the passes this pipeline records, for `GpuTimings::per_pass_ms`.
    pub fn with_name(mut self, name: impl Into<String>) -> Pipeline {
        self.name = name.into();
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn textures(&self) -> u32 {
        self.layouts.textures
    }

    /// The colour format the pipeline was built for.
    pub fn color_format(&self) -> vk::Format {
        self.color_format
    }

    pub(crate) fn raw(&self) -> vk::Pipeline {
        self.handle
    }

    pub(crate) fn set_layout(&self) -> vk::DescriptorSetLayout {
        self.layouts.set
    }

    pub(crate) fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.layouts.pipeline
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe { self.device.destroy_pipeline(self.handle, None) };
    }
}

fn shader_module(gpu: &Gpu, words: &[u32]) -> Result<vk::ShaderModule, RenderError> {
    if words.is_empty() {
        return Err(RenderError::EmptyShader);
    }
    let info = vk::ShaderModuleCreateInfo::default().code(words);
    Ok(unsafe { gpu.dev.create_shader_module(&info, None) }?)
}

/// Records the layer passes of one frame.
///
/// `draw` writes one fragment shader pass into a layer target's back image, with the target's
/// front image (and any other `inputs`) available as textures; afterwards the target's images
/// swap, so a feedback shader sees what it wrote last frame.
pub struct FrameRecorder<'a> {
    gpu: &'a Gpu,
    slot: usize,
    cmd: vk::CommandBuffer,
    draws: u32,
    names: Vec<String>,
    layers: Vec<(&'a LayerTarget, BlendMode, f32)>,
    finished: bool,
}

impl<'a> FrameRecorder<'a> {
    /// Claims the next frame slot, waiting only if that slot is still two frames behind.
    pub fn new(gpu: &'a Gpu) -> FrameRecorder<'a> {
        let (slot, pool, cmd, query_pool, last_value) = {
            let mut frames = gpu.frames.borrow_mut();
            let slot = frames.next_slot;
            frames.next_slot = (slot + 1) % FRAMES_IN_FLIGHT;
            let frame = &frames.slots[slot];
            (slot, frame.pool, frame.cmd, frame.query_pool, frame.last_value)
        };
        if last_value > 0 {
            crate::timing::wait_timeline(gpu, last_value).expect("wait for frame slot");
        }
        unsafe {
            gpu.dev
                .reset_command_pool(pool, vk::CommandPoolResetFlags::empty())
                .expect("reset frame command pool");
            gpu.dev
                .reset_descriptor_pool(
                    gpu.frames.borrow().slots[slot].descriptor_pool,
                    vk::DescriptorPoolResetFlags::empty(),
                )
                .expect("reset frame descriptor pool");
            gpu.dev.reset_query_pool(query_pool, 0, 2 * MAX_DRAWS_PER_FRAME as u32);
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            gpu.dev
                .begin_command_buffer(cmd, &begin)
                .expect("begin frame command buffer");
        }
        FrameRecorder {
            gpu,
            slot,
            cmd,
            draws: 0,
            names: Vec::new(),
            layers: Vec::new(),
            finished: false,
        }
    }

    /// The GPU this frame belongs to.
    pub fn gpu(&self) -> &'a Gpu {
        self.gpu
    }

    /// Registers a layer's output for the stage's compositor, in draw order.
    ///
    /// `Stage::present` composites these (blend mode and opacity included) into the swapchain
    /// image. An app that composites itself (`Compositor::composite`) can leave this empty.
    pub fn layer(&mut self, target: &'a LayerTarget, blend: BlendMode, opacity: f32) {
        self.layers.push((target, blend, opacity));
    }

    /// The layer stack registered with `layer`, in draw order.
    pub fn layers(&self) -> &[(&'a LayerTarget, BlendMode, f32)] {
        &self.layers
    }

    /// Records one pass: `target`'s back image is the colour attachment (cleared), `inputs` are
    /// sampled textures, `uniforms` is the fragment shader's uniform block.
    pub fn draw(
        &mut self,
        pipeline: &Pipeline,
        target: &LayerTarget,
        uniforms: &[u8],
        inputs: &[&LayerTarget],
    ) -> Result<(), RenderError> {
        if self.finished {
            return Err(RenderError::FrameFinished);
        }
        if self.draws as usize >= MAX_DRAWS_PER_FRAME {
            return Err(RenderError::TooManyDraws);
        }
        if inputs.len() != pipeline.layouts.textures as usize {
            return Err(RenderError::InputCountMismatch {
                expected: pipeline.layouts.textures,
                got: inputs.len() as u32,
            });
        }
        let range = pipeline.layouts.uniform_range;
        if uniforms.len() as u64 > range {
            return Err(RenderError::UniformTooLarge {
                size: uniforms.len() as u64,
                max: range,
            });
        }

        // Uniform ring: this frame's slot, this draw's slice of it.
        let stride = self.gpu.uniform_stride;
        let offset = (self.slot as u64 * MAX_DRAWS_PER_FRAME as u64 + u64::from(self.draws)) * stride;
        unsafe {
            let ring = &self.gpu.uniform_ring;
            ring.zero(offset, range);
            ring.write(offset, uniforms);
        }

        // One descriptor set per draw: the uniform slice plus this draw's inputs.
        let set = {
            let pool = self.gpu.frames.borrow().slots[self.slot].descriptor_pool;
            let set_layouts = [pipeline.layouts.set];
            let allocate = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&set_layouts);
            unsafe { self.gpu.dev.allocate_descriptor_sets(&allocate) }?[0]
        };
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(self.gpu.uniform_ring.buffer)
            .offset(0)
            .range(range)];
        let image_infos: Vec<vk::DescriptorImageInfo> = inputs
            .iter()
            .map(|input| {
                vk::DescriptorImageInfo::default()
                    .image_view(input.front_view())
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            })
            .collect();
        let mut writes = Vec::with_capacity(1 + image_infos.len());
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .buffer_info(&buffer_info),
        );
        for (index, info) in image_infos.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1 + index as u32 * 2)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(std::slice::from_ref(info)),
            );
        }
        unsafe {
            self.gpu.dev.update_descriptor_sets(&writes, &[]);
        }

        // Inputs must be readable; the invariant is that they already are, but a target that was
        // written earlier in this frame is explicitly moved back by its own pass.
        for input in inputs {
            debug_assert_eq!(
                input.front_layout(),
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                "layer target must be readable when sampled"
            );
        }

        let extent = vk::Rect2D::default().extent(vk::Extent2D {
            width: target.width(),
            height: target.height(),
        });
        let viewport = vk::Viewport::default()
            .width(target.width() as f32)
            .height(target.height() as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        unsafe {
            cmd_barrier(
                &self.gpu.dev,
                self.cmd,
                target.back_image(),
                to_color_attachment(),
            );

            let query = self.draws * 2;
            self.gpu.dev.cmd_write_timestamp(
                self.cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                {
                    let frames = self.gpu.frames.borrow();
                    frames.slots[self.slot].query_pool
                },
                query,
            );

            let clear = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            };
            let attachment = vk::RenderingAttachmentInfo::default()
                .image_view(target.back_view())
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear);
            let attachments = [attachment];
            let rendering = vk::RenderingInfo::default()
                .render_area(extent)
                .layer_count(1)
                .color_attachments(&attachments);
            self.gpu
                .dev
                .cmd_begin_rendering(self.cmd, &rendering);
            self.gpu.dev.cmd_bind_pipeline(
                self.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.handle,
            );
            self.gpu.dev.cmd_set_viewport(self.cmd, 0, &[viewport]);
            self.gpu.dev.cmd_set_scissor(self.cmd, 0, &[extent]);
            self.gpu.dev.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.layouts.pipeline,
                0,
                &[set],
                &[offset as u32],
            );
            self.gpu.dev.cmd_draw(self.cmd, 3, 1, 0, 0);
            self.gpu.dev.cmd_end_rendering(self.cmd);

            self.gpu.dev.cmd_write_timestamp(
                self.cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                {
                    let frames = self.gpu.frames.borrow();
                    frames.slots[self.slot].query_pool
                },
                query + 1,
            );

            cmd_barrier(
                &self.gpu.dev,
                self.cmd,
                target.back_image(),
                to_shader_read(),
            );
        }
        target.set_back_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        target.swap();
        self.draws += 1;
        self.names.push(pipeline.name.clone());
        Ok(())
    }

    /// Ends and submits the frame. Returns the ticket to wait on with `Gpu::wait_frame`.
    pub fn submit(mut self) -> FrameTicket {
        self.finished = true;
        let value = {
            let mut frames = self.gpu.frames.borrow_mut();
            let value = frames.next_value + 1;
            frames.next_value = value;
            frames.slots[self.slot].last_value = value;
            value
        };
        unsafe {
            self.gpu
                .dev
                .end_command_buffer(self.cmd)
                .expect("end frame command buffer");
        }
        let timeline = self.gpu.frames.borrow().timeline;
        let command_buffers = [self.cmd];
        let signal = [timeline];
        let signal_values = [value];
        let mut timeline_info =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);
        let submit = vk::SubmitInfo::default()
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal)
            .push_next(&mut timeline_info);
        unsafe {
            self.gpu
                .dev
                .queue_submit(self.gpu.queue, &[submit], vk::Fence::null())
                .expect("submit frame");
        }
        FrameTicket {
            slot: self.slot,
            value,
            passes: self.draws,
            names: std::mem::take(&mut self.names),
        }
    }
}

impl Drop for FrameRecorder<'_> {
    fn drop(&mut self) {
        // A recorder dropped without `submit` leaves its command buffer in the recording state;
        // the slot's pool is reset the next time the slot is used. Nothing is submitted.
    }
}

impl Gpu {
    /// Waits for a submitted frame and reads its GPU timestamps.
    ///
    /// A failed wait (device lost) yields empty timings instead of a panic: the stage must not
    /// show a dialog mid-set, and the failure resurfaces on the next submit.
    pub fn wait_frame(&self, ticket: FrameTicket) -> GpuTimings {
        if crate::timing::wait_timeline(self, ticket.value).is_err() {
            return GpuTimings::default();
        }
        crate::timing::read(self, &ticket)
    }
}
