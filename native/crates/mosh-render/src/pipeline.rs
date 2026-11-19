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

use crate::RenderError;
use crate::device::{FRAMES_IN_FLIGHT, Gpu, MAX_DRAWS_PER_FRAME, MAX_UNIFORM_BYTES};
use crate::target::{
    Barrier, LayerTarget, OFFSCREEN_FORMAT, cmd_barrier, from_transfer_write,
    to_color_attachment_discard, to_shader_read, to_transfer_dst,
};
use crate::timing::{FrameTicket, GpuTimings};
use crate::upload::VideoTexture;

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
        DescriptorLayouts::with_sampler(gpu, uniform_size, textures, gpu.sampler)
    }

    /// The same layout with a specific immutable sampler. Block-compressed textures are not
    /// required to support linear filtering, so a HAP pipeline asks for the one the device can
    /// actually use (`Gpu::supports_linear_filter`).
    pub(crate) fn with_sampler(
        gpu: &Gpu,
        uniform_size: usize,
        textures: u32,
        sampler: vk::Sampler,
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
        let immutable_samplers = [sampler];
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
        Pipeline::fullscreen_with_format(gpu, frag_spirv, uniform_size, textures, OFFSCREEN_FORMAT)
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
        Pipeline::with_layouts(gpu, &layouts, frag_spirv, color_format, None, "pipeline")
    }

    /// A pipeline whose sampled inputs are read with `filter`: `LINEAR` for ordinary formats,
    /// `NEAREST` for a block-compressed one on a device that does not report
    /// `SAMPLED_IMAGE_FILTER_LINEAR` for it (HAP video: `mosh_render::upload::HapCoverPass`).
    pub fn fullscreen_with_filter(
        gpu: &Gpu,
        frag_spirv: &[u32],
        uniform_size: usize,
        textures: u32,
        filter: vk::Filter,
    ) -> Result<Pipeline, RenderError> {
        let sampler = match filter {
            vk::Filter::NEAREST => gpu.nearest_sampler,
            _ => gpu.sampler,
        };
        let layouts = Arc::new(DescriptorLayouts::with_sampler(
            gpu,
            uniform_size,
            textures,
            sampler,
        )?);
        Pipeline::with_layouts(
            gpu,
            &layouts,
            frag_spirv,
            OFFSCREEN_FORMAT,
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
        // The fragment entry point is whatever the module declares: the crate's own shaders say
        // `fs_main`, the web app's extracted modules say `fs`.
        let fragment_entry =
            fragment_entry_point(frag_spirv).unwrap_or_else(|| std::ffi::CString::from(c"fs_main"));
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex)
                .name(c"vs_main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment)
                .name(&fragment_entry),
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
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
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
            Ok(mut pipelines) => pipelines
                .pop()
                .ok_or(RenderError::MissingFeature("pipeline"))?,
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

/// Which image of a [`LayerTarget`] a pass samples.
#[derive(Clone, Copy)]
pub enum TargetSample<'a> {
    /// The target's current (most recently written) image.
    Current(&'a LayerTarget),
    /// The target's previous image (the other half of a ping-pong pair, or the same image on a
    /// single-image target). Never pass `Previous(target)` when `target` is also the pass's
    /// destination.
    Previous(&'a LayerTarget),
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
            (
                slot,
                frame.pool,
                frame.cmd,
                frame.query_pool,
                frame.last_value,
            )
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
            gpu.dev
                .reset_query_pool(query_pool, 0, 2 * MAX_DRAWS_PER_FRAME as u32);
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
        if inputs.len() != pipeline.layouts.textures as usize {
            return Err(RenderError::InputCountMismatch {
                expected: pipeline.layouts.textures,
                got: inputs.len() as u32,
            });
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
        let views: Vec<vk::ImageView> = inputs.iter().map(|input| input.front_view()).collect();
        self.pass(pipeline, target, uniforms, &views)
    }

    /// Like [`Self::draw`], but each input chooses either the target's current image or the
    /// previous image of a ping-pong pair.
    pub fn draw_samples(
        &mut self,
        pipeline: &Pipeline,
        target: &LayerTarget,
        uniforms: &[u8],
        inputs: &[TargetSample<'_>],
    ) -> Result<(), RenderError> {
        if inputs.len() != pipeline.layouts.textures as usize {
            return Err(RenderError::InputCountMismatch {
                expected: pipeline.layouts.textures,
                got: inputs.len() as u32,
            });
        }
        let mut views = Vec::with_capacity(inputs.len());
        for input in inputs {
            match *input {
                TargetSample::Current(src) => {
                    debug_assert_eq!(
                        src.front_layout(),
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        "layer target front must be readable when sampled"
                    );
                    views.push(src.front_view());
                }
                TargetSample::Previous(src) => {
                    debug_assert_ne!(
                        src.id(),
                        target.id(),
                        "cannot sample Previous of the destination target while writing it"
                    );
                    debug_assert_eq!(
                        src.back_layout(),
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        "layer target back must be readable when sampled"
                    );
                    views.push(src.back_view());
                }
            }
        }
        self.pass(pipeline, target, uniforms, &views)
    }

    /// One pass that samples a [`VideoTexture`] instead of layer targets: this is how an overlay
    /// layer's cover pass joins the frame (`upload::CoverPass::record`). `pipeline` declares
    /// exactly one sampled input.
    pub fn draw_video(
        &mut self,
        pipeline: &Pipeline,
        target: &LayerTarget,
        uniforms: &[u8],
        video: &VideoTexture,
    ) -> Result<(), RenderError> {
        if pipeline.layouts.textures != 1 {
            return Err(RenderError::InputCountMismatch {
                expected: pipeline.layouts.textures,
                got: 1,
            });
        }
        self.pass(pipeline, target, uniforms, &[video.view()])
    }

    /// A pass that samples two [`VideoTexture`]s: HAP Alpha's colour plane plus its separate BC4
    /// alpha plane (`upload::HapCoverPass`). `pipeline` declares exactly two sampled inputs.
    pub fn draw_video_pair(
        &mut self,
        pipeline: &Pipeline,
        target: &LayerTarget,
        uniforms: &[u8],
        color: &VideoTexture,
        alpha: &VideoTexture,
    ) -> Result<(), RenderError> {
        if pipeline.layouts.textures != 2 {
            return Err(RenderError::InputCountMismatch {
                expected: pipeline.layouts.textures,
                got: 2,
            });
        }
        self.pass(pipeline, target, uniforms, &[color.view(), alpha.view()])
    }

    /// Records `video`'s pending CPU frame into this frame: its staging buffer to its image, left
    /// readable for [`Self::draw_video`].
    ///
    /// No submission and no fence wait: the copy is part of the frame's command buffer, which is
    /// what the staging buffer per frame in flight is for (`upload::VideoTexture`). Does nothing
    /// when no frame is pending; timed as the `overlay.upload` pass.
    pub fn upload_video(&mut self, video: &mut VideoTexture) -> Result<(), RenderError> {
        if self.finished {
            return Err(RenderError::FrameFinished);
        }
        let Some(staging) = video.pending_staging() else {
            return Ok(());
        };
        if self.draws as usize >= MAX_DRAWS_PER_FRAME {
            return Err(RenderError::TooManyDraws);
        }
        if staging != self.slot {
            // The pending bytes went into another frame slot's staging buffer (a recorder was
            // created and dropped between `upload` and this call). Wait for whatever that slot
            // last submitted before a copy reads those bytes.
            crate::timing::wait_timeline(self.gpu, self.gpu.slot_last_value(staging))?;
        }
        let (width, height) = (video.width(), video.height());
        let image = video.image();
        let old_layout = video.image_layout();
        let staging_buffer = video.staging_buffer(staging);
        let query = self.draws * 2;
        self.draws += 1;
        self.names.push("overlay.upload".to_string());
        unsafe {
            self.write_timestamp(query, vk::PipelineStageFlags::TOP_OF_PIPE);
            // Staging buffer -> video image, then readable by the cover pass below.
            cmd_barrier(
                &self.gpu.dev,
                self.cmd,
                image,
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
                    width,
                    height,
                    depth: 1,
                });
            self.gpu.dev.cmd_copy_buffer_to_image(
                self.cmd,
                staging_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
            cmd_barrier(
                &self.gpu.dev,
                self.cmd,
                image,
                Barrier {
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    src_stage: vk::PipelineStageFlags::TRANSFER,
                    src_access: vk::AccessFlags::TRANSFER_WRITE,
                    dst_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                    dst_access: vk::AccessFlags::SHADER_READ,
                },
            );
            self.write_timestamp(query + 1, vk::PipelineStageFlags::BOTTOM_OF_PIPE);
        }
        video.mark_copied();
        Ok(())
    }

    /// Records a clear of `target`'s back image to transparent black and swaps it, inside this
    /// frame and with no submission: a layer with no picture this frame is black instead of last
    /// frame's picture.
    pub fn clear(&mut self, target: &LayerTarget) -> Result<(), RenderError> {
        if self.finished {
            return Err(RenderError::FrameFinished);
        }
        unsafe {
            cmd_barrier(
                &self.gpu.dev,
                self.cmd,
                target.back_image(),
                to_transfer_dst(),
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
            self.gpu.dev.cmd_clear_color_image(
                self.cmd,
                target.back_image(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear,
                &[range],
            );
            cmd_barrier(
                &self.gpu.dev,
                self.cmd,
                target.back_image(),
                from_transfer_write(),
            );
        }
        target.set_back_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        target.swap();
        Ok(())
    }

    /// Writes one timestamp of this frame's query pair for pass `draws`.
    fn write_timestamp(&self, query: u32, stage: vk::PipelineStageFlags) {
        let query_pool = self.gpu.frames.borrow().slots[self.slot].query_pool;
        unsafe {
            self.gpu
                .dev
                .cmd_write_timestamp(self.cmd, stage, query_pool, query);
        }
    }

    /// The shared body of `draw` and `draw_video`: one pass from resolved image views.
    fn pass(
        &mut self,
        pipeline: &Pipeline,
        target: &LayerTarget,
        uniforms: &[u8],
        views: &[vk::ImageView],
    ) -> Result<(), RenderError> {
        if self.finished {
            return Err(RenderError::FrameFinished);
        }
        if self.draws as usize >= MAX_DRAWS_PER_FRAME {
            return Err(RenderError::TooManyDraws);
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
        let offset =
            (self.slot as u64 * MAX_DRAWS_PER_FRAME as u64 + u64::from(self.draws)) * stride;
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
        let image_infos: Vec<vk::DescriptorImageInfo> = views
            .iter()
            .map(|view| {
                vk::DescriptorImageInfo::default()
                    .image_view(*view)
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
                to_color_attachment_discard(),
            );

            let query = self.draws * 2;
            self.write_timestamp(query, vk::PipelineStageFlags::TOP_OF_PIPE);

            // Every fullscreen pass draws one unblended triangle covering the whole target with no
            // `discard`, so clearing before writing is a wasted full-target VRAM write.
            let attachment = vk::RenderingAttachmentInfo::default()
                .image_view(target.back_view())
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE);
            let attachments = [attachment];
            let rendering = vk::RenderingInfo::default()
                .render_area(extent)
                .layer_count(1)
                .color_attachments(&attachments);
            self.gpu.dev.cmd_begin_rendering(self.cmd, &rendering);
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

            self.write_timestamp(query + 1, vk::PipelineStageFlags::BOTTOM_OF_PIPE);

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
            crate::device::count_submission();
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

/// The name of the first fragment-stage `OpEntryPoint` in a SPIR-V module, if any.
pub(crate) fn fragment_entry_point(words: &[u32]) -> Option<std::ffi::CString> {
    const OP_ENTRY_POINT: u32 = 15;
    const EXECUTION_MODEL_FRAGMENT: u32 = 4;
    let mut at = 5; // the SPIR-V header is five words
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let opcode = words[at] & 0xffff;
        if count == 0 || at + count > words.len() {
            return None;
        }
        if opcode == OP_ENTRY_POINT && count >= 4 && words[at + 1] == EXECUTION_MODEL_FRAGMENT {
            let mut bytes = Vec::new();
            'name: for word in &words[at + 3..at + count] {
                for byte in word.to_le_bytes() {
                    if byte == 0 {
                        break 'name;
                    }
                    bytes.push(byte);
                }
            }
            return std::ffi::CString::new(bytes).ok();
        }
        at += count;
    }
    None
}

#[cfg(test)]
mod entry_point_tests {
    use super::fragment_entry_point;

    fn spirv(src: &str) -> Vec<u32> {
        let module = naga::front::wgsl::parse_str(src).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
            .unwrap()
    }

    #[test]
    fn reads_fs_and_fs_main() {
        let fs = spirv("@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }");
        let fs_main = spirv("@fragment fn fs_main() -> @location(0) vec4f { return vec4f(1.0); }");
        assert_eq!(fragment_entry_point(&fs).unwrap().to_str().unwrap(), "fs");
        assert_eq!(
            fragment_entry_point(&fs_main).unwrap().to_str().unwrap(),
            "fs_main"
        );
        assert_eq!(fragment_entry_point(&[]), None);
    }
}
