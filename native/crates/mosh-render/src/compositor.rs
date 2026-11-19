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
//! # Adjustment and Mask layers
//!
//! A `Mask` layer's picture is the opacity of the layer above it, and an `Adjustment` layer splits
//! the stack: the layers below it are composited, its clip's effects run on that composite, and the
//! layers above are drawn over it. Both need a pass that can do more than the plain one:
//!
//! * [`StackInput`] carries an optional mask per layer, which `compositor_stack_masked.frag.wgsl`
//!   folds into the layer's opacity (`Mask` layers draw nothing themselves).
//! * [`Compositor::composite_over`] starts the accumulator from a **base** image — the composite of
//!   the layers below an adjustment — instead of transparent black.
//! * [`Compositor::composite_into`] writes that intermediate composite into a [`LayerTarget`]
//!   rather than the final image, so the effects pass can read it and the next segment can start
//!   from it.
//!
//! A frame with no `Adjustment` and no `Mask` layer takes the plain path: one pass, the same
//! shader, the same uniform writes, byte for byte the same read-back.
//!
//! The fragment shaders' uniform blocks are
//! `{ opacity: f32, luma_threshold: f32, flags: u32, pad: u32 }` (per layer),
//! `{ count: u32, pad: u32 x3, layers: [vec4<f32>; 8] }` (the stack) and
//! `{ count: u32, flags: u32, pad: u32 x2, layers: [vec4<f32>; 8], masks: [vec4<f32>; 8] }`
//! (the masked stack).

use std::sync::Arc;

use ash::vk;
use mosh_core::show::BlendMode;

use crate::device::{align_up, Gpu, MappedBuffer};
use crate::pipeline::{Blend, DescriptorLayouts, Pipeline};
use crate::target::{
    cmd_barrier, to_color_attachment, to_color_attachment_discard, Barrier, LayerTarget,
    OutputImage, OFFSCREEN_FORMAT,
};
use crate::RenderError;

/// Bytes in the compositor's per-layer uniform block (`COMPOSITOR_UNIFORM_SIZE` in the WGSL).
pub(crate) const COMPOSITOR_UNIFORM_SIZE: usize = 16;
/// Front door for the uniform ring when the show grows past this: layers are 3-4 today.
const MAX_LAYERS: usize = 24;
/// Layers the single stack pass can take (three visualizer layers plus an overlay, with room).
pub const MAX_STACK: usize = 8;
/// Bytes of the stack pass's uniform block: a header plus one `vec4<f32>` per stack layer.
const COMPOSITOR_STACK_UNIFORM_SIZE: usize = 16 + MAX_STACK * 16;
/// Bytes of the masked stack pass's uniform block (`struct Stack` in the masked WGSL): a header,
/// one `vec4<f32>` per layer and one per mask.
const MASKED_UNIFORM_SIZE: usize = 16 + MAX_STACK * 16 + MAX_STACK * 16;
/// Sampled images one masked pass binds: the base image, [`MAX_STACK`] layers, [`MAX_STACK`]
/// masks. One more than Vulkan guarantees (16), so the pipeline is only built when the adapter
/// reports room for it.
const MASKED_TEXTURES: u32 = 1 + 2 * MAX_STACK as u32;
/// Luma below this fraction is keyed out; the overlay's luma key is refined in Task 2.3.
const LUMA_KEY_THRESHOLD: f32 = 0.08;

/// One layer for the mask- and base-capable composite passes.
///
/// `mask` is the picture of a `Mask` layer directly below this one: its luminance multiplies the
/// layer's opacity, so black hides the layer and white leaves it alone. The stacked [`BlendMode`]
/// and opacity are the show's own.
#[derive(Clone, Copy)]
pub struct StackInput<'a> {
    pub target: &'a LayerTarget,
    pub blend: BlendMode,
    pub opacity: f32,
    pub mask: Option<&'a LayerTarget>,
}

impl<'a> StackInput<'a> {
    /// A layer with no mask: what a `Visualizer` or `Overlay` layer is when nothing below it is a
    /// `Mask`.
    pub fn new(target: &'a LayerTarget, blend: BlendMode, opacity: f32) -> StackInput<'a> {
        StackInput {
            target,
            blend,
            opacity,
            mask: None,
        }
    }

    /// A layer masked by `mask`: the picture of the `Mask` layer directly below it.
    pub fn masked(
        target: &'a LayerTarget,
        blend: BlendMode,
        opacity: f32,
        mask: &'a LayerTarget,
    ) -> StackInput<'a> {
        StackInput {
            target,
            blend,
            opacity,
            mask: Some(mask),
        }
    }
}

/// Where a composite pass writes.
///
/// A frame with no `Adjustment` layer is one pass into `Output` (offscreen) or `Image` (the stage's
/// acquired swapchain image). With one, the layers below it go into a `Layer` target first, its
/// effects run on that, and the layers above are composited over it.
#[derive(Clone, Copy)]
pub enum CompositeTarget<'a> {
    /// An offscreen output image (tests, the budget bench): the pass clears it and leaves it
    /// readable, which is how `Compositor::composite` has always worked.
    Output(&'a OutputImage),
    /// A layer target: its back image is written and then left readable, so the target holds the
    /// composite (its front image) when the pass returns. A half-float intermediate of a
    /// multi-segment stack, or an `Adjustment` layer's input.
    Layer(&'a LayerTarget),
    /// A caller-owned view — the stage's acquired swapchain image — already in
    /// `COLOR_ATTACHMENT_OPTIMAL` and left there: the caller records the layout transitions, which
    /// is what `Stage::present_image` does. Nothing is cleared (the stack pass writes every pixel).
    Image {
        view: vk::ImageView,
        extent: vk::Extent2D,
        format: vk::Format,
    },
}

impl CompositeTarget<'_> {
    /// The colour format the pass's pipeline must be built for.
    pub fn format(&self) -> vk::Format {
        match self {
            CompositeTarget::Output(out) => out.format(),
            CompositeTarget::Layer(layer) => layer.format(),
            CompositeTarget::Image { format, .. } => *format,
        }
    }

    /// The size the pass renders at.
    pub fn extent(&self) -> vk::Extent2D {
        match self {
            CompositeTarget::Output(out) => vk::Extent2D {
                width: out.width(),
                height: out.height(),
            },
            CompositeTarget::Layer(layer) => vk::Extent2D {
                width: layer.width(),
                height: layer.height(),
            },
            CompositeTarget::Image { extent, .. } => *extent,
        }
    }

    fn clear_value(&self) -> vk::ClearValue {
        vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 0.0],
            },
        }
    }

    /// Whether the pass clears the target first. The stack pass writes every pixel, so an image
    /// that is already in the attachment layout (the swapchain) does not need a clear: a clear of
    /// a 1920x1080 target is another 8 MB write.
    fn load_op(&self) -> vk::AttachmentLoadOp {
        match self {
            CompositeTarget::Image { .. } => vk::AttachmentLoadOp::DONT_CARE,
            _ => vk::AttachmentLoadOp::CLEAR,
        }
    }
}

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
    /// The same pass with a mask per layer and a base image
    /// (`compositor_stack_masked.frag.wgsl`), in this compositor's format. `None` on an adapter
    /// without room for [`MASKED_TEXTURES`] sampled images in one stage.
    masked: Option<Pipeline>,
    /// The masked pass for a half-float intermediate, when this compositor writes some other
    /// format: an `Adjustment` layer's input and output are layer targets.
    masked_offscreen: Option<Pipeline>,
    /// A 1x1 target bound to the stack pass's unused inputs (the shader samples at most `count`).
    empty: LayerTarget,
    pool: vk::DescriptorPool,
    uniforms: MappedBuffer,
    allocations: crate::device::Allocations,
    stride: u64,
    /// Where the masked pass's uniform block starts: after the per-layer slices and the stack
    /// block, aligned like every other uniform range.
    masked_offset: u64,
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

        // The masked pass is `Option`: seventeen sampled images in one stage is one more than
        // Vulkan guarantees, so a device without room gets the plain path only (and an error, not
        // a validation message, when a stack with a mask or an adjustment asks for it).
        let masked_layouts = Arc::new(DescriptorLayouts::new(
            gpu,
            MASKED_UNIFORM_SIZE,
            MASKED_TEXTURES,
        )?);
        let masked = if sampled_image_limit(gpu) >= MASKED_TEXTURES {
            Some(Pipeline::with_layouts(
                gpu,
                &masked_layouts,
                crate::shaders::COMPOSITOR_STACK_MASKED_FRAG,
                format,
                None,
                "composite.masked",
            )?)
        } else {
            None
        };
        let masked_offscreen = if masked.is_some() && format != OFFSCREEN_FORMAT {
            Some(Pipeline::with_layouts(
                gpu,
                &masked_layouts,
                crate::shaders::COMPOSITOR_STACK_MASKED_FRAG,
                OFFSCREEN_FORMAT,
                None,
                "composite.masked.offscreen",
            )?)
        } else {
            None
        };

        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .descriptor_count(MAX_LAYERS as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count((MAX_LAYERS + MAX_STACK + MASKED_TEXTURES as usize) as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count((MAX_LAYERS + MAX_STACK + MASKED_TEXTURES as usize) as u32),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(MAX_LAYERS as u32)
            .pool_sizes(&sizes);
        let pool = unsafe { gpu.dev.create_descriptor_pool(&pool_info, None) }?;

        // The per-layer uniform slices are laid out with the device's required alignment, after
        // the stack pass's own block (which is written once per frame at the end of the buffer),
        // then the masked pass's block.
        let stride = align_up(
            COMPOSITOR_UNIFORM_SIZE as u64,
            gpu.min_uniform_offset_alignment(),
        );
        let masked_offset = align_up(
            stride * MAX_LAYERS as u64 + COMPOSITOR_STACK_UNIFORM_SIZE as u64,
            gpu.min_uniform_offset_alignment(),
        );
        let bytes = masked_offset + MASKED_UNIFORM_SIZE as u64;
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
            masked,
            masked_offscreen,
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
            masked_offset,
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

    /// True when this compositor can composite masks and base images: an adapter with room for
    /// [`MASKED_TEXTURES`] sampled images in one stage. Every adapter this app runs on does;
    /// without it, `Stack`'s Adjustment and Mask layers report a missing feature instead of
    /// drawing.
    pub fn supports_masks(&self) -> bool {
        self.masked.is_some()
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
        let layers: Vec<StackInput<'_>> = layers
            .iter()
            .map(|(target, blend, opacity)| StackInput::new(target, *blend, *opacity))
            .collect();
        write_stack_uniform(&self.uniforms, self.stack_offset(), &layers);
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
        bind_stack_set(self, set, &layers);
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
    ///
    /// This is the plain path: no masks, no base image, the same pass a frame with no
    /// `Adjustment` and no `Mask` layer took before those layers existed.
    pub fn composite(
        &mut self,
        layers: &[(&LayerTarget, BlendMode, f32)],
        out: &OutputImage,
    ) -> Result<(), RenderError> {
        let layers: Vec<StackInput<'_>> = layers
            .iter()
            .map(|(target, blend, opacity)| StackInput::new(target, *blend, *opacity))
            .collect();
        self.segment(None, &layers, CompositeTarget::Output(out))
    }

    /// Composites `layers` over `base` into `out`: the layers above an `Adjustment` layer drawn
    /// over the adjusted group, or a whole stack that has a `Mask` layer in it.
    ///
    /// `base` must be readable (a [`LayerTarget`] used as a pass sink already is); `None` starts
    /// from transparent black, which is what [`Self::composite`] does. `out` is an
    /// `OutputImage`, the stage's acquired swapchain image, or a layer target
    /// ([`CompositeTarget`]).
    pub fn composite_over(
        &mut self,
        base: Option<&LayerTarget>,
        layers: &[StackInput<'_>],
        out: CompositeTarget<'_>,
    ) -> Result<(), RenderError> {
        self.segment(base, layers, out)
    }

    /// Composites `layers` over `base` into `target` instead of an `OutputImage`, leaving the
    /// result in `target`'s front image (readable, compositable, samplable).
    ///
    /// This is how an `Adjustment` layer is built: the layers below it are composited into a
    /// half-float intermediate, the clip's effects run on that, and the layers above start from
    /// it.
    pub fn composite_into(
        &mut self,
        base: Option<&LayerTarget>,
        layers: &[StackInput<'_>],
        target: &LayerTarget,
    ) -> Result<(), RenderError> {
        self.segment(base, layers, CompositeTarget::Layer(target))
    }

    /// Composites one segment of a stack: `layers` over `base`, into `target`.
    ///
    /// The plain path (no base, no mask, a final image) is the single stack pass of
    /// `compositor_stack.frag.wgsl`, exactly as it was before Adjustment and Mask layers existed;
    /// everything else goes through `compositor_stack_masked.frag.wgsl`, which adds a mask input
    /// per layer and a base image to start the accumulator from.
    fn segment(
        &mut self,
        base: Option<&LayerTarget>,
        layers: &[StackInput<'_>],
        target: CompositeTarget<'_>,
    ) -> Result<(), RenderError> {
        if layers.len() > MAX_LAYERS {
            return Err(RenderError::TooManyLayers {
                max: MAX_LAYERS,
                got: layers.len(),
            });
        }
        let plain = base.is_none()
            && layers.iter().all(|layer| layer.mask.is_none())
            && !matches!(target, CompositeTarget::Layer(_));
        if !plain && layers.len() > MAX_STACK {
            return Err(RenderError::TooManyLayers {
                max: MAX_STACK,
                got: layers.len(),
            });
        }
        let extent = target.extent();
        if extent.width == 0 || extent.height == 0 {
            return Err(RenderError::ZeroSized);
        }
        let masked_pipeline = (!plain).then(|| self.masked_for(target.format()));
        let masked_pipeline = match masked_pipeline {
            Some(Some(pipeline)) => Some(pipeline),
            Some(None) => return Err(RenderError::MissingFeature("masked composite pass")),
            None => None,
        };

        // The uniform blocks. The mask pass reads one block for the whole stack; the plain pass
        // reads the per-layer slices (the per-layer fallback path) and the stack's own block.
        for (index, layer) in layers.iter().enumerate() {
            let mut block = [0u8; COMPOSITOR_UNIFORM_SIZE];
            block[0..4].copy_from_slice(&layer.opacity.clamp(0.0, 1.0).to_le_bytes());
            block[4..8].copy_from_slice(&LUMA_KEY_THRESHOLD.to_le_bytes());
            let flags: u32 = if layer.blend == BlendMode::LumaKey {
                1
            } else {
                0
            };
            block[8..12].copy_from_slice(&flags.to_le_bytes());
            let offset = index as u64 * self.stride;
            unsafe {
                self.uniforms.zero(offset, self.stride);
                self.uniforms.write(offset, &block);
            }
        }
        if plain {
            write_stack_uniform(&self.uniforms, self.stack_offset(), layers);
        } else {
            // A pass into an intermediate samples at `1 - uv.y`: the crate's fullscreen uv is
            // V-mirrored relative to `@builtin(position)`, and the intermediates must stay in the
            // layers' own orientation for the position-based effects and mosh passes. See the
            // masked shader's header.
            let flip = matches!(target, CompositeTarget::Layer(_));
            write_masked_uniform(&self.uniforms, self.masked_offset, base, layers, flip);
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

        // Single-pass stack shaders write every pixel without blending or discard, so clearing the
        // target first is a wasted full-output write. Only the >MAX_STACK per-layer fallback needs
        // an initial clear.
        let single_pass = masked_pipeline.is_some() || layers.len() <= MAX_STACK;

        // Into the attachment layout. An `OutputImage` carries its own layout; a `LayerTarget`'s
        // two images are always readable between passes; the stage's swapchain image is already
        // in the attachment layout and the caller owns the transitions.
        match &target {
            CompositeTarget::Output(out) => {
                let old_layout = if single_pass {
                    vk::ImageLayout::UNDEFINED
                } else {
                    out.layout()
                };
                if old_layout != vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
                    cmd_barrier(
                        &self.device,
                        self.cmd,
                        out.image(),
                        Barrier {
                            old_layout,
                            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                            src_stage: vk::PipelineStageFlags::FRAGMENT_SHADER
                                | vk::PipelineStageFlags::TRANSFER
                                | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                            src_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                            dst_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                            dst_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        },
                    );
                }
            }
            CompositeTarget::Layer(layer) => {
                cmd_barrier(
                    &self.device,
                    self.cmd,
                    layer.back_image(),
                    if single_pass {
                        to_color_attachment_discard()
                    } else {
                        to_color_attachment()
                    },
                );
            }
            CompositeTarget::Image { .. } => {}
        }

        let area = vk::Rect2D::default().extent(extent);
        let viewport = vk::Viewport::default()
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let load_op = if single_pass {
            vk::AttachmentLoadOp::DONT_CARE
        } else {
            target.load_op()
        };
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(match &target {
                CompositeTarget::Output(out) => out.view(),
                CompositeTarget::Layer(layer) => layer.back_view(),
                CompositeTarget::Image { view, .. } => *view,
            })
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(load_op)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(target.clear_value());
        let attachments = [attachment];

        unsafe {
            self.device.cmd_write_timestamp(
                self.cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool,
                0,
            );
            let rendering = vk::RenderingInfo::default()
                .render_area(area)
                .layer_count(1)
                .color_attachments(&attachments);
            self.device.cmd_begin_rendering(self.cmd, &rendering);
        }

        if let Some(pipeline) = masked_pipeline {
            let set = unsafe {
                let set_layouts = [pipeline.set_layout()];
                let allocate = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.pool)
                    .set_layouts(&set_layouts);
                self.device.allocate_descriptor_sets(&allocate)?[0]
            };
            bind_masked_set(self, set, base, layers);
            unsafe {
                self.device.cmd_bind_pipeline(
                    self.cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.raw(),
                );
                self.device.cmd_set_viewport(self.cmd, 0, &[viewport]);
                self.device.cmd_set_scissor(self.cmd, 0, &[area]);
                self.device.cmd_bind_descriptor_sets(
                    self.cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.pipeline_layout(),
                    0,
                    &[set],
                    &[self.masked_offset as u32],
                );
                self.device.cmd_draw(self.cmd, 3, 1, 0, 0);
            }
        } else if layers.len() <= MAX_STACK {
            self.draw_stack(layers, area, viewport)?;
        } else {
            for (index, layer) in layers.iter().enumerate() {
                let pipeline = &self.pipelines[blend_slot(layer.blend)];
                debug_assert_eq!(
                    layer.target.front_layout(),
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
                    .image_view(layer.target.front_view())
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
                    self.device.cmd_set_scissor(self.cmd, 0, &[area]);
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

        let output_barrier = || Barrier {
            old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_stage: vk::PipelineStageFlags::FRAGMENT_SHADER
                | vk::PipelineStageFlags::TRANSFER,
            dst_access: vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ,
        };
        match &target {
            CompositeTarget::Output(out) => {
                cmd_barrier(&self.device, self.cmd, out.image(), output_barrier());
            }
            CompositeTarget::Layer(layer) => {
                cmd_barrier(
                    &self.device,
                    self.cmd,
                    layer.back_image(),
                    output_barrier(),
                );
                layer.set_back_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            }
            CompositeTarget::Image { .. } => {}
        }

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
        match &target {
            CompositeTarget::Output(out) => {
                out.set_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            }
            CompositeTarget::Layer(layer) => layer.swap(),
            CompositeTarget::Image { .. } => {}
        }
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

    /// The masked pipeline for `format`, if this device can run one.
    fn masked_for(&self, format: vk::Format) -> Option<&Pipeline> {
        if format == self.format {
            self.masked.as_ref()
        } else if format == OFFSCREEN_FORMAT {
            self.masked_offscreen.as_ref()
        } else {
            None
        }
    }

    /// Offset of the stack pass's uniform block: after the per-layer slices.
    fn stack_offset(&self) -> u64 {
        self.stride * MAX_LAYERS as u64
    }

    /// Records the whole layer stack as one pass: every layer sampled once, the blend chain in
    /// registers, one write of the output image.
    fn draw_stack(
        &self,
        layers: &[StackInput<'_>],
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

/// How many sampled images one shader stage may bind on this device.
fn sampled_image_limit(gpu: &Gpu) -> u32 {
    unsafe { gpu.instance.get_physical_device_properties(gpu.physical) }
        .limits
        .max_per_stage_descriptor_sampled_images
}

/// The `vec4` the stack shaders take per layer: (opacity, mode, luma threshold, flip V).
fn layer_params(layer: &StackInput<'_>) -> [f32; 4] {
    [
        layer.opacity.clamp(0.0, 1.0),
        blend_mode_value(layer.blend),
        if layer.blend == BlendMode::LumaKey {
            LUMA_KEY_THRESHOLD
        } else {
            0.0
        },
        0.0,
    ]
}

/// Fills the stack pass's uniform block: the count, then (opacity, mode, luma threshold, flip)
/// per layer.
fn write_stack_uniform(uniforms: &MappedBuffer, offset: u64, layers: &[StackInput<'_>]) {
    let mut block = [0u8; COMPOSITOR_STACK_UNIFORM_SIZE];
    block[0..4].copy_from_slice(&(layers.len() as u32).to_le_bytes());
    for (index, layer) in layers.iter().enumerate() {
        let at = 16 + index * 16;
        for (slot, value) in layer_params(layer).iter().enumerate() {
            block[at + slot * 4..at + slot * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    unsafe {
        uniforms.zero(offset, COMPOSITOR_STACK_UNIFORM_SIZE as u64);
        uniforms.write(offset, &block);
    }
}

/// Fills the masked pass's uniform block: the count, the flags, then (opacity, mode, luma
/// threshold, flip) per layer and the mask input slot per layer (-1 when it has no mask).
///
/// `flip` sets bit 1 of the flags: the pass samples at `1 - uv.y`, which a pass that writes an
/// intermediate does (see the shader's header).
fn write_masked_uniform(
    uniforms: &MappedBuffer,
    offset: u64,
    base: Option<&LayerTarget>,
    layers: &[StackInput<'_>],
    flip: bool,
) {
    let mut block = [0u8; MASKED_UNIFORM_SIZE];
    block[0..4].copy_from_slice(&(layers.len() as u32).to_le_bytes());
    let flags: u32 = if base.is_some() { 1 } else { 0 } | if flip { 2 } else { 0 };
    block[4..8].copy_from_slice(&flags.to_le_bytes());
    for (index, layer) in layers.iter().enumerate() {
        let at = 16 + index * 16;
        for (slot, value) in layer_params(layer).iter().enumerate() {
            block[at + slot * 4..at + slot * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        let mask_at = 16 + MAX_STACK * 16 + index * 16;
        let slot = if layer.mask.is_some() { index as f32 } else { -1.0 };
        block[mask_at..mask_at + 4].copy_from_slice(&slot.to_le_bytes());
    }
    unsafe {
        uniforms.zero(offset, MASKED_UNIFORM_SIZE as u64);
        uniforms.write(offset, &block);
    }
}

/// Writes the stack pass's descriptor set: its uniform slice and one image per layer, with the
/// compositor's 1x1 target standing in for the slots past `layers.len()`.
fn bind_stack_set(
    compositor: &Compositor,
    set: vk::DescriptorSet,
    layers: &[StackInput<'_>],
) {
    let buffer_info = [vk::DescriptorBufferInfo::default()
        .buffer(compositor.uniforms.buffer)
        .offset(0)
        .range(COMPOSITOR_STACK_UNIFORM_SIZE as u64)];
    let empty = compositor.empty.front_view();
    let mut images: Vec<vk::DescriptorImageInfo> = Vec::with_capacity(MAX_STACK);
    for index in 0..MAX_STACK {
        let view = match layers.get(index) {
            Some(layer) => {
                debug_assert_eq!(
                    layer.target.front_layout(),
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    "layers must be readable when composited"
                );
                layer.target.front_view()
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
    unsafe {
        compositor.device.update_descriptor_sets(&writes, &[]);
    }
}

/// Writes the masked pass's descriptor set: its uniform slice, the base image (or the 1x1
/// stand-in), one image per layer and one per mask.
fn bind_masked_set(
    compositor: &Compositor,
    set: vk::DescriptorSet,
    base: Option<&LayerTarget>,
    layers: &[StackInput<'_>],
) {
    let buffer_info = [vk::DescriptorBufferInfo::default()
        .buffer(compositor.uniforms.buffer)
        .offset(0)
        .range(MASKED_UNIFORM_SIZE as u64)];
    let empty = compositor.empty.front_view();
    let readable = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    let mut views: Vec<vk::ImageView> = Vec::with_capacity(MASKED_TEXTURES as usize);
    views.push(match base {
        Some(base) => {
            debug_assert_eq!(
                base.front_layout(),
                readable,
                "the base image must be readable when composited over"
            );
            base.front_view()
        }
        None => empty,
    });
    for index in 0..MAX_STACK {
        views.push(match layers.get(index) {
            Some(layer) => {
                debug_assert_eq!(
                    layer.target.front_layout(),
                    readable,
                    "layers must be readable when composited"
                );
                layer.target.front_view()
            }
            None => empty,
        });
    }
    for index in 0..MAX_STACK {
        views.push(match layers.get(index).and_then(|layer| layer.mask) {
            Some(mask) => {
                debug_assert_eq!(
                    mask.front_layout(),
                    readable,
                    "a mask must be readable when composited"
                );
                mask.front_view()
            }
            None => empty,
        });
    }
    let images: Vec<vk::DescriptorImageInfo> = views
        .iter()
        .map(|view| {
            vk::DescriptorImageInfo::default()
                .image_view(*view)
                .image_layout(readable)
        })
        .collect();
    let mut writes = Vec::with_capacity(1 + images.len());
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
    unsafe {
        compositor.device.update_descriptor_sets(&writes, &[]);
    }
}
