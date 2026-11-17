//! The stage output: a surface on a window, a swapchain, and the recovery policy that keeps the
//! show running when the swapchain goes out of date or the surface is lost (a monitor unplugged
//! mid-set, Review Focus item 5).
//!
//! The recovery logic lives in `SwapchainPolicy`, which takes Vulkan results and never touches a
//! window â€” `tests/offscreen.rs::surface_lost_recreates_swapchain` drives it directly. `Stage`
//! wraps it around real swapchain objects, which need a window and are therefore exercised by
//! the app (Task 2.6) rather than here.
//!
//! The stage never shows a dialog and never panics on a lost surface: `present` reports
//! `PresentResult { surface_lost }` and the app moves or hides the window.

use ash::vk;
use mosh_core::show::BlendMode;
use raw_window_handle::RawWindowHandle;

use crate::compositor::Compositor;
use crate::device::{Gpu, FRAMES_IN_FLIGHT};
use crate::target::{cmd_barrier, Barrier, LayerTarget};
use crate::timing::GpuTimings;
use crate::win::module_handle;
use crate::RenderError;

/// A monitor the stage can be sent to. The app maps winit's monitors to these ids; the renderer
/// only carries the choice, it does not switch display modes (Task 2.6 owns the window).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MonitorId(pub u32);

/// The colour format the stage asks a surface for (the first one in this format wins, then the
/// surface's own). It is the format the compositor writes, and the budget bench measures.
pub const STAGE_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;

/// What the stage reports after a frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PresentResult {
    /// The surface was lost (or rebuilt) during this frame: the app should re-place the window.
    pub surface_lost: bool,
}

/// What to do after a swapchain result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwapchainAction {
    Continue,
    SkipFrame,
    Recreate { rebuild_surface: bool },
}

/// Swapchain and surface recovery, without any Vulkan objects.
///
/// Fed by `Stage` from real `acquire_next_image` / `queue_present` results, and by tests with
/// injected results.
#[derive(Clone, Debug, Default)]
pub struct SwapchainPolicy {
    valid: bool,
    needs_recreate: bool,
    rebuild_surface: bool,
    pending_surface_rebuild: bool,
    surface_lost_seen: bool,
    extent: [u32; 2],
    recreations: u32,
    surface_recreations: u32,
    skipped_frames: u32,
}

impl SwapchainPolicy {
    pub fn new() -> SwapchainPolicy {
        SwapchainPolicy::default()
    }

    /// True while there is a swapchain that can be acquired from.
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// True when the next `present` must rebuild before acquiring.
    pub fn needs_recreate(&self) -> bool {
        self.needs_recreate
    }

    /// The size the swapchain was last built at.
    pub fn extent(&self) -> [u32; 2] {
        self.extent
    }

    pub fn recreations(&self) -> u32 {
        self.recreations
    }

    pub fn surface_recreations(&self) -> u32 {
        self.surface_recreations
    }

    pub fn skipped_frames(&self) -> u32 {
        self.skipped_frames
    }

    /// True once per loss; the stage forwards it in `PresentResult` and then clears it.
    pub fn take_surface_lost(&mut self) -> bool {
        std::mem::take(&mut self.surface_lost_seen)
    }

    fn mark_recreate(&mut self, rebuild_surface: bool) -> SwapchainAction {
        self.valid = false;
        self.needs_recreate = true;
        self.rebuild_surface |= rebuild_surface;
        self.surface_lost_seen |= rebuild_surface;
        SwapchainAction::Recreate { rebuild_surface }
    }

    /// `vkAcquireNextImageKHR`'s result.
    pub fn acquire_result(&mut self, result: Result<(), vk::Result>) -> SwapchainAction {
        match result {
            Ok(()) => SwapchainAction::Continue,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.mark_recreate(false),
            Err(vk::Result::ERROR_SURFACE_LOST_KHR) => self.mark_recreate(true),
            Err(vk::Result::TIMEOUT) | Err(vk::Result::NOT_READY) => {
                self.skipped_frames += 1;
                SwapchainAction::SkipFrame
            }
            // Device lost and friends: keep the show alive, try to rebuild; if the device is
            // really gone the next submit fails and the app restarts the renderer.
            Err(_) => self.mark_recreate(false),
        }
    }

    /// `vkQueuePresentKHR`'s result.
    pub fn present_result(&mut self, result: Result<(), vk::Result>) -> SwapchainAction {
        match result {
            Ok(()) => SwapchainAction::Continue,
            Err(vk::Result::ERROR_SURFACE_LOST_KHR) => self.mark_recreate(true),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                self.mark_recreate(false)
            }
            Err(_) => self.mark_recreate(false),
        }
    }

    /// The window was resized (or a monitor was plugged in or removed).
    pub fn resize(&mut self, width: u32, height: u32) -> SwapchainAction {
        if width == 0 || height == 0 {
            // Minimized, or the monitor went away mid-set: nothing to build at 0x0.
            self.skipped_frames += 1;
            self.needs_recreate = true;
            self.valid = false;
            return SwapchainAction::SkipFrame;
        }
        self.extent = [width, height];
        self.mark_recreate(false)
    }

    /// Starts a rebuild. Returns true when the surface itself must be recreated first (the
    /// window may have moved to another monitor, or lost its surface).
    pub fn begin_recreate(&mut self) -> bool {
        self.needs_recreate = false;
        self.pending_surface_rebuild = std::mem::take(&mut self.rebuild_surface);
        self.pending_surface_rebuild
    }

    /// Finishes a rebuild at `extent`. A zero extent (minimized, monitor gone) leaves the policy
    /// invalid and scheduled for another attempt, instead of creating a 0x0 swapchain.
    pub fn complete_recreate(&mut self, extent: [u32; 2]) -> bool {
        if extent[0] == 0 || extent[1] == 0 {
            self.valid = false;
            self.needs_recreate = true;
            self.skipped_frames += 1;
            return false;
        }
        self.extent = extent;
        self.valid = true;
        self.needs_recreate = false;
        self.recreations += 1;
        if self.pending_surface_rebuild {
            self.pending_surface_rebuild = false;
            self.surface_recreations += 1;
        }
        true
    }
}

/// The swapchain and its views.
struct Swapchain {
    handle: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    /// Layout each image was left in; `UNDEFINED` after a present.
    layouts: Vec<vk::ImageLayout>,
}

/// The stage output: one fullscreen, borderless window on the secondary monitor (the app places
/// the window; this type presents into it).
pub struct Stage {
    device: ash::Device,
    physical: vk::PhysicalDevice,
    queue: vk::Queue,

    surface_loader: ash::khr::surface::Instance,
    win32_loader: ash::khr::win32_surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    hwnd: isize,
    hinstance: isize,
    swapchain: Option<Swapchain>,
    policy: SwapchainPolicy,
    monitor: Option<MonitorId>,
    /// Built for the swapchain's colour format: it composites straight into the presented image.
    compositor: Option<Compositor>,
    image_available: [vk::Semaphore; FRAMES_IN_FLIGHT],
    render_finished: Vec<vk::Semaphore>,
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    frame_index: usize,
    timings: GpuTimings,
    last_error: Option<String>,
}

impl Stage {
    /// Creates the surface and the first swapchain for `window`.
    ///
    /// Returns `Err` instead of the plan's bare `Stage` when the surface cannot be created: the
    /// app (Task 2.6) needs the failure to fall back to a borderless window on the laptop screen
    /// or hide the stage, and "no stage output" must never crash a set.
    pub fn new(
        gpu: &Gpu,
        window: RawWindowHandle,
        fullscreen_monitor: Option<MonitorId>,
    ) -> Result<Stage, RenderError> {
        if !gpu.surface_extensions {
            return Err(RenderError::SurfaceUnsupported);
        }
        let (hwnd, hinstance) = win32_handles(window)?;

        let win32_loader = ash::khr::win32_surface::Instance::new(&gpu.entry, &gpu.instance);
        let surface_loader = ash::khr::surface::Instance::new(&gpu.entry, &gpu.instance);
        let surface_info = vk::Win32SurfaceCreateInfoKHR::default()
            .hwnd(hwnd)
            .hinstance(hinstance);
        let surface = unsafe { win32_loader.create_win32_surface(&surface_info, None) }?;
        let supported = unsafe {
            surface_loader.get_physical_device_surface_support(
                gpu.physical,
                gpu.queue_family,
                surface,
            )?
        };
        if !supported {
            unsafe { surface_loader.destroy_surface(surface, None) };
            return Err(RenderError::SurfaceUnsupported);
        }

        let swapchain_loader = ash::khr::swapchain::Device::new(&gpu.instance, &gpu.dev);
        let mut stage = Stage {
            device: gpu.dev.clone(),
            physical: gpu.physical,
            queue: gpu.queue,

            surface_loader,
            win32_loader,
            swapchain_loader,
            surface,
            hwnd,
            hinstance,
            swapchain: None,
            policy: SwapchainPolicy::new(),
            monitor: fullscreen_monitor,
            compositor: None,
            image_available: [vk::Semaphore::null(); FRAMES_IN_FLIGHT],
            render_finished: Vec::new(),
            cmd_pool: vk::CommandPool::null(),
            cmd: vk::CommandBuffer::null(),
            frame_index: 0,
            timings: GpuTimings::default(),
            last_error: None,
        };

        stage.image_available = [
            unsafe { gpu.dev.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? },
            unsafe { gpu.dev.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? },
        ];

        stage.cmd_pool = unsafe {
            gpu.dev.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(gpu.queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };
        stage.cmd = unsafe {
            gpu.dev.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(stage.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?
        }[0];

        if !stage.recreate(gpu) {
            let error = stage
                .last_error
                .clone()
                .map(RenderError::Stage)
                .unwrap_or(RenderError::SurfaceUnsupported);
            stage.destroy();
            return Err(error);
        }
        Ok(stage)
    }

    /// Which monitor the stage intends to cover (the app places the window).
    pub fn monitor(&self) -> Option<MonitorId> {
        self.monitor
    }

    /// The swapchain size in pixels.
    pub fn extent(&self) -> [u32; 2] {
        match &self.swapchain {
            Some(swapchain) => [swapchain.extent.width, swapchain.extent.height],
            None => [0, 0],
        }
    }

    /// The size the compositor works at: the swapchain's. Kept for the stats overlay and for
    /// callers that need to know what the presented frame's size is.
    pub fn composite_extent(&self) -> [u32; 2] {
        self.extent()
    }

    /// Timings of the last presented frame.
    pub fn timings(&self) -> &GpuTimings {
        &self.timings
    }

    /// The last internal failure, for the stats overlay. The stage itself keeps presenting.
    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    /// Tell the stage the window changed size. Takes effect on the next `present`.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.policy.resize(width, height);
    }

    /// Composites `recorder`'s layers and presents one frame.
    ///
    /// The recorder's GPU work is submitted and waited for first, so the layers are readable
    /// when the compositor samples them. The composite itself is the first pass of the present
    /// submission and writes the acquired swapchain image directly: no offscreen copy of the
    /// frame is made, so a four-layer frame writes the output once (V-budget: it used to be a
    /// full 16F composite plus a blit, 1.2 ms at 1080p).
    pub fn present(&mut self, recorder: crate::pipeline::FrameRecorder<'_>) -> PresentResult {
        let gpu = recorder.gpu();
        let layers: Vec<(&LayerTarget, BlendMode, f32)> = recorder.layers().to_vec();
        let ticket = recorder.submit();
        self.timings = gpu.wait_frame(ticket);

        if !self.ensure_swapchain(gpu) {
            return PresentResult {
                surface_lost: self.policy.take_surface_lost(),
            };
        }

        let image_index = match self.acquire(gpu) {
            Some(index) => index,
            None => {
                return PresentResult {
                    surface_lost: self.policy.take_surface_lost(),
                }
            }
        };

        if let Err(error) = self.present_image(image_index, &layers) {
            self.last_error = Some(error.to_string());
        }
        self.frame_index = (self.frame_index + 1) % FRAMES_IN_FLIGHT;
        PresentResult {
            surface_lost: self.policy.take_surface_lost(),
        }
    }

    fn ensure_swapchain(&mut self, gpu: &Gpu) -> bool {
        if self.policy.is_valid() && !self.policy.needs_recreate() {
            return true;
        }
        self.recreate(gpu)
    }

    /// Rebuilds the swapchain (and the surface, when the surface was lost). Returns false when
    /// there is nothing presentable (window minimized, monitor gone).
    fn recreate(&mut self, gpu: &Gpu) -> bool {
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        let rebuild_surface = self.policy.begin_recreate();
        self.destroy_swapchain();
        if rebuild_surface {
            // The window handle is still valid; the surface object is not.
            unsafe {
                self.surface_loader.destroy_surface(self.surface, None);
            }
            self.surface = vk::SurfaceKHR::null();
            let info = vk::Win32SurfaceCreateInfoKHR::default()
                .hwnd(self.hwnd)
                .hinstance(self.hinstance);
            match unsafe { self.win32_loader.create_win32_surface(&info, None) } {
                Ok(surface) => self.surface = surface,
                Err(error) => {
                    self.last_error = Some(RenderError::Vulkan(error).to_string());
                    self.policy.complete_recreate([0, 0]);
                    return false;
                }
            }
        }

        let built = (|| -> Result<Swapchain, RenderError> {
            let capabilities = unsafe {
                self.surface_loader
                    .get_physical_device_surface_capabilities(self.physical, self.surface)?
            };
            let formats = unsafe {
                self.surface_loader
                    .get_physical_device_surface_formats(self.physical, self.surface)?
            };
            let format = formats
                .iter()
                .find(|format| format.format == STAGE_FORMAT)
                .or_else(|| formats.first())
                .copied()
                .ok_or(RenderError::SurfaceUnsupported)?;

            let extent = if capabilities.current_extent.width != u32::MAX {
                capabilities.current_extent
            } else {
                let wanted = self.policy.extent();
                let wanted = if wanted[0] == 0 || wanted[1] == 0 {
                    [1280, 720]
                } else {
                    wanted
                };
                vk::Extent2D {
                    width: wanted[0].clamp(
                        capabilities.min_image_extent.width,
                        capabilities.max_image_extent.width,
                    ),
                    height: wanted[1].clamp(
                        capabilities.min_image_extent.height,
                        capabilities.max_image_extent.height,
                    ),
                }
            };
            if extent.width == 0 || extent.height == 0 {
                return Err(RenderError::ZeroSized);
            }

            let mut image_count = capabilities.min_image_count + 1;
            if capabilities.max_image_count > 0 {
                image_count = image_count.min(capabilities.max_image_count);
            }
            let composite_alpha = if capabilities
                .supported_composite_alpha
                .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
            {
                vk::CompositeAlphaFlagsKHR::OPAQUE
            } else {
                vk::CompositeAlphaFlagsKHR::INHERIT
            };
            let info = vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(capabilities.current_transform)
                .composite_alpha(composite_alpha)
                .present_mode(vk::PresentModeKHR::FIFO)
                .clipped(true);
            let handle = unsafe { self.swapchain_loader.create_swapchain(&info, None) }?;
            let images = unsafe {
                self.swapchain_loader
                    .get_swapchain_images(handle)?
            };
            let mut views = Vec::with_capacity(images.len());
            for image in &images {
                views.push(unsafe { gpu.dev.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(*image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format.format)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1),
                        ),
                    None,
                )? });
            }
            Ok(Swapchain {
                handle,
                format: format.format,
                extent,
                layouts: vec![vk::ImageLayout::UNDEFINED; images.len()],
                images,
                views,
            })
        })();

        let swapchain = match built {
            Ok(swapchain) => swapchain,
            Err(error) => {
                self.last_error = Some(error.to_string());
                self.policy.complete_recreate([0, 0]);
                return false;
            }
        };

        // Render-finished semaphores are per swapchain image.
        self.destroy_render_finished();
        let mut render_finished = Vec::with_capacity(swapchain.images.len());
        for _ in 0..swapchain.images.len() {
            match unsafe {
                gpu.dev
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            } {
                Ok(semaphore) => render_finished.push(semaphore),
                Err(error) => {
                    self.last_error = Some(RenderError::Vulkan(error).to_string());
                    self.swapchain = Some(swapchain);
                    self.render_finished = render_finished;
                    self.destroy_swapchain();
                    self.policy.complete_recreate([0, 0]);
                    return false;
                }
            }
        }
        self.render_finished = render_finished;

        // The compositor writes this swapchain's colour format, so the composite pass is the
        // present pass: no offscreen copy of the frame.
        let format = swapchain.format;
        let extent = [swapchain.extent.width, swapchain.extent.height];
        self.swapchain = Some(swapchain);

        let compositor_matches = self
            .compositor
            .as_ref()
            .map(|compositor| compositor.format() == format)
            .unwrap_or(false);
        if !compositor_matches {
            match Compositor::new_for_format(gpu, format) {
                Ok(compositor) => self.compositor = Some(compositor),
                Err(error) => {
                    self.last_error = Some(error.to_string());
                    self.destroy_swapchain();
                    self.policy.complete_recreate([0, 0]);
                    return false;
                }
            }
        }
        self.policy.complete_recreate(extent)
    }

    fn acquire(&mut self, gpu: &Gpu) -> Option<u32> {
        for _ in 0..2 {
            let swapchain = self.swapchain.as_ref()?;
            let semaphore = self.image_available[self.frame_index];
            let result = unsafe {
                self.swapchain_loader
                    .acquire_next_image(swapchain.handle, u64::MAX, semaphore, vk::Fence::null())
            };
            let flattened: Result<u32, vk::Result> = result.map(|(index, _)| index);
            match self.policy.acquire_result(flattened.map(|_| ())) {
                SwapchainAction::Continue => return flattened.ok(),
                SwapchainAction::Recreate { .. } => {
                    if !self.recreate(gpu) {
                        return None;
                    }
                }
                SwapchainAction::SkipFrame => return None,
            }
        }
        None
    }

    /// Composites `layers` into the acquired image and presents it.
    ///
    /// Compositing failed (or there is no compositor): the image is cleared to black and
    /// presented anyway, so an acquired image never starves the swapchain and the stage never
    /// freezes while the app recovers.
    fn present_image(
        &mut self,
        image_index: u32,
        layers: &[(&LayerTarget, BlendMode, f32)],
    ) -> Result<(), RenderError> {
        let (handle, extent, view, image, old_layout) = {
            let swapchain = self.swapchain.as_ref().ok_or(RenderError::ZeroSized)?;
            let index = image_index as usize;
            (
                swapchain.handle,
                swapchain.extent,
                *swapchain.views.get(index).ok_or(RenderError::ZeroSized)?,
                *swapchain.images.get(index).ok_or(RenderError::ZeroSized)?,
                swapchain.layouts.get(index).copied().unwrap_or(vk::ImageLayout::UNDEFINED),
            )
        };

        unsafe {
            self.device
                .reset_command_pool(self.cmd_pool, vk::CommandPoolResetFlags::empty())?;
            self.device.begin_command_buffer(
                self.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            cmd_barrier(
                &self.device,
                self.cmd,
                image,
                Barrier {
                    old_layout,
                    new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    src_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                    src_access: vk::AccessFlags::empty(),
                    dst_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    dst_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                },
            );
        }

        // The composite pass writes the swapchain image directly, cleared to black first.
        let composited = match self.compositor.as_ref() {
            Some(compositor) => compositor
                .record_into(self.cmd, layers, view, extent)
                .inspect_err(|error| self.last_error = Some(error.to_string()))
                .is_ok(),
            None => false,
        };
        if !composited {
            // A black clear beats a frozen window: one dynamic-rendering pass with a clear and no
            // draw. It uses the layout the image is already in.
            let clear = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            };
            let attachment = vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear);
            let attachments = [attachment];
            unsafe {
                self.device.cmd_begin_rendering(
                    self.cmd,
                    &vk::RenderingInfo::default()
                        .render_area(vk::Rect2D::default().extent(extent))
                        .layer_count(1)
                        .color_attachments(&attachments),
                );
                self.device.cmd_end_rendering(self.cmd);
            }
        }

        unsafe {
            cmd_barrier(
                &self.device,
                self.cmd,
                image,
                Barrier {
                    old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    src_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    src_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dst_stage: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    dst_access: vk::AccessFlags::empty(),
                },
            );
            self.device.end_command_buffer(self.cmd)?;
        }

        let wait_semaphores = [self.image_available[self.frame_index]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self
            .render_finished
            .get(image_index as usize)
            .copied()
            .ok_or(RenderError::ZeroSized)?];
        let command_buffers = [self.cmd];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);
        unsafe {
            crate::device::count_submission();
            self.device
                .queue_submit(self.queue, &[submit], vk::Fence::null())?;
        }

        let swapchains = [handle];
        let indices = [image_index];
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&indices);
        let result = unsafe { self.swapchain_loader.queue_present(self.queue, &present) };

        // After a present the image's layout is undefined again.
        if let Some(layout) = self
            .swapchain
            .as_mut()
            .and_then(|swapchain| swapchain.layouts.get_mut(image_index as usize))
        {
            *layout = vk::ImageLayout::UNDEFINED;
        }
        // The action is "recreate before the next acquire"; the policy remembers that itself.
        let _ = self.policy.present_result(result.map(|_| ()));
        Ok(())
    }

    fn destroy_swapchain(&mut self) {
        if let Some(swapchain) = self.swapchain.take() {
            unsafe {
                for view in &swapchain.views {
                    self.device.destroy_image_view(*view, None);
                }
                self.swapchain_loader
                    .destroy_swapchain(swapchain.handle, None);
            }
        }
        self.destroy_render_finished();
    }

    fn destroy_render_finished(&mut self) {
        unsafe {
            for semaphore in self.render_finished.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
        }
    }

    fn destroy(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        self.destroy_swapchain();
        self.compositor = None;
        unsafe {
            for semaphore in self.image_available {
                if semaphore != vk::Semaphore::null() {
                    self.device.destroy_semaphore(semaphore, None);
                }
            }
            if self.cmd_pool != vk::CommandPool::null() {
                self.device.destroy_command_pool(self.cmd_pool, None);
            }
            if self.surface != vk::SurfaceKHR::null() {
                self.surface_loader.destroy_surface(self.surface, None);
                self.surface = vk::SurfaceKHR::null();
            }
        }
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        self.destroy();
    }
}

fn win32_handles(window: RawWindowHandle) -> Result<(isize, isize), RenderError> {
    match window {
        RawWindowHandle::Win32(handle) => {
            let hinstance = handle
                .hinstance
                .map(|hinstance| hinstance.get())
                .unwrap_or_else(module_handle);
            Ok((handle.hwnd.get(), hinstance))
        }
        _ => Err(RenderError::UnsupportedWindow),
    }
}
