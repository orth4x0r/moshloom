//! Instance, explicit GPU selection, the device, queues, the timeline semaphore, the two frame
//! slots, and the small allocation helpers the rest of the crate uses.
//!
//! Design (`docs/superpowers/specs/2026-11-17-native-port-design.md`, "Rendering"):
//! one device on the **integrated** GPU, chosen explicitly, because it drives the laptop panel
//! and the external output and so needs no copy between GPUs. The discrete GPU is only ever
//! picked when the caller asks for it by preference or by name.
//!
//! Validation layers are on in debug builds and every message is collected, so a test can
//! assert that nothing was reported (`Gpu::validation_messages`).

use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ash::vk;

use crate::RenderError;

/// Frames in flight: two command buffers, two query pools, two descriptor pools, one timeline
/// semaphore. Never more: the stage is a 60 fps fullscreen window, not a throughput benchmark.
pub const FRAMES_IN_FLIGHT: usize = 2;

/// Draws recorded into one frame, shared by every layer. A layer is several passes (a scene, its
/// helpers such as 12 reaction-diffusion steps, mosh 2-3, effects 1, transition 1-2), and a fluid
/// layer alone is 9 + up to 30 pressure iterations; 256 fits three heavy layers plus the overlay
/// and the compositor with room to spare (the uniform ring is 256 x 2 x stride, a few hundred KB).
pub const MAX_DRAWS_PER_FRAME: usize = 256;

/// Upper bound on a per-draw uniform block. The web app's per-layer uniform block (T, hue,
/// vari, dens, aux0..2, l0..l2, mel[24], palT/caT/melT/rdT, p0..p7) is 192 bytes; the text
/// layer's is already 1008, so the cap is one field away from being hit and sits at 4096.
///
/// Vulkan guarantees `maxUniformBufferRange` of at least 16384, so a 4096-byte range is safe on
/// every device. The cost is the ring in [`Gpu::new`]: its stride is
/// `align_up(this, minUniformBufferOffsetAlignment)` and Vulkan caps that alignment at 256, so the
/// stride is 4096 on every device: 4096 bytes per draw, 256 draws per frame slot and
/// [`FRAMES_IN_FLIGHT`] slots, 2 MiB of host-visible memory held once, up from 512 KiB at 1024.
pub const MAX_UNIFORM_BYTES: u64 = 4096;

const QUERIES_PER_FRAME: u32 = 2 * MAX_DRAWS_PER_FRAME as u32;

/// Which GPU to render on.
///
/// The reference machine has an integrated `AMD Radeon(TM) Graphics` (the one that drives the
/// panel and the stage output) and a discrete `AMD Radeon RX 6700S` (never the default).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuPreference {
    Integrated,
    Discrete,
    ByName(String),
}

impl GpuPreference {
    fn describe(&self) -> String {
        match self {
            GpuPreference::Integrated => "integrated".to_string(),
            GpuPreference::Discrete => "discrete".to_string(),
            GpuPreference::ByName(name) => format!("name containing {name:?}"),
        }
    }
}

/// How many `vkQueueSubmit` calls this crate has made (frames, one-shot uploads, clears,
/// readbacks, presents). A test can prove that a per-frame pass is *inside* the frame's submit
/// instead of in a submission of its own (`mosh-visuals`' `tests/overlay.rs`: an overlay copy plus
/// its cover pass are one submission). A wait on a fence or a timeline semaphore is what those
/// extra submissions cost, so the count is the thing worth pinning.
static QUEUE_SUBMISSIONS: AtomicU64 = AtomicU64::new(0);

/// `vkQueueSubmit` calls this crate has made so far, process-wide.
pub fn submission_count() -> u64 {
    QUEUE_SUBMISSIONS.load(Ordering::Relaxed)
}

pub(crate) fn count_submission() {
    QUEUE_SUBMISSIONS.fetch_add(1, Ordering::Relaxed);
}

/// Live device memory the renderer holds, in bytes and allocations.
///
/// Every image and buffer this crate allocates is added here and every `Drop` that frees one
/// subtracts it again, so [`Gpu::allocated_bytes`] is a *live* figure: on an integrated GPU it is
/// system RAM the process is holding, which is what the design's "< 500 MB" memory budget is
/// about. The per-target numbers a budget test prints come from this same counter.
///
/// Keyed by the `vk::DeviceMemory` handle, so a release never has to remember a size and the
/// figure is the driver's own aligned allocation size, not the requested byte count.
#[derive(Clone, Default)]
pub(crate) struct Allocations(Rc<RefCell<AllocationsInner>>);

#[derive(Default)]
struct AllocationsInner {
    live: Vec<(vk::DeviceMemory, u64)>,
    bytes: u64,
}

impl Allocations {
    pub(crate) fn add(&self, memory: vk::DeviceMemory, bytes: u64) {
        let mut inner = self.0.borrow_mut();
        inner.live.push((memory, bytes));
        inner.bytes += bytes;
    }

    /// A freed allocation. Releasing a handle this tracker never saw (or saw already) is a no-op:
    /// a budget line must never turn into a wrap-around on a double drop.
    pub(crate) fn release(&self, memory: vk::DeviceMemory) {
        let mut inner = self.0.borrow_mut();
        if let Some(index) = inner.live.iter().position(|(handle, _)| *handle == memory) {
            let (_, bytes) = inner.live.swap_remove(index);
            inner.bytes = inner.bytes.saturating_sub(bytes);
        }
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.0.borrow().bytes
    }

    pub(crate) fn count(&self) -> u64 {
        self.0.borrow().live.len() as u64
    }
}

/// A mapped, host-visible buffer. Used for the per-frame uniform ring.
pub(crate) struct MappedBuffer {    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) ptr: *mut u8,
    pub(crate) len: u64,
}

impl MappedBuffer {
    /// # Safety
    /// `offset + bytes <= self.len`, and no GPU read of the same range is in flight.
    pub(crate) unsafe fn write(&self, offset: u64, bytes: &[u8]) {
        debug_assert!(offset + bytes.len() as u64 <= self.len);
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(offset as usize), bytes.len());
        }
    }

    /// # Safety
    /// `offset + bytes <= self.len`.
    pub(crate) unsafe fn zero(&self, offset: u64, bytes: u64) {
        debug_assert!(offset + bytes <= self.len);
        unsafe {
            ptr::write_bytes(self.ptr.add(offset as usize), 0, bytes as usize);
        }
    }
}

pub(crate) struct FrameSlot {
    pub(crate) pool: vk::CommandPool,
    pub(crate) cmd: vk::CommandBuffer,
    pub(crate) query_pool: vk::QueryPool,
    pub(crate) descriptor_pool: vk::DescriptorPool,
    pub(crate) last_value: u64,
}

pub(crate) struct FrameState {
    pub(crate) timeline: vk::Semaphore,
    pub(crate) next_value: u64,
    pub(crate) next_slot: usize,
    pub(crate) slots: Vec<FrameSlot>,
}

pub(crate) struct OneShot {
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
}

/// A message reported by the Vulkan validation layers, kept so tests can assert emptiness.
struct DebugReporter {
    loader: ash::ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
    messages: Arc<Mutex<Vec<String>>>,
    /// Box<Arc<..>> passed as the callback's user data.
    user_data: *mut c_void,
    enabled: bool,
}

// The user data pointer is a Box<Arc<Mutex<Vec<String>>>> we own; Vulkan never touches it on
// another thread except inside the callback, which locks the mutex.
unsafe impl Send for DebugReporter {}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user_data: *mut c_void,
) -> vk::Bool32 {
    unsafe {
        if data.is_null() || user_data.is_null() {
            return vk::FALSE;
        }
        let arc = &*(user_data as *const Arc<Mutex<Vec<String>>>);
        let label = if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
            "error"
        } else {
            "warning"
        };
        let message = {
            let data = &*data;
            if data.p_message.is_null() {
                String::new()
            } else {
                CStr::from_ptr(data.p_message).to_string_lossy().into_owned()
            }
        };
        if let Ok(mut messages) = arc.lock() {
            messages.push(format!("{label}: {message}"));
        }
        vk::FALSE
    }
}

/// The Vulkan device everything else in the crate renders through.
///
/// Neither shared nor used from two threads at once: it is owned by the render thread. It is
/// `Send` so the app can build it on one thread and hand it to the render thread; children keep
/// a clone of the `ash::Device` handle and must be dropped before the `Gpu`.
pub struct Gpu {
    pub(crate) dev: ash::Device,
    pub(crate) instance: ash::Instance,
    pub(crate) physical: vk::PhysicalDevice,
    pub(crate) entry: ash::Entry,
    pub(crate) queue: vk::Queue,
    pub(crate) queue_family: u32,
    adapter_name: String,
    pub(crate) memory: vk::PhysicalDeviceMemoryProperties,
    pub(crate) timestamp_period_ns: f32,
    pub(crate) timestamp_valid_bits: u32,
    pub(crate) sampler: vk::Sampler,
    /// The same sampler with nearest filtering. Block-compressed (BCn) formats are not required
    /// to support linear filtering, so a pipeline that samples HAP blocks picks between the two
    /// by asking [`Gpu::supports_linear_filter`].
    pub(crate) nearest_sampler: vk::Sampler,
    /// `textureCompressionBC` was available and enabled: HAP textures can be created.
    texture_compression_bc: bool,
    pub(crate) uniform_ring: MappedBuffer,
    pub(crate) uniform_stride: u64,
    uniform_alignment: u64,
    pub(crate) frames: RefCell<FrameState>,
    pub(crate) one_shot: RefCell<OneShot>,
    /// Live device memory this renderer holds (all images and buffers of the crate).
    pub(crate) allocations: Allocations,
    debug: Option<DebugReporter>,
    /// `VK_KHR_surface` + `VK_KHR_win32_surface` were enabled, so a `Stage` can be built.
    pub(crate) surface_extensions: bool,
}

impl Gpu {
    /// Creates a Vulkan 1.3 device on the GPU the preference names.
    ///
    /// `Integrated` is the reference: the laptop panel and the stage output both hang off it.
    pub fn new(preference: GpuPreference) -> Result<Gpu, RenderError> {
        unsafe { Self::new_inner(preference) }
    }

    unsafe fn new_inner(preference: GpuPreference) -> Result<Gpu, RenderError> {
        let entry = unsafe { ash::Entry::load() }
            .map_err(|error| RenderError::Loader(error.to_string()))?;

        let available_extensions =
            unsafe { retry_incomplete(|| entry.enumerate_instance_extension_properties(None)) }?;
        let has_extension = |name: &CStr| {
            available_extensions
                .iter()
                .any(|properties| extension_name(properties) == name)
        };

        let mut instance_extensions: Vec<*const c_char> = Vec::new();
        let surface_extensions = has_extension(ash::khr::surface::NAME)
            && has_extension(ash::khr::win32_surface::NAME);
        if surface_extensions {
            instance_extensions.push(ash::khr::surface::NAME.as_ptr());
            instance_extensions.push(ash::khr::win32_surface::NAME.as_ptr());
        }
        let debug_utils = cfg!(debug_assertions) && has_extension(ash::ext::debug_utils::NAME);
        if debug_utils {
            instance_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        // Validation layers in debug builds, and only when they are installed.
        let available_layers =
            unsafe { retry_incomplete(|| entry.enumerate_instance_layer_properties()) }?;
        let validation_available = available_layers.iter().any(|properties| {
            properties
                .layer_name_as_c_str()
                .map(|name| name == c"VK_LAYER_KHRONOS_validation")
                .unwrap_or(false)
        });
        let use_validation = cfg!(debug_assertions) && validation_available;
        let layer_names: Vec<*const c_char> = if use_validation {
            vec![c"VK_LAYER_KHRONOS_validation".as_ptr()]
        } else {
            Vec::new()
        };

        let app_name = CString::new("Moshloom").expect("no nul");
        let engine_name = CString::new("mosh-render").expect("no nul");
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);
        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions)
            .enabled_layer_names(&layer_names);
        let instance = unsafe { entry.create_instance(&instance_info, None) }?;

        let messages = Arc::new(Mutex::new(Vec::new()));
        let debug = if debug_utils {
            let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let user_data = Box::into_raw(Box::new(Arc::clone(&messages))) as *mut c_void;
            let severities = vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR;
            let types = vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE;
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(severities)
                .message_type(types)
                .pfn_user_callback(Some(debug_callback))
                .user_data(user_data);
            match unsafe { loader.create_debug_utils_messenger(&info, None) } {
                Ok(messenger) => Some(DebugReporter {
                    loader,
                    messenger,
                    messages: Arc::clone(&messages),
                    user_data,
                    enabled: true,
                }),
                Err(_) => {
                    drop(unsafe { Box::from_raw(user_data as *mut Arc<Mutex<Vec<String>>>) });
                    None
                }
            }
        } else {
            None
        };

        // Guess which GPU to use before creating the device.
        let physical_devices =
            unsafe { retry_incomplete(|| instance.enumerate_physical_devices()) }?;
        let mut seen = Vec::new();
        let mut chosen = None;
        for physical in physical_devices {
            let properties = unsafe { instance.get_physical_device_properties(physical) };
            let name = properties
                .device_name_as_c_str()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "unknown".to_string());
            let matches = match &preference {
                GpuPreference::Integrated => {
                    properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
                }
                GpuPreference::Discrete => {
                    properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                }
                GpuPreference::ByName(wanted) => {
                    name.to_lowercase().contains(&wanted.to_lowercase())
                }
            };
            seen.push(format!("{name} ({:?})", properties.device_type));
            if matches && chosen.is_none() {
                chosen = Some((physical, properties, name));
            }
        }
        let (physical, properties, adapter_name) =
            chosen.ok_or_else(|| RenderError::NoSuitableDevice {
                preference: preference.describe(),
                seen: seen.clone(),
            })?;

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let (queue_family, timestamp_valid_bits) = queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| {
                family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && family.queue_count > 0
            })
            .map(|(index, family)| (index as u32, family.timestamp_valid_bits))
            .ok_or(RenderError::NoGraphicsQueue)?;

        let device_extensions_available = unsafe {
            retry_incomplete(|| instance.enumerate_device_extension_properties(physical))
        }?;
        let mut device_extensions: Vec<*const c_char> = Vec::new();
        if device_extensions_available
            .iter()
            .any(|properties| extension_name(properties) == ash::khr::swapchain::NAME)
        {
            device_extensions.push(ash::khr::swapchain::NAME.as_ptr());
        }

        // Vulkan 1.3 core: dynamic rendering (no render passes) and timeline semaphores
        // (one counter instead of per-frame fences).
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .host_query_reset(true);
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
        let mut features2 = vk::PhysicalDeviceFeatures2::default();
        // Chain: features2 -> features13 -> features12.
        unsafe {
            push_chain(&mut features13, &mut features12);
            push_chain(&mut features2, &mut features13);
        }
        unsafe { instance.get_physical_device_features2(physical, &mut features2) };
        if features13.dynamic_rendering != vk::TRUE {
            return Err(RenderError::MissingFeature("dynamicRendering"));
        }
        if features12.timeline_semaphore != vk::TRUE {
            return Err(RenderError::MissingFeature("timelineSemaphore"));
        }
        if features12.host_query_reset != vk::TRUE {
            return Err(RenderError::MissingFeature("hostQueryReset"));
        }
        // The 1.0 feature set sits in `features2.features`, not in the pNext chain: ask for the
        // one feature this crate needs (block-compressed textures, which HAP uploads into) and
        // leave the rest off, instead of enabling everything the query reported.
        let texture_compression_bc = features2.features.texture_compression_bc == vk::TRUE;
        features2.features = vk::PhysicalDeviceFeatures::default();
        if texture_compression_bc {
            features2.features.texture_compression_bc = vk::TRUE;
        }

        let priorities = [1.0f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let mut device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions);
        unsafe { push_chain(&mut device_info, &mut features2) };
        let device = unsafe { instance.create_device(physical, &device_info, None) }?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        // A shared linear-clamp sampler: every layer texture is sampled the same way, so the
        // sampler lives in the descriptor set layout as an immutable one.
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(1.0);
        let sampler = unsafe { device.create_sampler(&sampler_info, None) }?;
        // Nearest, for formats without linear filtering (block-compressed textures on drivers
        // that do not report SAMPLED_IMAGE_FILTER_LINEAR for them).
        let nearest_sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(0.0);
        let nearest_sampler = unsafe { device.create_sampler(&nearest_sampler_info, None) }?;

        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };

        // One timeline semaphore for every submit in the crate.
        let mut semaphore_type = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let semaphore_create = vk::SemaphoreCreateInfo::default().push_next(&mut semaphore_type);
        let timeline = unsafe { device.create_semaphore(&semaphore_create, None) }?;

        let uniform_alignment = properties
            .limits
            .min_uniform_buffer_offset_alignment
            .max(16);
        let uniform_stride = align_up(MAX_UNIFORM_BYTES, uniform_alignment);
        let uniform_bytes = uniform_stride * FRAMES_IN_FLIGHT as u64 * MAX_DRAWS_PER_FRAME as u64;
        let allocations = Allocations::default();
        let (uniform_buffer, uniform_memory) = create_buffer(
            &device,
            &memory,
            &allocations,
            uniform_bytes,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let uniform_ptr = unsafe {
            device.map_memory(
                uniform_memory,
                0,
                uniform_bytes,
                vk::MemoryMapFlags::empty(),
            )
        }? as *mut u8;

        let mut slots = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            let pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let pool = unsafe { device.create_command_pool(&pool_info, None) }?;
            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = unsafe { device.allocate_command_buffers(&allocate_info) }?[0];
            let query_info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(QUERIES_PER_FRAME);
            let query_pool = unsafe { device.create_query_pool(&query_info, None) }?;
            let sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                    .descriptor_count(MAX_DRAWS_PER_FRAME as u32),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(MAX_DRAWS_PER_FRAME as u32 * 8),
                // The samplers are immutable in the set layout, but the pool still declares the
                // type (validation warns otherwise).
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLER)
                    .descriptor_count(MAX_DRAWS_PER_FRAME as u32 * 8),
            ];
            let descriptor_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(MAX_DRAWS_PER_FRAME as u32)
                .pool_sizes(&sizes);
            let descriptor_pool = unsafe { device.create_descriptor_pool(&descriptor_info, None) }?;
            slots.push(FrameSlot {
                pool,
                cmd,
                query_pool,
                descriptor_pool,
                last_value: 0,
            });
        }

        // One-shot command buffer for uploads, clears and readbacks.
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let one_shot_pool = unsafe { device.create_command_pool(&pool_info, None) }?;
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(one_shot_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let one_shot_cmd = unsafe { device.allocate_command_buffers(&allocate_info) }?[0];
        let fence_info = vk::FenceCreateInfo::default();
        let one_shot_fence = unsafe { device.create_fence(&fence_info, None) }?;

        Ok(Gpu {
            dev: device,
            instance,
            physical,
            entry,
            queue,
            queue_family,
            adapter_name,
            memory,
            timestamp_period_ns: properties.limits.timestamp_period,
            timestamp_valid_bits,
            sampler,
            nearest_sampler,
            texture_compression_bc,
            uniform_ring: MappedBuffer {
                buffer: uniform_buffer,
                memory: uniform_memory,
                ptr: uniform_ptr,
                len: uniform_bytes,
            },
            uniform_stride,
            uniform_alignment,
            frames: RefCell::new(FrameState {
                timeline,
                next_value: 0,
                next_slot: 0,
                slots,
            }),
            one_shot: RefCell::new(OneShot {
                pool: one_shot_pool,
                cmd: one_shot_cmd,
                fence: one_shot_fence,
            }),
            allocations,
            debug,
            surface_extensions,
        })
    }

    /// The name of the GPU the device was created on, e.g. `AMD Radeon(TM) Graphics`.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// True when the device has `textureCompressionBC` and this renderer enabled it, so
    /// block-compressed (HAP) textures can be created and sampled.
    pub fn texture_compression_bc(&self) -> bool {
        self.texture_compression_bc
    }

    /// The device's properties for one format: what it can be used for.
    pub fn format_properties(&self, format: vk::Format) -> vk::FormatProperties {
        unsafe {
            self.instance
                .get_physical_device_format_properties(self.physical, format)
        }
    }

    /// True when `format` can be sampled with linear filtering on this device. Block-compressed
    /// formats are not required to support it; a pipeline that samples them uses the result to
    /// choose the linear or the nearest sampler, instead of getting a validation error.
    pub fn supports_linear_filter(&self, format: vk::Format) -> bool {
        self.format_properties(format)
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
    }

    /// Every validation-layer message seen so far. Tests assert this is empty.
    pub fn validation_messages(&self) -> Vec<String> {
        match &self.debug {
            Some(reporter) => reporter
                .messages
                .lock()
                .map(|messages| messages.clone())
                .unwrap_or_else(|poisoned| poisoned.into_inner().clone()),
            None => Vec::new(),
        }
    }

    /// True when the validation layer was requested and is collecting messages.
    pub fn validation_enabled(&self) -> bool {
        self.debug.as_ref().is_some_and(|reporter| reporter.enabled)
    }

    /// Device-local memory properties of the selected GPU.
    pub fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.memory
    }

    /// Bytes of device memory this renderer currently holds: every image and buffer of the crate
    /// that has not been dropped, as allocated (driver alignment included).
    ///
    /// On the reference machine the GPU is integrated, so this is system memory the process is
    /// paying for: the design's "< 500 MB with 4 videos" budget is checked against it (plus the
    /// decoded video frames and the driver's own books).
    pub fn allocated_bytes(&self) -> u64 {
        self.allocations.bytes()
    }

    /// How many live image and buffer allocations that is, for a budget line that says where the
    /// bytes went.
    pub fn allocation_count(&self) -> u64 {
        self.allocations.count()
    }

    pub(crate) fn allocations(&self) -> Allocations {
        self.allocations.clone()
    }

    /// Nanoseconds per GPU timestamp tick, from the device limits.
    pub fn timestamp_period_ns(&self) -> f32 {
        self.timestamp_period_ns
    }

    /// The device's `minUniformBufferOffsetAlignment`, the stride a uniform ring must use.
    pub(crate) fn min_uniform_offset_alignment(&self) -> u64 {
        self.uniform_alignment
    }

    /// The frame slot the next [`crate::FrameRecorder::new`] will claim. A staging buffer this
    /// slot owns is the one the next frame's copy reads, so `upload::VideoTexture::upload` writes
    /// that buffer and waits for the slot's previous submission first.
    pub(crate) fn next_frame_slot(&self) -> usize {
        self.frames.borrow().next_slot
    }

    /// The timeline value `slot` last signalled, or 0 if it never submitted.
    pub(crate) fn slot_last_value(&self, slot: usize) -> u64 {
        self.frames.borrow().slots[slot].last_value
    }

    /// Records one short command buffer and waits for it. Used for clears, readbacks and the
    /// present blit, never for a per-frame layer pass (those go through `FrameRecorder`).
    pub(crate) fn one_shot(
        &self,
        record: impl FnOnce(vk::CommandBuffer),
    ) -> Result<(), RenderError> {
        let one_shot = self.one_shot.borrow();
        unsafe {
            self.dev
                .reset_command_pool(one_shot.pool, vk::CommandPoolResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.dev.begin_command_buffer(one_shot.cmd, &begin)?;
            record(one_shot.cmd);
            self.dev.end_command_buffer(one_shot.cmd)?;
            self.dev.reset_fences(&[one_shot.fence])?;
            let command_buffers = [one_shot.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            count_submission();
            self.dev.queue_submit(self.queue, &[submit], one_shot.fence)?;
            self.dev
                .wait_for_fences(&[one_shot.fence], true, u64::MAX)?;
        }
        Ok(())
    }

    pub(crate) fn find_memory_type(
        &self,
        requirements: &vk::MemoryRequirements,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32, RenderError> {
        for index in 0..self.memory.memory_type_count {
            let supported = self.memory.memory_types[index as usize].property_flags;
            if requirements.memory_type_bits & (1 << index) != 0 && supported.contains(properties) {
                return Ok(index);
            }
        }
        Err(RenderError::MissingMemoryType)
    }

    pub(crate) fn create_image(
        &self,
        extent: vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<(vk::Image, vk::DeviceMemory), RenderError> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { self.dev.create_image(&info, None) }?;
        let requirements = unsafe { self.dev.get_image_memory_requirements(image) };
        let memory_type = self.find_memory_type(
            &requirements,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let memory = unsafe { self.dev.allocate_memory(&allocate, None) }?;
        unsafe { self.dev.bind_image_memory(image, memory, 0) }?;
        self.allocations.add(memory, requirements.size);
        Ok((image, memory))
    }

    /// Creates (and counts) a buffer. The companion of [`Self::create_image`] for the crate's own
    /// upload, uniform and staging buffers.
    pub(crate) fn new_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), RenderError> {
        create_buffer(&self.dev, &self.memory, &self.allocations, size, usage, properties)
    }

    pub(crate) fn create_image_view(
        &self,
        image: vk::Image,
        format: vk::Format,
    ) -> Result<vk::ImageView, RenderError> {
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        Ok(unsafe { self.dev.create_image_view(&info, None) }?)
    }
}

// The device is used from one thread at a time; `Send` lets the app create it on one thread and
// hand it to the render thread. `Sync` is deliberately not implemented: nothing here is safe to
// use from two threads at once (recorded command buffers and `Cell` image layouts).
unsafe impl Send for Gpu {}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.dev.device_wait_idle();
        }
        // Children first: every object the device owns, then the device and the instance drop
        // themselves (`ash::Device` and `ash::Instance` have `Drop`).
        unsafe {
            let frames = self.frames.get_mut();
            for slot in &frames.slots {
                self.dev.destroy_descriptor_pool(slot.descriptor_pool, None);
                self.dev.destroy_query_pool(slot.query_pool, None);
                self.dev.destroy_command_pool(slot.pool, None);
            }
            self.dev.destroy_semaphore(frames.timeline, None);

            let one_shot = self.one_shot.get_mut();
            self.dev.destroy_fence(one_shot.fence, None);
            self.dev.destroy_command_pool(one_shot.pool, None);

            self.dev.unmap_memory(self.uniform_ring.memory);
            self.dev.destroy_buffer(self.uniform_ring.buffer, None);
            self.dev.free_memory(self.uniform_ring.memory, None);
            self.allocations.release(self.uniform_ring.memory);
            self.dev.destroy_sampler(self.sampler, None);
            self.dev.destroy_sampler(self.nearest_sampler, None);
        }
        if let Some(reporter) = self.debug.take() {
            unsafe {
                reporter
                    .loader
                    .destroy_debug_utils_messenger(reporter.messenger, None);
                drop(Box::from_raw(
                    reporter.user_data as *mut Arc<Mutex<Vec<String>>>,
                ));
            }
        }
    }
}

pub(crate) fn align_up(value: u64, alignment: u64) -> u64 {
    debug_assert!(alignment > 0);
    value.div_ceil(alignment) * alignment
}

/// Appends `next` (and its own chain) to the end of `head`'s pNext chain.
///
/// `ash` generates `push_next` for many structs but not for a struct that is itself a feature
/// chain member with a lifetime (`PhysicalDeviceVulkan13Features`), so the chain is built by
/// hand. Every Vulkan struct here starts with `s_type`/`p_next`, which is what `BaseOutStructure`
/// describes.
///
/// # Safety
/// `next` and everything in its own chain must stay alive and must not move while `head` is
/// passed to Vulkan.
unsafe fn push_chain<H, T>(head: &mut H, next: &mut T) {
    unsafe {
        let head = (head as *mut H).cast::<vk::BaseOutStructure>();
        let next = (next as *mut T).cast::<vk::BaseOutStructure>();
        let mut tail = next;
        while !(*tail).p_next.is_null() {
            tail = (*tail).p_next;
        }
        // `next`'s chain now runs after whatever `head` already had.
        (*tail).p_next = (*head).p_next;
        (*head).p_next = next;
    }
}

pub(crate) fn extension_name(properties: &vk::ExtensionProperties) -> &CStr {
    properties.extension_name_as_c_str().unwrap_or(c"")
}

/// Attempts at one enumeration before an `INCOMPLETE` is reported as an error.
const ENUMERATION_ATTEMPTS: u32 = 8;

/// Runs a two-call Vulkan enumeration until it stops answering `VK_INCOMPLETE`.
///
/// `VK_INCOMPLETE` is a success code, not a failure: an enumeration returns it when the buffer is
/// smaller than the number of items that exist, because that count changed between the count query
/// and the fetch (`vkEnumeratePhysicalDevices`, `vkEnumerate*ExtensionProperties`,
/// `vkGetPhysicalDeviceSurfaceFormatsKHR`, `vkGetSwapchainImagesKHR`). The answer is to ask
/// again, never to fail: this retries up to [`ENUMERATION_ATTEMPTS`] times before it reports one.
///
/// `ash`'s helpers loop over the *fetch* call, but their **count query** goes through
/// `vk::Result::result()`, and ash's `result_with_success` treats every code except `VK_SUCCESS`
/// as an error, `VK_INCOMPLETE` included. That is the gap this closes: with a loader that
/// multiplexes several ICDs (the reference machine has an integrated and a discrete Radeon and a
/// virtual display driver), a device list rebuilt in the middle of the call makes the count query
/// itself answer `VK_INCOMPLETE`, which [`Gpu::new`] used to hand back as
/// `Vulkan(INCOMPLETE)` in about one workspace run in five.
pub(crate) fn retry_incomplete<T>(
    mut enumerate: impl FnMut() -> Result<T, vk::Result>,
) -> Result<T, RenderError> {
    for _ in 0..ENUMERATION_ATTEMPTS {
        match enumerate() {
            Ok(items) => return Ok(items),
            Err(vk::Result::INCOMPLETE) => std::thread::yield_now(),
            Err(error) => return Err(RenderError::Vulkan(error)),
        }
    }
    Err(RenderError::Vulkan(vk::Result::INCOMPLETE))
}

pub(crate) fn create_buffer(
    device: &ash::Device,
    memory: &vk::PhysicalDeviceMemoryProperties,
    allocations: &Allocations,
    size: u64,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory), RenderError> {
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&info, None) }?;
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mut memory_type = None;
    for index in 0..memory.memory_type_count {
        let supported = memory.memory_types[index as usize].property_flags;
        if requirements.memory_type_bits & (1 << index) != 0 && supported.contains(properties) {
            memory_type = Some(index);
            break;
        }
    }
    let memory_type = memory_type.ok_or(RenderError::MissingMemoryType)?;
    let allocate = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);
    let allocation = unsafe { device.allocate_memory(&allocate, None) }?;
    unsafe { device.bind_buffer_memory(buffer, allocation, 0) }?;
    allocations.add(allocation, requirements.size);
    Ok((buffer, allocation))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of the helper: `VK_INCOMPLETE` is an answer to retry, not a failure. The
    /// synthetic call counts its attempts so the retry itself is pinned, not just the result.
    #[test]
    fn incomplete_is_retried_not_reported() {
        let mut attempts = 0;
        let devices = retry_incomplete(|| {
            attempts += 1;
            if attempts < 4 {
                Err(vk::Result::INCOMPLETE)
            } else {
                Ok(vec!["integrated", "discrete"])
            }
        })
        .expect("INCOMPLETE is a success code");
        assert_eq!(devices, ["integrated", "discrete"]);
        assert_eq!(attempts, 4, "it asks again while the answer is INCOMPLETE");
    }

    /// A call that never settles must end as an error at the bound, not spin forever.
    #[test]
    fn incomplete_stops_after_the_bound() {
        let mut attempts = 0;
        let result: Result<Vec<u32>, RenderError> = retry_incomplete(|| {
            attempts += 1;
            Err(vk::Result::INCOMPLETE)
        });
        assert_eq!(result, Err(RenderError::Vulkan(vk::Result::INCOMPLETE)));
        assert_eq!(attempts, ENUMERATION_ATTEMPTS);
    }

    /// A real error must not be retried and must keep its code.
    #[test]
    fn a_real_error_passes_through_at_once() {
        let mut attempts = 0;
        let result: Result<Vec<u32>, RenderError> = retry_incomplete(|| {
            attempts += 1;
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY)
        });
        assert_eq!(
            result,
            Err(RenderError::Vulkan(vk::Result::ERROR_OUT_OF_HOST_MEMORY))
        );
        assert_eq!(attempts, 1);
    }
}
