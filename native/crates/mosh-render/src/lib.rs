//! mosh-render: the Vulkan renderer for the native app.
//!
//! See `docs/superpowers/plans/2026-11-17-native-port.md`, Task 1.7. The crate owns the device
//! (explicitly the **integrated** GPU), the per-layer render targets with their feedback
//! ping-pong, fullscreen fragment pipelines built from SPIR-V words, the compositor with blend
//! modes, the stage's swapchain with out-of-date/lost recovery, GPU timestamps and the
//! `FlashLimiter`.
//!
//! It does not depend on `mosh-shaders`: it takes SPIR-V as `&[u32]` and the app wires the two
//! together.
//!
//! ## Using it
//!
//! ```no_run
//! # use mosh_core::show::BlendMode;
//! # use mosh_render::{Compositor, FrameRecorder, Gpu, GpuPreference, LayerTarget, OutputImage, Pipeline};
//! # fn example() -> Result<(), mosh_render::RenderError> {
//! let gpu = Gpu::new(GpuPreference::Integrated)?;
//! let flow = LayerTarget::new(&gpu, 1280, 720);
//! let out = OutputImage::new(&gpu, 1280, 720);
//! let shader: &[u32] = &[]; // a fragment shader from mosh-shaders
//! // let pipeline = Pipeline::fullscreen(&gpu, shader, 192, 1)?;
//!
//! let mut recorder = FrameRecorder::new(&gpu);
//! recorder.layer(&flow, BlendMode::Normal, 1.0);
//! let mut compositor = Compositor::new(&gpu)?;
//! compositor.composite(recorder.layers(), &out)?;
//! let timings = gpu.wait_frame(recorder.submit());
//! assert!(timings.frame_ms >= 0.0);
//! # Ok(())
//! # }
//! ```
//!
//! Targets, pipelines and the compositor keep a clone of the device handle: drop them before the
//! `Gpu`. The crate is built for one render thread (`Gpu` is `Send` but not `Sync`).
//!
//! ## Deviations from the plan's interface list, for the lanes that consume this crate
//!
//! - `Stage::new` returns `Result<Stage, RenderError>`: a surface can fail to be created, and the
//!   app (Task 2.6) must be able to fall back to a borderless window or hide the stage instead of
//!   panicking. `PresentResult { surface_lost }` is as specified.
//! - `MonitorId` is this crate's own `u32` id, not winit's handle: the app maps its monitors and
//!   places the window; `Stage` only carries the choice.
//! - `Compositor::composite` and `FrameRecorder::draw` return `Result<(), RenderError>` instead
//!   of nothing, so a mis-sized uniform block or a wrong input count is an error rather than a
//!   validation message. `RenderPlan` code can `?` them.
//! - `FrameRecorder` also has `layer(&mut self, target, blend, opacity)` / `layers()`: the layer
//!   stack `Stage::present` composites, because `present(recorder)` has no other way to learn it.
//! - `Gpu::read_back` is as specified (RGBA8); `Gpu::read_back_output` is its `OutputImage`
//!   counterpart, and `Gpu::wait_frame(ticket)` / `GpuTimings` are how a frame's timings are read.
//! - `Pipeline::with_name` names a pipeline's pass in `GpuTimings::per_pass_ms`.
//! - `Stage` adds `resize`, `extent`, `timings`, `last_error` and `monitor` for the app's window
//!   management and stats overlay.
//! - `FrameRecorder` adds `upload_video`, `draw_video` and `clear`: a video frame's staging copy,
//!   an overlay cover pass and a transparent-black clear are recorded into the frame's own command
//!   buffer instead of a one-shot submission the CPU waits on (V-perf, Task 2.8's follow-up).
//! - `submission_count()` (device) counts `vkQueueSubmit` calls process-wide, so a test can prove
//!   that a per-frame pass did not add one.

use std::fmt;

mod compositor;
mod device;
mod flash;
mod pipeline;
mod shaders;
mod swapchain;
mod target;
mod timing;
pub mod upload;
mod win;

pub use compositor::Compositor;
pub use device::submission_count;
pub use device::{Gpu, GpuPreference};
pub use flash::{FlashLimiter, FLASH_INTERVAL};
pub use pipeline::{FrameRecorder, Pipeline};
pub use swapchain::{MonitorId, PresentResult, Stage, SwapchainAction, SwapchainPolicy};
pub use target::{LayerTarget, OutputImage};
pub use timing::{FrameTicket, GpuTimings};
pub use upload::{CoverPass, VideoTexture};

/// Everything this crate can fail at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderError {
    /// The Vulkan loader (`vulkan-1.dll`) could not be loaded.
    Loader(String),
    Vulkan(ash::vk::Result),
    /// No GPU matched the requested preference.
    NoSuitableDevice {
        preference: String,
        seen: Vec<String>,
    },
    NoGraphicsQueue,
    /// A Vulkan 1.3 feature the renderer requires is missing.
    MissingFeature(&'static str),
    MissingMemoryType,
    /// A required device extension (or the whole surface stack) is unavailable.
    SurfaceUnsupported,
    /// The window handle is not a Win32 handle.
    UnsupportedWindow,
    ZeroSized,
    /// A uniform block larger than the crate's ring (`MAX_UNIFORM_BYTES`).
    UniformTooLarge {
        size: u64,
        max: u64,
    },
    TooManyDraws,
    TooManyLayers {
        max: usize,
        got: usize,
    },
    InputCountMismatch {
        expected: u32,
        got: u32,
    },
    EmptyShader,
    FrameFinished,
    /// A swapchain or surface problem the stage could not recover from, as text for the stats
    /// overlay (the stage keeps running).
    Stage(String),
}

impl fmt::Display for RenderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RenderError::Loader(message) => write!(formatter, "Vulkan loader: {message}"),
            RenderError::Vulkan(result) => write!(formatter, "Vulkan error: {result:?}"),
            RenderError::NoSuitableDevice { preference, seen } => {
                write!(formatter, "no GPU matching {preference}; devices seen: {seen:?}")
            }
            RenderError::NoGraphicsQueue => write!(formatter, "no graphics queue family"),
            RenderError::MissingFeature(feature) => {
                write!(formatter, "missing Vulkan feature: {feature}")
            }
            RenderError::MissingMemoryType => write!(formatter, "no matching memory type"),
            RenderError::SurfaceUnsupported => {
                write!(formatter, "surface or swapchain unsupported on this device")
            }
            RenderError::UnsupportedWindow => write!(formatter, "window handle is not Win32"),
            RenderError::ZeroSized => write!(formatter, "target has a zero extent"),
            RenderError::UniformTooLarge { size, max } => {
                write!(formatter, "uniform block of {size} bytes (max {max})")
            }
            RenderError::TooManyDraws => write!(formatter, "too many draws in one frame"),
            RenderError::TooManyLayers { max, got } => {
                write!(formatter, "{got} layers composited (max {max})")
            }
            RenderError::InputCountMismatch { expected, got } => {
                write!(formatter, "pipeline wants {expected} inputs, got {got}")
            }
            RenderError::EmptyShader => write!(formatter, "empty SPIR-V module"),
            RenderError::FrameFinished => write!(formatter, "frame already submitted"),
            RenderError::Stage(message) => write!(formatter, "stage: {message}"),
        }
    }
}

impl std::error::Error for RenderError {}

impl From<ash::vk::Result> for RenderError {
    fn from(result: ash::vk::Result) -> RenderError {
        RenderError::Vulkan(result)
    }
}
