//! The window shell: two windows and one render thread.
//!
//! The design's shape, once:
//!
//! - the **stage** window never shows UI. With a second monitor it is borderless fullscreen
//!   there. On a single screen it is an ordinary window *beside* the control window, so the
//!   controls are never covered; `F11` (in either window) takes it in and out of fullscreen, and
//!   `Esc` on the stage leaves fullscreen and brings the control window to the front.
//!   `crate::monitor::window_plan` decides both windows' places.
//! - the **control** window is the editor: its own winit window on its own OpenGL context
//!   (`crate::editor`), painted only when something changes and at most 30 times a second, so it
//!   costs the stage nothing while the performer is not touching it. Closing it quits the app.
//! - the control window's two pictures of the show, the **stage preview** and the **clip
//!   thumbnails**, ride in the stage's own frames and are sent over as bytes (`crate::preview`):
//!   no wait on the render thread, at most one thumbnail per frame, the preview only while the
//!   control window is on screen and showing it.
//! - the **render thread** owns the analysis, the show and the GPU. It presents with a FIFO
//!   swapchain, so the frame loop blocks on the display and never busy-waits.
//!
//! Nothing here is exercised by the crate's tests: windowed runs are coordinator-only on the
//! reference machine (AGENTS.md). The decisions it implements are unit-tested in `core`,
//! `monitor` and `budget`.
//!
//! Recovery, in the order it is tried: a render error rebuilds the stage (surface and swapchain);
//! if that fails the device is recreated together with the visuals; if that fails too the app
//! keeps ticking and tries again a second later. No dialog, no exit, no panic: a set must survive
//! a monitor unplug, a device reset or a driver restart.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use mosh_core::{
    AnalysisFrame, BlendMode, Clip, ControlEvent, Layer, LayerKind, Project, RenderPlan, ShowMode,
};
use mosh_render::{
    BlockFormat, CompositeTarget, FramePacer, FrameRecorder, Gpu, GpuPreference, LayerTarget,
    RenderError, Stage, WaitTimer,
};
use mosh_video::hap::HapTextureFormat;
use mosh_visuals::stack::{Stack, StackLayer};
use mosh_visuals::{LayerSlot, Visuals};
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, NamedKey};
use winit::monitor::MonitorHandle;
use winit::platform::windows::MonitorHandleExtWindows;
use winit::raw_window_handle::HasWindowHandle;
use winit::window::{Fullscreen, Window, WindowId};

use crate::budget::{BudgetWatch, WARMUP_FRAMES};
use crate::core::{AppCore, Lamps, Status, default_project, window_title};
use crate::editor::model::Command as EditorCommand;
use crate::editor::ui::{StageInfo, UiAction};
use crate::editor::{EditorModel, EditorUi, EditorWindow};
use crate::monitor::{
    Area, Chrome, MonitorInfo, SavedLayout, SavedWindow, StageWindow, WindowPlan, format_layouts,
    parse_layouts, recall, remember, setup_key, usable, window_plan,
};
use crate::preview::{
    PREVIEW_FPS, PREVIEW_WIDTH, Picture, PreviewClock, THUMB_SIZE, ThumbQueue, ThumbRequest,
    has_thumbnail, preview_size, thumb_frame,
};
use crate::win::WindowHandle;

/// How often the monitor list is re-read, so an unplugged or replugged stage is noticed without
/// the human touching anything.
const MONITOR_CHECK: Duration = Duration::from_secs(2);
/// Stats are sent to the window thread at this rate.
const STATUS_INTERVAL: Duration = Duration::from_millis(100);
/// How often the control window repaints for a changed status: ten times a second moves the
/// playhead and the meters smoothly enough to read, and an unchanged status paints nothing.
const STATUS_PAINT: Duration = Duration::from_millis(100);
/// Backoff after a failed rebuild.
const RETRY_AFTER: Duration = Duration::from_millis(500);
/// How often the same render error is logged again.
const ERROR_REPEAT: Duration = Duration::from_secs(10);
/// The tick when there is nothing to present, in place of a vsync wait.
const IDLE_TICK: Duration = Duration::from_millis(16);

/// The renderer and every target it needs, rebuilt as a unit.
///
/// Field order is the drop order: `mosh-render` says targets, pipelines and the compositor keep a
/// clone of the device handle and must be dropped before the `Gpu`, so the device is last.
struct Scene {
    stage: Option<Stage>,
    /// The whole stack composited, when the show has an Adjustment or a Mask layer: the stack
    /// writes it and the stage presents it as one layer. Output-sized; dropped on a resize.
    stacked: Option<LayerTarget>,
    /// Adjustment and Mask layers (`mosh_visuals::stack`): allocates nothing until a show has one.
    stack: Stack,
    slots: Vec<LayerSlot>,
    /// Where a clip thumbnail is drawn (`crate::preview`), built on first use.
    thumb: Option<LayerTarget>,
    visuals: Option<Visuals>,
    /// The internal size `visuals` was built for.
    visuals_size: (u32, u32),
    gpu: Option<Gpu>,
    out: (u32, u32),
}

impl Scene {
    /// A device, the visuals and one slot per layer. The stage is attached separately, because a
    /// surface can fail while the show is perfectly able to keep drawing.
    fn new(out: (u32, u32), layers: &[Layer]) -> Result<Scene, RenderError> {
        // The reference machine's integrated GPU is the one the design budgets against. A machine
        // without one (a development desktop) still runs the show on its only GPU, on a path that
        // is never the reference: the log says so, once.
        let gpu = match Gpu::new(GpuPreference::Integrated) {
            Ok(gpu) => gpu,
            Err(error) => {
                static LOGGED: std::sync::Once = std::sync::Once::new();
                LOGGED.call_once(|| {
                    eprintln!(
                        "ilanaVisual: integrated GPU unavailable ({error}); falling back to the discrete GPU"
                    )
                });
                Gpu::new(GpuPreference::Discrete)?
            }
        };
        let (w, h) = internal_size(out, 0.5);
        let visuals = Visuals::new(&gpu, w, h)?;
        let mut scene = Scene {
            stage: None,
            stacked: None,
            stack: Stack::new(),
            slots: Vec::with_capacity(layers.len()),
            thumb: None,
            visuals: Some(visuals),
            visuals_size: (w, h),
            gpu: Some(gpu),
            out,
        };
        scene.rebuild_slots(layers);
        Ok(scene)
    }

    fn gpu(&self) -> Result<&Gpu, RenderError> {
        self.gpu
            .as_ref()
            .ok_or_else(|| RenderError::Stage("no device".into()))
    }

    fn rebuild_slots(&mut self, layers: &[Layer]) {
        let gpu = match self.gpu.as_ref() {
            Some(gpu) => gpu,
            None => return,
        };
        self.slots.clear();
        for (index, layer) in layers.iter().enumerate() {
            let (w, h) = internal_size(self.out, layer.internal_scale);
            self.slots.push(LayerSlot::new(gpu, index, w, h));
        }
    }

    /// Resize the layer stack. The device and the window survive; only targets are rebuilt.
    fn rescale(&mut self, out: (u32, u32), layers: &[Layer]) -> Result<(), RenderError> {
        let gpu = self.gpu()?;
        let (w, h) = internal_size(out, 0.5);
        // The visuals (every pipeline, the fluid and mosh targets) are rebuilt only when their
        // size changes. Starting up used to build them twice (once for the window's size, once
        // more when the swapchain reported the same size), which was most of the first second.
        if self.visuals.is_none() || self.visuals_size != (w, h) {
            // Build the new visuals before dropping the old ones: if this fails the old scene
            // is still intact and the next frame can try again.
            let visuals = Visuals::new(gpu, w, h)?;
            self.visuals = Some(visuals);
            self.visuals_size = (w, h);
        }
        self.out = out;
        self.stacked = None;
        self.rebuild_slots(layers);
        if let Some(stage) = self.stage.as_mut() {
            stage.resize(out.0, out.1);
        }
        Ok(())
    }

    /// Rebuild the surface and the swapchain on `handle`.
    fn attach_stage(&mut self, handle: WindowHandle, monitor: Option<u32>) -> Result<(), RenderError> {
        // Drop the old stage before asking for a new one: a lost surface usually means the window
        // moved to another monitor.
        self.stage = None;
        let gpu = self.gpu()?;
        let raw = handle.raw().ok_or(RenderError::UnsupportedWindow)?;
        let stage = Stage::new(gpu, raw, monitor.map(mosh_render::MonitorId))?;
        let extent = stage.extent();
        if extent[0] > 0 && extent[1] > 0 {
            self.out = (extent[0], extent[1]);
        }
        self.stage = Some(stage);
        Ok(())
    }

    fn detach_stage(&mut self) {
        self.stage = None;
    }
}

/// The internal render size of a layer: the output size times its scale, never zero.
fn internal_size(out: (u32, u32), scale: f32) -> (u32, u32) {
    let scale = scale.clamp(0.125, 1.0);
    (
        ((out.0 as f32 * scale) as u32).max(16),
        ((out.1 as f32 * scale) as u32).max(16),
    )
}

/// The texture format a HAP plane uploads as.
fn block_format(format: HapTextureFormat) -> BlockFormat {
    match format {
        HapTextureFormat::Bc1 => BlockFormat::Bc1,
        HapTextureFormat::Bc3 => BlockFormat::Bc3,
        HapTextureFormat::Bc3YCoCg => BlockFormat::Bc3YCoCg,
        HapTextureFormat::Bc4Alpha => BlockFormat::Bc4,
    }
}

/// Video for overlay layers, opened on first use and kept.
#[derive(Default)]
struct VideoBank {
    sources: BTreeMap<u32, mosh_video::VideoSource>,
    failed: BTreeSet<u32>,
}

impl VideoBank {
    fn frame_for(&mut self, clip: &Clip, t_s: f64) -> Option<&mosh_video::VideoFrame> {
        let path = clip.video.as_ref()?;
        if !self.sources.contains_key(&clip.id) {
            if self.failed.contains(&clip.id) {
                return None;
            }
            match mosh_video::VideoSource::open(path) {
                Ok(source) => {
                    self.sources.insert(clip.id, source);
                }
                Err(error) => {
                    eprintln!("ilanaVisual: overlay video {} ({path}): {error}", clip.id);
                    self.failed.insert(clip.id);
                    return None;
                }
            }
        }
        self.sources.get_mut(&clip.id)?.frame_at(t_s)
    }
}

/// What one presented frame measured, and the pictures that came back with it.
struct Outcome {
    cpu_ms: f32,
    gpu_ms: f32,
    surface_lost: bool,
    /// A preview picture recorded two frames ago.
    preview: Option<Picture>,
    /// Thumbnails recorded two frames ago, by key.
    thumbs: Vec<(u64, Picture)>,
    /// The thumbnail asked for this frame could not be drawn.
    thumb_failed: Option<u64>,
}

/// The control window's pictures a frame is asked to take along.
#[derive(Default)]
struct Pictures {
    /// Blit the presented image into the preview.
    preview: bool,
    /// Draw this clip thumbnail at the end of the frame.
    thumb: Option<ThumbRequest>,
}

/// Draw every layer of `plan` and present the result. `frame_start` is when the frame's CPU work
/// began, so `cpu_ms` covers analysis, the show and the recording but not the vsync wait.
#[allow(clippy::too_many_arguments)]
fn render_frame(
    scene: &mut Scene,
    plan: &RenderPlan,
    analysis: &AnalysisFrame,
    t: f32,
    frame_start: Instant,
    videos: &mut VideoBank,
    project: &Project,
    pictures: Pictures,
) -> Result<Outcome, RenderError> {
    let Some(gpu) = scene.gpu.as_ref() else {
        return Err(RenderError::Stage("no device".into()));
    };

    // Media goes into a layer before the layer is drawn: a video clip's frame, the camera a
    // camera clip names, a text clip's words. Each kind reads only its own clips, so a
    // `camera:` or `text:` source is never opened as a video file.
    if let Some(visuals) = scene.visuals.as_mut() {
        for (index, state) in plan.layers.iter().enumerate() {
            let Some(clip_id) = state.clip else { continue };
            let Some(clip) = project.clips.iter().find(|clip| clip.id == clip_id) else {
                continue;
            };
            let media = clip.video.as_deref().unwrap_or("");
            match state.visualizer.as_deref() {
                Some("overlay") => {
                    let Some(video) = videos.frame_for(clip, analysis.time_s) else {
                        continue;
                    };
                    match video.hap() {
                        // HAP: the compressed blocks go straight to the GPU, no CPU decode. A
                        // device without block compression leaves the layer transparent.
                        Some(hap) => {
                            let Some(format) = hap.format.map(block_format) else {
                                continue;
                            };
                            if !visuals.overlay.hap_supported() {
                                continue;
                            }
                            let alpha = hap.alpha.as_ref().map(|plane| plane.data.as_slice());
                            visuals.overlay.set_hap_frame(
                                gpu, index, video.w, video.h, format, &hap.data, alpha,
                            )?;
                        }
                        None => {
                            visuals
                                .overlay
                                .set_frame(gpu, index, video.w, video.h, &video.bgra)?;
                        }
                    }
                }
                Some("camera") => {
                    // `camera:` alone is the default camera; naming the same one again is free.
                    let device = media.strip_prefix("camera:").unwrap_or("").trim();
                    visuals
                        .camera
                        .set_source(index, (!device.is_empty()).then_some(device));
                }
                Some("text") => {
                    // The same words again are free: glyphs are laid out on a change only.
                    let words = media.strip_prefix("text:").unwrap_or("");
                    visuals.text.set_text(gpu, index, words);
                }
                _ => {}
            }
        }
    }

    // One slot per layer, with at least as many as the plan draws.
    while scene.slots.len() < plan.layers.len() {
        let index = scene.slots.len();
        let (w, h) = internal_size(scene.out, 0.5);
        scene.slots.push(LayerSlot::new(gpu, index, w, h));
    }

    // What each layer is: an Adjustment or a Mask changes how the stack is composited.
    let kinds: Vec<LayerKind> = plan
        .layers
        .iter()
        .map(|state| {
            project
                .session
                .layers
                .get(state.layer)
                .map_or(LayerKind::Visualizer, |layer| layer.kind)
        })
        .collect();
    let stacked = kinds
        .iter()
        .any(|kind| matches!(kind, LayerKind::Adjustment | LayerKind::Mask));

    let mut recorder = FrameRecorder::new(gpu);
    {
        let Some(visuals) = scene.visuals.as_mut() else {
            return Err(RenderError::Stage("no visuals".into()));
        };
        for (index, state) in plan.layers.iter().enumerate() {
            // An Adjustment layer draws nothing of its own: the stack applies its clip's effects
            // to the layers below it.
            if kinds[index] == LayerKind::Adjustment {
                continue;
            }
            visuals.draw_layer(&mut recorder, &mut scene.slots[index], state, analysis, t)?;
        }
        if !stacked {
            for (index, state) in plan.layers.iter().enumerate() {
                recorder.layer(scene.slots[index].output(), state.blend, state.opacity);
            }
        }
    }
    // Wall time of the stack's composite, which is mostly waiting for the GPU (the layers, then
    // the adjustment's own passes): it is counted as GPU time, not as render-thread work.
    let mut stack_ms = 0.0f32;
    if stacked {
        // A stack with an Adjustment or a Mask is composited by `mosh_visuals::stack` into an
        // output-sized target, and the stage presents that as one layer. A plain stack never
        // comes here: it stays one composite pass straight into the swapchain image.
        let out = scene.out;
        let target = scene
            .stacked
            .get_or_insert_with(|| LayerTarget::new(gpu, out.0, out.1));
        let flow = scene
            .visuals
            .as_ref()
            .and_then(|visuals| visuals.fluid.velocity());
        let layers: Vec<StackLayer<'_>> = plan
            .layers
            .iter()
            .enumerate()
            .map(|(index, state)| {
                StackLayer::new(
                    &scene.slots[index],
                    kinds[index],
                    state.blend,
                    state.opacity,
                    state,
                )
            })
            .collect();
        let composed = Instant::now();
        scene.stack.compose(
            recorder,
            flow,
            &layers,
            CompositeTarget::Layer(target),
            analysis,
            t,
        )?;
        stack_ms = composed.elapsed().as_secs_f32() * 1_000.0;
        recorder = FrameRecorder::new(gpu);
        recorder.layer(target, BlendMode::Normal, 1.0);
    }

    // A clip thumbnail, when this frame carries one: one more scene pass at the end of the
    // frame's own recorder, into a small target of its own. It is drawn after the layers, with
    // the frame's clock and spectrum, so the tables the scenes share are left as the stage's
    // own draws left them. A scene that fails to draw costs the thumbnail, never the frame.
    let mut capture = None;
    let mut thumb_failed = None;
    if let Some(request) = pictures.thumb.as_ref()
        && scene.stage.is_some()
    {
        let target = scene
            .thumb
            .get_or_insert_with(|| LayerTarget::new_single(gpu, THUMB_SIZE.0, THUMB_SIZE.1));
        let drawn = match scene.visuals.as_mut() {
            Some(visuals) => visuals.scenes.draw(
                &mut recorder,
                target,
                &request.visualizer,
                &thumb_frame(analysis),
                &request.params,
                t,
            ),
            None => Err(RenderError::Stage("no visuals".into())),
        };
        match drawn {
            Ok(()) => capture = Some((&*target, request.key)),
            Err(_) => thumb_failed = Some(request.key),
        }
    } else if let Some(request) = pictures.thumb.as_ref() {
        thumb_failed = Some(request.key);
    }

    let cpu_ms = (frame_start.elapsed().as_secs_f32() * 1_000.0 - stack_ms).max(0.0);
    let Some(stage) = scene.stage.as_mut() else {
        // No window to present in: the layers were still drawn, so the next monitor that appears
        // shows a current frame instead of a black one.
        drop(recorder);
        return Ok(Outcome {
            cpu_ms,
            gpu_ms: stack_ms,
            surface_lost: false,
            preview: None,
            thumbs: Vec::new(),
            thumb_failed,
        });
    };
    // The pictures are copied out inside the present submission and picked up two presents
    // later; neither size call does anything unless the size changed.
    let extent = stage.extent();
    let preview_ready = stage
        .set_preview_size(
            gpu,
            Some(preview_size((extent[0], extent[1]), PREVIEW_WIDTH)),
        )
        .is_ok();
    if pictures.preview && preview_ready {
        stage.request_preview();
    }
    if capture.is_some() && stage.set_capture_size(gpu, THUMB_SIZE.0, THUMB_SIZE.1).is_err() {
        thumb_failed = capture.take().map(|(_, key)| key);
    }
    let present = stage.present_with_capture(recorder, capture);
    let preview = stage.take_preview().map(Picture::from);
    let mut thumbs = Vec::new();
    while let Some(picture) = stage.take_capture() {
        thumbs.push((picture.tag, Picture::from(picture)));
    }
    Ok(Outcome {
        cpu_ms,
        gpu_ms: stage.timings().frame_ms + stack_ms,
        surface_lost: present.surface_lost,
        preview,
        thumbs,
        thumb_failed,
    })
}

/// Commands from the window thread to the render thread.
enum Command {
    Resize(u32, u32),
    /// The stage window (re)appeared or moved: rebuild the surface on this handle.
    ReplaceSurface {
        handle: WindowHandle,
        size: (u32, u32),
        monitor: Option<u32>,
    },
    /// There is no stage window: keep the show running, present nothing.
    DetachSurface,
    SetMode(ShowMode),
    /// An edited project from the control window: applied on the next bar line.
    SetProject(Project),
    /// A control event from the control window (a launch, a stop, a live parameter).
    Control(ControlEvent),
    /// Send preview frames of the stage, or stop sending them.
    SetPreview(bool),
    /// Draw one clip thumbnail.
    Thumb(ThumbRequest),
    Quit,
}

/// Events from the render thread to the window thread.
#[derive(Clone, Debug)]
pub enum UserEvent {
    Status(Status),
    /// The surface was lost (a monitor went away): re-place the stage window.
    SurfaceLost,
    /// A line for the terminal (a recovered error, a missing device).
    Note(String),
    /// One frame of the stage preview.
    Preview(Picture),
    /// A clip thumbnail, or `None` when that look has no picture.
    Thumb { key: u64, picture: Option<Picture> },
    /// A thumbnail request that could not be taken now: the control window asks again.
    ThumbRetry(u64),
}

struct RenderHandle {
    tx: Sender<Command>,
    thread: Option<JoinHandle<()>>,
}

/// The render thread's state.
struct RenderThread {
    core: AppCore,
    project_layers: Vec<Layer>,
    scene: Option<Scene>,
    videos: VideoBank,
    handle: Option<WindowHandle>,
    monitor: Option<u32>,
    size: (u32, u32),
    rx: Receiver<Command>,
    proxy: EventLoopProxy<UserEvent>,
    budget: BudgetWatch,
    /// `ILANAVISUAL_BUDGET_LOG=1`: log every budget window's averages, not only an exceed (for
    /// measuring a steady state; the exceed line is latched and says nothing after the first).
    budget_log: Option<Instant>,
    /// The frame rate cap: the stage renders at most this many frames a second whatever the
    /// display's refresh is (the reference panel is 120 Hz).
    pacer: FramePacer,
    /// The wait the pacer's deadlines use: a high-resolution waitable timer, because the system
    /// tick would turn a 16.6 ms wait into 31 ms.
    timer: WaitTimer,
    paced_frames: u64,
    /// The control window is showing the stage preview.
    preview_on: bool,
    preview_clock: PreviewClock,
    /// Clip thumbnails waiting for a frame: one per stage frame, none while the renderer warms up.
    thumb_queue: ThumbQueue,
    lamps: Lamps,
    started: Instant,
    last_status: Option<Instant>,
    last_error: Option<(Instant, String)>,
    retry_at: Option<Instant>,
}

impl RenderThread {
    #[allow(clippy::too_many_arguments)]
    fn new(
        project: Project,
        handle: Option<WindowHandle>,
        size: (u32, u32),
        monitor: Option<u32>,
        rx: Receiver<Command>,
        proxy: EventLoopProxy<UserEvent>,
        fps: u32,
    ) -> RenderThread {
        let project_layers = project.session.layers.clone();
        RenderThread {
            core: AppCore::open(project),
            project_layers,
            scene: None,
            videos: VideoBank::default(),
            handle,
            monitor,
            size,
            rx,
            proxy,
            budget: BudgetWatch::new(),
            budget_log: std::env::var_os("ILANAVISUAL_BUDGET_LOG").map(|_| Instant::now()),
            pacer: FramePacer::new(fps),
            timer: WaitTimer::new(),
            paced_frames: 0,
            preview_on: false,
            preview_clock: PreviewClock::new(PREVIEW_FPS),
            thumb_queue: ThumbQueue::new(),
            lamps: Lamps::default(),
            started: Instant::now(),
            last_status: None,
            last_error: None,
            retry_at: Some(Instant::now()),
        }
    }

    fn apply(&mut self, command: Command) {
        match command {
            Command::Resize(width, height) => {
                if width == 0 || height == 0 {
                    return;
                }
                self.size = (width, height);
                let result = match self.scene.as_mut() {
                    Some(scene) => scene.rescale(self.size, &self.project_layers),
                    None => Ok(()),
                };
                if let Err(error) = result {
                    self.recover(&error);
                }
            }
            Command::ReplaceSurface {
                handle,
                size,
                monitor,
            } => {
                self.handle = Some(handle);
                self.monitor = monitor;
                if size.0 > 0 && size.1 > 0 {
                    self.size = size;
                }
                if self.scene.is_none() {
                    self.build();
                    return;
                }
                let result = match self.scene.as_mut() {
                    Some(scene) => scene
                        .attach_stage(handle, self.monitor)
                        .and_then(|()| scene.rescale(self.size, &self.project_layers)),
                    None => Ok(()),
                };
                match result {
                    Ok(()) => self.retry_at = None,
                    Err(error) => self.recover(&error),
                }
            }
            Command::DetachSurface => {
                self.handle = None;
                if let Some(scene) = self.scene.as_mut() {
                    scene.detach_stage();
                }
            }
            Command::SetMode(mode) => self.core.set_mode(mode),
            Command::SetProject(project) => {
                // A layer stack that changed shape (how many layers, how large each renders)
                // needs its targets rebuilt. A clip edit does not, and neither does an opacity,
                // a blend mode or a layer kind: those must not touch the GPU at all, or dragging
                // an opacity would reallocate every target on every tick.
                let shape = |layers: &[Layer]| -> Vec<f32> {
                    layers.iter().map(|layer| layer.internal_scale).collect()
                };
                if shape(&project.session.layers) != shape(&self.project_layers) {
                    let layers = project.session.layers.clone();
                    if let Some(scene) = self.scene.as_mut() {
                        scene.rebuild_slots(&layers);
                    }
                }
                self.project_layers = project.session.layers.clone();
                self.core.queue_project(project);
            }
            Command::Control(event) => self.core.queue_control(event),
            Command::SetPreview(on) => self.preview_on = on,
            Command::Thumb(request) => {
                if !has_thumbnail(&request.visualizer) {
                    self.send_thumb(request.key, None);
                } else if let Some(dropped) = self.thumb_queue.push(request) {
                    // The queue was full: the control window is told, and asks again if the
                    // clip is still on screen.
                    let _ = self.proxy.send_event(UserEvent::ThumbRetry(dropped.key));
                }
            }
            Command::Quit => {}
        }
    }

    /// One frame: wait for its slot, analyse, decide, draw, present, report.
    ///
    /// The wait is the frame rate cap. `Stage::present` presents in FIFO, so the display paces
    /// the loop: at 120 Hz a free-running loop draws 120 frames a second, twice the rate the
    /// budget is written for and twice the GPU time per second. The pacer waits out the rest of
    /// the frame's interval (16.67 ms at 60 fps) with a high-resolution timer, never a spin.
    fn frame(&mut self) {
        if self.pacer.is_capped() {
            let wait = self.pacer.wait_for(Instant::now());
            if !wait.is_zero() {
                self.timer.wait(wait);
                self.pacer.resync_if_stalled(Instant::now());
                self.paced_frames += 1;
            }
        }
        let frame_start = Instant::now();
        let frame = self.core.step(frame_start);
        self.lamps.note(frame.analysis.events, frame_start);

        // Building the renderer (a device, every pipeline) is not a frame: it happens before
        // the first one and after a device loss, takes as long as it takes, and is logged on its
        // own line. The frame's CPU time is counted from after it, so the 2 ms budget line means
        // "a frame's work", not "the frame in which the app started".
        let mut work_start = frame_start;
        if self.scene.is_none() && self.retry_at.is_none_or(|at| frame_start >= at) {
            self.build();
            self.thumb_queue.renderer_rebuilt();
            // The frames that follow a build pay for first uses; they are reported as warm-up,
            // and the budget's window starts after them.
            self.budget.warm_up(WARMUP_FRAMES);
            self.note(format!(
                "renderer built in {:.0} ms (not a frame: the budget counts from the next one)",
                frame_start.elapsed().as_secs_f32() * 1_000.0
            ));
            work_start = Instant::now();
        }
        // A window that exists without a stage (a failed surface, a monitor change) gets its
        // stage rebuilt on the retry interval.
        let wants_stage = self.scene.as_ref().is_some_and(|scene| scene.stage.is_none());
        if wants_stage && self.handle.is_some() && self.retry_at.is_none_or(|at| frame_start >= at) {
            self.attach_stage();
            work_start = Instant::now();
        }

        let mut cpu_ms = 0.0;
        let mut gpu_ms = 0.0;
        let has_stage = self.scene.as_ref().is_some_and(|scene| scene.stage.is_some());
        if has_stage {
            // The control window's pictures ride in this frame: the preview when one is due and
            // wanted, and at most one clip thumbnail (none while the renderer warms up).
            let pictures = Pictures {
                preview: self.preview_on && self.preview_clock.due(frame_start),
                thumb: self.thumb_queue.next_frame(),
            };
            let asked = pictures.thumb.as_ref().map(|request| request.visualizer.clone());
            if let Some(scene) = self.scene.as_mut() {
                let t = self.started.elapsed().as_secs_f32();
                match render_frame(
                    scene,
                    &frame.plan,
                    &frame.analysis,
                    t,
                    work_start,
                    &mut self.videos,
                    self.core.show().project(),
                    pictures,
                ) {
                    Ok(outcome) => {
                        cpu_ms = outcome.cpu_ms;
                        gpu_ms = outcome.gpu_ms;
                        if outcome.surface_lost {
                            let _ = self.proxy.send_event(UserEvent::SurfaceLost);
                        }
                        if let Some(picture) = outcome.preview {
                            let _ = self.proxy.send_event(UserEvent::Preview(picture));
                        }
                        for (key, picture) in outcome.thumbs {
                            self.thumb_queue.settled(key);
                            self.send_thumb(key, Some(picture));
                        }
                        if let Some(key) = outcome.thumb_failed {
                            self.thumb_queue.settled(key);
                            self.note_once(format!(
                                "thumbnail of {} could not be drawn",
                                asked.as_deref().unwrap_or("a clip")
                            ));
                            self.send_thumb(key, None);
                        }
                    }
                    Err(error) => self.recover(&error),
                }
            }
            // A picture that never came back (its frame was not presented): the control window
            // asks again instead of waiting for it for ever.
            for key in self.thumb_queue.lost() {
                let _ = self.proxy.send_event(UserEvent::ThumbRetry(key));
            }
        } else {
            // Nothing to present: the show keeps its clock, and the loop sleeps instead of
            // spinning on an acquire that can never succeed.
            std::thread::sleep(IDLE_TICK);
        }

        for line in self.budget.sample(cpu_ms, gpu_ms, Instant::now()) {
            eprintln!("ilanaVisual: {line}");
        }
        if let Some(at) = self.budget_log
            && at.elapsed() >= Duration::from_secs(1)
        {
            self.budget_log = Some(Instant::now());
            eprintln!(
                "ilanaVisual: budget window: render thread {:.2} ms, GPU {:.2} ms per frame",
                self.budget.last_cpu_ms(),
                self.budget.last_gpu_ms()
            );
        }

        if self
            .last_status
            .is_none_or(|at| frame_start.saturating_duration_since(at) >= STATUS_INTERVAL)
        {
            self.last_status = Some(frame_start);
            let status =
                Status::from_frame(&frame, gpu_ms, cpu_ms, self.lamps.levels(frame_start));
            let _ = self.proxy.send_event(UserEvent::Status(status));
        }
    }

    fn send_thumb(&self, key: u64, picture: Option<Picture>) {
        let _ = self.proxy.send_event(UserEvent::Thumb { key, picture });
    }

    fn build(&mut self) {
        self.scene = None;
        match Scene::new(self.size, &self.project_layers) {
            Ok(mut scene) => {
                if let Some(handle) = self.handle
                    && let Err(error) = scene.attach_stage(handle, self.monitor)
                {
                    self.note_once(format!("stage could not be built: {error}"));
                }
                // The swapchain may have picked its own extent; size the layer stack to the
                // window the app placed.
                if let Err(error) = scene.rescale(self.size, &self.project_layers) {
                    self.note_once(format!("layer stack could not be built: {error}"));
                }
                self.scene = Some(scene);
                self.retry_at = None;
            }
            Err(error) => {
                self.note_once(format!("renderer could not be built: {error}"));
                self.retry_at = Some(Instant::now() + RETRY_AFTER);
            }
        }
    }

    fn attach_stage(&mut self) {
        let Some(handle) = self.handle else { return };
        let result = match self.scene.as_mut() {
            Some(scene) => scene.attach_stage(handle, self.monitor),
            None => return,
        };
        match result {
            Ok(()) => self.retry_at = None,
            Err(error) => {
                self.note_once(format!("stage could not be built: {error}"));
                self.retry_at = Some(Instant::now() + RETRY_AFTER);
            }
        }
    }

    /// Rebuild after a render error: the stage first, the device and the visuals if that fails.
    fn recover(&mut self, error: &RenderError) {
        let now = Instant::now();
        self.note_once(format!("render error: {error}; rebuilding"));
        if self.retry_at.is_some_and(|at| now < at) {
            return;
        }
        self.retry_at = Some(now + RETRY_AFTER);

        if self.scene.is_some() {
            let stage_step = match (self.scene.as_mut(), self.handle) {
                (Some(scene), Some(handle)) => scene.attach_stage(handle, self.monitor),
                _ => Err(RenderError::Stage("no surface".into())),
            };
            if stage_step.is_ok() {
                let rescale = match self.scene.as_mut() {
                    Some(scene) => scene.rescale(self.size, &self.project_layers),
                    None => Ok(()),
                };
                if rescale.is_ok() {
                    self.retry_at = None;
                    return;
                }
            }
            // The surface could not be rebuilt on this device: assume the device is gone too.
            self.scene = None;
        }
        self.build();
    }

    /// One terminal line per distinct failure per [`ERROR_REPEAT`], so a rebuild loop cannot bury
    /// the first message in per-frame noise.
    fn note_once(&mut self, text: String) {
        let now = Instant::now();
        let repeat = self
            .last_error
            .as_ref()
            .is_none_or(|(at, last)| *last != text || at.elapsed() >= ERROR_REPEAT);
        if repeat {
            self.last_error = Some((now, text.clone()));
            self.note(text);
        }
    }

    fn note(&self, text: String) {
        let _ = self.proxy.send_event(UserEvent::Note(text));
    }
}

/// Run one frame loop until the window thread asks it to stop.
///
/// A panic in one frame must not end a set: the renderer is dropped, rebuilt and the loop
/// continues. (A driver that aborts the process cannot be caught; nothing can.)
fn render_main(mut thread: RenderThread) {
    loop {
        let mut quit = false;
        while let Ok(command) = thread.rx.try_recv() {
            if matches!(command, Command::Quit) {
                quit = true;
                break;
            }
            thread.apply(command);
        }
        if quit {
            break;
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| thread.frame()));
        if result.is_err() {
            thread.scene = None;
            thread.note_once("render thread panicked; rebuilding the renderer".to_string());
            thread.retry_at = Some(Instant::now() + RETRY_AFTER);
            std::thread::sleep(RETRY_AFTER);
        }
    }
}

/// The window thread's state.
struct Shell {
    /// The editing model behind the control window: it owns the project the app runs.
    model: EditorModel,
    editor_ui: EditorUi,
    note: Option<String>,
    proxy: EventLoopProxy<UserEvent>,
    stage_window: Option<Window>,
    /// The control window and its OpenGL context. Absent when it could not be built: the show
    /// keeps running, because a set does not end because an editor window failed.
    editor: Option<EditorWindow>,
    monitors: Vec<MonitorHandle>,
    monitor_list: Vec<MonitorInfo>,
    /// Where the two windows are, as last applied.
    plan: Option<WindowPlan>,
    /// The performer's `F11` choice; `None` until they make one (fullscreen on a second
    /// monitor, a window beside the controls on a single one).
    fullscreen: Option<bool>,
    render: Option<RenderHandle>,
    /// The stage's frame rate cap (`--fps`, default 60): a 120 Hz panel would otherwise draw
    /// twice the frames the budget is written for.
    fps: u32,
    title: Option<String>,
    last_status_paint: Option<Instant>,
    next_monitor_check: Option<Instant>,
    /// Whether the render thread was last told to send preview frames.
    preview_sent: bool,
    /// What a window frame adds to a client area: a guess from the display's scale until the
    /// control window exists, then its measured frame.
    chrome: Chrome,
    /// Where the performer left the windows, per monitor setup (`windows.txt`).
    layouts: Vec<(String, SavedLayout)>,
    /// A window was moved or resized since the layouts were last written.
    layout_dirty: bool,
    started: bool,
}

impl Shell {
    fn new(
        project: Project,
        path: Option<String>,
        note: Option<String>,
        proxy: EventLoopProxy<UserEvent>,
        fps: u32,
    ) -> Shell {
        let mut editor_ui = EditorUi {
            recent: read_recent(),
            ..EditorUi::default()
        };
        if let Some(path) = path.as_deref() {
            editor_ui.project_path = path.to_string();
        }
        let mut model = EditorModel::new(project).with_path(path);
        if let Some(note) = note.as_deref() {
            model.note(note);
        }
        Shell {
            model,
            editor_ui,
            note,
            proxy,
            stage_window: None,
            editor: None,
            monitors: Vec::new(),
            monitor_list: Vec::new(),
            plan: None,
            fullscreen: None,
            render: None,
            fps,
            title: None,
            last_status_paint: None,
            next_monitor_check: None,
            preview_sent: false,
            chrome: Chrome::estimate(1.0),
            layouts: read_layouts(),
            layout_dirty: false,
            started: false,
        }
    }

    /// The saved layout for the monitors that are plugged in now.
    fn saved_layout(&self) -> Option<SavedLayout> {
        recall(&self.layouts, &setup_key(&self.monitor_list))
    }

    /// A planned outer rectangle on a monitor, as a desktop position and a client size.
    fn planned(&self, monitor: u32, area: Area) -> Option<SavedWindow> {
        let info = self.monitor_list.iter().find(|info| info.index == monitor)?;
        let (width, height) = area.inner(self.chrome);
        Some(SavedWindow {
            x: info.x + area.x,
            y: info.y + area.y,
            width,
            height,
        })
    }

    /// Where the control window goes: where the performer left it on this monitor setup, if
    /// that is still on a screen, and the plan's place otherwise.
    fn control_place(&self) -> Option<SavedWindow> {
        let saved = self
            .saved_layout()
            .and_then(|layout| layout.control)
            .filter(|window| usable(*window, &self.monitor_list, self.chrome));
        saved.or_else(|| {
            let (monitor, area) = self.plan?.control?;
            self.planned(monitor, area)
        })
    }

    /// Where a windowed stage goes, the same way.
    fn stage_place(&self, monitor: u32, area: Area) -> Option<SavedWindow> {
        let saved = self
            .saved_layout()
            .and_then(|layout| layout.stage)
            .filter(|window| usable(*window, &self.monitor_list, self.chrome));
        saved.or_else(|| self.planned(monitor, area))
    }

    /// Remember where the windows are now, for this monitor setup, and write the file.
    fn save_layout(&mut self) {
        if !std::mem::take(&mut self.layout_dirty) {
            return;
        }
        let mut layout = self.saved_layout().unwrap_or_default();
        if let Some(place) = self.editor.as_ref().and_then(EditorWindow::placement) {
            layout.control = Some(place);
        }
        let windowed = matches!(
            self.plan.map(|plan| plan.stage),
            Some(StageWindow::Windowed { .. })
        );
        if windowed
            && let Some(window) = self.stage_window.as_ref()
            && !window.is_minimized().unwrap_or(false)
            && window.fullscreen().is_none()
            && let Ok(position) = window.outer_position()
        {
            let size = window.inner_size();
            if size.width > 0 && size.height > 0 {
                layout.stage = Some(SavedWindow {
                    x: position.x,
                    y: position.y,
                    width: size.width,
                    height: size.height,
                });
            }
        }
        let key = setup_key(&self.monitor_list);
        if recall(&self.layouts, &key) != Some(layout) {
            remember(&mut self.layouts, &key, layout);
            write_layouts(&self.layouts);
        }
    }

    fn send(&self, command: Command) {
        if let Some(render) = self.render.as_ref() {
            let _ = render.tx.send(command);
        }
    }

    fn refresh_monitors(&mut self, event_loop: &ActiveEventLoop) {
        let primary = event_loop.primary_monitor();
        self.monitors.clear();
        self.monitor_list.clear();
        for (index, monitor) in event_loop.available_monitors().enumerate() {
            let size = monitor.size();
            let position = monitor.position();
            let name = monitor.name().unwrap_or_else(|| format!("monitor {index}"));
            let is_primary = match primary.as_ref() {
                Some(primary) => same_monitor(primary, &monitor),
                None => index == 0,
            };
            let mut info = MonitorInfo::new(
                index as u32,
                name,
                is_primary,
                size.width,
                size.height,
            )
            .at(position.x, position.y);
            // The work area is what windows are placed in: the desktop without the taskbar.
            if let Some((x, y, width, height)) = crate::win::work_area(monitor.hmonitor()) {
                info = info.with_work(Area {
                    x,
                    y,
                    width,
                    height,
                });
            }
            self.monitor_list.push(info);
            self.monitors.push(monitor);
        }
    }

    /// The scale of the monitor the control window opens on, for the first guess at a window
    /// frame's size.
    fn control_scale(&self) -> f64 {
        let index = self
            .monitor_list
            .iter()
            .position(|info| info.primary)
            .unwrap_or(0);
        self.monitors
            .get(index)
            .map_or(1.0, MonitorHandle::scale_factor)
    }

    fn monitor(&self, index: u32) -> Option<MonitorHandle> {
        self.monitors.get(index as usize).cloned()
    }

    fn create_control_window(&mut self, event_loop: &ActiveEventLoop) {
        let title = window_title(self.model.path(), self.model.modified());
        let place = self.control_place();
        let remembered = self
            .saved_layout()
            .and_then(|layout| layout.control)
            .is_some_and(|window| Some(window) == place);
        // Building a GL context is the one place the window thread talks to a driver that may be
        // absent or broken. A panic there must not take the show with it: the stage keeps running
        // on the render thread and the app logs one line.
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            EditorWindow::new(event_loop, &title, place)
        }));
        match built {
            Ok(Ok(editor)) => {
                self.log(format!(
                    "control window: {}x{} on OpenGL, at most 30 fps, painted on change",
                    editor.window().inner_size().width,
                    editor.window().inner_size().height
                ));
                // The frame was a guess until now. With the real one, a planned place is planned
                // again (a remembered place is already exact: it was saved as it is restored).
                let measured = editor.chrome();
                if measured != self.chrome && measured.height > 0 {
                    self.chrome = measured;
                    self.plan = Some(window_plan(&self.monitor_list, self.fullscreen, measured));
                    if !remembered && let Some(place) = self.control_place() {
                        editor.place(place);
                    }
                }
                self.title = Some(title);
                self.editor = Some(editor);
            }
            Ok(Err(error)) => self.log(format!("control window could not be created: {error}")),
            Err(_) => self.log(
                "control window could not be created: the GL driver panicked building the context"
                    .to_string(),
            ),
        }
    }

    /// Paint the control window if it asked for a frame, and apply what the performer did.
    fn paint_editor(&mut self, now: Instant) {
        let Some(editor) = self.editor.as_mut() else {
            return;
        };
        if !editor.wants_paint(now) {
            return;
        }
        self.editor_ui.stage = StageInfo {
            fullscreen: self.plan.is_some_and(|plan| plan.stage.is_fullscreen()),
            shared_monitor: self.plan.is_some_and(|plan| plan.shared_monitor),
        };
        // A panic in the UI (or in the driver, on the paint path) must not take the show with it:
        // the stage is on the render thread, and the editor is the part that can be given up.
        let painted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let commands = editor.draw(&mut self.editor_ui, &self.model);
            (commands, editor.take_error())
        }));
        match painted {
            Ok((commands, error)) => {
                if let Some(error) = error {
                    self.log(error);
                }
                self.run_editor_commands(commands);
                self.run_ui_requests();
            }
            Err(_) => {
                self.editor = None;
                self.send(Command::SetPreview(false));
                self.log(
                    "control window failed while painting; the editor is gone, the show is not"
                        .to_string(),
                );
            }
        }
    }

    /// Apply one frame's worth of editor commands and hand the result to the show.
    fn run_editor_commands(&mut self, commands: Vec<EditorCommand>) {
        let mut applied = false;
        for command in commands {
            // The file a command opens or saves, for the recent list.
            let file = match &command {
                EditorCommand::OpenProject { path } | EditorCommand::SaveProject { path } => {
                    Some(path.clone())
                }
                EditorCommand::Save => self.model.path().map(str::to_string),
                _ => None,
            };
            applied = true;
            // An error is already in the status strip: the model put it there.
            let Ok(outcome) = self.model.apply(command) else {
                continue;
            };
            // The mode is a show-level action: the stage hears it now, not at the next bar.
            if let Some(mode) = outcome.set_mode {
                self.send(Command::SetMode(mode));
            }
            if let Some(event) = outcome.launch {
                self.send(Command::Control(event));
            }
            for event in outcome.live {
                self.send(Command::Control(event));
            }
            if let Some(path) = file {
                self.editor_ui.note_recent(&path);
                self.editor_ui.project_path = path;
                write_recent(&self.editor_ui.recent);
            }
        }
        if let Some(project) = self.model.handoff() {
            self.send(Command::SetProject(project));
        }
        if applied {
            self.update_title();
            if let Some(editor) = self.editor.as_mut() {
                // An applied edit changes what the views show: paint it, at the 30 fps ceiling.
                editor.request_paint(Instant::now());
            }
        }
    }

    /// What the widgets asked of the shell this frame: the stage window, thumbnails, the preview.
    fn run_ui_requests(&mut self) {
        for action in std::mem::take(&mut self.editor_ui.actions) {
            match action {
                UiAction::ToggleStageFullscreen => self.toggle_fullscreen(),
            }
        }
        for request in self.editor_ui.thumbs.take_requests() {
            self.send(Command::Thumb(request));
        }
        self.sync_preview();
    }

    /// Tell the render thread whether anyone is looking at the preview: it is drawn only while
    /// the control window is on screen and showing it.
    fn sync_preview(&mut self) {
        let visible = self
            .editor
            .as_ref()
            .is_some_and(|editor| !editor.is_hidden());
        let wanted = visible && self.editor_ui.wants_preview();
        if wanted != self.preview_sent {
            self.preview_sent = wanted;
            self.send(Command::SetPreview(wanted));
        }
    }

    /// The control window's title: the show's file, and a star while it has unsaved edits.
    fn update_title(&mut self) {
        let title = window_title(self.model.path(), self.model.modified());
        if self.title.as_deref() == Some(title.as_str()) {
            return;
        }
        if let Some(editor) = self.editor.as_ref() {
            editor.window().set_title(&title);
        }
        self.title = Some(title);
    }

    /// `F11`: the stage in and out of fullscreen. Leaving fullscreen on a shared screen brings
    /// the control window back in front, which is the whole point of the key.
    fn toggle_fullscreen(&mut self) {
        let now = self.plan.is_some_and(|plan| plan.stage.is_fullscreen());
        self.fullscreen = Some(!now);
        self.apply_plan();
        if now {
            if let Some(editor) = self.editor.as_ref() {
                editor.bring_to_front();
            }
        } else if let Some(stage) = self.stage_window.as_ref() {
            // Going fullscreen: the stage comes to the front and takes the keyboard, so that
            // `Esc` and `F11` reach it when it covers the controls.
            stage.focus_window();
        }
        if let Some(editor) = self.editor.as_mut() {
            editor.request_paint(Instant::now());
        }
    }

    /// Leave fullscreen (if the stage is in it) and bring the control window to the front.
    fn back_to_controls(&mut self) {
        if self.plan.is_some_and(|plan| plan.stage.is_fullscreen())
            && self.plan.is_some_and(|plan| plan.shared_monitor)
        {
            self.fullscreen = Some(false);
            self.apply_plan();
        }
        if let Some(editor) = self.editor.as_mut() {
            editor.bring_to_front();
            editor.request_paint(Instant::now());
        }
    }

    /// Create the stage window for the current plan, if there is anything to present on.
    fn create_stage(&mut self, event_loop: &ActiveEventLoop) {
        let Some(plan) = self.plan else {
            self.send(Command::DetachSurface);
            return;
        };
        self.log(stage_label(plan.stage, &self.monitor_list));
        let Some(monitor) = plan.stage.monitor().and_then(|index| self.monitor(index)) else {
            // No monitor at all: leave the stage absent and keep the show running.
            self.send(Command::DetachSurface);
            return;
        };
        let attributes = Window::default_attributes()
            .with_title("ilanaVisual - stage (F11 fullscreen, Esc back to the controls)")
            .with_visible(true);
        let attributes = match plan.stage {
            StageWindow::Fullscreen { .. } => attributes
                .with_decorations(false)
                .with_fullscreen(Some(Fullscreen::Borderless(Some(monitor)))),
            StageWindow::Windowed { monitor, area } => match self.stage_place(monitor, area) {
                Some(place) => attributes
                    .with_decorations(true)
                    .with_inner_size(PhysicalSize::new(place.width, place.height))
                    .with_position(PhysicalPosition::new(place.x, place.y)),
                None => attributes.with_decorations(true),
            },
            StageWindow::Hidden => attributes,
        };
        match event_loop.create_window(attributes) {
            Ok(window) => {
                self.stage_window = Some(window);
                self.send_replace_surface();
            }
            Err(error) => {
                self.log(format!("stage window could not be created: {error}"));
                self.send(Command::DetachSurface);
            }
        }
    }

    fn send_replace_surface(&mut self) {
        let Some(window) = self.stage_window.as_ref() else {
            self.send(Command::DetachSurface);
            return;
        };
        let Ok(handle) = window.window_handle() else {
            self.send(Command::DetachSurface);
            return;
        };
        let Some(handle) = WindowHandle::from_raw(handle.as_raw()) else {
            self.log("the stage window is not a Win32 window".to_string());
            self.send(Command::DetachSurface);
            return;
        };
        let size = window.inner_size();
        self.send(Command::ReplaceSurface {
            handle,
            size: (size.width.max(1), size.height.max(1)),
            monitor: self.plan.and_then(|plan| plan.stage.monitor()),
        });
    }

    fn start_render(&mut self) {
        if self.render.is_some() {
            return;
        }
        let (handle, size) = match self.stage_window.as_ref() {
            Some(window) => {
                let handle = window
                    .window_handle()
                    .ok()
                    .and_then(|handle| WindowHandle::from_raw(handle.as_raw()));
                let size = window.inner_size();
                (handle, (size.width.max(1), size.height.max(1)))
            }
            None => (None, (1920, 1080)),
        };
        let (tx, rx) = channel();
        let project = self.model.project().clone();
        let monitor = self.plan.and_then(|plan| plan.stage.monitor());
        let proxy = self.proxy.clone();
        let fps = self.fps;
        match std::thread::Builder::new()
            .name("mosh-render".into())
            .spawn(move || {
                render_main(RenderThread::new(project, handle, size, monitor, rx, proxy, fps))
            }) {
            Ok(thread) => {
                self.render = Some(RenderHandle {
                    tx,
                    thread: Some(thread),
                })
            }
            Err(error) => self.log(format!("render thread could not start: {error}")),
        }
    }

    /// Work out where the windows go now and move the stage there if that changed.
    fn apply_plan(&mut self) {
        let plan = window_plan(&self.monitor_list, self.fullscreen, self.chrome);
        let previous = self.plan.replace(plan);
        if previous.map(|plan| plan.stage) == Some(plan.stage) {
            return;
        }
        let Some(window) = self.stage_window.as_ref() else {
            return;
        };
        self.log(stage_label(plan.stage, &self.monitor_list));
        match plan.stage {
            StageWindow::Hidden => {
                window.set_visible(false);
                self.send(Command::DetachSurface);
                return;
            }
            StageWindow::Fullscreen { monitor } => {
                let Some(monitor) = self.monitor(monitor) else {
                    return;
                };
                window.set_visible(true);
                window.set_decorations(false);
                window.set_fullscreen(Some(Fullscreen::Borderless(Some(monitor))));
            }
            StageWindow::Windowed { monitor, area } => {
                let Some(place) = self.stage_place(monitor, area) else {
                    return;
                };
                window.set_visible(true);
                window.set_fullscreen(None);
                window.set_decorations(true);
                window.set_outer_position(PhysicalPosition::new(place.x, place.y));
                let _ = window.request_inner_size(PhysicalSize::new(place.width, place.height));
            }
        }
        self.send_replace_surface();
    }

    fn log(&self, text: String) {
        eprintln!("ilanaVisual: {text}");
    }

    fn shutdown(&mut self) {
        self.save_layout();
        self.send(Command::Quit);
        if let Some(mut render) = self.render.take()
            && let Some(thread) = render.thread.take()
        {
            let _ = thread.join();
        }
    }
}

/// Where the stage is, for the log.
fn stage_label(stage: StageWindow, monitors: &[MonitorInfo]) -> String {
    let name_of = |index: u32| {
        monitors
            .iter()
            .find(|monitor| monitor.index == index)
            .map_or_else(|| "monitor".to_string(), |monitor| monitor.name.clone())
    };
    match stage {
        StageWindow::Fullscreen { monitor } => {
            format!("stage on {} (fullscreen; F11 or Esc leaves it)", name_of(monitor))
        }
        StageWindow::Windowed { monitor, area } => format!(
            "stage on {} (window {}x{} beside the controls; F11 for fullscreen)",
            name_of(monitor),
            area.width,
            area.height
        ),
        StageWindow::Hidden => "stage hidden (no monitor)".to_string(),
    }
}

/// `MonitorHandle` has no identity of its own, so two handles are the same monitor when they
/// agree on the name, size and position (all the OS tells us).
fn same_monitor(a: &MonitorHandle, b: &MonitorHandle) -> bool {
    a.name() == b.name() && a.size() == b.size() && a.position() == b.position()
}

/// Where the recent-projects list is kept: `%APPDATA%\ilanaVisual\recent.txt`, one path a line.
fn recent_file() -> Option<std::path::PathBuf> {
    let base = std::env::var_os("APPDATA")?;
    Some(std::path::PathBuf::from(base).join("ilanaVisual").join("recent.txt"))
}

fn read_recent() -> Vec<String> {
    let Some(path) = recent_file() else {
        return Vec::new();
    };
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut recent = Vec::new();
    // The file is newest first; `push_recent` puts each path first, so feed it oldest first.
    for line in text.lines().rev() {
        crate::editor::ui::push_recent(&mut recent, line);
    }
    recent
}

/// Where the windows' remembered places are kept: `%APPDATA%\ilanaVisual\windows.txt`.
fn layouts_file() -> Option<std::path::PathBuf> {
    recent_file().map(|path| path.with_file_name("windows.txt"))
}

fn read_layouts() -> Vec<(String, SavedLayout)> {
    layouts_file()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .map(|text| parse_layouts(&text))
        .unwrap_or_default()
}

/// Best effort, like the recent list.
fn write_layouts(layouts: &[(String, SavedLayout)]) {
    let Some(path) = layouts_file() else {
        return;
    };
    if let Some(folder) = path.parent() {
        let _ = std::fs::create_dir_all(folder);
    }
    let _ = std::fs::write(path, format_layouts(layouts));
}

/// Best effort: a list that cannot be written is not worth a line during a rehearsal.
fn write_recent(recent: &[String]) {
    let Some(path) = recent_file() else {
        return;
    };
    if let Some(folder) = path.parent() {
        let _ = std::fs::create_dir_all(folder);
    }
    let _ = std::fs::write(path, recent.join("\n"));
}

impl ApplicationHandler<UserEvent> for Shell {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.started {
            return;
        }
        self.started = true;
        if let Some(note) = self.note.clone() {
            self.log(note);
        }
        self.log(if self.fps == 0 {
            "stage frame rate: uncapped (FIFO, the panel's refresh)".to_string()
        } else {
            format!(
                "stage frame rate: at most {} fps (--fps, default {}), whatever the panel's refresh",
                self.fps,
                mosh_render::DEFAULT_MAX_FPS
            )
        });
        self.refresh_monitors(event_loop);
        self.chrome = Chrome::estimate(self.control_scale());
        self.plan = Some(window_plan(&self.monitor_list, self.fullscreen, self.chrome));
        self.create_control_window(event_loop);
        self.create_stage(event_loop);
        self.start_render();
        // The stage was created last and took the focus: the performer works in the controls.
        if let Some(editor) = self.editor.as_ref() {
            editor.bring_to_front();
        }
        self.next_monitor_check = Some(Instant::now() + MONITOR_CHECK);
        event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + MONITOR_CHECK));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let is_control = self
            .editor
            .as_ref()
            .map(|editor| editor.window().id())
            == Some(window_id);
        let is_stage = self.stage_window.as_ref().map(Window::id) == Some(window_id);

        // The control window's keys, clicks and scrolls all go to egui: every shortcut is handled
        // in `editor::ui`, where a key typed into a text field is known not to be one.
        if is_control
            && let Some(editor) = self.editor.as_mut()
        {
            editor.on_window_event(&event);
        }

        match event {
            WindowEvent::CloseRequested => {
                if is_control {
                    // The human closed the window they drive the app from: that is a quit.
                    event_loop.exit();
                }
                // A stage window close request is never a reason to end a set: it is ignored.
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } if is_stage => {
                // The stage has the keyboard when it is fullscreen over the controls: these two
                // keys are the way back.
                match logical_key {
                    Key::Named(NamedKey::Escape) => self.back_to_controls(),
                    Key::Named(NamedKey::F11) => self.toggle_fullscreen(),
                    _ => {}
                }
            }
            WindowEvent::Moved(_) => {
                if is_stage || is_control {
                    self.layout_dirty = true;
                }
            }
            WindowEvent::Resized(size) => {
                if is_stage || is_control {
                    self.layout_dirty = true;
                }
                if is_stage && size.width > 0 && size.height > 0 {
                    self.send(Command::Resize(size.width, size.height));
                }
                if is_control {
                    // Minimised or restored: the preview follows the window.
                    self.sync_preview();
                }
            }
            WindowEvent::Destroyed => {
                if is_stage {
                    self.stage_window = None;
                    self.send(Command::DetachSurface);
                } else if is_control {
                    self.editor = None;
                    self.send(Command::SetPreview(false));
                }
            }
            _ => {}
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Status(status) => {
                // The top bar is part of the control window: a changed number is a reason to
                // paint it, an unchanged one is not, and it is throttled so a running show paints
                // the window ten times a second for its playhead, not sixty.
                let changed = self.model.set_status(status);
                let due = self
                    .last_status_paint
                    .is_none_or(|at| at.elapsed() >= STATUS_PAINT);
                if changed
                    && due
                    && let Some(editor) = self.editor.as_mut()
                {
                    editor.request_paint(Instant::now());
                    self.last_status_paint = Some(Instant::now());
                }
            }
            UserEvent::Preview(picture) => {
                if let Some(editor) = self.editor.as_mut() {
                    self.editor_ui.set_preview(editor.ctx(), picture);
                    editor.request_paint(Instant::now());
                }
            }
            UserEvent::Thumb { key, picture } => {
                if let Some(editor) = self.editor.as_mut() {
                    self.editor_ui.thumbs.deliver(editor.ctx(), key, picture);
                    editor.request_paint(Instant::now());
                }
            }
            UserEvent::ThumbRetry(key) => self.editor_ui.thumbs.retry(key),
            UserEvent::SurfaceLost => {
                // The stage's surface went away (a monitor was unplugged): re-place the window
                // where the monitors now say it goes, and rebuild the surface on its handle.
                self.refresh_monitors(event_loop);
                self.apply_plan();
                self.send_replace_surface();
            }
            UserEvent::Note(text) => self.log(text),
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        if self.next_monitor_check.is_none_or(|at| now >= at) {
            self.next_monitor_check = Some(now + MONITOR_CHECK);
            self.refresh_monitors(event_loop);
            self.apply_plan();
            let hidden = self
                .plan
                .is_some_and(|plan| plan.stage == StageWindow::Hidden);
            if self.stage_window.is_none() && !hidden {
                // A monitor came back after the stage had no window at all.
                self.create_stage(event_loop);
                self.send_replace_surface();
            }
            self.sync_preview();
            // Where the performer put the windows, written at most this often.
            self.save_layout();
        }
        // The control window paints on change only, so an idle app sleeps until the next monitor
        // check (or the next frame the editor asked for).
        self.paint_editor(now);
        let mut next = self.next_monitor_check.unwrap_or(now + MONITOR_CHECK);
        if let Some(editor) = self.editor.as_ref()
            && let Some(at) = editor.next_paint_at()
        {
            next = next.min(at);
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(next));
    }
}

/// Run the app: the project from the first argument, or the default Autopilot show.
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (path, fps) = parse_args(&args);
    let (project, path, note) = match path {
        None => (default_project(), None, None),
        Some(path) => match mosh_show::io::load(&path) {
            Ok(project) => (project, Some(path), None),
            Err(error) => (
                default_project(),
                None,
                Some(format!(
                    "project {path} could not be read ({error}); running the default show"
                )),
            ),
        },
    };

    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    event_loop.set_control_flow(ControlFlow::Wait);
    let proxy = event_loop.create_proxy();
    let mut shell = Shell::new(project, path, note, proxy, fps);
    event_loop.run_app(&mut shell)?;
    shell.shutdown();
    Ok(())
}

/// The command line: `ilanaVisual [project.ron] [--fps <n>]`.
///
/// `--fps` is the stage's frame rate cap, `0` for no cap; absent means
/// [`mosh_render::DEFAULT_MAX_FPS`]. A cap is what keeps a 120 Hz panel from drawing 120 frames a
/// second, which is twice the GPU time per second the design's budget is written for.
pub fn parse_args(args: &[String]) -> (Option<String>, u32) {
    let mut path = None;
    let mut fps = mosh_render::DEFAULT_MAX_FPS;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--fps" => {
                if let Some(value) = args.get(index + 1).and_then(|value| value.parse().ok()) {
                    fps = value;
                }
                index += 2;
            }
            other => {
                if path.is_none() {
                    path = Some(other.to_string());
                }
                index += 1;
            }
        }
    }
    (path, fps)
}

#[cfg(test)]
mod tests {
    use super::parse_args;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn the_frame_cap_is_a_setting_that_defaults_to_sixty() {
        assert_eq!(parse_args(&args(&[])), (None, 60));
        assert_eq!(parse_args(&args(&["show.ron"])), (Some("show.ron".into()), 60));
        assert_eq!(
            parse_args(&args(&["show.ron", "--fps", "30"])),
            (Some("show.ron".into()), 30)
        );
        assert_eq!(
            parse_args(&args(&["--fps", "120", "show.ron"])),
            (Some("show.ron".into()), 120)
        );
        // 0 is the uncapped setting, and a missing or unreadable value leaves the default.
        assert_eq!(parse_args(&args(&["--fps", "0"])), (None, 0));
        assert_eq!(parse_args(&args(&["--fps"])), (None, 60));
        assert_eq!(parse_args(&args(&["--fps", "fast"])), (None, 60));
    }
}
