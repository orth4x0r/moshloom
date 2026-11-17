//! The window shell: two windows and one render thread.
//!
//! The design's shape, once:
//!
//! - the **stage** window is a borderless fullscreen window on the first non-primary monitor, or
//!   a borderless window on the laptop screen when that monitor is missing. It never shows UI.
//! - the **control** window is the editor: its own winit window on its own OpenGL context
//!   (`crate::editor`), painted only when something changes and at most 30 times a second, so it
//!   costs the stage nothing while the performer is not touching it.
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

use mosh_core::{AnalysisFrame, Clip, ControlEvent, Layer, Project, RenderPlan, ShowMode};
use mosh_render::{FramePacer, FrameRecorder, Gpu, GpuPreference, RenderError, Stage, WaitTimer};
use mosh_visuals::{LayerSlot, Visuals};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, NamedKey};
use winit::monitor::MonitorHandle;
use winit::raw_window_handle::HasWindowHandle;
use winit::window::{Fullscreen, Window, WindowId};

use crate::budget::BudgetWatch;
use crate::core::{AppCore, SourceName, Status, default_project, stage_placement_label, title_for};
use crate::editor::model::Command as EditorCommand;
use crate::editor::{EditorModel, EditorUi, EditorWindow};
use crate::monitor::{MonitorInfo, StagePlacement, monitor_choice};
use crate::win::WindowHandle;

/// How often the monitor list is re-read, so an unplugged or replugged stage is noticed without
/// the human touching anything.
const MONITOR_CHECK: Duration = Duration::from_secs(2);
/// Stats are sent to the window thread at this rate; the control window's title is set at most
/// 30 times a second regardless.
const STATUS_INTERVAL: Duration = Duration::from_millis(100);
/// How often the control window repaints for a changed status line: the numbers are for a glance,
/// not for the stage, so four times a second is plenty and keeps the editor's cost at nothing.
const STATUS_PAINT: Duration = Duration::from_millis(250);
const TITLE_INTERVAL: Duration = Duration::from_millis(33);
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
    slots: Vec<LayerSlot>,
    visuals: Option<Visuals>,
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
            slots: Vec::with_capacity(layers.len()),
            visuals: Some(visuals),
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
        // Build the new visuals before dropping the old ones: if this fails the old scene is
        // still intact and the next frame can try again.
        let visuals = Visuals::new(gpu, w, h)?;
        self.visuals = Some(visuals);
        self.out = out;
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

/// What one presented frame measured.
struct Outcome {
    cpu_ms: f32,
    gpu_ms: f32,
    surface_lost: bool,
}

/// Draw every layer of `plan` and present the result. `frame_start` is when the frame's CPU work
/// began, so `cpu_ms` covers analysis, the show and the recording but not the vsync wait.
fn render_frame(
    scene: &mut Scene,
    plan: &RenderPlan,
    analysis: &AnalysisFrame,
    t: f32,
    frame_start: Instant,
    videos: &mut VideoBank,
    project: &Project,
) -> Result<Outcome, RenderError> {
    let Some(gpu) = scene.gpu.as_ref() else {
        return Err(RenderError::Stage("no device".into()));
    };

    // Overlay video goes into the layer's texture before the layer is drawn.
    for (index, state) in plan.layers.iter().enumerate() {
        let Some(clip_id) = state.clip else { continue };
        let Some(clip) = project.clips.iter().find(|clip| clip.id == clip_id) else {
            continue;
        };
        let Some(video) = videos.frame_for(clip, analysis.time_s) else {
            continue;
        };
        let (width, height) = (video.w, video.h);
        if let Some(visuals) = scene.visuals.as_mut() {
            visuals
                .overlay
                .set_frame(gpu, index, width, height, &video.bgra)?;
        }
    }

    // One slot per layer, with at least as many as the plan draws.
    while scene.slots.len() < plan.layers.len() {
        let index = scene.slots.len();
        let (w, h) = internal_size(scene.out, 0.5);
        scene.slots.push(LayerSlot::new(gpu, index, w, h));
    }

    let mut recorder = FrameRecorder::new(gpu);
    {
        let Some(visuals) = scene.visuals.as_mut() else {
            return Err(RenderError::Stage("no visuals".into()));
        };
        for (index, state) in plan.layers.iter().enumerate() {
            visuals.draw_layer(&mut recorder, &mut scene.slots[index], state, analysis, t)?;
        }
        for (index, state) in plan.layers.iter().enumerate() {
            recorder.layer(scene.slots[index].output(), state.blend, state.opacity);
        }
    }

    let cpu_ms = frame_start.elapsed().as_secs_f32() * 1_000.0;
    let Some(stage) = scene.stage.as_mut() else {
        // No window to present in: the layers were still drawn, so the next monitor that appears
        // shows a current frame instead of a black one.
        drop(recorder);
        return Ok(Outcome {
            cpu_ms,
            gpu_ms: 0.0,
            surface_lost: false,
        });
    };
    let present = stage.present(recorder);
    Ok(Outcome {
        cpu_ms,
        gpu_ms: stage.timings().frame_ms,
        surface_lost: present.surface_lost,
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
    /// A control event from the control window (a preview launch).
    Control(ControlEvent),
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
    /// The frame rate cap: the stage renders at most this many frames a second whatever the
    /// display's refresh is (the reference panel is 120 Hz).
    pacer: FramePacer,
    /// The wait the pacer's deadlines use: a high-resolution waitable timer, because the system
    /// tick would turn a 16.6 ms wait into 31 ms.
    timer: WaitTimer,
    paced_frames: u64,
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
            pacer: FramePacer::new(fps),
            timer: WaitTimer::new(),
            paced_frames: 0,
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
            Command::SetMode(mode) => self.core.show_mut().set_mode(mode),
            Command::SetProject(project) => {
                // A layer stack that changed shape needs its targets rebuilt; a clip edit does
                // not, and must not touch the GPU at all.
                if project.session.layers != self.project_layers {
                    self.project_layers = project.session.layers.clone();
                    let layers = self.project_layers.clone();
                    if let Some(scene) = self.scene.as_mut() {
                        scene.rebuild_slots(&layers);
                    }
                }
                self.core.queue_project(project);
            }
            Command::Control(event) => self.core.queue_control(event),
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

        if self.scene.is_none() && self.retry_at.is_none_or(|at| frame_start >= at) {
            self.build();
        }
        // A window that exists without a stage (a failed surface, a monitor change) gets its
        // stage rebuilt on the retry interval.
        let wants_stage = self.scene.as_ref().is_some_and(|scene| scene.stage.is_none());
        if wants_stage && self.handle.is_some() && self.retry_at.is_none_or(|at| frame_start >= at) {
            self.attach_stage();
        }

        let mut cpu_ms = 0.0;
        let mut gpu_ms = 0.0;
        let has_stage = self.scene.as_ref().is_some_and(|scene| scene.stage.is_some());
        if has_stage {
            if let Some(scene) = self.scene.as_mut() {
                let t = self.started.elapsed().as_secs_f32();
                match render_frame(
                    scene,
                    &frame.plan,
                    &frame.analysis,
                    t,
                    frame_start,
                    &mut self.videos,
                    self.core.show().project(),
                ) {
                    Ok(outcome) => {
                        cpu_ms = outcome.cpu_ms;
                        gpu_ms = outcome.gpu_ms;
                        if outcome.surface_lost {
                            let _ = self.proxy.send_event(UserEvent::SurfaceLost);
                        }
                    }
                    Err(error) => self.recover(&error),
                }
            }
        } else {
            // Nothing to present: the show keeps its clock, and the loop sleeps instead of
            // spinning on an acquire that can never succeed.
            std::thread::sleep(IDLE_TICK);
        }

        for line in self.budget.sample(cpu_ms, gpu_ms, Instant::now()) {
            eprintln!("ilanaVisual: {line}");
        }

        if self
            .last_status
            .is_none_or(|at| frame_start.saturating_duration_since(at) >= STATUS_INTERVAL)
        {
            self.last_status = Some(frame_start);
            let status = Status {
                mode: frame.mode.into(),
                source: SourceName(frame.source),
                tempo_bpm: frame.analysis.tempo_bpm,
                bar: frame.analysis.bar_index,
                bar_phase: frame.analysis.bar_phase,
                level: frame.analysis.level,
                gpu_ms,
                cpu_ms,
            };
            let _ = self.proxy.send_event(UserEvent::Status(status));
        }
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
    /// The control window and its OpenGL context (Task 2.7). Absent when it could not be built:
    /// the show keeps running, because a set does not end because an editor window failed.
    editor: Option<EditorWindow>,
    monitors: Vec<MonitorHandle>,
    monitor_list: Vec<MonitorInfo>,
    placement: Option<StagePlacement>,
    render: Option<RenderHandle>,
    /// The stage's frame rate cap (`--fps`, default 60): a 120 Hz panel would otherwise draw
    /// twice the frames the budget is written for.
    fps: u32,
    title: Option<String>,
    last_title_at: Option<Instant>,
    last_status_paint: Option<Instant>,
    next_monitor_check: Option<Instant>,
    started: bool,
}

impl Shell {
    fn new(
        project: Project,
        note: Option<String>,
        proxy: EventLoopProxy<UserEvent>,
        fps: u32,
    ) -> Shell {
        Shell {
            model: EditorModel::new(project),
            editor_ui: EditorUi::default(),
            note,
            proxy,
            stage_window: None,
            editor: None,
            monitors: Vec::new(),
            monitor_list: Vec::new(),
            placement: None,
            render: None,
            fps,
            title: None,
            last_title_at: None,
            last_status_paint: None,
            next_monitor_check: None,
            started: false,
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
            let name = monitor.name().unwrap_or_else(|| format!("monitor {index}"));
            let is_primary = match primary.as_ref() {
                Some(primary) => same_monitor(primary, &monitor),
                None => index == 0,
            };
            self.monitor_list.push(MonitorInfo::new(
                index as u32,
                name,
                is_primary,
                size.width,
                size.height,
            ));
            self.monitors.push(monitor);
        }
    }

    fn monitor(&self, index: u32) -> Option<MonitorHandle> {
        self.monitors.get(index as usize).cloned()
    }

    fn create_control_window(&mut self, event_loop: &ActiveEventLoop) {
        let title = self
            .note
            .as_deref()
            .unwrap_or("ilanaVisual · control")
            .to_string();
        // Building a GL context is the one place the window thread talks to a driver that may be
        // absent or broken. A panic there must not take the show with it: the stage keeps running
        // on the render thread and the app logs one line.
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            EditorWindow::new(event_loop, &title)
        }));
        match built {
            Ok(Ok(editor)) => {
                self.log(format!(
                    "control window: {}x{} on OpenGL, at most 30 fps, painted on change",
                    editor.window().inner_size().width,
                    editor.window().inner_size().height
                ));
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
            }
            Err(_) => {
                self.editor = None;
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
            match self.model.apply(command) {
                Ok(outcome) => {
                    applied = true;
                    // The mode is a show-level action: the stage hears it now, not at the next bar.
                    if let Some(mode) = outcome.set_mode {
                        self.send(Command::SetMode(mode));
                    }
                    if let Some(event) = outcome.launch {
                        self.send(Command::Control(event));
                    }
                }
                // The model already put the reason in the status strip.
                Err(_) => applied = true,
            }
        }
        if let Some(project) = self.model.handoff() {
            self.send(Command::SetProject(project));
        }
        if applied
            && let Some(editor) = self.editor.as_mut()
        {
            // An applied edit changes what the screens show: paint it, at the 30 fps ceiling.
            editor.request_paint(Instant::now());
        }
    }

    /// Create the stage window for the current monitor list, if there is anything to present on.
    fn create_stage(&mut self, event_loop: &ActiveEventLoop) {
        self.refresh_monitors(event_loop);
        let placement = monitor_choice(&self.monitor_list);
        self.placement = Some(placement);
        self.log(stage_placement_label(placement, &self.monitor_list));
        let Some(index) = placement.monitor() else {
            // No monitor at all: leave the stage absent and keep the show running.
            self.send(Command::DetachSurface);
            return;
        };
        let Some(monitor) = self.monitor(index) else {
            self.send(Command::DetachSurface);
            return;
        };
        let size = monitor.size();
        let mut attributes = Window::default_attributes()
            .with_title("ilanaVisual stage")
            .with_decorations(false)
            .with_visible(true);
        if placement.is_fullscreen() {
            attributes = attributes.with_fullscreen(Some(Fullscreen::Borderless(Some(monitor))));
        } else {
            attributes = attributes
                .with_inner_size(size)
                .with_position(monitor.position());
        }
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
            monitor: self.placement.and_then(StagePlacement::monitor),
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
        let monitor = self.placement.and_then(StagePlacement::monitor);
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

    /// Move the stage window to wherever `monitor_choice` now says it goes.
    fn update_stage_placement(&mut self, event_loop: &ActiveEventLoop) {
        let placement = monitor_choice(&self.monitor_list);
        if Some(placement) == self.placement {
            return;
        }
        self.placement = Some(placement);

        if placement.is_hidden() {
            if let Some(window) = self.stage_window.as_ref() {
                window.set_visible(false);
            }
            self.send(Command::DetachSurface);
            return;
        }
        if self.stage_window.is_none() {
            self.create_stage(event_loop);
            return;
        }
        let Some(index) = placement.monitor() else {
            return;
        };
        let Some(monitor) = self.monitor(index) else {
            return;
        };
        self.log(stage_placement_label(placement, &self.monitor_list));
        if let Some(window) = self.stage_window.as_ref() {
            apply_stage_placement(window, placement, &monitor);
        }
        self.send_replace_surface();
    }

    fn log(&self, text: String) {
        eprintln!("ilanaVisual: {text}");
    }

    fn shutdown(&mut self) {
        self.send(Command::Quit);
        if let Some(mut render) = self.render.take()
            && let Some(thread) = render.thread.take()
        {
            let _ = thread.join();
        }
    }
}

/// Put the stage window where `placement` says it belongs.
fn apply_stage_placement(window: &Window, placement: StagePlacement, monitor: &MonitorHandle) {
    match placement {
        StagePlacement::Fullscreen(_) => {
            window.set_visible(true);
            window.set_fullscreen(Some(Fullscreen::Borderless(Some(monitor.clone()))));
        }
        StagePlacement::Laptop(_) => {
            window.set_visible(true);
            window.set_fullscreen(None);
            window.set_decorations(false);
            window.set_outer_position(monitor.position());
            let _ = window.request_inner_size(monitor.size());
        }
        StagePlacement::Hidden => window.set_visible(false),
    }
}

/// `MonitorHandle` has no identity of its own, so two handles are the same monitor when they
/// agree on the name, size and position (all the OS tells us).
fn same_monitor(a: &MonitorHandle, b: &MonitorHandle) -> bool {
    a.name() == b.name() && a.size() == b.size() && a.position() == b.position()
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
        self.create_control_window(event_loop);
        self.create_stage(event_loop);
        self.start_render();
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

        // The editor sees its own events first: a keystroke it used (a path, a name) must not also
        // be read as a mode shortcut.
        let mut consumed = false;
        if is_control
            && let Some(editor) = self.editor.as_mut()
        {
            consumed = editor.on_window_event(&event);
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
                        ..
                    },
                ..
            } => {
                if !is_control || consumed {
                    return;
                }
                match logical_key {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    // Mode keys, for rehearsing with the editor window closed. The Project screen
                    // is the real control; these stay because they work with the window hidden.
                    Key::Character(key) if key.as_str() == "1" => {
                        self.send(Command::SetMode(ShowMode::Arrangement))
                    }
                    Key::Character(key) if key.as_str() == "2" => {
                        self.send(Command::SetMode(ShowMode::Session))
                    }
                    Key::Character(key) if key.as_str() == "3" => {
                        self.send(Command::SetMode(ShowMode::Autopilot))
                    }
                    _ => {}
                }
            }
            WindowEvent::Resized(size) => {
                if is_stage && size.width > 0 && size.height > 0 {
                    self.send(Command::Resize(size.width, size.height));
                }
            }
            WindowEvent::Destroyed => {
                if is_stage {
                    self.stage_window = None;
                    self.placement = None;
                    self.send(Command::DetachSurface);
                } else if is_control {
                    self.editor = None;
                }
            }
            _ => {}
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Status(status) => {
                // The status strip is part of the control window: a changed number is a reason to
                // paint it, an unchanged one is not, and it is throttled to four times a second so
                // a running show does not repaint the editor ten times a second for a bar phase.
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
                let Some(title) = title_for(self.title.as_deref(), &status) else {
                    return;
                };
                let due = self
                    .last_title_at
                    .is_none_or(|at| at.elapsed() >= TITLE_INTERVAL);
                if !due {
                    return;
                }
                if let Some(editor) = self.editor.as_ref() {
                    editor.window().set_title(&title);
                }
                self.title = Some(title);
                self.last_title_at = Some(Instant::now());
            }
            UserEvent::SurfaceLost => {
                // The stage's surface went away (a monitor was unplugged): re-place the window
                // where the monitors now say it goes, and rebuild the surface on its handle.
                self.refresh_monitors(event_loop);
                self.update_stage_placement(event_loop);
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
            self.update_stage_placement(event_loop);
            if self.stage_window.is_none() && !self.placement.is_some_and(StagePlacement::is_hidden) {
                // A monitor came back after the stage had no window at all.
                self.create_stage(event_loop);
                self.send_replace_surface();
            }
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
    let (project, note) = match path.as_deref() {
        None => (default_project(), None),
        Some(path) => match mosh_show::io::load(path) {
            Ok(project) => (project, None),
            Err(error) => (
                default_project(),
                Some(format!(
                    "project {path} could not be read ({error}); running the default show"
                )),
            ),
        },
    };

    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    event_loop.set_control_flow(ControlFlow::Wait);
    let proxy = event_loop.create_proxy();
    let mut shell = Shell::new(project, note, proxy, fps);
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
