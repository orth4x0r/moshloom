//! The control window: the editors' winit window, on its own OpenGL context.
//!
//! **Why OpenGL and not Vulkan.** The stage is a Vulkan device (`mosh-render`, ash) driven by a
//! render thread that presents with FIFO on the stage's swapchain. A second renderer on that
//! device would share its queue and its pacing: a control-window repaint could push a stage frame
//! late, which is the one thing the design forbids ("the control window must cost nothing on
//! stage"). `egui_glow` paints the control window through the desktop OpenGL driver (WGL on
//! Windows) on a context that the Vulkan device cannot see, from the window thread, and only when
//! this module decides a frame is due (at most 30 a second, and nothing at all while the performer
//! is not touching it). The alternative in the brief — an ash renderer on its own device and
//! queue — costs a second Vulkan device and a second set of pipelines for the same result.
//!
//! Swap interval is `DontWait`: this window is painted on the same thread that runs the winit
//! event loop, so waiting for vsync here would hold up the stage window's events. The frame limit
//! is [`MAX_PERIOD`] instead.
//!
//! Nothing in here is covered by the crate's tests: windowed runs are coordinator-only
//! (AGENTS.md). The model and the widget tree it paints are tested without a window.

use std::sync::Arc;
use std::time::{Duration, Instant};

use egui::{FullOutput, ViewportId};
use glutin::config::{ConfigTemplateBuilder, GlConfig};
use glutin::context::{ContextApi, ContextAttributesBuilder, PossiblyCurrentContext, Version};
use glutin::display::GetGlDisplay;
use glutin::prelude::*;
use glutin::surface::{Surface, SurfaceAttributesBuilder, SwapInterval, WindowSurface};
use glutin_winit::{DisplayBuilder, GlWindow};
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::raw_window_handle::HasWindowHandle;
use winit::window::{Window, WindowAttributes};

use crate::editor::model::{Command, EditorModel};
use crate::editor::ui::{self, EditorUi};

/// The control window never paints more often than this, however loudly egui asks.
pub const MAX_PERIOD: Duration = Duration::from_millis(33);

/// A repaint delay this long means "nothing is pending": the window goes idle instead of waking
/// up a minute later for a frame nothing asked for.
const IDLE_AFTER: Duration = Duration::from_secs(1);

/// The editor's background, matching egui's dark theme.
const CLEAR: [f32; 4] = [0.055, 0.055, 0.07, 1.0];

/// The control window and everything needed to paint it.
pub struct EditorWindow {
    window: Window,
    egui: egui::Context,
    state: egui_winit::State,
    surface: Surface<WindowSurface>,
    context: PossiblyCurrentContext,
    painter: egui_glow::Painter,
    /// When the next frame is wanted; `None` while nothing asks for one.
    next_paint: Option<Instant>,
    last_paint: Option<Instant>,
    error: Option<String>,
}

impl EditorWindow {
    /// Build the control window and its OpenGL context.
    ///
    /// The window is created through glutin's builder so the config it gets is the one the
    /// context is made with; that is why the shell does not create it with `create_window`.
    pub fn new(event_loop: &ActiveEventLoop, title: &str) -> Result<EditorWindow, Box<dyn std::error::Error>> {
        let attributes = control_attributes(title);
        let template = ConfigTemplateBuilder::new()
            .with_alpha_size(8)
            .with_depth_size(0)
            .with_multisampling(0);
        let (window, gl_config) = DisplayBuilder::new()
            .with_window_attributes(Some(attributes))
            .build(event_loop, template, |configs| {
                // Any config the driver offers will do; take the one with the most samples so a
                // rounded egui shape does not crawl, which the platform chooses for us.
                configs
                    .reduce(|best, config| {
                        if config.num_samples() > best.num_samples() {
                            config
                        } else {
                            best
                        }
                    })
                    .expect("no OpenGL configuration for the control window")
            })?;
        let window = window.ok_or("the platform created no control window")?;

        let raw = window.window_handle()?.as_raw();
        let context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::OpenGl(Some(Version::new(3, 3))))
            .build(Some(raw));
        let display = gl_config.display();
        // Ask for GL 3.3 (what egui_glow's shaders assume on desktop); a driver that cannot give
        // it may still give a context the painter accepts, and a failure here is logged by the
        // shell rather than ending a set.
        let not_current = unsafe { display.create_context(&gl_config, &context_attributes) }
            .or_else(|_| {
                let plain = ContextAttributesBuilder::new().build(Some(raw));
                unsafe { display.create_context(&gl_config, &plain) }
            })?;
        let surface_attributes = window
            .build_surface_attributes(SurfaceAttributesBuilder::<WindowSurface>::new())?;
        let surface = unsafe { display.create_window_surface(&gl_config, &surface_attributes)? };
        let context = not_current.make_current(&surface)?;
        // Never block the event loop thread on a swapchain: see the module docs.
        surface.set_swap_interval(&context, SwapInterval::DontWait)?;

        let gl = unsafe {
            glow::Context::from_loader_function_cstr(|name| display.get_proc_address(name))
        };
        let painter = egui_glow::Painter::new(Arc::new(gl), "", None, true)?;

        let egui = egui::Context::default();
        egui.set_visuals(egui::Visuals::dark());
        let state = egui_winit::State::new(
            egui.clone(),
            ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(painter.max_texture_side()),
        );

        Ok(EditorWindow {
            window,
            egui,
            state,
            surface,
            context,
            painter,
            next_paint: Some(Instant::now()),
            last_paint: None,
            error: None,
        })
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    /// Ask for a frame (after a window event, a status line, or an applied command).
    pub fn request_paint(&mut self, now: Instant) {
        self.next_paint = Some(now);
    }

    /// Whether a frame should be painted now: something asked for one, the 30 fps limit since the
    /// last one has passed, and the window is on screen.
    ///
    /// A minimised control window paints nothing at all (the design's "can be hidden during a
    /// set"): every event is still handled, so restoring it paints the current state.
    pub fn wants_paint(&self, now: Instant) -> bool {
        !self.hidden() && self.next_paint_at().is_some_and(|at| now >= at)
    }

    /// When the window next wants a frame, including the 30 fps floor, if anything is pending.
    pub fn next_paint_at(&self) -> Option<Instant> {
        if self.hidden() {
            return None;
        }
        let asked = self.next_paint?;
        Some(match self.last_paint {
            Some(at) => asked.max(at + MAX_PERIOD),
            None => asked,
        })
    }

    fn hidden(&self) -> bool {
        self.window.is_minimized().unwrap_or(false)
    }

    /// Forward one window event to egui. Returns whether egui used it (so the shell does not also
    /// treat a typed `1` as a mode key).
    pub fn on_window_event(&mut self, event: &WindowEvent) -> bool {
        if let WindowEvent::Resized(size) = event
            && size.width > 0
            && size.height > 0
        {
            self.window.resize_surface(&self.surface, &self.context);
        }
        let response = self.state.on_window_event(&self.window, event);
        if response.repaint {
            self.request_paint(Instant::now());
        }
        response.consumed
    }

    /// Build one frame of widgets, paint it and return the commands the performer asked for.
    pub fn draw(&mut self, state: &mut EditorUi, model: &EditorModel) -> Vec<Command> {
        let input = self.state.take_egui_input(&self.window);
        let mut commands = Vec::new();
        let output = self.egui.run_ui(input, |root| {
            commands = ui::draw(root, state, model);
        });
        self.paint(output);
        commands
    }

    fn paint(&mut self, output: FullOutput) {
        let FullOutput {
            platform_output,
            shapes,
            textures_delta,
            pixels_per_point,
            viewport_output,
            ..
        } = output;
        self.state.handle_platform_output(&self.window, platform_output);

        let primitives = self.egui.tessellate(shapes, pixels_per_point);
        let size = self.window.inner_size();
        let screen = [size.width.max(1), size.height.max(1)];
        self.painter.clear(screen, CLEAR);
        // The painter drains the delta as it uploads, which is why it is passed by `&mut` and not
        // copied: a `TexturesDelta` dropped with unapplied deltas panics (by design).
        let mut textures_delta = textures_delta;
        self.painter.paint_and_update_textures(
            screen,
            pixels_per_point,
            &primitives,
            &mut textures_delta,
        );
        if let Err(error) = self.surface.swap_buffers(&self.context) {
            self.error = Some(format!("control window swap: {error}"));
        }

        let now = Instant::now();
        self.last_paint = Some(now);
        // egui asks for another frame when it is animating or a text cursor is blinking; the
        // 30 fps floor keeps a busy editor from spinning the event loop.
        let delay = viewport_output
            .get(&ViewportId::ROOT)
            .map_or(MAX_PERIOD, |viewport| viewport.repaint_delay);
        self.next_paint = if delay >= IDLE_AFTER {
            None
        } else {
            now.checked_add(delay.max(MAX_PERIOD))
        };
    }

    /// The last painting failure, once, for the shell to log.
    pub fn take_error(&mut self) -> Option<String> {
        self.error.take()
    }
}

impl Drop for EditorWindow {
    fn drop(&mut self) {
        // The painter's buffers belong to this context: make it current before they are deleted.
        let _ = self.context.make_current(&self.surface);
        self.painter.destroy();
    }
}

/// Kept out of `new` so the text does not fight the GL setup.
fn control_attributes(title: &str) -> WindowAttributes {
    Window::default_attributes()
        .with_title(title)
        .with_inner_size(PhysicalSize::new(1000, 700))
        .with_min_inner_size(PhysicalSize::new(560, 360))
}
