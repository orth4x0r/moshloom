//! The control window: the editors, the model behind them and the window that paints them.
//!
//! ```text
//! widgets (ui/) --Command--> EditorModel (model.rs) --Project--> AppCore at the next bar
//! ```
//!
//! The split is the point:
//!
//! * [`model::EditorModel`] owns a [`mosh_core::Project`] and applies commands. It has no window,
//!   no GPU and no clock, so every edit's behaviour is unit-tested in `tests/editors.rs`.
//! * [`ui::draw`] maps widgets to commands and nothing else: it reads the model and returns the
//!   commands the performer asked for. That is what makes a frame measurable without a window
//!   (`tests/ui.rs` builds both views headless, at three window sizes).
//! * [`theme`] is the one place colours, sizes and spacing come from.
//! * [`window::EditorWindow`] is the only part that needs a machine with a display: a winit window
//!   with its own OpenGL context, painted with `egui_glow`. The stage's Vulkan device, queue and
//!   frame pacing are never touched from here.
//!
//! The interface is Ableton's: two main views, **Arrangement** (the timeline that follows Live's
//! playhead) and **Session** (the clip grid), toggled with `Tab`; a browser on the left to drag
//! visualizers and presets from; a detail panel at the bottom for the selected clip
//! (`Shift+Tab`); a top bar that says where the show is.
//!
//! Edits are not applied to the running show where they are made. The shell collects them, takes
//! the whole project from the model and hands it to `AppCore`, which swaps it in on the next bar
//! line (see `AppCore::queue_project`).

pub mod model;
pub mod theme;
pub mod ui;
pub mod window;

pub use model::{
    ClipSource, Command, EditorError, EditorModel, EditorOutcome, Panels, SpanRef, View, bars_at,
};
pub use ui::EditorUi;
pub use window::EditorWindow;
