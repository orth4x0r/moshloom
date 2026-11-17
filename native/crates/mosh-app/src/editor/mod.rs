//! Task 2.7's control window: the editors, the model behind them and the window that paints them.
//!
//! ```text
//! widgets (ui.rs) --Command--> EditorModel (model.rs) --Project--> AppCore at the next bar
//! ```
//!
//! The split is the point of the task:
//!
//! * [`model::EditorModel`] owns a [`mosh_core::Project`] and applies commands. It has no window,
//!   no GPU and no clock, so every screen's behaviour is unit-tested in `tests/editors.rs`.
//! * [`ui::draw`] maps widgets to commands and nothing else: it reads the model and returns the
//!   commands the performer asked for. That is what makes an idle frame measurable without a
//!   window.
//! * [`window::EditorWindow`] is the only part that needs a machine with a display: a winit window
//!   with its own OpenGL context, painted with `egui_glow`. The stage's Vulkan device, queue and
//!   frame pacing are never touched from here.
//!
//! Edits are not applied to the running show where they are made. The shell collects them, takes
//! the whole project from the model and hands it to `AppCore`, which swaps it in on the next bar
//! line (see `AppCore::queue_project`).

pub mod model;
pub mod ui;
pub mod window;

pub use model::{Command, EditorError, EditorModel, EditorOutcome, Screen, bars_at};
pub use ui::EditorUi;
pub use window::EditorWindow;
