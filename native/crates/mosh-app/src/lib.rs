//! mosh-app: the native live show. See docs/superpowers/plans/2026-11-17-native-port.md, Task 2.6.
//!
//! The crate is split so that everything except the window shell is testable without a window:
//!
//! | module | owns |
//! | --- | --- |
//! | [`core`] | `AppCore`: the analyzer, the show, audio input switching, control events, song markers, one `RenderPlan` per frame |
//! | [`budget`] | the design's performance budgets and the once-per-exceeded-budget log |
//! | [`monitor`] | `monitor_choice`: which monitor the stage takes, from whatever monitors exist |
//! | [`shell`] | the winit event loop, the two windows and the render thread |
pub mod budget;
pub mod core;
pub mod monitor;
pub mod shell;
mod win;
