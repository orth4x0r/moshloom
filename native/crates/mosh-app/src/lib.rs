//! mosh-app: the native live show. See docs/superpowers/plans/2026-11-17-native-port.md, Task 2.6.
//!
//! The crate is split so that everything except the window shell is testable without a window:
//!
//! | module | owns |
//! | --- | --- |
//! | [`core`] | `AppCore`: the analyzer, the show, audio input switching, control events, song markers, one `RenderPlan` per frame |
//! | [`budget`] | the design's performance budgets and the once-per-exceeded-budget log |
//! | [`editor`] | the control window: the editing model (no window), the theme, the egui views, the window itself |
//! | [`leds`] | the APC mini's pad LEDs as a mirror of the Session grid |
//! | [`monitor`] | which monitor the stage takes and where both windows go, from whatever monitors exist |
//! | [`preview`] | the stage preview and clip thumbnails: sizes, rate, the offscreen readbacks |
//! | [`shell`] | the winit event loop, the two windows and the render thread |
pub mod budget;
pub mod core;
pub mod editor;
pub mod leds;
pub mod monitor;
pub mod preview;
pub mod shell;
mod win;
