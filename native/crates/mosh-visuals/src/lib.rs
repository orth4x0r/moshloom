//! mosh-visuals: turns one layer's `LayerState` into pixels, on top of mosh-render.
//!
//! Phase 2 of docs/superpowers/plans/2026-11-17-native-port.md. The coordinator froze the
//! signatures in this crate (like mosh-core in Phase 0); each Phase 2 lane fills in one module:
//!
//! | module | lane | task |
//! | --- | --- | --- |
//! | `visuals.rs` (this facade's bodies) and `scenes` | V-scenes | 2.1 |
//! | `mosh` | V-mosh | 2.2 |
//! | `effects`, `transitions` | V-fx | 2.3 |
//! | `fluid` | V-fluid | 2.4 |
//! | `overlay` | V-overlay | 2.8 |
//!
//! A layer is drawn as a fixed chain, each stage writing a different target so no target is
//! written twice in one frame (a `LayerTarget` is a ping-pong pair whose input side is the
//! previous frame):
//!
//! ```text
//! scene (or fluid, or overlay video) -> slot.scene
//!   -> mosh, if the clip has a Mosh effect slot with amount > 0     -> slot.post[0]
//!   -> every other effect slot, in ONE pass (like the web app's FX) -> slot.post[1]
//!   -> transition, while one is active                              -> slot.post[2]
//! ```
//!
//! A stage that has nothing to do returns `Ok(false)` and the chain keeps the previous output.
//! `LayerSlot::output()` is the picture the compositor blends.

pub mod effects;
pub mod fluid;
pub mod mosh;
pub mod overlay;
pub mod scenes;
pub mod transitions;
mod visuals;

pub use visuals::{LayerSlot, Stage, Visuals};
