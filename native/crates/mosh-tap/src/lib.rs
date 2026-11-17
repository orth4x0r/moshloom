//! mosh-tap: the two Moshloom VST3 plugins (plan Task 1.2).
//!
//! - **Moshloom Tap**, an audio effect for Ableton's master: passes audio through bit-exact and
//!   writes the block plus the transport to shared memory (role `master`, or `drums` / `bass` /
//!   `vocal` / `other` for a stem tap).
//! - **Moshloom Control**, an instrument for a MIDI track: silent stereo output, writes notes, CCs
//!   and program changes plus the transport to role `control`.
//!
//! Shape:
//!
//! - `host`, `tap`, `control`, `sink` are plain Rust: the whole of `process()` lives in
//!   `TapCore::push_interleaved` and `ControlCore::process`, plus `host::interleave_into`. They are
//!   tested with a fake sink, with no plugin host and no `mosh-ipc`.
//! - `glue` is the only module that touches nih-plug's host types.
//! - `MoshloomTap` / `MoshloomControl` are the nih-plug plugins: parameters, `initialize()`
//!   (where the writer is created), `process()` (which only copies and pushes) and `deactivate()`.
//! - `ipc_sink` adapts `mosh-ipc`'s `IpcWriter`, behind the `ipc` cargo feature (off by default
//!   until Task 1.1 lands).
//!
//! The audio callback never allocates, never locks and never creates the writer: `tests/alloc.rs`
//! counts allocations across 10 000 blocks with a counting global allocator.

pub mod control;
pub mod glue;
pub mod host;
pub mod sink;
pub mod tap;

#[cfg(feature = "ipc")]
pub mod ipc_sink;

pub use control::{ControlCore, ControlStats, HostEvent, MoshloomControl};
pub use host::{HostTransport, interleave_into};
pub use sink::{BlockSink, NullSink, PluginSink};
pub use tap::{MoshloomTap, TapCore};

// Both plugins live in one binary; the VST3 factory reports two classes. `cargo xtask bundle
// mosh-tap` (see `xtask/`) bundles it once, and Live lists Moshloom Tap and Moshloom Control
// separately.
nih_plug::nih_export_vst3!(MoshloomTap, MoshloomControl);
