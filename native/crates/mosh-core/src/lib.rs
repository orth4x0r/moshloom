//! mosh-core: the shared contract between every native lane.
//!
//! Frozen in Phase 0 of docs/superpowers/plans/2026-11-17-native-port.md (Appendix A).
//! Changes go through the coordinator, never through a lane.

pub mod analysis;
pub mod control;
pub mod ipc_layout;
pub mod show;
pub mod transport;

pub use analysis::*;
pub use control::*;
pub use ipc_layout::*;
pub use show::*;
pub use transport::*;
