//! Adapter from `mosh-ipc`'s writer (plan Task 1.1) to [`BlockSink`].
//!
//! Compiled only with the `ipc` cargo feature, which stays off until `mosh-ipc` lands. The API
//! coded against here is the one in the plan's Task 1.1:
//!
//! ```ignore
//! IpcWriter::create(role: ShowRole, sample_rate: u32, channels: u16) -> Result<IpcWriter, IpcError>
//! IpcWriter::push_block(&mut self, t: Option<&Transport>, interleaved: &[f32], frames: u32)
//! IpcWriter::push_midi(&mut self, e: &MidiEvent)
//! ```

use mosh_core::ipc_layout::ShowRole;
use mosh_core::transport::{MidiEvent, Transport};
use mosh_ipc::IpcWriter;

use crate::sink::BlockSink;

/// The real writer. Created once in `Plugin::initialize()`.
pub struct IpcSink {
    writer: IpcWriter,
}

impl IpcSink {
    /// Open this instance's region, or `None` if it could not be created (the plugin then passes
    /// audio through and writes nothing).
    pub fn create(role: ShowRole, sample_rate: u32, channels: u16) -> Option<IpcSink> {
        IpcWriter::create(role, sample_rate, channels)
            .ok()
            .map(|writer| IpcSink { writer })
    }

    pub fn is_connected(&self) -> bool {
        true
    }
}

impl std::fmt::Debug for IpcSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IpcSink").finish_non_exhaustive()
    }
}

impl BlockSink for IpcSink {
    fn push_block(&mut self, transport: Option<&Transport>, interleaved: &[f32], frames: u32) {
        self.writer.push_block(transport, interleaved, frames);
    }

    fn push_midi(&mut self, event: &MidiEvent) {
        self.writer.push_midi(event);
    }
}
