//! Where a plugin pushes what it sees: the shared-memory writer, or nothing.
//!
//! `BlockSink` is the whole of what the audio callback may do, so a test fake and the real
//! `mosh-ipc` writer are interchangeable.

use mosh_core::ipc_layout::ShowRole;
use mosh_core::transport::{MidiEvent, Transport};

/// The writer side of the IPC contract (Task 1.1), reduced to the two calls a plugin makes.
///
/// Both must be realtime-safe: no allocation, no lock, no blocking wait.
pub trait BlockSink {
    /// Push one interleaved block. `frames * channels` samples of `interleaved` are read.
    fn push_block(&mut self, transport: Option<&Transport>, interleaved: &[f32], frames: u32);
    /// Push one MIDI event, stamped with its position in ppq.
    fn push_midi(&mut self, event: &MidiEvent);
}

/// A sink that throws everything away: what the plugins use when there is no shared memory to
/// write to (the app is not running, or the writer could not be created).
#[derive(Clone, Copy, Debug, Default)]
pub struct NullSink;

impl BlockSink for NullSink {
    fn push_block(&mut self, _transport: Option<&Transport>, _interleaved: &[f32], _frames: u32) {}

    fn push_midi(&mut self, _event: &MidiEvent) {}
}

/// The sink a plugin instance actually owns.
///
/// With the `ipc` feature off (the default until `mosh-ipc` lands) this is always `Null`, and the
/// plugins are pure pass-throughs. `PluginSink::create` is only ever called from
/// `Plugin::initialize()`: the audio thread never maps memory.
#[derive(Debug)]
pub enum PluginSink {
    Null(NullSink),
    #[cfg(feature = "ipc")]
    Ipc(crate::ipc_sink::IpcSink),
}

impl PluginSink {
    /// Open this plugin instance's region, falling back to `Null` when that fails.
    ///
    /// A missing mapping is not an error: the plugin must pass audio through with no app running.
    pub fn create(role: ShowRole, sample_rate: u32, channels: u16) -> PluginSink {
        #[cfg(feature = "ipc")]
        {
            match crate::ipc_sink::IpcSink::create(role, sample_rate, channels) {
                Some(sink) => return PluginSink::Ipc(sink),
                None => nih_plug::nih_warn!(
                    "mosh-tap: no shared memory for role {}, passing audio through without a writer",
                    role.name()
                ),
            }
        }

        let _ = (role, sample_rate, channels);
        PluginSink::Null(NullSink)
    }

    /// Whether this instance is writing to the app.
    pub fn is_connected(&self) -> bool {
        match self {
            PluginSink::Null(_) => false,
            #[cfg(feature = "ipc")]
            PluginSink::Ipc(_) => true,
        }
    }
}

impl BlockSink for PluginSink {
    fn push_block(&mut self, transport: Option<&Transport>, interleaved: &[f32], frames: u32) {
        match self {
            PluginSink::Null(sink) => sink.push_block(transport, interleaved, frames),
            #[cfg(feature = "ipc")]
            PluginSink::Ipc(sink) => sink.push_block(transport, interleaved, frames),
        }
    }

    fn push_midi(&mut self, event: &MidiEvent) {
        match self {
            PluginSink::Null(sink) => sink.push_midi(event),
            #[cfg(feature = "ipc")]
            PluginSink::Ipc(sink) => sink.push_midi(event),
        }
    }
}

/// The source roles one Moshloom Tap instance may take. `ShowRole::Control` is not in here: only
/// Moshloom Control writes it.
pub const ROLE_SLOTS: i32 = 5;

/// The `role` parameter's plain value (0..=4) as a `ShowRole`. Out-of-range values clamp, so the
/// audio thread never panics on a host sending something odd.
pub fn role_for_index(index: i32) -> ShowRole {
    match index {
        i if i <= 0 => ShowRole::Master,
        1 => ShowRole::Drums,
        2 => ShowRole::Bass,
        3 => ShowRole::Vocal,
        _ => ShowRole::Other,
    }
}

/// The `role` parameter value for a role, or `None` for `ShowRole::Control`.
pub fn role_index_of(role: ShowRole) -> Option<i32> {
    match role {
        ShowRole::Master => Some(0),
        ShowRole::Drums => Some(1),
        ShowRole::Bass => Some(2),
        ShowRole::Vocal => Some(3),
        ShowRole::Other => Some(4),
        ShowRole::Control => None,
    }
}

/// The `role` parameter's label.
pub fn role_label(index: i32) -> &'static str {
    match role_for_index(index) {
        ShowRole::Master => "master",
        ShowRole::Drums => "drums",
        ShowRole::Bass => "bass",
        ShowRole::Vocal => "vocal",
        _ => "other",
    }
}

/// The `role` parameter value for a label, for typed parameter entry.
pub fn role_index_from_label(label: &str) -> Option<i32> {
    let label = label.trim();
    (0..ROLE_SLOTS).find(|index| role_label(*index).eq_ignore_ascii_case(label))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_round_trip() {
        for index in 0..ROLE_SLOTS {
            assert_eq!(role_index_from_label(role_label(index)), Some(index));
        }
        assert_eq!(role_index_from_label("Master"), Some(0));
        assert_eq!(role_index_from_label("nope"), None);
    }

    #[test]
    fn the_null_sink_is_never_connected() {
        assert!(!PluginSink::Null(NullSink).is_connected());
    }
}
