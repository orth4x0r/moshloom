//! Shared-memory layout between the plugins (writers) and the app (reader).
//!
//! Region = `IpcHeader` + `[IpcBlock; RING_BLOCKS]` + `[IpcMidiEvent; MIDI_RING]`.
//! Mapping name: `Local\moshloom.<role>.v1`, role in lowercase (see `ShowRole::name`).
//! Sizes are asserted in tests/layout.rs; changing them changes `IPC_VERSION`.

use crate::transport::{MidiEvent, Transport};
use std::sync::atomic::AtomicU64;

pub const IPC_MAGIC: u32 = 0x4D4F_5348; // "MOSH"
pub const IPC_VERSION: u32 = 1;
pub const RING_BLOCKS: usize = 256;
pub const BLOCK_MAX_FRAMES: usize = 1024;
/// BLOCK_MAX_FRAMES * 2 channels.
pub const BLOCK_MAX_SAMPLES: usize = 2048;
pub const MIDI_RING: usize = 1024;

/// Which plugin instance a region belongs to. Version 1 of the app reads `Master` and `Control`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShowRole {
    Master = 0,
    Drums = 1,
    Bass = 2,
    Vocal = 3,
    Other = 4,
    Control = 16,
}

impl ShowRole {
    pub fn name(self) -> &'static str {
        match self {
            ShowRole::Master => "master",
            ShowRole::Drums => "drums",
            ShowRole::Bass => "bass",
            ShowRole::Vocal => "vocal",
            ShowRole::Other => "other",
            ShowRole::Control => "control",
        }
    }

    /// `Local\moshloom.<role>.v1`
    pub fn mapping_name(self) -> String {
        format!("Local\\moshloom.{}.v{}", self.name(), IPC_VERSION)
    }

    pub fn from_u32(v: u32) -> Option<ShowRole> {
        Some(match v {
            0 => ShowRole::Master,
            1 => ShowRole::Drums,
            2 => ShowRole::Bass,
            3 => ShowRole::Vocal,
            4 => ShowRole::Other,
            16 => ShowRole::Control,
            _ => return None,
        })
    }
}

/// `Transport` as stored in shared memory. 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TransportRaw {
    pub tempo_bpm: f64,
    pub ppq: f64,
    pub bar_start_ppq: f64,
    pub sample_pos: i64,
    pub sig_num: u16,
    pub sig_den: u16,
    pub playing: u8,
    /// 0 = no transport in this block (the host gave none).
    pub valid: u8,
    pub _pad: [u8; 26],
}

impl TransportRaw {
    pub fn from_transport(t: Option<&Transport>) -> TransportRaw {
        match t {
            None => TransportRaw::default(),
            Some(t) => TransportRaw {
                tempo_bpm: t.tempo_bpm,
                ppq: t.ppq,
                bar_start_ppq: t.bar_start_ppq,
                sample_pos: t.sample_pos,
                sig_num: t.sig_num,
                sig_den: t.sig_den,
                playing: t.playing as u8,
                valid: 1,
                _pad: [0; 26],
            },
        }
    }

    pub fn to_transport(&self) -> Option<Transport> {
        (self.valid != 0).then_some(Transport {
            tempo_bpm: self.tempo_bpm,
            ppq: self.ppq,
            bar_start_ppq: self.bar_start_ppq,
            sig_num: self.sig_num,
            sig_den: self.sig_den,
            playing: self.playing != 0,
            sample_pos: self.sample_pos,
        })
    }
}

/// Region header. 128 bytes.
#[repr(C)]
#[derive(Debug)]
pub struct IpcHeader {
    pub magic: u32,
    pub version: u32,
    pub role: u32,
    pub sample_rate: u32,
    pub channels: u32,
    pub block_max_frames: u32,
    pub ring_blocks: u32,
    pub midi_ring: u32,
    /// Total blocks ever written; slot = write_index % RING_BLOCKS.
    pub write_index: AtomicU64,
    /// Total MIDI events ever written; slot = midi_write_index % MIDI_RING.
    pub midi_write_index: AtomicU64,
    /// Blocks the writer overwrote before they could have been read (writer-side estimate).
    pub dropped: AtomicU64,
    /// Writer's monotonic clock in nanoseconds, updated every block.
    pub heartbeat_ns: AtomicU64,
    pub _pad: [u8; 64],
}

/// One audio block slot, 8320 bytes, 64-byte aligned. `seq` is a seqlock: odd while being written.
#[repr(C, align(64))]
#[derive(Debug)]
pub struct IpcBlock {
    pub seq: AtomicU64,
    pub frames: u32,
    pub channels: u32,
    pub _pad: [u8; 48],
    pub transport: TransportRaw,
    /// Interleaved, `frames * channels` used.
    pub samples: [f32; BLOCK_MAX_SAMPLES],
}

/// One MIDI event slot, 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct IpcMidiEvent {
    pub ppq: f64,
    pub status: u8,
    pub data1: u8,
    pub data2: u8,
    pub _pad: [u8; 5],
}

impl IpcMidiEvent {
    pub fn from_event(e: &MidiEvent) -> IpcMidiEvent {
        IpcMidiEvent { ppq: e.ppq, status: e.status, data1: e.data1, data2: e.data2, _pad: [0; 5] }
    }

    pub fn to_event(&self) -> MidiEvent {
        MidiEvent { ppq: self.ppq, status: self.status, data1: self.data1, data2: self.data2 }
    }
}

/// Byte offsets of the three parts of a region.
pub const HEADER_BYTES: usize = std::mem::size_of::<IpcHeader>();
pub const BLOCKS_OFFSET: usize = HEADER_BYTES;
pub const MIDI_OFFSET: usize = BLOCKS_OFFSET + RING_BLOCKS * std::mem::size_of::<IpcBlock>();
pub const REGION_BYTES: usize = MIDI_OFFSET + MIDI_RING * std::mem::size_of::<IpcMidiEvent>();
