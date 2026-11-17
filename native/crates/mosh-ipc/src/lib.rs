//! mosh-ipc: the shared-memory rings between the Ableton plugins (writers) and the app (reader).
//!
//! Task 1.1 of docs/superpowers/plans/2026-11-17-native-port.md. The layout itself is frozen in
//! `mosh-core::ipc_layout`; this crate owns the mapping, the seqlock and the poll loop.
//!
//! ```no_run
//! use mosh_core::ShowRole;
//! use mosh_ipc::{IpcReader, IpcWriter};
//!
//! // In the plugin (audio callback): create once, push per block.
//! let mut writer = IpcWriter::create(ShowRole::Master, 48_000, 2)?;
//! writer.push_block(None, &[0.0; 512], 256);
//!
//! // In the app: attach at any time, poll when it suits.
//! let mut reader = IpcReader::open(ShowRole::Master)?;
//! let mut blocks = Vec::new();
//! let mut midi = Vec::new();
//! let stats = reader.poll(&mut blocks, &mut midi);
//! # Ok::<(), mosh_ipc::IpcError>(())
//! ```
//!
//! # What the writer promises
//!
//! - `push_block` and `push_midi` never allocate, never lock, never block. They copy into a
//!   pre-mapped region and bump a heartbeat. Safe to call from an audio callback.
//! - A full ring overwrites its oldest block: the newest data always wins, which is what a
//!   visualizer wants after a stall.
//!
//! # What the reader promises
//!
//! - `poll` never returns a torn block: a slot that is being written, or that changed while it
//!   was copied, is skipped.
//! - `poll` says how many blocks it never saw (`PollStats::dropped`) and whether the writer's
//!   heartbeat is younger than 500 ms (`PollStats::writer_alive`).
//! - Attaching late, or attaching after the writer exited, is not an error state: the mapping
//!   lives while either side holds it.
//!
//! # Drop accounting
//!
//! The frozen `IpcHeader` has no field for the reader's position, so the writer cannot know which
//! of its overwrites the reader had already consumed. `PollStats::dropped` is therefore computed
//! by the reader, where the information is: blocks skipped when attaching (`write_index` was more
//! than a ring ahead), blocks that fell off the tail while the reader was behind, and blocks that
//! were in flight and torn. `IpcHeader::dropped` counts slot reuses from the writer's side
//! instead: an upper bound on drops, never lower.

#![deny(unsafe_op_in_unsafe_fn)]

mod ring;
mod win;

use std::fmt;

use mosh_core::{
    AudioBlock, BLOCK_MAX_FRAMES, IPC_MAGIC, IPC_VERSION, MIDI_RING, MidiEvent, REGION_BYTES,
    RING_BLOCKS, ShowRole, Transport,
};
use std::sync::atomic::Ordering;

/// A writer heartbeat older than this means the writer is gone (or stalled).
pub const HEARTBEAT_TIMEOUT_MS: u64 = 500;
const HEARTBEAT_TIMEOUT_NS: u64 = HEARTBEAT_TIMEOUT_MS * 1_000_000;

/// What one `IpcReader::poll` did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PollStats {
    /// Blocks appended to the caller's `out` by this poll.
    pub new_blocks: u32,
    /// Blocks this reader never saw, counted from the moment it attached: overwritten before a
    /// first read, fell off the tail while it was behind, or caught mid-write and skipped.
    pub dropped: u64,
    /// The writer's heartbeat is younger than 500 ms.
    pub writer_alive: bool,
}

/// Everything that can go wrong while attaching to a region.
#[derive(Debug)]
pub enum IpcError {
    /// No writer is running for this role (or the app is not allowed to open the object).
    Open(std::io::Error),
    /// The region exists but could not be created or mapped.
    Map(std::io::Error),
    /// Channel count outside 1..=2: `IpcBlock` holds at most two interleaved channels.
    BadChannels(u16),
    /// Channel count outside 1..=2 or zero sample rate.
    BadFormat { sample_rate: u32, channels: u16 },
    /// The mapping exists but does not carry this crate's magic: not a moshloom region.
    BadMagic(u32),
    /// A different IPC version wrote it. The mapping name carries the version, so this means a
    /// writer built against another `mosh-core`.
    VersionMismatch { found: u32, expected: u32 },
}

impl fmt::Display for IpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IpcError::Open(e) => write!(f, "no moshloom writer: {e}"),
            IpcError::Map(e) => write!(f, "shared memory mapping failed: {e}"),
            IpcError::BadChannels(c) => {
                write!(
                    f,
                    "{c} channels: the IPC region carries 1 or 2 interleaved channels"
                )
            }
            IpcError::BadFormat {
                sample_rate,
                channels,
            } => {
                write!(
                    f,
                    "bad stream format: {sample_rate} Hz, {channels} channels"
                )
            }
            IpcError::BadMagic(m) => write!(f, "not a moshloom region (magic {m:#010x})"),
            IpcError::VersionMismatch { found, expected } => {
                write!(f, "IPC version {found}, this build speaks {expected}")
            }
        }
    }
}

impl std::error::Error for IpcError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            IpcError::Open(e) | IpcError::Map(e) => Some(e),
            _ => None,
        }
    }
}

/// The writer half: one per plugin instance, created in `initialize()`, never in `process()`.
pub struct IpcWriter {
    /// Owns the mapped view; `region` points into it.
    _mapping: win::Mapping,
    region: ring::Region,
    role: ShowRole,
    sample_rate: u32,
    channels: u16,
    write_index: u64,
    midi_write_index: u64,
    overwrites: u64,
}

// SAFETY: the mapping and the region are owned by exactly one thread at a time (the writer is
// moved to the audio thread); cross-thread access to the data is through the region's atomics.
unsafe impl Send for IpcWriter {}

impl IpcWriter {
    /// Create (or take over) `Local\moshloom.<role>.v1` and initialise its header.
    ///
    /// Call from `initialize()`, not from the audio callback: this touches the whole 2 MiB region.
    pub fn create(role: ShowRole, sample_rate: u32, channels: u16) -> Result<IpcWriter, IpcError> {
        if sample_rate == 0 || channels == 0 || channels > 2 {
            return Err(IpcError::BadFormat {
                sample_rate,
                channels,
            });
        }
        let mapping =
            win::Mapping::create(&role.mapping_name(), REGION_BYTES).map_err(IpcError::Map)?;
        // SAFETY: `Mapping` guarantees a `REGION_BYTES`-long, page-aligned view.
        let region = unsafe { ring::Region::new(mapping.base(), mapping.bytes()) };
        region.initialize(role as u32, sample_rate, channels, win::monotonic_ns());
        Ok(IpcWriter {
            _mapping: mapping,
            region,
            role,
            sample_rate,
            channels,
            write_index: 0,
            midi_write_index: 0,
            overwrites: 0,
        })
    }

    /// Push one block of interleaved audio with the transport of that block.
    ///
    /// `frames` is clamped to `BLOCK_MAX_FRAMES` and to what `interleaved` actually holds, so a
    /// host handing over an oversized buffer cannot make the writer read past it. A block that
    /// leaves no frames is dropped (the heartbeat still moves, so the writer still looks alive).
    ///
    /// Never allocates, never blocks. When the ring is full the oldest block is overwritten.
    pub fn push_block(&mut self, t: Option<&Transport>, interleaved: &[f32], frames: u32) {
        let channels = self.channels as usize;
        let frames = (frames as usize)
            .min(BLOCK_MAX_FRAMES)
            .min(interleaved.len() / channels.max(1));
        let header = self.region.header();
        header
            .heartbeat_ns
            .store(win::monotonic_ns(), Ordering::Release);
        if frames == 0 {
            return;
        }

        let index = self.write_index;
        self.region
            .write_block(index, t, interleaved, frames, channels);
        self.write_index = index + 1;
        if index >= RING_BLOCKS as u64 {
            // The slot this write reused was written RING_BLOCKS writes ago: at worst it was
            // still unread. The exact number is the reader's (`PollStats::dropped`).
            self.overwrites += 1;
            header.dropped.store(self.overwrites, Ordering::Relaxed);
        }
    }

    /// Push one MIDI event, stamped with the position the host reported for it.
    /// Never allocates, never blocks; a full ring overwrites the oldest event.
    pub fn push_midi(&mut self, e: &MidiEvent) {
        let index = self.midi_write_index;
        self.region.write_midi(index, e);
        self.midi_write_index = index + 1;
        self.region
            .header()
            .heartbeat_ns
            .store(win::monotonic_ns(), Ordering::Release);
    }

    pub fn role(&self) -> ShowRole {
        self.role
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Blocks written since the writer was created.
    pub fn blocks_written(&self) -> u64 {
        self.write_index
    }

    /// Writes that reused a ring slot. With a reader that never lagged this counts slot reuse,
    /// not lost data; `PollStats::dropped` is the exact number the reader missed.
    pub fn overwrites(&self) -> u64 {
        self.overwrites
    }
}

/// The reader half: the app owns one per role and polls it on its audio or render thread.
pub struct IpcReader {
    _mapping: win::Mapping,
    region: ring::Region,
    role: ShowRole,
    sample_rate: u32,
    channels: u16,
    read_index: u64,
    midi_read_index: u64,
    attached: bool,
    dropped_blocks: u64,
    dropped_midi: u64,
    scratch: AudioBlock,
}

// SAFETY: as for the writer — one owner, cross-thread access through the region's atomics.
unsafe impl Send for IpcReader {}

impl IpcReader {
    /// Attach to `Local\moshloom.<role>.v1`. Fails with `IpcError::Open` when no writer is
    /// running — the app retries; this is the normal state before Live starts.
    pub fn open(role: ShowRole) -> Result<IpcReader, IpcError> {
        let mapping =
            win::Mapping::open(&role.mapping_name(), REGION_BYTES).map_err(IpcError::Open)?;
        // SAFETY: `Mapping` guarantees a `REGION_BYTES`-long, page-aligned view.
        let region = unsafe { ring::Region::new(mapping.base(), mapping.bytes()) };
        let header = region.header();
        let magic = header.magic;
        if magic != IPC_MAGIC {
            return Err(IpcError::BadMagic(magic));
        }
        if header.version != IPC_VERSION {
            return Err(IpcError::VersionMismatch {
                found: header.version,
                expected: IPC_VERSION,
            });
        }
        let sample_rate = header.sample_rate;
        let channels = header.channels as u16;
        if sample_rate == 0 || channels == 0 || channels > 2 {
            return Err(IpcError::BadFormat {
                sample_rate,
                channels,
            });
        }
        Ok(IpcReader {
            _mapping: mapping,
            region,
            role,
            sample_rate,
            channels,
            read_index: 0,
            midi_read_index: 0,
            attached: false,
            dropped_blocks: 0,
            dropped_midi: 0,
            scratch: AudioBlock::default(),
        })
    }

    /// Take everything the writer published since the last poll.
    ///
    /// Blocks and events are appended to `out`/`midi` (which the caller may keep and clear).
    /// The first poll attaches at the oldest block still in the ring, so a reader that starts
    /// mid-song gets the last 256 blocks rather than waiting for new ones.
    ///
    /// If the writer restarts while the reader is attached (the plugin is removed and loaded
    /// again mid-set), `write_index` goes backwards. That is a new producer, not lag: the reader
    /// rebinds to its ring instead of waiting for the counter to catch up.
    pub fn poll(&mut self, out: &mut Vec<AudioBlock>, midi: &mut Vec<MidiEvent>) -> PollStats {
        let header = self.region.header();
        let write_index = header.write_index.load(Ordering::Acquire);
        let midi_write_index = header.midi_write_index.load(Ordering::Acquire);

        if self.attached
            && (write_index < self.read_index || midi_write_index < self.midi_read_index)
        {
            self.attached = false;
        }
        if !self.attached {
            self.attached = true;
            let first = write_index.saturating_sub(RING_BLOCKS as u64);
            self.dropped_blocks = first;
            self.read_index = first;
            let first_midi = midi_write_index.saturating_sub(MIDI_RING as u64);
            self.dropped_midi = first_midi;
            self.midi_read_index = first_midi;
        }

        // Blocks that fell off the tail while this reader was behind.
        let lag = write_index.saturating_sub(self.read_index);
        if lag > RING_BLOCKS as u64 {
            let missed = lag - RING_BLOCKS as u64;
            self.dropped_blocks += missed;
            self.read_index = write_index - RING_BLOCKS as u64;
        }

        let mut new_blocks = 0u32;
        let mut scratch = std::mem::take(&mut self.scratch);
        while self.read_index < write_index {
            match self.region.read_block(self.read_index, &mut scratch) {
                Some(()) => {
                    scratch.sample_rate = self.sample_rate;
                    out.push(scratch.clone());
                    new_blocks += 1;
                }
                None => {
                    // A block being written, or one the writer reused mid-copy. Lost, not torn.
                    self.dropped_blocks += 1;
                }
            }
            self.read_index += 1;
        }
        self.scratch = scratch;

        let midi_lag = midi_write_index.saturating_sub(self.midi_read_index);
        if midi_lag > MIDI_RING as u64 {
            let missed = midi_lag - MIDI_RING as u64;
            self.dropped_midi += missed;
            self.midi_read_index = midi_write_index - MIDI_RING as u64;
        }
        while self.midi_read_index < midi_write_index {
            match self.region.read_midi(self.midi_read_index) {
                Some(event) => midi.push(event),
                None => self.dropped_midi += 1,
            }
            self.midi_read_index += 1;
        }

        let heartbeat = header.heartbeat_ns.load(Ordering::Acquire);
        let now = win::monotonic_ns();
        let age = now.saturating_sub(heartbeat);
        PollStats {
            new_blocks,
            dropped: self.dropped_blocks,
            writer_alive: heartbeat != 0 && age < HEARTBEAT_TIMEOUT_NS,
        }
    }

    pub fn role(&self) -> ShowRole {
        self.role
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// MIDI events the reader never saw — reported for completeness; `PollStats` carries the
    /// block count because that is the one the audio path acts on.
    pub fn midi_dropped(&self) -> u64 {
        self.dropped_midi
    }
}
