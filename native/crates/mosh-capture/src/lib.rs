//! `mosh-capture`: Windows WASAPI **loopback** capture of a render endpoint, the fallback
//! audio input for the app when the tap plugin is not installed.
//!
//! See `docs/superpowers/plans/2026-11-17-native-port.md`, Task 1.6. Loopback has no transport,
//! so every [`AudioBlock`] this crate produces carries `transport: None`; the app's causal
//! tempo tracker (Task 2.5) is what gives loopback mode a beat.
//!
//! # Shape
//!
//! * [`Loopback::start`] opens the endpoint and runs the capture loop on its own thread. Every
//!   windows/WASAPI call lives in `win.rs`, and WASAPI's objects are not `Send`, so the endpoint
//!   never leaves that thread.
//! * Blocks cross the thread boundary through a lock-free SPSC ring ([`rtrb`]). When the ring is
//!   full the capture thread **drops and counts** instead of blocking, and [`Loopback::poll`]
//!   only ever *tries* to read, so neither side can stall the other.
//! * Blocks are [`BLOCK_FRAMES`] (256) frames at the device rate, `f32` interleaved, whatever
//!   packet sizes WASAPI hands over.
//! * Silence is not an error. A loopback endpoint that renders nothing either sends no packets
//!   at all or sends silent ones, so the capture thread tops the stream up against its own clock
//!   and the block stream never stops.
//!
//! # Example
//!
//! ```no_run
//! use mosh_capture::Loopback;
//!
//! let mut capture = Loopback::start(None)?; // default render endpoint
//! let mut blocks = Vec::new();
//! loop {
//!     capture.poll(&mut blocks); // never blocks
//!     for block in blocks.drain(..) {
//!         assert!(block.transport.is_none());
//!         // hand block.samples to mosh-analysis
//!     }
//!     std::thread::sleep(std::time::Duration::from_millis(5));
//! }
//! # Ok::<(), mosh_capture::CaptureError>(())
//! ```

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;

use mosh_core::AudioBlock;

#[cfg(windows)]
mod win;

#[cfg(not(windows))]
compile_error!(
    "mosh-capture is Windows-only: it captures a WASAPI loopback endpoint (see the native port design)"
);

/// Frames per block, at the device rate.
pub const BLOCK_FRAMES: usize = 256;

/// Blocks the ring holds before the capture thread starts dropping. 64 blocks is ~341 ms at
/// 48 kHz: enough for a slow consumer frame, small enough that a stalled app does not eat memory.
const RING_BLOCKS: usize = 64;

/// What the capture thread has done so far. Shared with the capture thread; read through the
/// [`Loopback`] accessors.
#[derive(Debug, Default)]
pub(crate) struct Counters {
    dropped: AtomicU64,
    synthetic: AtomicU64,
    packet_frames: AtomicU64,
}

impl Counters {
    fn add_dropped(&self) {
        self.dropped.fetch_add(1, Ordering::Relaxed);
    }

    fn add_synthetic(&self) {
        self.synthetic.fetch_add(1, Ordering::Relaxed);
    }

    fn add_packet_frames(&self, frames: u64) {
        self.packet_frames.fetch_add(frames, Ordering::Relaxed);
    }
}

/// Why loopback capture could not start or why it stopped.
///
/// The variants carry the message, not the source error, so the app can log one line and keep
/// running (the show never blocks on audio).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CaptureError {
    /// No render endpoint matched the requested name.
    DeviceNotFound(String),
    /// The endpoint exists but WASAPI refused an operation on it.
    Device(String),
    /// The endpoint's format could not be captured as 32-bit float.
    UnsupportedFormat(String),
    /// The capture thread could not be started, or died without reporting an error.
    Thread(String),
}

impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceNotFound(what) => write!(f, "no such render endpoint: {what}"),
            Self::Device(what) => write!(f, "render endpoint failed: {what}"),
            Self::UnsupportedFormat(what) => write!(f, "unsupported loopback format: {what}"),
            Self::Thread(what) => write!(f, "capture thread failed: {what}"),
        }
    }
}

impl std::error::Error for CaptureError {}

/// The render endpoints this machine can capture, by friendly name, in enumeration order.
///
/// An enumeration failure returns an empty list: this is a picker, not a start path, and the
/// caller finds out what is wrong from [`Loopback::start`].
pub fn list_render_devices() -> Vec<String> {
    win::list_render_devices()
}

/// A running loopback capture. Drop it to stop and join the capture thread.
pub struct Loopback {
    consumer: rtrb::Consumer<f32>,
    sample_rate: u32,
    channels: u16,
    stop: Arc<AtomicBool>,
    counters: Arc<Counters>,
    error: Arc<OnceLock<CaptureError>>,
    thread: Option<JoinHandle<()>>,
}

impl Loopback {
    /// Start loopback capture of `device` (a friendly name, matched exactly first and then by
    /// case-insensitive substring), or of the default render endpoint when `None`.
    ///
    /// Returns once the endpoint is open and the stream is running, or with the reason it is
    /// not. Errors after that point surface through [`Loopback::error`].
    pub fn start(device: Option<&str>) -> Result<Self, CaptureError> {
        let stop = Arc::new(AtomicBool::new(false));
        let counters = Arc::new(Counters::default());
        let error: Arc<OnceLock<CaptureError>> = Arc::new(OnceLock::new());
        let requested = device.map(str::to_owned);

        // Rendezvous with the capture thread: it hands back the ring's consumer and the format
        // it actually opened, so `start` can report a failure synchronously.
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(0);

        let thread = {
            let stop = Arc::clone(&stop);
            let counters = Arc::clone(&counters);
            let error = Arc::clone(&error);
            std::thread::Builder::new()
                .name("mosh-capture".into())
                .spawn(move || win::run(requested, stop, counters, error, ready_tx))
                .map_err(|e| CaptureError::Thread(e.to_string()))?
        };

        match ready_rx.recv() {
            Ok(Ok(opened)) => Ok(Self {
                consumer: opened.consumer,
                sample_rate: opened.sample_rate,
                channels: opened.channels,
                stop,
                counters,
                error,
                thread: Some(thread),
            }),
            Ok(Err(e)) => {
                let _ = thread.join();
                Err(e)
            }
            Err(_) => {
                let _ = thread.join();
                Err(CaptureError::Thread(
                    "the capture thread exited before opening the endpoint".into(),
                ))
            }
        }
    }

    /// Append every block that has arrived to `out`. Never blocks: an empty ring is a no-op.
    ///
    /// Only whole [`BLOCK_FRAMES`] blocks are handed out, so a caller that polls at frame rate
    /// gets 0..n blocks and never a partial one.
    pub fn poll(&mut self, out: &mut Vec<AudioBlock>) {
        let block_len = BLOCK_FRAMES * self.channels as usize;
        if block_len == 0 {
            return;
        }
        loop {
            let mut samples = vec![0.0f32; block_len];
            if self.consumer.pop_entire_slice(&mut samples).is_err() {
                return;
            }
            out.push(AudioBlock {
                transport: None,
                sample_rate: self.sample_rate,
                channels: self.channels,
                samples,
            });
        }
    }

    /// Sample rate of the endpoint.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Channels of the endpoint (interleaved samples per frame).
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Blocks dropped because the consumer was more than [`RING_BLOCKS`] blocks behind.
    pub fn dropped_blocks(&self) -> u64 {
        self.counters.dropped.load(Ordering::Relaxed)
    }

    /// Zero blocks the capture thread synthesized because the endpoint sent no audio for them.
    pub fn synthetic_blocks(&self) -> u64 {
        self.counters.synthetic.load(Ordering::Relaxed)
    }

    /// Frames the endpoint actually delivered in packets (the rest of the stream is
    /// [`Loopback::synthetic_blocks`]).
    pub fn packet_frames(&self) -> u64 {
        self.counters.packet_frames.load(Ordering::Relaxed)
    }

    /// The error that stopped capture, once the thread has stopped, or `None` while it runs.
    pub fn error(&self) -> Option<CaptureError> {
        self.error.get().cloned()
    }

    /// Whether the capture thread is still streaming.
    pub fn is_running(&self) -> bool {
        self.error.get().is_none() && !self.stop.load(Ordering::Acquire)
    }
}

impl Drop for Loopback {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            // The thread wakes from its wait within one block period, so this join is bounded.
            let _ = thread.join();
        }
    }
}

impl fmt::Debug for Loopback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Loopback")
            .field("sample_rate", &self.sample_rate)
            .field("channels", &self.channels)
            .field("dropped_blocks", &self.dropped_blocks())
            .field("synthetic_blocks", &self.synthetic_blocks())
            .field("packet_frames", &self.packet_frames())
            .field("error", &self.error())
            .finish_non_exhaustive()
    }
}

/// How many whole zero blocks the stream is behind `expected_frames` of wall-clock audio.
///
/// The capture thread tops the stream up with this after every wake-up, which is what keeps
/// blocks flowing while the endpoint sends nothing. It is a pure function so the policy can be
/// tested without a device.
pub(crate) fn blocks_behind(frames_pushed: u64, expected_frames: u64) -> u64 {
    expected_frames.saturating_sub(frames_pushed) / BLOCK_FRAMES as u64
}

/// Cuts arbitrary packets into fixed blocks. Pure logic, so it is unit-tested without a device.
pub(crate) struct BlockAssembler {
    channels: usize,
    frames_per_block: usize,
    pending: Vec<f32>,
}

impl BlockAssembler {
    pub(crate) fn new(channels: u16, frames_per_block: usize) -> Self {
        assert!(channels > 0, "a block needs at least one channel");
        assert!(frames_per_block > 0, "a block needs at least one frame");
        let channels = channels as usize;
        Self {
            channels,
            frames_per_block,
            pending: Vec::with_capacity(channels * frames_per_block * 2),
        }
    }

    fn block_len(&self) -> usize {
        self.frames_per_block * self.channels
    }

    /// Append interleaved samples, handing every complete block to `emit`.
    ///
    /// A trailing partial frame (fewer samples than a frame) is dropped: a half frame cannot be
    /// interleaved and WASAPI never produces one.
    pub(crate) fn push_packet(&mut self, interleaved: &[f32], mut emit: impl FnMut(&[f32])) {
        let whole_frame_samples = interleaved.len() - interleaved.len() % self.channels;
        if whole_frame_samples == 0 {
            return;
        }
        self.pending
            .extend_from_slice(&interleaved[..whole_frame_samples]);
        self.flush(&mut emit);
    }

    /// Append `frames` frames of digital silence, handing every complete block to `emit`.
    pub(crate) fn push_silence(&mut self, frames: usize, mut emit: impl FnMut(&[f32])) {
        if frames == 0 {
            return;
        }
        self.pending
            .resize(self.pending.len() + frames * self.channels, 0.0);
        self.flush(&mut emit);
    }

    fn flush(&mut self, emit: &mut impl FnMut(&[f32])) {
        let block_len = self.block_len();
        let mut done = 0;
        while self.pending.len() - done >= block_len {
            emit(&self.pending[done..done + block_len]);
            done += block_len;
        }
        if done > 0 {
            self.pending.drain(..done);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blocks_from(push: impl FnOnce(&mut BlockAssembler, &mut Vec<Vec<f32>>)) -> Vec<Vec<f32>> {
        let mut assembler = BlockAssembler::new(2, BLOCK_FRAMES);
        let mut blocks = Vec::new();
        push(&mut assembler, &mut blocks);
        blocks
    }

    #[test]
    fn assembler_emits_blocks_from_odd_packet_sizes() {
        // 1000 frames at 2 channels, in packet sizes WASAPI actually uses (480, 441, 79).
        let mut packets = Vec::new();
        for frames in [480usize, 441, 79] {
            packets.extend(std::iter::repeat_n(0.5f32, frames * 2));
        }
        assert_eq!(packets.len(), 2000);

        let blocks = blocks_from(|assembler, out| {
            assembler.push_packet(&packets, |block| out.push(block.to_vec()));
        });
        // 1000 frames = 3 full 256-frame blocks, 232 frames left pending.
        assert_eq!(blocks.len(), 3);
        assert!(blocks.iter().all(|b| b.len() == BLOCK_FRAMES * 2));
        assert!(blocks.iter().flatten().all(|s| *s == 0.5));
    }

    #[test]
    fn assembler_carries_pending_frames_between_packets() {
        let one_block = vec![1.0f32; BLOCK_FRAMES * 2];
        let mut rest = vec![0.25f32; (BLOCK_FRAMES - 100) * 2];
        rest = [vec![0.75f32; 100 * 2], rest].concat();

        let blocks = blocks_from(|assembler, out| {
            // The first packet is one sample short of a block, so nothing is emitted yet.
            assembler.push_packet(&one_block[..one_block.len() - 2], |block| {
                out.push(block.to_vec())
            });
            assert!(out.is_empty(), "a partial block must not be emitted");
            assembler.push_packet(&rest, |block| out.push(block.to_vec()));
        });

        assert_eq!(blocks.len(), 1);
        // The block is the 255 frames of the first packet, then the first frame of the second.
        assert!(blocks[0][..100].iter().all(|s| *s == 1.0));
        assert!(
            blocks[0][..(BLOCK_FRAMES - 1) * 2]
                .iter()
                .all(|s| *s == 1.0)
        );
        assert_eq!(&blocks[0][(BLOCK_FRAMES - 1) * 2..], &[0.75, 0.75]);
    }

    #[test]
    fn assembler_turns_silence_into_zero_blocks() {
        let blocks = blocks_from(|assembler, out| {
            assembler.push_silence(BLOCK_FRAMES, |block| out.push(block.to_vec()));
            assembler.push_silence(BLOCK_FRAMES * 2, |block| out.push(block.to_vec()));
        });
        assert_eq!(blocks.len(), 3);
        assert!(blocks.iter().flatten().all(|s| *s == 0.0));
    }

    #[test]
    fn assembler_ignores_a_partial_frame() {
        // 5 samples at 2 channels is 2 frames plus a stray sample.
        let blocks = blocks_from(|assembler, out| {
            assembler.push_packet(&[1.0, 2.0, 3.0, 4.0, 5.0], |block| out.push(block.to_vec()));
            assembler.push_silence(BLOCK_FRAMES - 2, |block| out.push(block.to_vec()));
        });
        assert_eq!(blocks.len(), 1);
        assert_eq!(&blocks[0][..4], &[1.0, 2.0, 3.0, 4.0]);
        // The stray 5.0 never becomes a frame.
        assert!(blocks[0][4..].iter().all(|s| *s == 0.0));
    }

    #[test]
    fn blocks_behind_fills_a_silent_second_at_the_block_rate() {
        // One second of 48 kHz audio that the endpoint never sent.
        assert_eq!(blocks_behind(0, 48_000), 48_000 / BLOCK_FRAMES as u64);
        assert_eq!(blocks_behind(0, 48_000), 187);
    }

    #[test]
    fn blocks_behind_tolerates_less_than_one_block_of_latency() {
        // A packet is late but the stream is not a whole block behind: no zeros.
        assert_eq!(blocks_behind(1_000, 1_000), 0);
        assert_eq!(blocks_behind(1_000, 1_255), 0);
        assert_eq!(blocks_behind(1_000, 1_256), 1);
    }

    #[test]
    fn blocks_behind_is_zero_when_packets_are_ahead_of_the_clock() {
        // Drivers that deliver in bursts push the stream ahead; that is not a reason to insert
        // silence, and saturating means a burst never underflows.
        assert_eq!(blocks_behind(5_000, 1_000), 0);
        assert_eq!(blocks_behind(0, 0), 0);
    }

    #[test]
    fn loopback_is_send_and_capture_error_is_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send::<Loopback>();
        assert_send_sync::<CaptureError>();
    }

    #[test]
    fn block_frames_is_256() {
        assert_eq!(BLOCK_FRAMES, 256);
    }
}
