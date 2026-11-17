//! The plain types the plugin cores work with.
//!
//! `nih-plug`'s `ProcessContext` can only be built by a VST3/CLAP wrapper, and its `Transport`
//! keeps the position fields private, so the cores take this crate's own `HostTransport` instead
//! (see `glue.rs` for the one conversion) and the tests need no plugin host.

use mosh_core::ipc_layout::BLOCK_MAX_FRAMES;
use mosh_core::transport::Transport;

/// One block of host transport, in `mosh-core` terms.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HostTransport {
    pub tempo_bpm: f64,
    /// Position at the *start* of the block, in quarter-note beats.
    pub ppq: f64,
    /// Start of the bar that contains `ppq`.
    pub bar_start_ppq: f64,
    pub sig_num: u16,
    pub sig_den: u16,
    pub playing: bool,
    pub sample_pos: i64,
}

impl HostTransport {
    /// The transport to hand to the shared-memory writer.
    pub fn to_transport(self) -> Transport {
        Transport {
            tempo_bpm: self.tempo_bpm,
            ppq: self.ppq,
            bar_start_ppq: self.bar_start_ppq,
            sig_num: self.sig_num,
            sig_den: self.sig_den,
            playing: self.playing,
            sample_pos: self.sample_pos,
        }
    }

    /// The transport `frames` samples later, for the second and later chunks of an oversized host
    /// block. `bar_start_ppq` is kept: a chunk that crosses a bar line is the next chunk's job.
    pub fn advanced(self, frames: u32, sample_rate: f32) -> HostTransport {
        HostTransport {
            ppq: self.ppq + beats_for_samples(frames, sample_rate, self.tempo_bpm),
            sample_pos: self.sample_pos + frames as i64,
            ..self
        }
    }
}

/// Quarter-note beats covered by `frames` samples at `tempo_bpm`.
pub fn beats_for_samples(frames: u32, sample_rate: f32, tempo_bpm: f64) -> f64 {
    if sample_rate <= 0.0 {
        return 0.0;
    }
    frames as f64 / sample_rate as f64 * tempo_bpm / 60.0
}

/// The largest host block one push covers. Bigger blocks are split by `TapCore`/`ControlCore`.
pub const MAX_BLOCK_FRAMES: usize = BLOCK_MAX_FRAMES;

/// Interleave `count` frames starting at `start_frame` from `channels` into `dst`.
///
/// Returns the number of samples written, or 0 when `dst` is too small or a channel is too short
/// (both of which mean "do not push this block", never "allocate"). Reads from `channels`, never
/// writes to them: the audio callback must not alter the samples it passes through.
pub fn interleave_into(
    dst: &mut [f32],
    channels: &[&mut [f32]],
    start_frame: usize,
    count: usize,
) -> usize {
    let channel_count = channels.len();
    if channel_count == 0 || count == 0 {
        return 0;
    }

    let end_frame = start_frame.saturating_add(count);
    if dst.len() < count * channel_count || channels.iter().any(|channel| channel.len() < end_frame)
    {
        return 0;
    }

    for (frame, slot) in dst.chunks_exact_mut(channel_count).take(count).enumerate() {
        let source_frame = start_frame + frame;
        for (sample, channel) in slot.iter_mut().zip(channels.iter()) {
            *sample = channel[source_frame];
        }
    }

    count * channel_count
}
