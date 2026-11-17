//! Musical time from Ableton (via the plugins) and the audio blocks that carry it.

/// Ableton's transport for one audio block. All musical time is in ppq (quarter-note beats).
/// Bars come from `bar_start_ppq` and the time signature, never from `ppq / 4`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Transport {
    pub tempo_bpm: f64,
    pub ppq: f64,
    pub bar_start_ppq: f64,
    pub sig_num: u16,
    pub sig_den: u16,
    pub playing: bool,
    pub sample_pos: i64,
}

/// One block of audio as the app sees it, from the plugin (with transport) or loopback (without).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct AudioBlock {
    pub transport: Option<Transport>,
    pub sample_rate: u32,
    pub channels: u16,
    /// Interleaved samples, `frames * channels` long.
    pub samples: Vec<f32>,
}

impl AudioBlock {
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels as usize
        }
    }
}

/// A MIDI message from the Moshloom Control plugin, stamped with Live's position.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct MidiEvent {
    pub ppq: f64,
    pub status: u8,
    pub data1: u8,
    pub data2: u8,
}
