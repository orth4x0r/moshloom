//! Moshloom Control: an instrument for a MIDI track.
//!
//! It outputs silence and writes the MIDI notes, CCs and program changes Live sends it, stamped
//! with the transport position, to the `control` region. Two halves: [`ControlCore`], the whole of
//! the audio callback in testable form, and [`MoshloomControl`], the nih-plug plugin that owns it.

use std::sync::Arc;

use mosh_core::ipc_layout::{BLOCK_MAX_FRAMES, BLOCK_MAX_SAMPLES, ShowRole};
use mosh_core::transport::MidiEvent;
use nih_plug::prelude::*;

use crate::glue::{host_event, host_transport};
use crate::host::{HostTransport, beats_for_samples};
use crate::sink::{BlockSink, PluginSink};

/// The instrument's silent stereo output.
pub const CONTROL_CHANNELS: u16 = 2;

/// How many MIDI events one block may hand over. A 16th-note chord on a 32-frame block stays far
/// below this; beyond it, events are dropped and counted rather than allocating.
pub const CONTROL_EVENT_CAPACITY: usize = 256;

/// One MIDI event from the host, in this crate's own terms so tests need no plugin host.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HostEvent {
    NoteOn {
        timing: u32,
        channel: u8,
        note: u8,
        velocity: f32,
    },
    NoteOff {
        timing: u32,
        channel: u8,
        note: u8,
    },
    ControlChange {
        timing: u32,
        channel: u8,
        cc: u8,
        value: f32,
    },
    ProgramChange {
        timing: u32,
        channel: u8,
        program: u8,
    },
}

/// What one block did, for tests and for the event counter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ControlStats {
    pub blocks: u32,
    pub midi: u32,
}

/// The MIDI event for one host event, or `None` if the event has no MIDI equivalent.
///
/// Timing is sample-accurate within the block: `ppq = block ppq + timing / sample_rate * tempo`,
/// so a CC 128 samples into a 120 BPM block at 48 kHz lands 1/187.5 of a beat late.
pub fn event_to_midi(
    event: &HostEvent,
    transport: Option<&HostTransport>,
    sample_rate: f32,
) -> Option<MidiEvent> {
    let (timing, status, data1, data2) = match *event {
        HostEvent::NoteOn {
            timing,
            channel,
            note,
            velocity,
        } => (
            timing,
            0x90 | (channel & 0x0f),
            note,
            // A note on at zero velocity would read as a note off, so it clamps to 1.
            (velocity * 127.0).round().clamp(1.0, 127.0) as u8,
        ),
        HostEvent::NoteOff {
            timing,
            channel,
            note,
        } => (timing, 0x80 | (channel & 0x0f), note, 0),
        HostEvent::ControlChange {
            timing,
            channel,
            cc,
            value,
        } => (
            timing,
            0xb0 | (channel & 0x0f),
            cc,
            (value * 127.0).round().clamp(0.0, 127.0) as u8,
        ),
        HostEvent::ProgramChange {
            timing,
            channel,
            program,
        } => (timing, 0xc0 | (channel & 0x0f), program, 0),
    };

    let base_ppq = transport.map(|t| t.ppq).unwrap_or(0.0);
    let tempo_bpm = transport.map(|t| t.tempo_bpm).unwrap_or(0.0);

    Some(MidiEvent {
        ppq: base_ppq + beats_for_samples(timing, sample_rate, tempo_bpm),
        status,
        data1,
        data2,
    })
}

/// What the audio callback does: a silent block for the transport and heartbeat, then the events.
pub struct ControlCore<S: BlockSink> {
    sink: S,
    sample_rate: f32,
    channels: u16,
    /// Always zeros, allocated once: the instrument's output and the blocks it pushes.
    silence: Vec<f32>,
}

impl<S: BlockSink> ControlCore<S> {
    pub fn new(sink: S, sample_rate: f32, channels: u16) -> Self {
        let channel_count = channels as usize;
        // Whole frames only, so a push never splits a frame.
        let frames = BLOCK_MAX_SAMPLES.checked_div(channel_count).unwrap_or(0);

        Self {
            sink,
            sample_rate,
            channels,
            silence: vec![0.0; frames * channel_count],
        }
    }

    pub fn sink(&self) -> &S {
        &self.sink
    }

    pub fn sink_mut(&mut self) -> &mut S {
        &mut self.sink
    }

    /// Push this block's silence and events. Never allocates and never blocks.
    pub fn process(
        &mut self,
        transport: Option<&HostTransport>,
        events: &[HostEvent],
        frames: u32,
    ) -> ControlStats {
        let mut stats = ControlStats::default();

        let channels = self.channels as usize;
        let max_frames = self
            .silence
            .len()
            .checked_div(channels)
            .map_or(0, |frames| frames.min(BLOCK_MAX_FRAMES));
        let mut done = 0u32;
        while done < frames && max_frames > 0 {
            let chunk = (frames - done).min(max_frames as u32);
            let len = chunk as usize * channels;
            let advanced = transport
                .copied()
                .map(|t| t.advanced(done, self.sample_rate))
                .map(HostTransport::to_transport);

            self.sink
                .push_block(advanced.as_ref(), &self.silence[..len], chunk);

            done += chunk;
            stats.blocks += 1;
        }

        for event in events {
            if let Some(midi) = event_to_midi(event, transport, self.sample_rate) {
                self.sink.push_midi(&midi);
                stats.midi += 1;
            }
        }

        stats
    }
}

/// Moshloom Control has no parameters: what it sends is what Live's clip sends it.
#[derive(Debug, Default)]
pub struct ControlParams;

// SAFETY: `ControlParams` holds nothing, and the plugin keeps it in an `Arc` that outlives the
// wrapper's parameter mapping.
unsafe impl Params for ControlParams {
    fn param_map(&self) -> Vec<(String, ParamPtr, String)> {
        Vec::new()
    }
}

/// A fixed-capacity list of events, so `process()` never allocates.
struct EventBuf {
    events: [HostEvent; CONTROL_EVENT_CAPACITY],
    len: usize,
    /// Events that did not fit, kept as a number so a host flooding us is visible in a test and
    /// never a silent loss.
    overflowed: u32,
}

impl EventBuf {
    fn new() -> Self {
        Self {
            events: [HostEvent::NoteOff {
                timing: 0,
                channel: 0,
                note: 0,
            }; CONTROL_EVENT_CAPACITY],
            len: 0,
            overflowed: 0,
        }
    }

    fn push(&mut self, event: HostEvent) {
        if self.len < CONTROL_EVENT_CAPACITY {
            self.events[self.len] = event;
            self.len += 1;
        } else {
            self.overflowed = self.overflowed.saturating_add(1);
        }
    }

    fn as_slice(&self) -> &[HostEvent] {
        &self.events[..self.len]
    }

    fn clear(&mut self) {
        self.len = 0;
        self.overflowed = 0;
    }
}

pub struct MoshloomControl {
    params: Arc<ControlParams>,
    /// Created in `initialize()`; the audio thread only pushes into it.
    core: Option<ControlCore<PluginSink>>,
    events: EventBuf,
}

impl Default for MoshloomControl {
    fn default() -> Self {
        Self {
            params: Arc::new(ControlParams),
            core: None,
            events: EventBuf::new(),
        }
    }
}

impl Plugin for MoshloomControl {
    const NAME: &'static str = "Moshloom Control";
    const VENDOR: &'static str = "Moshloom";
    const URL: &'static str = "https://github.com/orth4x0r/moshloom";
    const EMAIL: &'static str = "noreply@moshloom.invalid";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        // An instrument: no main input, a silent stereo output. Live refuses MIDI-only plugins.
        main_input_channels: None,
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    /// `MidiCCs` so Live's CCs (the APC-style knob map in Task 1.10) arrive as well as notes.
    const MIDI_INPUT: MidiConfig = MidiConfig::MidiCCs;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        let sample_rate = buffer_config.sample_rate;

        // The writer is created here and nowhere else.
        self.core = Some(ControlCore::new(
            PluginSink::create(
                ShowRole::Control,
                sample_rate.round() as u32,
                CONTROL_CHANNELS,
            ),
            sample_rate,
            CONTROL_CHANNELS,
        ));

        true
    }

    fn deactivate(&mut self) {
        self.core = None;
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // An instrument's output is silence, whatever the host handed us.
        for channel in buffer.as_slice() {
            channel.fill(0.0);
        }

        let transport = host_transport(context.transport());

        let Self { core, events, .. } = self;
        let Some(core) = core.as_mut() else {
            return ProcessStatus::KeepAlive;
        };

        events.clear();
        while let Some(event) = context.next_event() {
            if let Some(event) = host_event(&event) {
                events.push(event);
            }
        }

        let frames = buffer.samples() as u32;
        core.process(transport.as_ref(), events.as_slice(), frames);

        // Keep processing even while silent, or Live would suspend us and the transport would
        // stop flowing to the app.
        ProcessStatus::KeepAlive
    }
}

impl Vst3Plugin for MoshloomControl {
    const VST3_CLASS_ID: [u8; 16] = *b"MoshloomCtl00001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Instrument,
        Vst3SubCategory::Synth,
        Vst3SubCategory::Tools,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullSink;

    #[test]
    fn zero_channels_pushes_no_block_but_still_sends_midi() {
        let mut core = ControlCore::new(NullSink, 48_000.0, 0);
        let stats = core.process(
            None,
            &[HostEvent::NoteOn {
                timing: 0,
                channel: 0,
                note: 36,
                velocity: 1.0,
            }],
            256,
        );
        assert_eq!(stats, ControlStats { blocks: 0, midi: 1 });
    }

    #[test]
    fn zero_frames_pushes_no_block() {
        let mut core = ControlCore::new(NullSink, 48_000.0, 2);
        assert_eq!(core.process(None, &[], 0), ControlStats::default());
    }

    #[test]
    fn the_event_buffer_stops_at_its_capacity() {
        let mut buf = EventBuf::new();
        for _ in 0..CONTROL_EVENT_CAPACITY + 5 {
            buf.push(HostEvent::NoteOff {
                timing: 0,
                channel: 0,
                note: 1,
            });
        }
        assert_eq!(buf.as_slice().len(), CONTROL_EVENT_CAPACITY);
        assert_eq!(buf.overflowed, 5);

        buf.clear();
        assert!(buf.as_slice().is_empty());
        assert_eq!(buf.overflowed, 0);
    }
}
