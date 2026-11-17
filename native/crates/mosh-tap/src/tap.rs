//! Moshloom Tap: an audio effect for Ableton's master.
//!
//! It passes audio through untouched and writes the block plus Ableton's transport to shared
//! memory. Two halves: [`TapCore`], the whole of the audio callback in testable form, and
//! [`MoshloomTap`], the nih-plug plugin that owns it.

use std::sync::Arc;

use mosh_core::ipc_layout::{BLOCK_MAX_FRAMES, BLOCK_MAX_SAMPLES, ShowRole};
use nih_plug::prelude::*;

use crate::glue::host_transport;
use crate::host::{HostTransport, interleave_into};
use crate::sink::{
    BlockSink, PluginSink, ROLE_SLOTS, role_for_index, role_index_from_label, role_label,
};

/// Ableton's master insert is stereo, and the IPC contract's block holds exactly that.
pub const TAP_CHANNELS: u16 = 2;

/// What the audio callback does, with the sink and the sample rate as parameters.
pub struct TapCore<S: BlockSink> {
    sink: S,
    sample_rate: f32,
    channels: u16,
}

impl<S: BlockSink> TapCore<S> {
    pub fn new(sink: S, sample_rate: f32, channels: u16) -> Self {
        Self {
            sink,
            sample_rate,
            channels,
        }
    }

    pub fn sink(&self) -> &S {
        &self.sink
    }

    pub fn sink_mut(&mut self) -> &mut S {
        &mut self.sink
    }

    /// Copy one host block to the writer without touching it, and return the number of blocks
    /// pushed. A block longer than `BLOCK_MAX_FRAMES` is split, with the transport advanced for
    /// each chunk so the app's alignment stays sample-exact.
    ///
    /// Never allocates and never blocks.
    pub fn push_interleaved(
        &mut self,
        transport: Option<&HostTransport>,
        samples: &[f32],
        frames: u32,
    ) -> u32 {
        let channels = self.channels as usize;
        if channels == 0 {
            return 0;
        }

        let frames = frames.min((samples.len() / channels) as u32);
        let mut done = 0u32;
        let mut pushed = 0u32;
        while done < frames {
            let chunk = (frames - done).min(BLOCK_MAX_FRAMES as u32);
            let start = done as usize * channels;
            let end = start + chunk as usize * channels;
            let advanced = transport
                .copied()
                .map(|t| t.advanced(done, self.sample_rate))
                .map(HostTransport::to_transport);

            self.sink
                .push_block(advanced.as_ref(), &samples[start..end], chunk);

            done += chunk;
            pushed += 1;
        }

        pushed
    }
}

/// The plugin's one parameter: which source this instance is.
#[derive(Params)]
pub struct TapParams {
    /// Takes effect when the plugin is (re)initialised, i.e. when Live starts the audio engine or
    /// the device is reloaded: the writer is created there, never on the audio thread.
    #[id = "role"]
    pub role: IntParam,
}

impl Default for TapParams {
    fn default() -> Self {
        Self {
            role: IntParam::new(
                "Role",
                0,
                IntRange::Linear {
                    min: 0,
                    max: ROLE_SLOTS - 1,
                },
            )
            .with_value_to_string(Arc::new(|value: i32| role_label(value).to_string()))
            .with_string_to_value(Arc::new(|label: &str| role_index_from_label(label))),
        }
    }
}

pub struct MoshloomTap {
    params: Arc<TapParams>,
    /// Created in `initialize()`; the audio thread only pushes into it.
    core: Option<TapCore<PluginSink>>,
    /// Interleaving scratch, sized in `initialize()` and never grown in `process()`.
    scratch: Vec<f32>,
}

impl Default for MoshloomTap {
    fn default() -> Self {
        Self {
            params: Arc::new(TapParams::default()),
            core: None,
            scratch: Vec::new(),
        }
    }
}

impl Plugin for MoshloomTap {
    const NAME: &'static str = "ilanaVisual Tap";
    const VENDOR: &'static str = "Ilana Audio";
    const URL: &'static str = "";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
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
        let role: ShowRole = role_for_index(self.params.role.value());

        // The writer is created here and nowhere else. Failing to open it is fine: the plugin
        // then passes audio through and writes nothing.
        self.core = Some(TapCore::new(
            PluginSink::create(role, sample_rate.round() as u32, TAP_CHANNELS),
            sample_rate,
            TAP_CHANNELS,
        ));
        self.scratch = vec![0.0; BLOCK_MAX_SAMPLES];

        true
    }

    fn deactivate(&mut self) {
        // Unmap the region; the app sees the writer's heartbeat stop and keeps playing.
        self.core = None;
        self.scratch = Vec::new();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let transport = host_transport(context.transport());

        // Borrow the two fields separately: `core` and `scratch` are one `&mut self` otherwise.
        let Self { core, scratch, .. } = self;
        let Some(core) = core.as_mut() else {
            return ProcessStatus::Normal;
        };

        let frames = buffer.samples();
        if frames == 0 || scratch.len() < BLOCK_MAX_SAMPLES {
            return ProcessStatus::Normal;
        }

        let channels = buffer.as_slice();
        let mut done = 0usize;
        while done < frames {
            let chunk = (frames - done).min(BLOCK_MAX_FRAMES);
            let written = interleave_into(scratch.as_mut_slice(), channels, done, chunk);
            if written == 0 {
                break;
            }

            core.push_interleaved(transport.as_ref(), &scratch[..written], chunk as u32);
            done += chunk;
        }

        // The audio itself is untouched: the host's buffers are only ever read here.
        ProcessStatus::Normal
    }
}

impl Vst3Plugin for MoshloomTap {
    const VST3_CLASS_ID: [u8; 16] = *b"ilanaVisualTap01";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Tools];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::NullSink;

    #[test]
    fn zero_channels_pushes_nothing() {
        let mut core = TapCore::new(NullSink, 48_000.0, 0);
        assert_eq!(core.push_interleaved(None, &[1.0, 2.0], 2), 0);
    }

    #[test]
    fn an_empty_block_pushes_nothing() {
        let mut core = TapCore::new(NullSink, 48_000.0, 2);
        assert_eq!(core.push_interleaved(None, &[], 0), 0);
    }

    #[test]
    fn a_short_slice_shrinks_the_block() {
        // 3 samples of stereo is one frame, not one and a half.
        let mut core = TapCore::new(NullSink, 48_000.0, 2);
        assert_eq!(core.push_interleaved(None, &[0.0, 1.0, 2.0], 4), 1);
    }
}
