//! The Windows half of `mosh-capture`: WASAPI loopback of a render endpoint.
//!
//! Every Win32/WASAPI call in this crate lives in this module. WASAPI's COM objects are not
//! `Send`, so the endpoint is opened *on the capture thread* and only plain data — the ring's
//! consumer, the sample rate, the channel count — crosses back to the caller.
//!
//! Loopback specifics that shaped this code:
//!
//! * Capturing from a render endpoint is loopback: `AudioClient::initialize_client` sets
//!   `AUDCLNT_STREAMFLAGS_LOOPBACK` for the `Render` device / `Capture` stream pair.
//! * A silent endpoint may deliver **no packets at all** (a virtual endpoint with no renderer
//!   does exactly that) or a stream of *silent* packets (an APO or a Realtek endpoint keeps its
//!   stream alive). Silence is therefore topped up from this thread's own clock, so the
//!   consumer keeps seeing blocks at the block rate either way.
//! * The event can time out even while the stream is healthy (`EventTimeout`), so a wait
//!   timeout is not an error; a real WASAPI error is.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::SyncSender;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use wasapi::{
    AudioCaptureClient, AudioClient, Device, DeviceCollection, DeviceEnumerator, Direction, Handle,
    SampleType, StreamMode, WasapiError, WaveFormat, initialize_mta,
};

use crate::{BLOCK_FRAMES, BlockAssembler, CaptureError, Counters, RING_BLOCKS, blocks_behind};

/// The most zero blocks one wake-up may synthsize before resyncing the clock. A longer stall is
/// a late thread, not silence we owe the consumer.
const MAX_SILENT_BURST: usize = 64;

/// What the capture thread hands back once the endpoint is open: the ring reader plus the
/// format it actually negotiated.
pub(crate) struct Opened {
    pub(crate) consumer: rtrb::Consumer<f32>,
    pub(crate) sample_rate: u32,
    pub(crate) channels: u16,
}

fn device_error(error: WasapiError) -> CaptureError {
    match error {
        WasapiError::DeviceNotFound(name) => CaptureError::DeviceNotFound(name),
        other => CaptureError::Device(other.to_string()),
    }
}

/// Friendly names of the render endpoints, in enumeration order. Empty when enumeration itself
/// fails: this feeds a device picker, and `Loopback::start` is where a failure gets reported.
pub(crate) fn list_render_devices() -> Vec<String> {
    let _ = initialize_mta();
    let Ok(enumerator) = DeviceEnumerator::new() else {
        return Vec::new();
    };
    let Ok(collection) = enumerator.get_device_collection(Direction::Render) else {
        return Vec::new();
    };
    let count = collection.get_nbr_devices().unwrap_or(0);
    (0..count)
        .filter_map(|index| {
            collection
                .get_device_at_index(index)
                .ok()?
                .get_friendlyname()
                .ok()
        })
        .collect()
}

/// Pick a render endpoint by friendly name: an exact match wins, otherwise the first
/// case-insensitive substring match. The error lists what was actually available.
fn find_device(collection: &DeviceCollection, wanted: &str) -> Result<Device, CaptureError> {
    let needle = wanted.to_lowercase();
    let count = collection.get_nbr_devices().map_err(device_error)?;
    let mut near_match: Option<Device> = None;
    let mut names: Vec<String> = Vec::with_capacity(count as usize);
    for index in 0..count {
        let device = collection
            .get_device_at_index(index)
            .map_err(device_error)?;
        let name = device.get_friendlyname().map_err(device_error)?;
        if name == wanted {
            return Ok(device);
        }
        if near_match.is_none() && name.to_lowercase().contains(&needle) {
            near_match = Some(device);
        }
        names.push(name);
    }
    near_match.ok_or_else(|| {
        CaptureError::DeviceNotFound(format!("{wanted} (available: {})", names.join(" | ")))
    })
}

/// An open loopback client, plus the format it negotiated. Lives only on the capture thread.
struct Stream {
    client: AudioClient,
    capture: AudioCaptureClient,
    event: Handle,
    sample_rate: u32,
    channels: u16,
    bytes_per_frame: usize,
    max_frames: usize,
}

/// Open loopback capture of `device` (default render endpoint when `None`).
fn open(device: Option<&str>) -> Result<Stream, CaptureError> {
    let _ = initialize_mta();
    let enumerator = DeviceEnumerator::new().map_err(device_error)?;
    let device = match device {
        Some(name) => {
            let collection = enumerator
                .get_device_collection(Direction::Render)
                .map_err(device_error)?;
            find_device(&collection, name)?
        }
        None => enumerator
            .get_default_device(Direction::Render)
            .map_err(device_error)?,
    };

    let mut client = device.get_iaudioclient().map_err(device_error)?;
    let mix = client.get_mixformat().map_err(device_error)?;
    let sample_rate = mix.get_samplespersec();
    let channels = mix.get_nchannels();
    if sample_rate == 0 || channels == 0 {
        return Err(CaptureError::UnsupportedFormat(format!(
            "endpoint reports {sample_rate} Hz, {channels} ch"
        )));
    }
    let (default_period, _minimum_period) = client.get_device_period().map_err(device_error)?;

    // Event driven shared mode. `initialize_client` turns this pair of directions — a render
    // *device* asked for a capture *stream* — into a loopback stream.
    let mode = StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns: default_period,
    };
    initialize_float_loopback(&mut client, &mix, sample_rate, channels, mode)?;

    let event = client.set_get_eventhandle().map_err(device_error)?;
    let capture = client.get_audiocaptureclient().map_err(device_error)?;
    let max_frames = client.get_buffer_size().map_err(device_error)? as usize;
    if max_frames == 0 {
        return Err(CaptureError::Device(
            "endpoint reports a zero-length buffer".into(),
        ));
    }
    Ok(Stream {
        client,
        capture,
        event,
        sample_rate,
        channels,
        bytes_per_frame: channels as usize * size_of::<f32>(),
        max_frames,
    })
}

/// Ask for 32-bit float at the endpoint's rate and channel count, twice: the default channel
/// mask first, then no mask (some drivers reject the default). If both are refused, accept the
/// endpoint's own mix format — but only when it is already float, because this crate does not
/// convert integer PCM.
fn initialize_float_loopback(
    client: &mut AudioClient,
    mix: &WaveFormat,
    sample_rate: u32,
    channels: u16,
    mode: StreamMode,
) -> Result<(), CaptureError> {
    let rate = sample_rate as usize;
    let channels = channels as usize;
    let candidates = [
        WaveFormat::new(32, 32, SampleType::Float, rate, channels, None),
        WaveFormat::new(32, 32, SampleType::Float, rate, channels, Some(0)),
    ];
    let mut last_error: Option<WasapiError> = None;
    for format in &candidates {
        match client.initialize_client(format, Direction::Capture, mode, None) {
            Ok(()) => return Ok(()),
            Err(error) => last_error = Some(error),
        }
    }
    if mix.get_subformat().ok() == Some(SampleType::Float)
        && client
            .initialize_client(mix, Direction::Capture, mode, None)
            .is_ok()
    {
        return Ok(());
    }
    Err(CaptureError::UnsupportedFormat(format!(
        "{sample_rate} Hz / {channels} ch as f32 was refused (mix format: {} bits, {:?}); last WASAPI error: {}",
        mix.get_bitspersample(),
        mix.get_subformat().err().map(|e| e.to_string()),
        last_error
            .map(|e| e.to_string())
            .unwrap_or_else(|| "none".into()),
    )))
}

/// Stop the stream when the capture loop leaves, on success or on error.
struct StopOnDrop<'a>(&'a AudioClient);

impl Drop for StopOnDrop<'_> {
    fn drop(&mut self) {
        let _ = self.0.stop_stream();
    }
}

/// The capture thread: open the endpoint, hand its format back, then stream until `stop`.
pub(crate) fn run(
    device: Option<String>,
    stop: Arc<AtomicBool>,
    counters: Arc<Counters>,
    error: Arc<OnceLock<CaptureError>>,
    ready: SyncSender<Result<Opened, CaptureError>>,
) {
    let stream = match open(device.as_deref()) {
        Ok(stream) => stream,
        Err(failure) => {
            let _ = ready.send(Err(failure));
            return;
        }
    };

    let block_len = BLOCK_FRAMES * stream.channels as usize;
    let (producer, consumer) = rtrb::RingBuffer::new(RING_BLOCKS * block_len);
    let opened = Opened {
        consumer,
        sample_rate: stream.sample_rate,
        channels: stream.channels,
    };
    if ready.send(Ok(opened)).is_err() {
        return; // the caller gave up while we were opening the endpoint
    }

    if let Err(failure) = stream.capture(producer, &stop, &counters) {
        let _ = error.set(failure);
    }
}

/// Hand one fixed-size block to the ring. Full ring: count the block and move on — the capture
/// thread must never wait for the consumer.
fn push_block(producer: &mut rtrb::Producer<f32>, block: &[f32], counters: &Counters) {
    match producer.write_chunk_uninit(block.len()) {
        Ok(chunk) => {
            let _ = chunk.fill_from_iter(block.iter().copied());
        }
        Err(_) => counters.add_dropped(),
    }
}

impl Stream {
    /// Stream packets into the ring until `stop` is set or WASAPI fails.
    ///
    /// # The silence clock
    ///
    /// A loopback endpoint that renders nothing may send **no packets at all** (this machine's
    /// virtual ASUS endpoint does exactly that), so a block stream built only from packets stops
    /// exactly when the music stops — and the app's analysis starves. What arrives is therefore
    /// continuously topped up against the wall clock: every wake-up, the frames pushed so far
    /// are compared with `elapsed * sample_rate`, and whole zero blocks are pushed whenever the
    /// stream is a block behind.
    ///
    /// Comparing against the clock rather than "did this wake-up get a packet" matters on an
    /// endpoint whose driver keeps sending *silent* packets (an APO; this machine's Realtek and
    /// Oculus endpoints behave that way): those packets already account for the time, so no
    /// zeros are added — with the naive test the stream was overproduced by 13%. Late packets
    /// get one block of silence and then take it back out of the clock's debt, so the long-run
    /// frame count stays within one block of wall time.
    fn capture(
        &self,
        mut producer: rtrb::Producer<f32>,
        stop: &AtomicBool,
        counters: &Counters,
    ) -> Result<(), CaptureError> {
        let frame_bytes = self.bytes_per_frame;
        let channels = self.channels as usize;
        let block_frames = BLOCK_FRAMES as u64;
        let block_period =
            Duration::from_secs_f64(BLOCK_FRAMES as f64 / f64::from(self.sample_rate));
        let rate = f64::from(self.sample_rate);
        // Wake about twice per block period: often enough that a timed-out event never leaves
        // the device buffer unread, and it is also the top-up clock's granularity.
        let timeout_ms = (block_period.as_secs_f64() * 500.0)
            .round()
            .clamp(2.0, 10.0) as u32;

        let mut frame_capacity = self.max_frames;
        let mut bytes = vec![0u8; frame_capacity * frame_bytes];
        let mut samples = vec![0.0f32; frame_capacity * channels];
        let mut assembler = BlockAssembler::new(self.channels, BLOCK_FRAMES);

        self.client.start_stream().map_err(device_error)?;
        let _stop_on_drop = StopOnDrop(&self.client);

        // The clock starts when the stream does: opening an endpoint can take tens of
        // milliseconds, and that time is not audio we owe the consumer.
        let mut clock_start = Instant::now();
        let mut frames_pushed = 0u64;

        while !stop.load(Ordering::Acquire) {
            match self.event.wait_for_event(timeout_ms) {
                Ok(()) | Err(WasapiError::EventTimeout) => {}
                Err(failure) => return Err(device_error(failure)),
            }

            loop {
                let next_frames = match self.capture.get_next_packet_size() {
                    Ok(Some(next_frames)) => next_frames as usize,
                    // Exclusive mode never reports a packet size; we are always shared.
                    Ok(None) => 0,
                    Err(failure) => return Err(device_error(failure)),
                };
                if next_frames == 0 {
                    break;
                }
                if next_frames > frame_capacity {
                    frame_capacity = next_frames;
                    bytes.resize(frame_capacity * frame_bytes, 0);
                    samples.resize(frame_capacity * channels, 0.0);
                }

                let wanted_bytes = next_frames * frame_bytes;
                let (frames, _info) = self
                    .capture
                    .read_from_device(&mut bytes[..wanted_bytes])
                    .map_err(device_error)?;
                if frames == 0 {
                    break;
                }

                let wanted_samples = frames as usize * channels;
                let raw = &bytes[..frames as usize * frame_bytes];
                let (raw_samples, _trailing) = raw.as_chunks::<{ size_of::<f32>() }>();
                for (slot, raw_sample) in samples[..wanted_samples].iter_mut().zip(raw_samples) {
                    *slot = f32::from_le_bytes(*raw_sample);
                }
                assembler.push_packet(&samples[..wanted_samples], |block| {
                    push_block(&mut producer, block, counters);
                });
                frames_pushed += u64::from(frames);
                counters.add_packet_frames(u64::from(frames));
            }

            // Top the stream up to the wall clock: silence the endpoint never sends us, and
            // packets that are late, both become zero blocks instead of a stalled consumer.
            let expected = (clock_start.elapsed().as_secs_f64() * rate) as u64;
            let mut due = blocks_behind(frames_pushed, expected);
            if due > MAX_SILENT_BURST as u64 {
                // A paused thread, not silence we owe: skip the gap rather than replay it.
                clock_start = Instant::now();
                frames_pushed = 0;
                due = 0;
            }
            for _ in 0..due {
                assembler.push_silence(BLOCK_FRAMES, |block| {
                    push_block(&mut producer, block, counters);
                });
                frames_pushed += block_frames;
                counters.add_synthetic();
            }
        }

        Ok(())
    }
}
