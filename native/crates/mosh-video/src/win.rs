//! Media Foundation: the only file in `mosh-video` that talks to Windows.
//!
//! A source reader is opened with hardware transforms enabled, so an H.264/HEVC decoder MFT can
//! run on the GPU; the reader's video processor then converts whatever the decoder emits into
//! top-down or bottom-up RGB32, which is BGRA in memory on little-endian Windows.
//!
//! Frame rate and size are read from the negotiated media types, not guessed from samples, so a
//! variable-frame-rate clip still gets a usable `fps` (the nominal rate, used only to decide which
//! frame covers a timestamp).

use std::path::Path;
use std::sync::OnceLock;

use windows::Win32::Media::MediaFoundation::{
    IMF2DBuffer, IMFAttributes, IMFMediaType, IMFSample, IMFSourceReader, MF_MT_DEFAULT_STRIDE,
    MF_MT_FRAME_RATE, MF_MT_FRAME_RATE_RANGE_MAX, MF_MT_FRAME_SIZE, MF_MT_INTERLACE_MODE,
    MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE, MF_PD_DURATION, MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS,
    MF_SOURCE_READER_ALL_STREAMS, MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING,
    MF_SOURCE_READER_FIRST_VIDEO_STREAM, MF_SOURCE_READER_MEDIASOURCE,
    MF_SOURCE_READERF_ENDOFSTREAM, MF_SOURCE_READERF_ERROR, MF_VERSION, MFCreateAttributes,
    MFCreateMediaType, MFCreateSourceReaderFromURL, MFMediaType_Video, MFSTARTUP_FULL, MFStartup,
    MFVideoFormat_RGB32, MFVideoInterlace_Progressive,
};
use windows::Win32::System::Com::StructuredStorage::{
    PROPVARIANT, PROPVARIANT_0, PROPVARIANT_0_0, PROPVARIANT_0_0_0, PropVariantToUInt64,
};
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx};
use windows::Win32::System::Variant::VT_I8;
use windows::core::{GUID, HSTRING, Interface};

use crate::{VideoError, VideoInfo};

/// No picture appeared for this many reads in a row: the reader is stuck rather than ticking.
const MAX_EMPTY_READS: usize = 16;

/// 8K RGBA: a real frame never needs more, so a bad media type cannot ask for a huge allocation.
const MAX_FRAME_BYTES: usize = 7680 * 4320 * 4;

/// The decoder for one file. Owned by [`crate::VideoSource`]; not `Send`, because Media Foundation
/// source readers are used from the thread that created them (COM apartment).
pub(crate) struct WinReader {
    reader: IMFSourceReader,
    w: usize,
    h: usize,
    /// Byte distance between rows in the negotiated RGB32 buffer. Not always `w * 4`: Media
    /// Foundation pads frames (this fixture is 160x90, the RGB32 buffer is 160x96 at pitch 640).
    pitch: usize,
    /// RGB32 from the reader is normally bottom-up: `MF_MT_DEFAULT_STRIDE` is negative, meaning
    /// that the first byte of the buffer is the bottom row of the picture.
    flip: bool,
}

impl WinReader {
    pub(crate) fn open(path: &Path) -> Result<(Self, VideoInfo), VideoError> {
        ensure_media_foundation()?;
        let url = file_url(path)?;

        let mut raw_attributes: Option<IMFAttributes> = None;
        unsafe { MFCreateAttributes(&mut raw_attributes, 4) }
            .map_err(|e| VideoError::Backend(format!("MFCreateAttributes: {}", describe(&e))))?;
        let attributes = raw_attributes
            .ok_or_else(|| VideoError::Backend("MFCreateAttributes returned null".into()))?;
        // Task 1.11: allow the reader to bind a hardware decoder MFT (DXVA), and to insert the
        // video processor MFT that produces the RGB32 output type asked for below.
        unsafe { attributes.SetUINT32(&MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, 1) }
            .map_err(|e| VideoError::Backend(format!("hardware transforms: {}", describe(&e))))?;
        unsafe { attributes.SetUINT32(&MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, 1) }
            .map_err(|e| VideoError::Backend(format!("video processing: {}", describe(&e))))?;

        let reader = unsafe { MFCreateSourceReaderFromURL(&url, &attributes) }.map_err(|e| {
            VideoError::Backend(format!("MFCreateSourceReaderFromURL: {}", describe(&e)))
        })?;

        let stream = stream_index();
        unsafe { reader.SetStreamSelection(all_streams_index(), false) }
            .map_err(|e| VideoError::Backend(format!("SetStreamSelection: {}", describe(&e))))?;
        unsafe { reader.SetStreamSelection(stream, true) }
            .map_err(|e| VideoError::Backend(format!("SetStreamSelection: {}", describe(&e))))?;

        let native = unsafe { reader.GetNativeMediaType(stream, 0) }.map_err(|e| {
            VideoError::Unsupported(format!(
                "no video stream in {}: {}",
                path.display(),
                describe(&e)
            ))
        })?;
        let frame_size = unsafe { native.GetUINT64(&MF_MT_FRAME_SIZE) }.unwrap_or(0);
        if frame_size == 0 {
            return Err(VideoError::Unsupported(format!(
                "{} has no frame size",
                path.display()
            )));
        }
        let fps = nominal_fps(&native);

        let output = unsafe { MFCreateMediaType() }
            .map_err(|e| VideoError::Backend(format!("MFCreateMediaType: {}", describe(&e))))?;
        unsafe { output.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video) }
            .map_err(|e| VideoError::Backend(format!("output major type: {}", describe(&e))))?;
        unsafe { output.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_RGB32) }
            .map_err(|e| VideoError::Backend(format!("output subtype: {}", describe(&e))))?;
        unsafe { output.SetUINT64(&MF_MT_FRAME_SIZE, frame_size) }
            .map_err(|e| VideoError::Backend(format!("output frame size: {}", describe(&e))))?;
        unsafe { output.SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32) }
            .map_err(|e| VideoError::Backend(format!("output interlace mode: {}", describe(&e))))?;
        if let Ok(rate) = unsafe { native.GetUINT64(&MF_MT_FRAME_RATE_RANGE_MAX) } {
            let _ = unsafe { output.SetUINT64(&MF_MT_FRAME_RATE, rate) };
        }
        unsafe { reader.SetCurrentMediaType(stream, None, &output) }.map_err(|e| {
            VideoError::Unsupported(format!(
                "{} cannot be converted to RGB32 (H.264/HEVC only in v1): {}",
                path.display(),
                describe(&e)
            ))
        })?;

        let current = unsafe { reader.GetCurrentMediaType(stream) }.ok();
        let (w, h) = current
            .as_ref()
            .and_then(|t| unsafe { t.GetUINT64(&MF_MT_FRAME_SIZE) }.ok())
            .map(|fs| ((fs >> 32) as u32, fs as u32))
            .filter(|(w, h)| *w > 0 && *h > 0)
            .unwrap_or(((frame_size >> 32) as u32, frame_size as u32));
        // The row stride and the row order are both the negotiator's to decide; a missing stride
        // means the tight top-down layout, which is what an RGB32 output type asks for.
        let (pitch, flip) = current
            .as_ref()
            .and_then(|t| unsafe { t.GetUINT32(&MF_MT_DEFAULT_STRIDE) }.ok())
            .map(|stride| {
                let stride = stride as i32;
                (stride.unsigned_abs() as usize, stride < 0)
            })
            .unwrap_or((w as usize * 4, false));
        let pitch = pitch.max(w as usize * 4);

        let duration = unsafe {
            reader.GetPresentationAttribute(MF_SOURCE_READER_MEDIASOURCE.0 as u32, &MF_PD_DURATION)
        }
        .ok()
        .and_then(|pv| unsafe { PropVariantToUInt64(&pv) }.ok())
        .map(|hundred_ns| hundred_ns as f64 / 1e7)
        .unwrap_or(0.0);
        if !duration.is_finite() || duration <= 0.0 {
            return Err(VideoError::Unsupported(format!(
                "{} reports no duration",
                path.display()
            )));
        }

        let info = VideoInfo {
            w,
            h,
            fps,
            duration_s: duration,
        };
        Ok((
            Self {
                reader,
                w: w as usize,
                h: h as usize,
                pitch,
                flip,
            },
            info,
        ))
    }

    /// Moves the reader to the keyframe at or before `t_s`; the next read starts from there.
    pub(crate) fn seek(&mut self, t_s: f64) -> Result<(), VideoError> {
        let position = (t_s.max(0.0) * 1e7).round().clamp(0.0, i64::MAX as f64) as i64;
        let value = propvariant_i64(position);
        unsafe { self.reader.SetCurrentPosition(&GUID::from_u128(0), &value) }
            .map_err(|e| VideoError::Backend(format!("SetCurrentPosition: {}", describe(&e))))
    }

    /// Decodes the next picture into `dst` and returns its presentation time, or `None` at the end
    /// of the stream. `dst` keeps its allocation between calls.
    ///
    /// The geometry fixed at `open` is the reader's output type, which the video processor
    /// negotiates once. `MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED` is therefore ignored: what the
    /// reader reports mid-stream is the decoder's own padded frame size (90 rows become 96 for
    /// H.264), not the RGB32 output requested here, and following it resized the frame buffer to
    /// the wrong geometry.
    pub(crate) fn read_frame(&mut self, dst: &mut Vec<u8>) -> Result<Option<f64>, VideoError> {
        let stream = stream_index();
        let mut empty_reads = 0;
        loop {
            let mut flags = 0u32;
            let mut timestamp = 0i64;
            let mut sample: Option<IMFSample> = None;
            unsafe {
                self.reader.ReadSample(
                    stream,
                    0,
                    None,
                    Some(&mut flags),
                    Some(&mut timestamp),
                    Some(&mut sample),
                )
            }
            .map_err(|e| VideoError::Backend(format!("ReadSample: {}", describe(&e))))?;

            if flags & (MF_SOURCE_READERF_ERROR.0 as u32) != 0 {
                return Err(VideoError::Decode("source reader reported an error".into()));
            }
            if let Some(sample) = sample {
                blit(dst, &sample, self.w, self.h, self.pitch, self.flip)?;
                return Ok(Some(timestamp as f64 / 1e7));
            }
            if flags & (MF_SOURCE_READERF_ENDOFSTREAM.0 as u32) != 0 {
                return Ok(None);
            }

            // A stream tick or a hole in the file: no picture yet, ask again.
            empty_reads += 1;
            if empty_reads >= MAX_EMPTY_READS {
                return Err(VideoError::Decode(format!(
                    "no picture after {MAX_EMPTY_READS} reads"
                )));
            }
        }
    }
}

/// Copies one decoded sample into `dst` as tightly packed BGRA. One contiguous buffer, no
/// allocation once `dst` is the right size.
fn blit(
    dst: &mut Vec<u8>,
    sample: &IMFSample,
    w: usize,
    h: usize,
    pitch: usize,
    flip: bool,
) -> Result<(), VideoError> {
    let row = w * 4;
    let need = row * h;
    if need == 0 {
        return Err(VideoError::Decode("frame has no pixels".into()));
    }
    if need > MAX_FRAME_BYTES {
        return Err(VideoError::Decode(format!(
            "frame of {w}x{h} is larger than 8K; refusing to allocate {need} bytes"
        )));
    }
    if dst.len() != need {
        dst.clear();
        dst.resize(need, 0);
    }

    let buffer = unsafe { sample.ConvertToContiguousBuffer() }
        .map_err(|e| VideoError::Backend(format!("ConvertToContiguousBuffer: {}", describe(&e))))?;

    // The preferred path: a 2D buffer knows its own stride.
    if let Ok(image) = buffer.cast::<IMF2DBuffer>() {
        let mut scanline: *mut u8 = std::ptr::null_mut();
        let mut buffer_pitch = 0i32;
        if unsafe { image.Lock2D(&mut scanline, &mut buffer_pitch) }.is_ok() {
            let copied = !scanline.is_null() && buffer_pitch != 0;
            if copied {
                let buffer_pitch = (buffer_pitch.unsigned_abs() as usize).max(row);
                copy_rows(dst, need, row, h, scanline, buffer_pitch, flip);
            }
            let _ = unsafe { image.Unlock2D() };
            if copied {
                return Ok(());
            }
        }
    }

    // Otherwise one contiguous allocation: the stride is the negotiated one from the media type,
    // never the buffer length divided by the height, because the buffer can carry padded rows
    // (160x90 is delivered as 160x96, which made that division come out as 682 instead of 640).
    let mut scanline: *mut u8 = std::ptr::null_mut();
    unsafe { buffer.Lock(&mut scanline, None, None) }
        .map_err(|e| VideoError::Backend(format!("IMFMediaBuffer::Lock: {}", describe(&e))))?;
    let available = unsafe { buffer.GetCurrentLength() }.unwrap_or(0) as usize;
    let stride = pitch.max(row);
    let required = (h - 1) * stride + row;
    let result = if scanline.is_null() {
        Err(VideoError::Decode(
            "decoded sample has a null buffer".into(),
        ))
    } else if required > available {
        // A media type that lies about its stride would otherwise read past the buffer.
        Err(VideoError::Decode(format!(
            "sample holds {available} bytes, {w}x{h} at stride {stride} needs {required}"
        )))
    } else {
        copy_rows(dst, need, row, h, scanline, stride, flip);
        Ok(())
    };
    let _ = unsafe { buffer.Unlock() };
    result
}

/// Row-wise copy from a locked Media Foundation buffer into top-down BGRA rows.
fn copy_rows(
    dst: &mut [u8],
    need: usize,
    row: usize,
    h: usize,
    src: *const u8,
    pitch: usize,
    flip: bool,
) {
    let dst = &mut dst[..need];
    for y in 0..h {
        let src_y = if flip { h - 1 - y } else { y };
        // SAFETY: `src` is a locked Media Foundation buffer with `h` rows, each `row` bytes long,
        // `pitch` bytes apart, as returned by `Lock2D`/`Lock` for this sample.
        let src_row = unsafe { std::slice::from_raw_parts(src.add(src_y * pitch), row) };
        dst[y * row..(y + 1) * row].copy_from_slice(src_row);
    }
}

fn stream_index() -> u32 {
    MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32
}

fn all_streams_index() -> u32 {
    MF_SOURCE_READER_ALL_STREAMS.0 as u32
}

/// `MF_MT_FRAME_RATE` packed as `(numerator << 32) | denominator`, falling back to the nominal
/// maximum rate, then to 30 fps for files that do not declare one.
fn nominal_fps(type_: &IMFMediaType) -> f64 {
    unsafe { type_.GetUINT64(&MF_MT_FRAME_RATE) }
        .ok()
        .and_then(unpack_ratio)
        .or_else(|| {
            unsafe { type_.GetUINT64(&MF_MT_FRAME_RATE_RANGE_MAX) }
                .ok()
                .and_then(unpack_ratio)
        })
        .unwrap_or(30.0)
}

fn unpack_ratio(packed: u64) -> Option<f64> {
    let numerator = (packed >> 32) as u32;
    let denominator = packed as u32;
    if numerator == 0 || denominator == 0 {
        return None;
    }
    Some(f64::from(numerator) / f64::from(denominator))
}

/// A `VT_I8` propvariant holding a 100 ns position, which is what `SetCurrentPosition` wants with
/// a null time-format GUID.
fn propvariant_i64(value: i64) -> PROPVARIANT {
    PROPVARIANT {
        Anonymous: PROPVARIANT_0 {
            Anonymous: std::mem::ManuallyDrop::new(PROPVARIANT_0_0 {
                vt: VT_I8,
                wReserved1: 0,
                wReserved2: 0,
                wReserved3: 0,
                Anonymous: PROPVARIANT_0_0_0 { hVal: value },
            }),
        },
    }
}

/// Media Foundation needs `MFStartup` once per process, and COM once per thread. Every call is
/// cheap after the first, so `open` just runs both.
fn ensure_media_foundation() -> Result<(), VideoError> {
    // SAFETY: no reserved pointer; MTA is what a decode thread wants. A thread that is already in
    // an apartment gets RPC_E_CHANGED_MODE, which is fine: Media Foundation works in both.
    let _ = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };

    static START: OnceLock<Result<(), String>> = OnceLock::new();
    START
        .get_or_init(|| match unsafe { MFStartup(MF_VERSION, MFSTARTUP_FULL) } {
            Ok(()) => Ok(()),
            Err(e) => Err(describe(&e)),
        })
        .clone()
        .map_err(VideoError::Backend)
}

/// A local path as the wide string the source reader accepts. `MFCreateSourceReaderFromURL` takes
/// a URL, and for a file on disk a plain Win32 path works; the `\\?\` prefix canonicalize adds is
/// stripped because Media Foundation's scheme handlers reject it.
fn file_url(path: &Path) -> Result<HSTRING, VideoError> {
    use std::os::windows::ffi::OsStrExt;

    if !path.is_file() {
        return Err(VideoError::NotFound(path.to_path_buf()));
    }
    let absolute =
        std::fs::canonicalize(path).map_err(|_| VideoError::NotFound(path.to_path_buf()))?;
    let text = absolute.to_string_lossy();
    let text = text.strip_prefix(r"\\?\").unwrap_or(&text);
    let wide: Vec<u16> = std::ffi::OsStr::new(text).encode_wide().collect();
    Ok(HSTRING::from_wide(&wide))
}

fn describe(error: &windows::core::Error) -> String {
    format!(
        "{} (0x{:08X})",
        error.message().trim_end_matches('\0'),
        error.code().0 as u32
    )
}
