//! Media Foundation video capture: the only Windows code behind [`super::CameraCapture`].
//!
//! A webcam is a video capture device source. `MFEnumDeviceSources` lists them, the friendly-name
//! attribute is what the clip editor shows, and `MFCreateSourceReaderFromMediaSource` opens one
//! into the same kind of source reader `win.rs` uses for a file. The reader is asked for RGB32
//! (BGRA in memory on little-endian Windows) with its video processor enabled, so the device's
//! NV12/YUY2/MJPEG stream is converted on the GPU where the driver offers it, and the negotiated
//! media type reports the real frame size, row stride and row order.
//!
//! The API is deliberately small: [`device_names`] for the editor, [`WinCamera::open`] and
//! [`WinCamera::read_into`] for the capture thread. Nothing here blocks the render thread, because
//! the render thread never calls it.
//!
//! This file mirrors `win.rs`'s sample blit and media-foundation startup instead of sharing them:
//! `win.rs` belongs to the file-decoding lane, and a camera sample arrives through a different
//! open path. The rules are the same ones that file learned:
//!
//! * the stride is the media type's, never the buffer length divided by the height (cameras pad
//!   rows exactly like decoders do);
//! * `MF_MT_DEFAULT_STRIDE`'s sign says which way the rows run, and a negative stride is a
//!   bottom-up buffer that has to be flipped on the way into a top-down frame;
//! * a media type that lies about its stride is refused rather than read past the buffer.

use std::ffi::c_void;
use std::sync::OnceLock;

use windows::Win32::Media::MediaFoundation::{
    IMF2DBuffer, IMFActivate, IMFAttributes, IMFMediaSource, IMFMediaType, IMFSample,
    IMFSourceReader, MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
    MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID, MF_MT_DEFAULT_STRIDE, MF_MT_FRAME_SIZE,
    MF_MT_INTERLACE_MODE, MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE, MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS,
    MF_SOURCE_READER_ALL_STREAMS, MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING,
    MF_SOURCE_READER_FIRST_VIDEO_STREAM, MF_SOURCE_READERF_ENDOFSTREAM, MF_SOURCE_READERF_ERROR,
    MF_VERSION, MFCreateAttributes, MFCreateMediaType, MFCreateSourceReaderFromMediaSource,
    MFEnumDeviceSources, MFMediaType_Video, MFSTARTUP_FULL, MFStartup, MFVideoFormat_RGB32,
    MFVideoInterlace_Progressive,
};
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx, CoTaskMemFree};
use windows::core::{Interface, PWSTR};

use super::{CameraError, CameraFrame, MAX_FRAME_BYTES};

/// No picture after this many consecutive reads: the reader is stuck rather than ticking. A camera
/// delivers stream ticks while it warms up, and a few in a row is normal.
const MAX_EMPTY_READS: usize = 16;

/// The friendly names of the video capture devices, in the order Windows lists them.
pub(crate) fn device_names() -> Result<Vec<String>, CameraError> {
    ensure_media_foundation()?;
    let activates = enumerate()?;
    Ok(activates
        .iter()
        .enumerate()
        .map(|(index, activate)| {
            friendly_name(activate).unwrap_or_else(|| format!("Camera {}", index + 1))
        })
        .collect())
}

/// One open camera, owned by the capture thread that created it (Media Foundation source readers
/// live in the COM apartment that created them).
pub(crate) struct WinCamera {
    reader: IMFSourceReader,
    name: String,
    w: usize,
    h: usize,
    /// Byte distance between rows in the negotiated RGB32 buffer; cameras pad rows.
    pitch: usize,
    /// RGB32 can arrive bottom-up: `MF_MT_DEFAULT_STRIDE` is negative, so the first byte of the
    /// buffer is the bottom row of the picture.
    flip: bool,
}

impl WinCamera {
    /// The device's friendly name.
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// The picture size the reader negotiates, which every frame will have.
    pub(crate) fn width(&self) -> u32 {
        self.w as u32
    }

    pub(crate) fn height(&self) -> u32 {
        self.h as u32
    }

    /// Opens `requested` (exactly the friendly name, case-insensitively) or, for `None`/`""`, the
    /// first device that opens. Trying every device for the default camera is what makes a machine
    /// whose first camera is held by another app still work.
    pub(crate) fn open(requested: Option<&str>) -> Result<WinCamera, CameraError> {
        ensure_media_foundation()?;
        let activates = enumerate()?;
        if activates.is_empty() {
            return Err(CameraError::NoDevice);
        }
        let candidates = candidates(&activates, requested)?;
        let mut last = CameraError::NoDevice;
        for (name, activate) in candidates {
            match open_device(activate, &name) {
                Ok(camera) => return Ok(camera),
                // A camera that cannot deliver RGB32 is as unusable as a busy one here: the layer
                // would be black either way, so try the next device and report the last reason.
                Err(error) => last = error,
            }
        }
        Err(last)
    }

    /// Reads the next picture into `frame`, blocking until the device delivers one. A read error
    /// (a camera that disappeared, a driver that stopped) closes the device when this returns: the
    /// caller retries the open.
    pub(crate) fn read_into(&mut self, frame: &mut CameraFrame) -> Result<(), CameraError> {
        let stream = video_stream();
        let mut empty_reads = 0;
        loop {
            let mut flags = 0u32;
            let mut sample: Option<IMFSample> = None;
            unsafe {
                self.reader
                    .ReadSample(stream, 0, None, Some(&mut flags), None, Some(&mut sample))
            }
            .map_err(|error| {
                CameraError::Backend(format!("{}: ReadSample: {}", self.name, describe(&error)))
            })?;

            if flags & (MF_SOURCE_READERF_ERROR.0 as u32) != 0 {
                return Err(CameraError::Backend(format!(
                    "{}: the reader reported an error (was the camera unplugged?)",
                    self.name
                )));
            }
            if let Some(sample) = sample {
                frame.resize(self.w as u32, self.h as u32);
                blit(
                    &sample, frame, self.w, self.h, self.pitch, self.flip, &self.name,
                )?;
                return Ok(());
            }
            if flags & (MF_SOURCE_READERF_ENDOFSTREAM.0 as u32) != 0 {
                return Err(CameraError::Backend(format!(
                    "{}: the camera stopped streaming",
                    self.name
                )));
            }
            empty_reads += 1;
            if empty_reads >= MAX_EMPTY_READS {
                return Err(CameraError::Backend(format!(
                    "{}: no picture after {MAX_EMPTY_READS} reads",
                    self.name
                )));
            }
        }
    }
}

/// The (name, activate) pairs to try, in order.
fn candidates<'a>(
    activates: &'a [IMFActivate],
    requested: Option<&str>,
) -> Result<Vec<(String, &'a IMFActivate)>, CameraError> {
    let named: Vec<(String, &IMFActivate)> = activates
        .iter()
        .enumerate()
        .map(|(index, activate)| {
            (
                friendly_name(activate).unwrap_or_else(|| format!("Camera {}", index + 1)),
                activate,
            )
        })
        .collect();
    match requested {
        None => Ok(named),
        Some(want) if want.trim().is_empty() => Ok(named),
        Some(want) => {
            let matching: Vec<(String, &IMFActivate)> = named
                .into_iter()
                .filter(|(name, _)| name == want || name.eq_ignore_ascii_case(want))
                .collect();
            if matching.is_empty() {
                Err(CameraError::NotFound(want.to_string()))
            } else {
                Ok(matching)
            }
        }
    }
}

/// Opens one device and negotiates an RGB32 video stream on it.
fn open_device(activate: &IMFActivate, name: &str) -> Result<WinCamera, CameraError> {
    let source: IMFMediaSource = unsafe { activate.ActivateObject() }.map_err(|error| {
        CameraError::Busy(format!("{name}: ActivateObject: {}", describe(&error)))
    })?;
    let attributes = reader_attributes()?;
    let reader =
        unsafe { MFCreateSourceReaderFromMediaSource(&source, &attributes) }.map_err(|error| {
            CameraError::Busy(format!(
                "{name}: MFCreateSourceReaderFromMediaSource: {}",
                describe(&error)
            ))
        })?;

    let stream = video_stream();
    unsafe { reader.SetStreamSelection(all_streams(), false) }.map_err(|error| {
        CameraError::Backend(format!("{name}: SetStreamSelection: {}", describe(&error)))
    })?;
    unsafe { reader.SetStreamSelection(stream, true) }.map_err(|error| {
        CameraError::Backend(format!("{name}: SetStreamSelection: {}", describe(&error)))
    })?;

    let native = unsafe { reader.GetNativeMediaType(stream, 0) }.map_err(|error| {
        CameraError::Unsupported(format!("{name}: no video stream: {}", describe(&error)))
    })?;
    let frame_size = unsafe { native.GetUINT64(&MF_MT_FRAME_SIZE) }.unwrap_or(0);
    if frame_size == 0 {
        return Err(CameraError::Unsupported(format!(
            "{name}: the video stream reports no frame size"
        )));
    }

    let output = unsafe { MFCreateMediaType() }.map_err(|error| {
        CameraError::Backend(format!("MFCreateMediaType: {}", describe(&error)))
    })?;
    unsafe { output.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video) }.map_err(|error| {
        CameraError::Backend(format!("{name}: output major type: {}", describe(&error)))
    })?;
    unsafe { output.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_RGB32) }.map_err(|error| {
        CameraError::Backend(format!("{name}: output subtype: {}", describe(&error)))
    })?;
    unsafe { output.SetUINT64(&MF_MT_FRAME_SIZE, frame_size) }.map_err(|error| {
        CameraError::Backend(format!("{name}: output frame size: {}", describe(&error)))
    })?;
    unsafe { output.SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32) }
        .map_err(|error| {
            CameraError::Backend(format!(
                "{name}: output interlace mode: {}",
                describe(&error)
            ))
        })?;
    // No frame rate is requested: the video processor then keeps the device's native rate, and
    // every UVC camera's range accepts its own rate where it may reject an invented one.
    unsafe { reader.SetCurrentMediaType(stream, None, &output) }.map_err(|error| {
        CameraError::Unsupported(format!(
            "{name}: the camera cannot deliver RGB32 video: {}",
            describe(&error)
        ))
    })?;

    let current = unsafe { reader.GetCurrentMediaType(stream) }.ok();
    let (w, h) = current
        .as_ref()
        .and_then(|media_type: &IMFMediaType| {
            unsafe { media_type.GetUINT64(&MF_MT_FRAME_SIZE) }.ok()
        })
        .map(|size| ((size >> 32) as u32, size as u32))
        .filter(|(w, h)| *w > 0 && *h > 0)
        .unwrap_or(((frame_size >> 32) as u32, frame_size as u32));
    let (pitch, flip) = current
        .as_ref()
        .and_then(|media_type: &IMFMediaType| {
            unsafe { media_type.GetUINT32(&MF_MT_DEFAULT_STRIDE) }.ok()
        })
        .map(|stride| {
            let stride = stride as i32;
            (stride.unsigned_abs() as usize, stride < 0)
        })
        .unwrap_or((w as usize * 4, false));
    let pitch = pitch.max(w as usize * 4);

    Ok(WinCamera {
        reader,
        name: name.to_string(),
        w: w as usize,
        h: h as usize,
        pitch,
        flip,
    })
}

/// Sample reader attributes: the video processor MFT is what turns a camera's YUV/MJPEG stream
/// into the RGB32 the layer texture wants, and hardware transforms let that run on the GPU.
fn reader_attributes() -> Result<IMFAttributes, CameraError> {
    let attributes = create_attributes()?;
    unsafe { attributes.SetUINT32(&MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, 1) }.map_err(
        |error| CameraError::Backend(format!("hardware transforms: {}", describe(&error))),
    )?;
    unsafe { attributes.SetUINT32(&MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, 1) }
        .map_err(|error| CameraError::Backend(format!("video processing: {}", describe(&error))))?;
    Ok(attributes)
}

/// A fresh attribute store.
fn create_attributes() -> Result<IMFAttributes, CameraError> {
    let mut raw: Option<IMFAttributes> = None;
    unsafe { MFCreateAttributes(&mut raw, 4) }.map_err(|error| {
        CameraError::Backend(format!("MFCreateAttributes: {}", describe(&error)))
    })?;
    raw.ok_or_else(|| CameraError::Backend("MFCreateAttributes returned null".into()))
}

/// Enumerates video capture devices. The returned array is Media Foundation's; each element is
/// cloned, then the array is freed, so the caller owns a `Vec` that outlives it.
fn enumerate() -> Result<Vec<IMFActivate>, CameraError> {
    let attributes = create_attributes()?;
    unsafe {
        attributes.SetGUID(
            &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
            &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID,
        )
    }
    .map_err(|error| CameraError::Backend(format!("device source type: {}", describe(&error))))?;

    let mut list: *mut Option<IMFActivate> = std::ptr::null_mut();
    let mut count = 0u32;
    unsafe { MFEnumDeviceSources(&attributes, &mut list, &mut count) }.map_err(|error| {
        CameraError::Backend(format!("MFEnumDeviceSources: {}", describe(&error)))
    })?;
    if list.is_null() || count == 0 {
        return Ok(Vec::new());
    }
    // SAFETY: `MFEnumDeviceSources` handed back an array of `count` CoTaskMemAlloc'd IMFActivate
    // pointers, valid until they are freed below; cloning each keeps the objects alive after.
    let activates: Vec<IMFActivate> = unsafe { std::slice::from_raw_parts(list, count as usize) }
        .iter()
        .filter_map(|entry| entry.clone())
        .collect();
    unsafe { CoTaskMemFree(Some(list as *const c_void)) };
    Ok(activates)
}

/// A device's friendly name, or `None` when it has none (an unnamed device still opens).
fn friendly_name(activate: &IMFActivate) -> Option<String> {
    let mut value = PWSTR::null();
    let mut length = 0u32;
    unsafe {
        activate
            .GetAllocatedString(
                &MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME,
                &mut value,
                &mut length,
            )
            .ok()?
    };
    let name = unsafe { value.to_string() }.ok();
    unsafe { CoTaskMemFree(Some(value.as_ptr() as *const c_void)) };
    name.filter(|name| !name.trim().is_empty())
}

/// Copies one camera sample into `frame` as tightly packed, top-down BGRA.
fn blit(
    sample: &IMFSample,
    frame: &mut CameraFrame,
    w: usize,
    h: usize,
    pitch: usize,
    flip: bool,
    name: &str,
) -> Result<(), CameraError> {
    let row = w * 4;
    let need = row * h;
    if need == 0 {
        return Err(CameraError::Backend(format!(
            "{name}: a frame with no pixels"
        )));
    }
    if need > MAX_FRAME_BYTES {
        return Err(CameraError::Backend(format!(
            "{name}: frame of {w}x{h} is larger than 8K; refusing to allocate {need} bytes"
        )));
    }
    if frame.bgra.len() != need {
        frame.bgra.clear();
        frame.bgra.resize(need, 0);
    }

    let buffer = unsafe { sample.ConvertToContiguousBuffer() }.map_err(|error| {
        CameraError::Backend(format!(
            "{name}: ConvertToContiguousBuffer: {}",
            describe(&error)
        ))
    })?;

    // The preferred path: a 2D buffer knows its own stride.
    if let Ok(image) = buffer.cast::<IMF2DBuffer>() {
        let mut scanline: *mut u8 = std::ptr::null_mut();
        let mut buffer_pitch = 0i32;
        if unsafe { image.Lock2D(&mut scanline, &mut buffer_pitch) }.is_ok() {
            let copied = !scanline.is_null() && buffer_pitch != 0;
            if copied {
                let buffer_pitch = (buffer_pitch.unsigned_abs() as usize).max(row);
                copy_rows(frame, need, row, h, scanline, buffer_pitch, flip);
            }
            let _ = unsafe { image.Unlock2D() };
            if copied {
                return Ok(());
            }
        }
    }

    // Otherwise one contiguous allocation: the stride is the negotiated one from the media type,
    // never the buffer length divided by the height.
    let mut scanline: *mut u8 = std::ptr::null_mut();
    unsafe { buffer.Lock(&mut scanline, None, None) }.map_err(|error| {
        CameraError::Backend(format!(
            "{name}: IMFMediaBuffer::Lock: {}",
            describe(&error)
        ))
    })?;
    let available = unsafe { buffer.GetCurrentLength() }.unwrap_or(0) as usize;
    let stride = pitch.max(row);
    let required = (h - 1) * stride + row;
    let result = if scanline.is_null() {
        Err(CameraError::Backend(format!(
            "{name}: the camera sample has a null buffer"
        )))
    } else if required > available {
        // A media type that lies about its stride would otherwise read past the buffer.
        Err(CameraError::Backend(format!(
            "{name}: sample holds {available} bytes, {w}x{h} at stride {stride} needs {required}"
        )))
    } else {
        copy_rows(frame, need, row, h, scanline, stride, flip);
        Ok(())
    };
    let _ = unsafe { buffer.Unlock() };
    result
}

/// Row-wise copy from a locked Media Foundation buffer into top-down BGRA rows.
fn copy_rows(
    frame: &mut CameraFrame,
    need: usize,
    row: usize,
    h: usize,
    src: *const u8,
    pitch: usize,
    flip: bool,
) {
    let dst = &mut frame.bgra[..need];
    for y in 0..h {
        let src_y = if flip { h - 1 - y } else { y };
        // SAFETY: `src` is a locked Media Foundation buffer with `h` rows, each `row` bytes long,
        // `pitch` bytes apart, as returned by `Lock2D`/`Lock` for this sample.
        let src_row = unsafe { std::slice::from_raw_parts(src.add(src_y * pitch), row) };
        dst[y * row..(y + 1) * row].copy_from_slice(src_row);
    }
}

fn video_stream() -> u32 {
    MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32
}

fn all_streams() -> u32 {
    MF_SOURCE_READER_ALL_STREAMS.0 as u32
}

/// Media Foundation needs `MFStartup` once per process (the call is reference-counted, so this is
/// safe next to `win.rs` doing the same), and COM once per thread.
fn ensure_media_foundation() -> Result<(), CameraError> {
    // SAFETY: no reserved pointer; MTA is what a capture thread wants. A thread already in an
    // apartment gets RPC_E_CHANGED_MODE, which is fine: Media Foundation works in both.
    let _ = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };

    static START: OnceLock<Result<(), String>> = OnceLock::new();
    START
        .get_or_init(|| match unsafe { MFStartup(MF_VERSION, MFSTARTUP_FULL) } {
            Ok(()) => Ok(()),
            Err(error) => Err(describe(&error)),
        })
        .clone()
        .map_err(|error| CameraError::Backend(format!("MFStartup: {error}")))
}

fn describe(error: &windows::core::Error) -> String {
    format!(
        "{} (0x{:08X})",
        error.message().trim_end_matches('\0'),
        error.code().0 as u32
    )
}
