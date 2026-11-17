//! mosh-video: overlay video decoding (Task 1.11 of the native port plan, plus the HAP/alpha
//! follow-up of Task 2.8).
//!
//! One clip per [`VideoSource`]. Two decoders behind one interface:
//!
//! * **HAP** (QuickTime `.mov` with a `Hap1`/`Hap5`/`HapY`/`HapA`/`HapM` track): the VJ standard
//!   for overlay video, and the one that keeps transparency. A frame is BCn (DXT/BC) blocks, so
//!   it is parsed, Snappy-decompressed and handed over as a GPU upload with no CPU pixel decode;
//!   `Hap5` carries straight alpha and `HapM` (HAP Q Alpha) a separate BC4 alpha plane, which is
//!   what lets an overlay be a cut-out graphic. See [`hap`], and `mosh-render`'s block-compressed
//!   texture upload for the GPU half.
//! * **Media Foundation** for everything else: H.264/HEVC with hardware transforms, handing back
//!   BGRA frames.
//!
//! [`VideoSource::open`] picks the HAP path for a HAP file and Media Foundation otherwise, so a
//! caller that just asks for frames at a time keeps working. Lookup is by time and the clip loops,
//! so a visual can run for a whole set off one file:
//!
//! ```no_run
//! # use mosh_video::VideoSource;
//! let mut source = VideoSource::open("fixtures/clip.mp4")?;
//! let info = source.info();
//! let frame = source.frame_at(info.duration_s + 0.5).expect("a frame, wrapped to the start");
//! if let Some(hap) = frame.hap() {
//!     // Hand `hap.data` to a compressed texture; no CPU decode happened.
//!     let _ = hap.format;
//! } else {
//!     assert_eq!(frame.bgra.len(), (frame.w * frame.h * 4) as usize);
//! }
//! # Ok::<(), mosh_video::VideoError>(())
//! ```
//!
//! Memory: one frame buffer per source, reused for every frame. A Media Foundation source keeps
//! one BGRA frame (8.3 MB at 1080p); a HAP source keeps the compressed sample plus one BCn plane
//! (0.25-0.5 MB at 1080p, so four HAP overlays cost less than one BGRA frame of CPU memory).
//! [`VideoSource::memory_bytes`] reports what a source holds. The decoders themselves hold their
//! own reference frames inside Media Foundation, which is the part the budget test has to measure.
//!
//! Windows only, like the rest of the native app: every Media Foundation call lives in `win.rs`.
//! The HAP path ([`hap`] + the demuxer behind it) is plain Rust with no Windows dependency, so it
//! is unit-tested on every machine.

use std::fmt;
use std::path::{Path, PathBuf};

pub mod hap;
mod mov;
mod win;

/// A decoded frame. Exactly one of the two payloads holds the picture:
///
/// * [`VideoFrame::bgra`] for Media Foundation decodes (BGRA, top row first, `w * h * 4` bytes);
/// * [`VideoFrame::hap`] for HAP, which is never decoded to RGBA: it is the BCn block payload a
///   compressed texture uploads straight to the GPU.
///
/// The same allocations are handed back for every frame from one [`VideoSource`]: compare
/// `bgra.as_ptr()` (or `hap.data.as_ptr()`) if you need to know whether the content changed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VideoFrame {
    /// Display width: the picture's size, not the 4x4-padded texture size.
    pub w: u32,
    /// Display height.
    pub h: u32,
    /// Media Foundation's RGB32 output: BGRA, top row first. Empty for a HAP frame.
    pub bgra: Vec<u8>,
    /// Presentation time of this frame on the clip's own timeline, in seconds from its start.
    pub pts_s: f64,
    /// The BCn payload of a HAP frame. `None` for Media Foundation frames.
    pub hap: Option<hap::HapFrame>,
}

impl VideoFrame {
    /// The HAP payload, when this frame came from a HAP file.
    pub fn hap(&self) -> Option<&hap::HapFrame> {
        self.hap.as_ref()
    }

    /// The size the frame's HAP texture is padded to: whole 4x4 blocks, so a 160x90 clip is
    /// 160x92 and a 1920x1080 clip needs no padding at all. `None` for a Media Foundation frame,
    /// whose rows are tightly packed.
    pub fn hap_texture_size(&self) -> Option<(u32, u32)> {
        self.hap
            .as_ref()
            .map(|_| (self.w.div_ceil(4) * 4, self.h.div_ceil(4) * 4))
    }
}

/// What a clip is, as the renderer needs it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VideoInfo {
    pub w: u32,
    pub h: u32,
    /// Nominal frame rate. Variable-frame-rate clips report their nominal rate; it is only used to
    /// decide which frame covers a timestamp, never to step the show.
    pub fps: f64,
    pub duration_s: f64,
}

/// Why a clip could not be opened or decoded. Never a panic: a missing or unreadable overlay is an
/// error the app shows and carries on from (hands-free rule).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VideoError {
    /// The path does not exist or is not a file.
    NotFound(PathBuf),
    /// The file exists but is not something v1 decodes: no video stream, no duration, a codec
    /// that cannot be converted to RGB32, or a HAP track whose tables or frames do not add up.
    Unsupported(String),
    /// A decode failed after the file was opened.
    Decode(String),
    /// A Media Foundation call failed. Carries the call, the message and the HRESULT.
    Backend(String),
}

impl fmt::Display for VideoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(path) => write!(f, "video not found: {}", path.display()),
            Self::Unsupported(reason) => write!(f, "unsupported video: {reason}"),
            Self::Decode(reason) => write!(f, "decode failed: {reason}"),
            Self::Backend(reason) => write!(f, "Media Foundation failed: {reason}"),
        }
    }
}

impl std::error::Error for VideoError {}

/// Moving forward more than this many seconds seeks instead of decoding every frame in between.
/// A show relocates the playhead when Live jumps, so the seek path is the common one.
const SEEK_FORWARD_S: f64 = 0.75;

/// Tolerance for comparing a requested time with a frame's presentation time.
const EPSILON_S: f64 = 1e-9;

/// Which decoder opened the file. HAP is plain Rust; Media Foundation needs `win.rs`.
enum Decoder {
    MediaFoundation(win::WinReader),
    Hap(Box<hap::HapReader>),
}

impl Decoder {
    fn seek(&mut self, t_s: f64) -> Result<(), VideoError> {
        match self {
            Self::MediaFoundation(reader) => reader.seek(t_s),
            Self::Hap(reader) => {
                reader.seek(t_s);
                Ok(())
            }
        }
    }

    /// Fills the frame's payload (`bgra` or `hap`) and returns its presentation time, or `None`
    /// at the end of the clip.
    fn read_frame(&mut self, frame: &mut VideoFrame) -> Result<Option<f64>, VideoError> {
        match self {
            Self::MediaFoundation(reader) => {
                let pts = reader.read_frame(&mut frame.bgra)?;
                if pts.is_some() {
                    frame.hap = None;
                }
                Ok(pts)
            }
            Self::Hap(reader) => {
                let hap_frame = frame.hap.get_or_insert_with(hap::HapFrame::default);
                let pts = reader.read_frame(hap_frame)?;
                if pts.is_some() {
                    frame.bgra.clear();
                }
                Ok(pts)
            }
        }
    }

    fn memory_bytes(&self) -> usize {
        match self {
            Self::MediaFoundation(_) => 0,
            Self::Hap(reader) => reader.memory_bytes(),
        }
    }
}

/// One decoded clip with a time-based, looping frame cursor.
pub struct VideoSource {
    info: VideoInfo,
    /// The one frame buffer. The decoder refills it in place.
    frame: VideoFrame,
    decoder: Decoder,
    /// Presentation time of the frame currently in `frame`, if any.
    cur_pts_s: Option<f64>,
    /// The last failure, kept because [`VideoSource::frame_at`] answers `None` rather than an
    /// error (the render loop must not stop for a bad overlay).
    last_error: Option<VideoError>,
}

impl VideoSource {
    /// Opens a clip and reads its size, frame rate and duration. Does not decode a frame yet
    /// (a HAP clip validates its first frame here, so a broken file fails at open).
    ///
    /// A QuickTime/MP4 file with a HAP track goes down [`hap`]'s path; everything else goes to
    /// Media Foundation. Callers do not choose.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, VideoError> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(VideoError::NotFound(path.to_path_buf()));
        }

        let (decoder, info) = match hap::HapReader::open_if_hap(path)? {
            Some((reader, info)) => (Decoder::Hap(Box::new(reader)), info),
            None => {
                let (reader, info) = win::WinReader::open(path)?;
                (Decoder::MediaFoundation(reader), info)
            }
        };
        let len = if matches!(decoder, Decoder::Hap(_)) {
            // HAP keeps its payload in `frame.hap`, which grows on the first read.
            0
        } else {
            frame_len(info.w, info.h)
        };
        Ok(Self {
            info,
            frame: VideoFrame {
                w: info.w,
                h: info.h,
                bgra: vec![0; len],
                pts_s: 0.0,
                hap: None,
            },
            decoder,
            cur_pts_s: None,
            last_error: None,
        })
    }

    pub fn info(&self) -> VideoInfo {
        self.info
    }

    /// True when this clip is HAP, so the renderer's overlay path must take
    /// [`VideoFrame::hap`] rather than [`VideoFrame::bgra`].
    pub fn is_hap(&self) -> bool {
        matches!(self.decoder, Decoder::Hap(_))
    }

    /// The failure that made the last [`VideoSource::frame_at`] return `None`, if any.
    pub fn last_error(&self) -> Option<&VideoError> {
        self.last_error.as_ref()
    }

    /// CPU memory this source holds, in bytes: the frame buffer(s), the demuxer's sample table
    /// and the compressed sample. The GPU textures an overlay draws into are not included.
    pub fn memory_bytes(&self) -> usize {
        let frame = self.frame.bgra.capacity()
            + self
                .frame
                .hap
                .as_ref()
                .map(|hap| {
                    hap.data.capacity()
                        + hap
                            .alpha
                            .as_ref()
                            .map(|alpha| alpha.data.capacity())
                            .unwrap_or(0)
                })
                .unwrap_or(0);
        frame + self.decoder.memory_bytes() + std::mem::size_of::<Self>()
    }

    /// The frame displayed at `t_s` on a looping timeline: `t_s` is reduced modulo the duration,
    /// so `duration + 0.5` is half a second into the second pass. Backwards jumps seek, small
    /// forwards jumps decode.
    ///
    /// `None` only when the clip cannot produce a frame at all; the reason is in
    /// [`VideoSource::last_error`].
    pub fn frame_at(&mut self, t_s: f64) -> Option<&VideoFrame> {
        let duration = self.info.duration_s;
        if !duration.is_finite() || duration <= 0.0 {
            return None;
        }
        let frame_dt = self.frame_dt();

        let mut t = if t_s.is_finite() { t_s % duration } else { 0.0 };
        if t < 0.0 {
            t += duration;
        }

        let seek = match self.cur_pts_s {
            None => true,
            Some(cur) => t + EPSILON_S < cur || t > cur + SEEK_FORWARD_S,
        };
        if seek {
            self.cur_pts_s = None;
            if let Err(e) = self.decoder.seek(t) {
                self.last_error = Some(e);
                return None;
            }
        }

        // Read until the frame in the buffer covers `t`. The clamp at the end of the stream keeps
        // the last frame, so asking for `duration - epsilon` never comes back empty.
        let max_reads = ((duration * self.info.fps).ceil() as usize).saturating_add(8);
        let mut reads = 0usize;
        loop {
            if let Some(cur) = self.cur_pts_s
                && t < cur + frame_dt - EPSILON_S
            {
                break;
            }
            match self.decoder.read_frame(&mut self.frame) {
                Ok(Some(pts_s)) => self.cur_pts_s = Some(pts_s),
                Ok(None) => {
                    if self.cur_pts_s.is_some() {
                        break;
                    }
                    // Nothing decoded since the seek: rewind and decode from the top.
                    self.cur_pts_s = None;
                    if let Err(e) = self.decoder.seek(0.0) {
                        self.last_error = Some(e);
                        return None;
                    }
                }
                Err(e) => {
                    self.last_error = Some(e);
                    return None;
                }
            }
            reads += 1;
            if reads > max_reads {
                self.last_error = Some(VideoError::Decode(format!(
                    "frame_at({t_s}) decoded {reads} frames without reaching t={t}"
                )));
                return None;
            }
        }

        self.frame.pts_s = self.cur_pts_s?;
        self.last_error = None;
        Some(&self.frame)
    }

    /// One nominal frame period, floored at 1 ms so a bogus frame rate cannot spin the read loop.
    fn frame_dt(&self) -> f64 {
        if self.info.fps > 0.0 {
            (1.0 / self.info.fps).max(1e-3)
        } else {
            1.0 / 30.0
        }
    }
}

fn frame_len(w: u32, h: u32) -> usize {
    (w as usize).saturating_mul(h as usize).saturating_mul(4)
}
