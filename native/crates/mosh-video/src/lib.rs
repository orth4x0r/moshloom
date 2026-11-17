//! mosh-video: overlay video decoding (Task 1.11 of the native port plan).
//!
//! One clip per [`VideoSource`]. Media Foundation decodes H.264/HEVC with hardware transforms and
//! hands back BGRA frames that upload straight into a layer texture. Lookup is by time, and the
//! clip loops, so a visual can run for a whole set off one file:
//!
//! ```no_run
//! # use mosh_video::VideoSource;
//! let mut source = VideoSource::open("fixtures/clip.mp4")?;
//! let info = source.info();
//! let frame = source.frame_at(info.duration_s + 0.5).expect("a frame, wrapped to the start");
//! assert_eq!(frame.bgra.len(), (frame.w * frame.h * 4) as usize);
//! # Ok::<(), mosh_video::VideoError>(())
//! ```
//!
//! Memory: one BGRA buffer per source, reused for every frame, so four 1080p overlays cost about
//! 33 MB of frame buffers and no per-frame allocation. The decoder itself holds its own reference
//! frames inside Media Foundation, which is the part the budget test has to measure.
//!
//! Windows only, like the rest of the native app: every Windows call lives in `win.rs`.

use std::fmt;
use std::path::{Path, PathBuf};

pub mod camera;
mod win;

/// A decoded frame, packed BGRA, top row first, `w * h * 4` bytes.
///
/// The same allocation is handed back for every frame from one [`VideoSource`]: compare
/// `bgra.as_ptr()` if you need to know whether the content changed.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoFrame {
    pub w: u32,
    pub h: u32,
    pub bgra: Vec<u8>,
    /// Presentation time of this frame on the clip's own timeline, in seconds from its start.
    pub pts_s: f64,
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
    /// The file exists but is not something v1 decodes: no video stream, no duration, or a codec
    /// that cannot be converted to RGB32.
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

/// One decoded clip with a time-based, looping frame cursor.
pub struct VideoSource {
    info: VideoInfo,
    /// The one frame buffer. `win::WinReader` refills it in place.
    frame: VideoFrame,
    reader: win::WinReader,
    /// Presentation time of the frame currently in `frame`, if any.
    cur_pts_s: Option<f64>,
    /// The last failure, kept because [`VideoSource::frame_at`] answers `None` rather than an
    /// error (the render loop must not stop for a bad overlay).
    last_error: Option<VideoError>,
}

impl VideoSource {
    /// Opens a clip and reads its size, frame rate and duration. Does not decode a frame yet.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, VideoError> {
        let path = path.as_ref();
        let (reader, info) = win::WinReader::open(path)?;
        let len = frame_len(info.w, info.h);
        Ok(Self {
            info,
            frame: VideoFrame {
                w: info.w,
                h: info.h,
                bgra: vec![0; len],
                pts_s: 0.0,
            },
            reader,
            cur_pts_s: None,
            last_error: None,
        })
    }

    pub fn info(&self) -> VideoInfo {
        self.info
    }

    /// The failure that made the last [`VideoSource::frame_at`] return `None`, if any.
    pub fn last_error(&self) -> Option<&VideoError> {
        self.last_error.as_ref()
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
            if let Err(e) = self.reader.seek(t) {
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
            match self.reader.read_frame(&mut self.frame.bgra) {
                Ok(Some(pts_s)) => self.cur_pts_s = Some(pts_s),
                Ok(None) => {
                    if self.cur_pts_s.is_some() {
                        break;
                    }
                    // Nothing decoded since the seek: rewind and decode from the top.
                    self.cur_pts_s = None;
                    if let Err(e) = self.reader.seek(0.0) {
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
