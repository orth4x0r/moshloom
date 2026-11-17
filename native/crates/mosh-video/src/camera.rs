//! Live camera capture: a webcam's picture on its own thread, newest frame wins.
//!
//! The camera layer needs a picture the render thread can take at any moment without ever waiting
//! for the device. So the device lives on a capture thread started by
//! [`CameraCapture::open`]: the thread opens the camera, reads frames as fast as the device makes
//! them, converts each into BGRA (Media Foundation's video processor does the YUV -> RGB32 step)
//! and publishes it into a one-slot mailbox. The render thread calls
//! [`CameraCapture::take_frame`], which is a `try_lock`: it takes the newest picture, or answers
//! `None` immediately when the capture thread is between frames. A picture the renderer did not
//! take yet is replaced by the next one, so a render thread that runs slower than the camera drops
//! frames instead of queueing them, and a render thread that runs faster redraws the newest frame.
//!
//! # Failure is a state, not an error on the render thread
//!
//! A missing camera, a camera another app holds, a device that was unplugged mid-set: all of them
//! end in the same place, a failed read or open that marks [`CameraCapture::is_failed`], closes the
//! device and tries again every [`RETRY_INTERVAL`]. The capture thread reports the reason through
//! [`CameraCapture::stats`] (`last_error`), which the clip editor can show; the render thread only
//! ever sees "no picture" and draws black.
//!
//! # Memory
//!
//! A frame buffer is reused through the mailbox: the capture thread takes a spare buffer, fills
//! it, publishes it; the renderer uploads it and hands it back with [`CameraCapture::recycle`].
//! At most [`POOL_LIMIT`] spare buffers plus the one published plus the one in the renderer are
//! alive, so a 1080p camera costs four frames of CPU memory (33 MB) and no per-frame allocation.
//!
//! Windows only, like the rest of the native app: every Windows call lives in `camera_win.rs`.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[path = "camera_win.rs"]
mod win;

/// How long a missing, busy or unplugged camera waits before another open attempt. A wrong name
/// recovers when the device shows up, a busy device recovers when the other app lets go, and a
/// cable that came loose recovers when it is plugged back in.
pub const RETRY_INTERVAL: Duration = Duration::from_secs(2);

/// Spare frame buffers kept for reuse.
const POOL_LIMIT: usize = 2;

/// 8K BGRA: a real camera frame never needs more, so a bad media type cannot ask for a huge
/// allocation.
pub(crate) const MAX_FRAME_BYTES: usize = 7680 * 4320 * 4;

/// Why a camera could not be opened or stopped streaming. Never a panic: the capture thread keeps
/// the reason for [`CameraCapture::stats`] and retries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CameraError {
    /// The machine reports no video capture devices at all.
    NoDevice,
    /// No device has that friendly name.
    NotFound(String),
    /// The device exists but will not open: another app holds it, or the driver refused.
    Busy(String),
    /// The device opened but cannot deliver RGB32 video (no video stream, no frame size).
    Unsupported(String),
    /// A Media Foundation call failed. Carries the call, the message and the HRESULT.
    Backend(String),
}

impl fmt::Display for CameraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDevice => f.write_str("no camera on this machine"),
            Self::NotFound(name) => write!(f, "camera not found: {name}"),
            Self::Busy(reason) => write!(f, "camera busy: {reason}"),
            Self::Unsupported(reason) => write!(f, "unsupported camera: {reason}"),
            Self::Backend(reason) => write!(f, "Media Foundation failed: {reason}"),
        }
    }
}

impl std::error::Error for CameraError {}

/// One captured picture: BGRA8, top row first, `w * h * 4` bytes.
///
/// The buffer is the capture thread's, borrowed by the renderer for one upload; hand it back with
/// [`CameraCapture::recycle`] so the next frame reuses it.
pub struct CameraFrame {
    pub w: u32,
    pub h: u32,
    pub bgra: Vec<u8>,
}

impl CameraFrame {
    /// An empty buffer for a `w x h` picture.
    fn sized(w: u32, h: u32) -> CameraFrame {
        CameraFrame {
            w,
            h,
            bgra: vec![0; frame_len(w, h)],
        }
    }

    /// Makes this buffer hold a `w x h` picture, keeping the allocation when it is already large
    /// enough. Only the capture thread calls this, before converting a sample into it.
    fn resize(&mut self, w: u32, h: u32) {
        let need = frame_len(w, h);
        if self.bgra.len() < need {
            self.bgra.resize(need, 0);
        } else if self.bgra.len() > need {
            self.bgra.truncate(need);
        }
        self.w = w;
        self.h = h;
    }
}

impl fmt::Debug for CameraFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CameraFrame")
            .field("w", &self.w)
            .field("h", &self.h)
            .field("bytes", &self.bgra.len())
            .finish()
    }
}

/// What the capture thread is doing, for the clip editor and for tests. A snapshot: the camera can
/// change state the moment after it is read.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CameraStats {
    /// Frames published to the mailbox since the capture started.
    pub frames: u64,
    /// Pictures replaced by a newer one before the renderer took them.
    pub dropped: u64,
    /// Open attempts, one per retry while the camera is missing, busy or unplugged.
    pub reopen_attempts: u64,
    /// True while a device is open and streaming.
    pub open: bool,
    /// The friendly name of the open device, when it is open.
    pub device: Option<String>,
    /// The size of the last published frame, once one has arrived.
    pub frame: Option<(u32, u32)>,
    /// The failure that stopped the last attempt, cleared when a picture arrives again.
    pub last_error: Option<String>,
}

/// The handover between the capture thread and the render thread. Never held across a device
/// call: the capture thread locks it only to swap buffers and update counters.
struct Mailbox {
    newest: Option<CameraFrame>,
    pool: Vec<CameraFrame>,
    stats: CameraStats,
}

impl Mailbox {
    fn publish(&mut self, frame: CameraFrame) {
        self.stats.frame = Some((frame.w, frame.h));
        if let Some(unseen) = self.newest.replace(frame) {
            // The renderer did not take the previous picture: keep the new one, drop the old.
            self.stats.dropped += 1;
            self.recycle(unseen);
        }
        self.stats.frames += 1;
    }

    fn take(&mut self) -> Option<CameraFrame> {
        self.newest.take()
    }

    fn recycle(&mut self, frame: CameraFrame) {
        if self.pool.len() < POOL_LIMIT {
            self.pool.push(frame);
        }
    }

    fn spare(&mut self) -> Option<CameraFrame> {
        self.pool.pop()
    }
}

/// One camera, streaming on its own thread. Drop it to stop the capture.
pub struct CameraCapture {
    shared: Arc<Mutex<Mailbox>>,
    /// The most recent attempt failed, so there is no picture to draw. Lock-free: the render
    /// thread reads it every frame.
    failed: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    /// The requested name, or `None` for the default camera, so [`Self::requested`] can say which
    /// device a layer was set to even while that device is missing.
    requested: Option<String>,
    thread: Option<JoinHandle<()>>,
}

impl CameraCapture {
    /// Starts capturing `device` (`None` or `Some("")` = the default camera). The device is opened
    /// on the capture thread, so this never blocks the caller; failures show up in
    /// [`Self::stats`] and are retried every [`RETRY_INTERVAL`].
    pub fn open(device: Option<&str>) -> CameraCapture {
        let requested = device
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_string);
        let shared = Arc::new(Mutex::new(Mailbox {
            newest: None,
            pool: Vec::new(),
            stats: CameraStats::default(),
        }));
        // No picture yet, so the layer is black until the first frame arrives.
        let failed = Arc::new(AtomicBool::new(true));
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let shared = Arc::clone(&shared);
            let failed = Arc::clone(&failed);
            let stop = Arc::clone(&stop);
            let requested_thread = requested.clone();
            let name = match &requested {
                Some(device) => format!("camera: {device}"),
                None => "camera: default".to_string(),
            };
            thread::Builder::new()
                .name(name)
                .spawn(move || capture_loop(shared, failed, stop, requested_thread))
                .ok()
        };
        if thread.is_none() {
            // The thread could not start: report it the same way a failed open is reported,
            // without touching any Windows API on the caller's thread.
            let reason = "the capture thread could not start".to_string();
            if let Ok(mut mailbox) = shared.lock() {
                mailbox.stats.last_error = Some(reason);
            }
        }
        CameraCapture {
            shared,
            failed,
            stop,
            requested,
            thread,
        }
    }

    /// The newest picture, or `None` when there is none yet or the capture thread is publishing
    /// one at this very moment. Never blocks.
    pub fn take_frame(&self) -> Option<CameraFrame> {
        let mut mailbox = match self.shared.try_lock() {
            Ok(mailbox) => mailbox,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => return None,
        };
        mailbox.take()
    }

    /// Hands a frame back so the capture thread reuses its buffer. Dropping it instead is correct
    /// too, just one allocation more.
    pub fn recycle(&self, frame: CameraFrame) {
        if let Ok(mut mailbox) = self.shared.lock() {
            mailbox.recycle(frame);
        }
    }

    /// A snapshot of the capture thread's state.
    pub fn stats(&self) -> CameraStats {
        match self.shared.lock() {
            Ok(mailbox) => mailbox.stats.clone(),
            Err(poisoned) => poisoned.into_inner().stats.clone(),
        }
    }

    /// True while the last open or read failed: the layer draws black and the capture thread is
    /// waiting to retry. Lock-free, so the render thread can ask every frame.
    pub fn is_failed(&self) -> bool {
        self.failed.load(Ordering::Relaxed)
    }

    /// The device this capture was asked for: `None` for the default camera.
    pub fn requested(&self) -> Option<&str> {
        self.requested.as_deref()
    }

    /// Asks the capture thread to stop without waiting for it. [`Drop`] does this too.
    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Stops capturing and waits for the thread, for tests and app shutdown. Blocks until the
    /// thread notices: at most one frame interval, unless the device itself is stuck in a read.
    pub fn shutdown(&mut self) {
        self.request_stop();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for CameraCapture {
    fn drop(&mut self) {
        self.request_stop();
        if let Some(thread) = self.thread.take() {
            // Never join on the dropping thread: a read is allowed to be blocked until the next
            // frame, and the render thread drops a layer's capture when the clip changes. The
            // reaper only holds the stopped thread, so a device that is slow to let go cannot
            // stall the show.
            let _ = thread::Builder::new()
                .name("camera-reaper".to_string())
                .spawn(move || {
                    let _ = thread.join();
                });
        }
    }
}

/// The capture thread: open, read, publish, and retry every [`RETRY_INTERVAL`] when anything goes
/// wrong. `shared` is locked only for bookkeeping and buffer swaps.
fn capture_loop(
    shared: Arc<Mutex<Mailbox>>,
    failed: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    requested: Option<String>,
) {
    while !stop.load(Ordering::Relaxed) {
        with_mailbox(&shared, |mailbox| {
            mailbox.stats.reopen_attempts += 1;
        });
        match win::WinCamera::open(requested.as_deref()) {
            Ok(mut camera) => {
                let name = camera.name().to_string();
                let (width, height) = (camera.width(), camera.height());
                with_mailbox(&shared, |mailbox| {
                    mailbox.stats.open = true;
                    mailbox.stats.device = Some(name);
                    mailbox.stats.last_error = None;
                });
                while !stop.load(Ordering::Relaxed) {
                    let mut frame = with_mailbox(&shared, Mailbox::spare)
                        .unwrap_or_else(|| CameraFrame::sized(width, height));
                    match camera.read_into(&mut frame) {
                        Ok(()) => {
                            failed.store(false, Ordering::Relaxed);
                            with_mailbox(&shared, |mailbox| {
                                mailbox.stats.last_error = None;
                                mailbox.publish(frame);
                            });
                        }
                        Err(error) => {
                            failed.store(true, Ordering::Relaxed);
                            with_mailbox(&shared, |mailbox| {
                                mailbox.stats.last_error = Some(error.to_string());
                            });
                            break;
                        }
                    }
                }
                with_mailbox(&shared, |mailbox| {
                    mailbox.stats.open = false;
                    mailbox.stats.device = None;
                });
            }
            Err(error) => {
                failed.store(true, Ordering::Relaxed);
                with_mailbox(&shared, |mailbox| {
                    mailbox.stats.open = false;
                    mailbox.stats.device = None;
                    mailbox.stats.last_error = Some(error.to_string());
                });
            }
        }
        wait_before_retry(&stop);
    }
}

/// Runs `f` on the mailbox, ignoring a poisoned lock (a panic while holding it must not stop the
/// capture or the renderer).
fn with_mailbox<T>(shared: &Mutex<Mailbox>, f: impl FnOnce(&mut Mailbox) -> T) -> T {
    match shared.lock() {
        Ok(mut mailbox) => f(&mut mailbox),
        Err(poisoned) => f(&mut poisoned.into_inner()),
    }
}

/// Sleeps in short steps so a stopped capture does not wait out the whole retry interval.
fn wait_before_retry(stop: &AtomicBool) {
    const STEP: Duration = Duration::from_millis(50);
    let mut waited = Duration::ZERO;
    while waited < RETRY_INTERVAL {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        thread::sleep(STEP);
        waited += STEP;
    }
}

fn frame_len(w: u32, h: u32) -> usize {
    (w as usize).saturating_mul(h as usize).saturating_mul(4)
}

/// The camera names Windows reports, in the order the system lists them. Blocking: the clip
/// editor calls it, the render thread does not.
pub fn devices() -> Vec<String> {
    win::device_names().unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: u32, h: u32, fill: u8) -> CameraFrame {
        CameraFrame {
            w,
            h,
            bgra: vec![fill; frame_len(w, h)],
        }
    }

    fn mailbox() -> Mailbox {
        Mailbox {
            newest: None,
            pool: Vec::new(),
            stats: CameraStats::default(),
        }
    }

    #[test]
    fn newest_frame_wins_and_the_replaced_one_is_dropped() {
        let mut mailbox = mailbox();
        mailbox.publish(frame(4, 2, 1));
        mailbox.publish(frame(4, 2, 2));
        mailbox.publish(frame(4, 2, 3));

        let newest = mailbox.take().expect("a frame");
        assert_eq!(newest.bgra[0], 3, "the newest picture is the one taken");
        assert_eq!(mailbox.stats.frames, 3);
        assert_eq!(mailbox.stats.dropped, 2);
        assert_eq!(
            mailbox.stats.frame,
            Some((4, 2)),
            "the stats say what size the last picture was"
        );
        assert!(mailbox.take().is_none(), "the mailbox holds one picture");
        println!(
            "mailbox: 3 published, {} dropped, {} spare",
            mailbox.stats.dropped,
            mailbox.pool.len()
        );
    }

    #[test]
    fn recycled_frames_are_reused_and_the_pool_is_bounded() {
        let mut mailbox = mailbox();
        for _ in 0..(POOL_LIMIT + 4) {
            mailbox.recycle(frame(2, 2, 0));
        }
        assert_eq!(
            mailbox.pool.len(),
            POOL_LIMIT,
            "the pool keeps {POOL_LIMIT} buffers"
        );

        let taken = mailbox.spare().expect("a spare buffer");
        assert_eq!(mailbox.pool.len(), POOL_LIMIT - 1);
        mailbox.recycle(taken);
        assert_eq!(mailbox.pool.len(), POOL_LIMIT, "a returned frame is reused");
        println!(
            "pool: {} buffers after {} recycles",
            mailbox.pool.len(),
            POOL_LIMIT + 5
        );
    }

    #[test]
    fn a_frame_resized_larger_keeps_its_allocation_when_it_fits() {
        let mut frame = CameraFrame::sized(4, 2);
        let capacity = frame.bgra.capacity();
        frame.resize(2, 2);
        assert_eq!(frame.bgra.len(), 16);
        assert_eq!((frame.w, frame.h), (2, 2));
        frame.resize(4, 2);
        assert_eq!(frame.bgra.len(), 32);
        assert!(frame.bgra.capacity() >= capacity);
        println!(
            "frame buffer: 4x2 -> 2x2 -> 4x2, len {} capacity {}",
            frame.bgra.len(),
            frame.bgra.capacity()
        );
    }

    #[test]
    fn failures_read_as_sentences() {
        assert_eq!(
            CameraError::NoDevice.to_string(),
            "no camera on this machine"
        );
        assert_eq!(
            CameraError::NotFound("USB Cam".into()).to_string(),
            "camera not found: USB Cam"
        );
        assert_eq!(
            CameraError::Busy("USB Cam: in use".into()).to_string(),
            "camera busy: USB Cam: in use"
        );
        assert_eq!(
            CameraError::Unsupported("no video stream".into()).to_string(),
            "unsupported camera: no video stream"
        );
        assert!(
            CameraError::Backend("ReadSample: failed (0x00000001)".into())
                .to_string()
                .starts_with("Media Foundation failed: ReadSample")
        );
    }

    #[test]
    fn a_capture_that_cannot_start_reports_it_without_panicking() {
        // `open` never touches Windows on the caller's thread, so this is safe everywhere: the
        // capture thread either finds a camera or records the reason.
        let mut capture = CameraCapture::open(Some("a camera that does not exist"));
        let stats = capture.stats();
        assert_eq!(capture.requested(), Some("a camera that does not exist"));
        assert!(
            capture.take_frame().is_none(),
            "no picture for a missing camera"
        );
        println!(
            "missing camera: attempts {}, error {:?}",
            stats.reopen_attempts, stats.last_error
        );
        capture.shutdown();
    }
}
