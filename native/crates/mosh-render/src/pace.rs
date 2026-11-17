//! Frame pacing: the stage renders at most `max_fps` frames per second, whatever the display's
//! refresh is.
//!
//! The stage presents in FIFO, so the display paces the loop: on the reference laptop's
//! **2560x1600 panel at 120 Hz**, `Stage::present` returns every 8.3 ms and the render thread
//! draws 120 frames a second — twice the budget's rate, at twice the GPU cost per second. The
//! design's budget (2 ms CPU, 6 ms GPU per frame at 1920x1080) is written for 60 fps, so the
//! stage must not render faster than that even when the panel can.
//!
//! Two pieces:
//!
//! * [`FramePacer`] — the policy, pure `Instant` arithmetic so it is tested without a clock:
//!   when the next frame is due, given a frame rate and when the last frame started.
//! * [`WaitTimer`] — the wait itself. `std::thread::sleep` is **not** usable for this: Windows
//!   quantises `Sleep` to the system tick, and a `Sleep(16)` on the reference machine takes
//!   31 ms (measured), i.e. 32 fps. A waitable timer created with
//!   `CREATE_WAITABLE_TIMER_HIGH_RESOLUTION` has ~0.5 ms accuracy and is per-timer, so it does
//!   not raise the system-wide timer resolution the way `timeBeginPeriod(1)` would (that costs
//!   battery, which is the resource this whole budget is about). Where the high-resolution timer
//!   is unavailable the fallback is a plain sleep, which caps the rate at *or below* the target:
//!   a cap that is occasionally slower is still a cap.
//!
//! The pacer never busy-waits and never spins: one wait per frame.

use std::time::{Duration, Instant};

/// The stage's default frame rate cap, and the rate the performance budget is written for.
pub const DEFAULT_MAX_FPS: u32 = 60;

/// A frame rate cap as `Instant` arithmetic.
///
/// The deadline is an absolute grid (`previous + interval`), not "this frame plus a sleep", so a
/// late frame does not push every later frame late with it. A frame that started on time is
/// waited for; a frame that is already late starts immediately.
#[derive(Clone, Debug)]
pub struct FramePacer {
    max_fps: u32,
    interval: Duration,
    /// When the next frame should start; `None` until the first frame is paced.
    next: Option<Instant>,
    frames_paced: u64,
    frames_late: u64,
}

impl Default for FramePacer {
    fn default() -> Self {
        FramePacer::new(DEFAULT_MAX_FPS)
    }
}

impl FramePacer {
    /// A pacer for at most `max_fps` frames a second. `0` means no cap.
    pub fn new(max_fps: u32) -> FramePacer {
        FramePacer {
            max_fps,
            interval: interval_for(max_fps),
            next: None,
            frames_paced: 0,
            frames_late: 0,
        }
    }

    pub fn max_fps(&self) -> u32 {
        self.max_fps
    }

    /// True when a cap is in force.
    pub fn is_capped(&self) -> bool {
        self.max_fps > 0
    }

    /// Change the cap. The deadline grid restarts on the next frame.
    pub fn set_max_fps(&mut self, max_fps: u32) {
        self.max_fps = max_fps;
        self.interval = interval_for(max_fps);
        self.next = None;
    }

    /// Frames this pacer has made the loop wait for.
    pub fn frames_paced(&self) -> u64 {
        self.frames_paced
    }

    /// Frames that arrived already past their deadline (the GPU or the driver was the limit, not
    /// the cap). A high count means the cap is not what is pacing the show.
    pub fn frames_late(&self) -> u64 {
        self.frames_late
    }

    /// How long to wait before starting a frame at `now`, and the instant that frame is due.
    ///
    /// Pure: the same `now` on the same state gives the same answer. `wait_for(now).0` is zero
    /// when the frame may start immediately.
    pub fn wait_for(&mut self, now: Instant) -> Duration {
        if !self.is_capped() {
            return Duration::ZERO;
        }
        let interval = self.interval;
        let deadline = match self.next {
            None => now + interval,
            Some(deadline) => deadline,
        };
        if now >= deadline {
            // Late: start now and keep the grid. If we are more than a whole interval behind
            // (a hitch, a window drag, a slow shader compile), resync instead of bursting frames
            // back to back to catch up.
            self.frames_late += 1;
            let behind = now.saturating_duration_since(deadline);
            self.next = Some(if behind >= interval {
                now + interval
            } else {
                deadline + interval
            });
            return Duration::ZERO;
        }
        self.next = Some(deadline + interval);
        self.frames_paced += 1;
        deadline.saturating_duration_since(now)
    }

    /// The loop woke up after a wait: was the frame late by more than one interval (a suspended
    /// laptop, a stalled present)? The grid resyncs, so a stall does not turn into a burst.
    pub fn resync_if_stalled(&mut self, now: Instant) {
        if let Some(deadline) = self.next
            && now > deadline + self.interval
        {
            self.next = Some(now + self.interval);
        }
    }
}

/// One frame's interval for `max_fps` (zero for no cap, which `frames_per_second` treats as 0).
fn interval_for(max_fps: u32) -> Duration {
    if max_fps == 0 {
        return Duration::ZERO;
    }
    Duration::from_secs_f64(1.0 / f64::from(max_fps))
}

/// A high-resolution wait, with no busy loop.
///
/// On Windows this is a waitable timer created with `CREATE_WAITABLE_TIMER_HIGH_RESOLUTION`
/// (Windows 10 1803+); everywhere else, and if the timer cannot be created, a plain
/// `std::thread::sleep`.
pub struct WaitTimer {
    #[cfg(windows)]
    handle: isize,
}

impl Default for WaitTimer {
    fn default() -> Self {
        WaitTimer::new()
    }
}

impl WaitTimer {
    pub fn new() -> WaitTimer {
        #[cfg(windows)]
        {
            WaitTimer {
                handle: win::create_high_resolution_timer(),
            }
        }
        #[cfg(not(windows))]
        {
            WaitTimer {}
        }
    }

    /// True when the platform's high-resolution timer is in use (a test and the startup log quote
    /// it: a fallback sleep quantises to the system tick and caps the rate lower).
    pub fn is_high_resolution(&self) -> bool {
        #[cfg(windows)]
        {
            self.handle != 0
        }
        #[cfg(not(windows))]
        {
            false
        }
    }

    /// Waits at least `duration`. Never spins: one wait, or one wait plus a re-arm if the timer
    /// fired early.
    pub fn wait(&self, duration: Duration) {
        if duration.is_zero() {
            return;
        }
        #[cfg(windows)]
        {
            if self.handle != 0 && win::wait(self.handle, duration) {
                return;
            }
        }
        std::thread::sleep(duration);
    }
}

impl Drop for WaitTimer {
    fn drop(&mut self) {
        #[cfg(windows)]
        {
            if self.handle != 0 {
                win::close(self.handle);
                self.handle = 0;
            }
        }
    }
}

// A timer handle is a kernel object; moving it between threads is fine (the crate uses one
// render thread at a time).
unsafe impl Send for WaitTimer {}

#[cfg(windows)]
mod win {
    use std::ffi::c_void;
    use std::time::Duration;

    /// `CREATE_WAITABLE_TIMER_HIGH_RESOLUTION` (winbase.h): a ~0.5 ms timer without
    /// `timeBeginPeriod`, so the system-wide timer resolution stays coarse and the laptop stays
    /// on its power budget.
    const CREATE_WAITABLE_TIMER_HIGH_RESOLUTION: u32 = 0x0000_0002;
    /// `TIMER_ALL_ACCESS` (winnt.h).
    const TIMER_ALL_ACCESS: u32 = 0x001F_0003;
    const WAIT_OBJECT_0: u32 = 0;
    /// `INFINITE` (winbase.h).
    const INFINITE: u32 = u32::MAX;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateWaitableTimerExW(
            attributes: *mut c_void,
            name: *const u16,
            flags: u32,
            desired_access: u32,
        ) -> isize;
        fn SetWaitableTimer(
            timer: isize,
            due_time: *const i64,
            period: i32,
            completion_routine: *mut c_void,
            argument: *mut c_void,
            resume: i32,
        ) -> i32;
        fn WaitForSingleObject(handle: isize, milliseconds: u32) -> u32;
        fn CloseHandle(handle: isize) -> i32;
    }

    /// A high-resolution waitable timer, or 0 when the platform refuses one.
    pub(super) fn create_high_resolution_timer() -> isize {
        let handle = unsafe {
            CreateWaitableTimerExW(
                std::ptr::null_mut(),
                std::ptr::null(),
                CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                TIMER_ALL_ACCESS,
            )
        };
        if handle == 0 || handle == -1 {
            0
        } else {
            handle
        }
    }

    /// Waits `duration` on `timer`. False when a call failed (the caller falls back to sleeping).
    pub(super) fn wait(timer: isize, duration: Duration) -> bool {
        // A relative due time is negative, in 100 ns units.
        let ticks = -(duration.as_nanos() as i64 / 100).max(1);
        let set = unsafe {
            SetWaitableTimer(
                timer,
                &ticks,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        if set == 0 {
            return false;
        }
        unsafe { WaitForSingleObject(timer, INFINITE) == WAIT_OBJECT_0 }
    }

    pub(super) fn close(timer: isize) {
        unsafe {
            CloseHandle(timer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sixty_fps_is_one_frame_each_sixteen_point_six_ms() {
        let pacer = FramePacer::new(60);
        assert!(pacer.is_capped());
        assert!((pacer.interval.as_secs_f64() * 1000.0 - 16.666).abs() < 0.01);
        assert_eq!(FramePacer::new(120).interval.as_millis(), 8);
        assert!(!FramePacer::new(0).is_capped());
    }

    /// A 120 Hz panel presents every 8.3 ms; the cap lets about every other frame through.
    #[test]
    fn a_120_hz_loop_is_halved() {
        let start = Instant::now();
        let mut pacer = FramePacer::new(60);
        let tick = Duration::from_secs_f64(1.0 / 120.0);
        let mut vblank = 0u64;
        let mut frames = 0u32;
        let mut now = start;
        // One second of a vsync-driven loop: the pacer sleeps out the part of the interval the
        // panel has not used, then the present snaps forward to the next vblank (FIFO).
        while now.saturating_duration_since(start) < Duration::from_secs(1) {
            let wait = pacer.wait_for(now);
            now += wait;
            let since = now.saturating_duration_since(start).as_secs_f64();
            vblank = vblank.max((since / tick.as_secs_f64()).ceil() as u64);
            now = start + tick * vblank as u32;
            frames += 1;
        }
        println!(
            "{vblank} vblanks at 120 Hz let {frames} frames through a 60 fps cap; paced {},
             late {}",
            pacer.frames_paced(),
            pacer.frames_late()
        );
        assert!(
            (55..=61).contains(&frames),
            "{vblank} vblanks let {frames} frames through, expected ~60"
        );
    }

    /// With no cap the pacer never asks for a wait and never reports a late frame.
    #[test]
    fn no_cap_never_waits() {
        let mut pacer = FramePacer::new(0);
        let start = Instant::now();
        for step in 0..100 {
            let now = start + Duration::from_millis(step);
            assert_eq!(pacer.wait_for(now), Duration::ZERO);
        }
        assert_eq!(pacer.frames_paced(), 0);
        assert_eq!(pacer.frames_late(), 0);
    }

    /// Frames that arrive far past their deadline resync instead of bursting: a hitch must not
    /// be followed by a run of back-to-back frames.
    #[test]
    fn a_stall_resyncs_instead_of_bursting() {
        let start = Instant::now();
        let mut pacer = FramePacer::new(60);
        // First frame: one interval from the start.
        assert_eq!(pacer.wait_for(start), Duration::from_secs_f64(1.0 / 60.0));
        // The app then stalls for a second (a driver reset, a window drag, the laptop waking up).
        let after_stall = start + Duration::from_secs(1);
        assert_eq!(pacer.wait_for(after_stall), Duration::ZERO, "start at once");
        assert_eq!(pacer.frames_late(), 1);
        // The next frames are a full interval apart again, not a burst.
        assert!(
            pacer.wait_for(after_stall) > Duration::from_millis(15),
            "the grid resynced to one interval from the stall"
        );
    }

    /// A frame whose own work overran by 2 ms keeps the grid: the *next* wait is 2 ms shorter
    /// rather than the whole show slowing down by 2 ms.
    #[test]
    fn a_slightly_late_frame_keeps_the_grid() {
        let start = Instant::now();
        let mut pacer = FramePacer::new(60);
        let interval = pacer.wait_for(start);
        let first = start + interval;
        // Frame 1 started on its deadline and took 2 ms longer than an interval to come back.
        let wait = pacer.wait_for(first + Duration::from_millis(2));
        assert!(
            wait < Duration::from_millis(15) && wait > Duration::from_millis(13),
            "expected ~14.7 ms (the grid kept, 2 ms paid back), got {wait:?}"
        );
        let second = first + Duration::from_millis(2) + wait;
        let third = second + pacer.wait_for(second);
        // Frame 3 takes 20 ms instead of nothing: it arrives past its deadline.
        let late = third + Duration::from_millis(20);
        assert_eq!(
            pacer.wait_for(late),
            Duration::ZERO,
            "late frames start at once"
        );
        assert_eq!(pacer.frames_late(), 1);
        let after = pacer.wait_for(late);
        assert!(
            after > Duration::from_millis(12) && after < Duration::from_millis(15),
            "the grid is still one interval, not reset to now: {after:?}"
        );
    }

    #[test]
    fn changing_the_cap_restarts_the_grid() {
        let start = Instant::now();
        let mut pacer = FramePacer::new(60);
        pacer.wait_for(start);
        assert!(pacer.wait_for(start) > Duration::ZERO);
        pacer.set_max_fps(30);
        assert_eq!(pacer.max_fps(), 30);
        assert!(
            pacer.wait_for(start + Duration::from_millis(20)) > Duration::from_millis(10),
            "a 30 fps cap asks for a ~33 ms grid"
        );
    }

    /// The wait itself: 12 waits of ~10 ms take about 120 ms of wall time, not the 31 ms per
    /// `Sleep(16)` the system tick would give.
    #[test]
    fn the_wait_timer_honours_a_ten_millisecond_wait() {
        let timer = WaitTimer::new();
        let start = Instant::now();
        for _ in 0..12 {
            timer.wait(Duration::from_millis(10));
        }
        let elapsed = start.elapsed();
        println!(
            "wait timer: high resolution {} | 12 x 10 ms took {:.1} ms",
            timer.is_high_resolution(),
            elapsed.as_secs_f64() * 1000.0
        );
        assert!(
            elapsed >= Duration::from_millis(120),
            "12 waits of 10 ms cannot take {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_millis(400),
            "12 waits of 10 ms took {elapsed:?}: the timer is quantised to the system tick"
        );
    }
}
