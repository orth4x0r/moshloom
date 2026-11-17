//! The crate's Windows-specific bits, kept in one module (AGENTS.md global constraint).
//!
//! A Vulkan surface is created from the window's `HWND` on the render thread, but a
//! `RawWindowHandle` is not `Send` (its enum covers platforms whose handles are not). The window
//! thread therefore passes the Win32 handle as plain integers and the render thread rebuilds the
//! raw handle for `mosh-render::Stage`.

use winit::raw_window_handle::{RawWindowHandle, Win32WindowHandle};
use std::num::NonZeroIsize;

/// A stage window's Win32 handle, safe to move to the render thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowHandle {
    pub hwnd: isize,
    pub hinstance: Option<isize>,
}

impl WindowHandle {
    /// The Win32 handle inside a raw handle, or `None` on any other platform.
    pub fn from_raw(handle: RawWindowHandle) -> Option<WindowHandle> {
        match handle {
            RawWindowHandle::Win32(handle) => Some(WindowHandle {
                hwnd: handle.hwnd.get(),
                hinstance: handle.hinstance.map(NonZeroIsize::get),
            }),
            _ => None,
        }
    }

    /// The raw handle `mosh-render` wants, rebuilt on the render thread.
    pub fn raw(self) -> Option<RawWindowHandle> {
        let hwnd = NonZeroIsize::new(self.hwnd)?;
        let mut handle = Win32WindowHandle::new(hwnd);
        handle.hinstance = self.hinstance.and_then(NonZeroIsize::new);
        Some(RawWindowHandle::Win32(handle))
    }
}

/// `RECT` (windef.h).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawRect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

/// `MONITORINFO` (winuser.h).
#[repr(C)]
struct RawMonitorInfo {
    size: u32,
    monitor: RawRect,
    work: RawRect,
    flags: u32,
}

#[link(name = "user32")]
unsafe extern "system" {
    fn GetMonitorInfoW(monitor: isize, info: *mut RawMonitorInfo) -> i32;
}

/// A monitor's work area (the desktop without the taskbar) as `(x, y, width, height)` relative
/// to the monitor's own top-left corner, or `None` when the system will not say.
///
/// winit reports a monitor's size and position but not its work area, and a window placed
/// against the whole monitor ends up under the taskbar.
pub fn work_area(hmonitor: isize) -> Option<(i32, i32, u32, u32)> {
    let mut info = RawMonitorInfo {
        size: std::mem::size_of::<RawMonitorInfo>() as u32,
        monitor: RawRect::default(),
        work: RawRect::default(),
        flags: 0,
    };
    // `info` is a properly sized, writable MONITORINFO; a stale handle makes the call fail,
    // which is the `None` below.
    let ok = unsafe { GetMonitorInfoW(hmonitor, &mut info) };
    if ok == 0 {
        return None;
    }
    work_in_monitor(
        (info.monitor.left, info.monitor.top, info.monitor.right, info.monitor.bottom),
        (info.work.left, info.work.top, info.work.right, info.work.bottom),
    )
}

/// A work rectangle in desktop coordinates, made relative to its monitor's corner.
fn work_in_monitor(
    monitor: (i32, i32, i32, i32),
    work: (i32, i32, i32, i32),
) -> Option<(i32, i32, u32, u32)> {
    let width = work.2.checked_sub(work.0)?;
    let height = work.3.checked_sub(work.1)?;
    if width <= 0 || height <= 0 {
        return None;
    }
    Some((work.0 - monitor.0, work.1 - monitor.1, width as u32, height as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_win32_handle_survives_the_trip_to_the_render_thread() {
        let hwnd = NonZeroIsize::new(0x1234).unwrap();
        let mut raw = Win32WindowHandle::new(hwnd);
        raw.hinstance = NonZeroIsize::new(0x42);
        let handle = WindowHandle::from_raw(RawWindowHandle::Win32(raw)).expect("win32");
        assert_eq!(handle.hwnd, 0x1234);
        assert_eq!(handle.hinstance, Some(0x42));
        let rebuilt = handle.raw().expect("win32");
        match rebuilt {
            RawWindowHandle::Win32(rebuilt) => {
                assert_eq!(rebuilt.hwnd, hwnd);
                assert_eq!(rebuilt.hinstance, NonZeroIsize::new(0x42));
            }
            _ => panic!("rebuilt handle is not Win32"),
        }
        // `WindowHandle` is what crosses the thread boundary.
        fn assert_send<T: Send>() {}
        assert_send::<WindowHandle>();
    }

    #[test]
    fn a_work_area_is_relative_to_its_monitor() {
        // The laptop panel with the taskbar at the bottom.
        assert_eq!(
            work_in_monitor((0, 0, 2560, 1600), (0, 0, 2560, 1516)),
            Some((0, 0, 2560, 1516))
        );
        // A second monitor to the right, taskbar on its left edge.
        assert_eq!(
            work_in_monitor((2560, 0, 4480, 1080), (2632, 0, 4480, 1080)),
            Some((72, 0, 1848, 1080))
        );
        // A monitor to the left of the primary has negative desktop coordinates.
        assert_eq!(
            work_in_monitor((-1920, 0, 0, 1080), (-1920, 0, 0, 1040)),
            Some((0, 0, 1920, 1040))
        );
        assert_eq!(work_in_monitor((0, 0, 1920, 1080), (0, 0, 0, 0)), None);
        // A handle that is not a monitor is refused by the system, not trusted.
        assert_eq!(work_area(0), None);
    }
}
