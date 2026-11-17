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
}
