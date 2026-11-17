//! The one Windows-specific call in this crate.
//!
//! `vkCreateWin32SurfaceKHR` wants the window's `HINSTANCE`. winit hands out a `RawWindowHandle`
//! with an optional `hinstance`; when it is missing the module handle is the right answer.

use std::ffi::c_void;

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetModuleHandleW(module_name: *const u16) -> *mut c_void;
}

/// The handle of this executable, for `VkWin32SurfaceCreateInfoKHR::hinstance`.
pub(crate) fn module_handle() -> isize {
    #[cfg(windows)]
    {
        // SAFETY: GetModuleHandleW(NULL) returns the handle of the calling process's module and
        // does not take ownership of anything.
        unsafe { GetModuleHandleW(std::ptr::null()) as isize }
    }
    #[cfg(not(windows))]
    {
        0
    }
}
