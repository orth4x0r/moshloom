//! The only Windows-specific module in this crate: named file mappings and the clock.
//!
//! Everything here is a thin, checked wrapper. The ring logic in `ring.rs` works on a raw
//! base pointer, so it can be tested without a mapping.

use std::ffi::OsStr;
use std::io;
use std::os::windows::ffi::OsStrExt;
use std::sync::atomic::{AtomicI64, Ordering};
use windows::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_ALL_ACCESS, FILE_MAP_READ, MEMORY_MAPPED_VIEW_ADDRESS,
    MapViewOfFile, OpenFileMappingW, PAGE_READWRITE, UnmapViewOfFile,
};
use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};

/// A named, pagefile-backed shared memory region plus one mapped view of it.
///
/// `Local\` names are per login session, which is the right scope: the plugins and the app run
/// as the same user. Dropping unmaps the view and closes the handle; the region itself lives on
/// while any other process still has it open.
pub(crate) struct Mapping {
    handle: HANDLE,
    view: MEMORY_MAPPED_VIEW_ADDRESS,
    bytes: usize,
}

// The mapping is a raw handle plus a pointer; whichever thread owns it is the only user of
// this struct, and the ring code is written against a plain base pointer.
unsafe impl Send for Mapping {}

impl Mapping {
    /// Create the region, or attach to an existing one of the same name. The caller overwrites
    /// the header afterwards, so a stale region from a crashed writer is harmless.
    pub(crate) fn create(name: &str, bytes: usize) -> io::Result<Mapping> {
        let wide = wide(name);
        let handle = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                None,
                PAGE_READWRITE,
                (bytes >> 32) as u32,
                (bytes & 0xFFFF_FFFF) as u32,
                windows::core::PCWSTR(wide.as_ptr()),
            )
        }
        .map_err(io_error)?;
        Self::map(handle, bytes, FILE_MAP_ALL_ACCESS)
    }

    /// Attach to an existing region. Fails with `ErrorKind::NotFound` when no writer is running,
    /// which the app treats as "no plugin yet" and retries.
    pub(crate) fn open(name: &str, bytes: usize) -> io::Result<Mapping> {
        let wide = wide(name);
        let handle = unsafe {
            OpenFileMappingW(FILE_MAP_READ.0, false, windows::core::PCWSTR(wide.as_ptr()))
        }
        .map_err(io_error)?;
        Self::map(handle, bytes, FILE_MAP_READ)
    }

    fn map(
        handle: HANDLE,
        bytes: usize,
        access: windows::Win32::System::Memory::FILE_MAP,
    ) -> io::Result<Mapping> {
        let view = unsafe { MapViewOfFile(handle, access, 0, 0, bytes) };
        if view.Value.is_null() {
            let err = io::Error::last_os_error();
            unsafe {
                let _ = CloseHandle(handle);
            }
            return Err(err);
        }
        Ok(Mapping {
            handle,
            view,
            bytes,
        })
    }

    pub(crate) fn base(&self) -> *mut u8 {
        self.view.Value.cast::<u8>()
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe {
            let _ = UnmapViewOfFile(self.view);
            let _ = CloseHandle(self.handle);
        }
    }
}

fn wide(name: &str) -> Vec<u16> {
    OsStr::new(name).encode_wide().chain(Some(0)).collect()
}

/// `windows` errors are HRESULTs. A Win32 failure is `HRESULT_FROM_WIN32(code)`, so the low 16
/// bits give back the error code that `std::io` already knows how to classify (`NotFound` for a
/// mapping that no writer created, `AccessDenied` for one owned by another user, and so on).
fn io_error(err: windows::core::Error) -> io::Error {
    let hr = err.code().0 as u32;
    if hr & 0xFFFF_0000 == 0x8007_0000 {
        io::Error::from_raw_os_error((hr & 0xFFFF) as i32)
    } else {
        io::Error::other(err.message())
    }
}

/// `QueryPerformanceCounter` ticks per second. Read once; it never changes while the machine runs.
static TICKS_PER_SECOND: AtomicI64 = AtomicI64::new(0);

fn ticks_per_second() -> i64 {
    let cached = TICKS_PER_SECOND.load(Ordering::Relaxed);
    if cached > 0 {
        return cached;
    }
    let mut frequency = 0i64;
    let frequency = if unsafe { QueryPerformanceFrequency(&mut frequency) }.is_ok() && frequency > 0
    {
        frequency
    } else {
        1_000_000
    };
    TICKS_PER_SECOND.store(frequency, Ordering::Relaxed);
    frequency
}

/// A machine-wide monotonic clock in nanoseconds: `QueryPerformanceCounter` ticks scaled by the
/// counter's frequency.
///
/// QPC is monotonic and shared by every process on the machine, so the heartbeat a plugin writes
/// and the age the app computes are directly comparable. No syscall, no allocation, and no
/// dependence on wall-clock time being stepped by NTP mid-set.
pub(crate) fn monotonic_ns() -> u64 {
    let mut ticks = 0i64;
    let ticks = if unsafe { QueryPerformanceCounter(&mut ticks) }.is_ok() && ticks > 0 {
        ticks as u64
    } else {
        0
    };
    let frequency = ticks_per_second() as u64;
    (ticks as u128 * 1_000_000_000 / frequency as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotonic_ns_moves_forward() {
        let a = monotonic_ns();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let b = monotonic_ns();
        assert!(b > a, "{b} is not after {a}");
        assert!(b - a >= 4_000_000, "5 ms of sleep measured {} ns", b - a);
        assert!(b - a < 5_000_000_000, "5 ms of sleep measured {} ns", b - a);
    }

    #[test]
    fn missing_mapping_is_not_found() {
        match Mapping::open("Local\\moshloom.does-not-exist.v1", 4096) {
            Ok(_) => panic!("a mapping was opened that no writer created"),
            Err(err) => assert_eq!(err.kind(), io::ErrorKind::NotFound),
        }
    }
}
