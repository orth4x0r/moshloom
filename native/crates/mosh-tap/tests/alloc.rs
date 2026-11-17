//! The audio callback must never allocate (design anti-goal: no dropouts caused by the plugin).
//!
//! This file is a test binary of its own with a single test on purpose: the harness runs tests in
//! parallel threads, and any allocation on another thread would land in the measured window.
//! With one test, the only other thread is the harness thread blocked joining this one.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use mosh_core::ipc_layout::BLOCK_MAX_SAMPLES;
use mosh_tap::control::{ControlCore, HostEvent};
use mosh_tap::host::{HostTransport, interleave_into};
use mosh_tap::sink::NullSink;
use mosh_tap::tap::TapCore;

struct Counting;

static COUNT: AtomicUsize = AtomicUsize::new(0);
static ARMED: AtomicBool = AtomicBool::new(false);

impl Counting {
    #[inline]
    fn note() {
        if ARMED.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::note();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::note();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::note();
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[test]
fn tap_and_control_process_do_not_allocate() {
    // Everything the callback may touch is allocated before the counter is armed: the cores
    // (scratch/silence buffers), the interleaving scratch and the channel slices.
    let mut tap = TapCore::new(NullSink, 48_000.0, 2);
    let mut control = ControlCore::new(NullSink, 48_000.0, 2);

    let mut left = vec![0.1f32; 256];
    let mut right = vec![0.2f32; 256];
    let mut scratch = vec![0.0f32; BLOCK_MAX_SAMPLES];
    let transport = HostTransport {
        tempo_bpm: 128.0,
        ppq: 64.5,
        bar_start_ppq: 64.0,
        sig_num: 4,
        sig_den: 4,
        playing: true,
        sample_pos: 250_000,
    };
    let events = [
        HostEvent::NoteOn {
            timing: 0,
            channel: 0,
            note: 36,
            velocity: 1.0,
        },
        HostEvent::NoteOff {
            timing: 12,
            channel: 0,
            note: 36,
        },
        HostEvent::ControlChange {
            timing: 48,
            channel: 0,
            cc: 48,
            value: 0.5,
        },
        HostEvent::ProgramChange {
            timing: 96,
            channel: 0,
            program: 3,
        },
    ];

    // Warm up: the first block may touch lazily initialised state.
    let written = interleave_into(&mut scratch, &[&mut left[..], &mut right[..]], 0, 256);
    tap.push_interleaved(Some(&transport), &scratch[..written], 256);
    control.process(Some(&transport), &events, 256);

    ARMED.store(true, Ordering::SeqCst);
    COUNT.store(0, Ordering::SeqCst);
    for _ in 0..10_000 {
        let written = interleave_into(&mut scratch, &[&mut left[..], &mut right[..]], 0, 256);
        tap.push_interleaved(Some(&transport), &scratch[..written], 256);
        control.process(Some(&transport), &events, 256);
    }
    ARMED.store(false, Ordering::SeqCst);

    let allocations = COUNT.load(Ordering::SeqCst);
    assert_eq!(
        allocations, 0,
        "{allocations} allocations in 10 000 process blocks"
    );
}
