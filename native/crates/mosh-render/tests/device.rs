//! `Gpu::new` under concurrency.
//!
//! Every test binary of the workspace creates Vulkan instances and devices while this one does,
//! and on the reference machine an enumeration's count can change between its count query and its
//! fetch: the loader multiplexes an integrated Radeon, a discrete Radeon and a virtual display
//! driver. `VK_INCOMPLETE` is that, and it is a success code. A creation path that turns it into
//! an error fails intermittently (`mosh-visuals`' `scenes.rs:15` panicked with
//! `Vulkan(INCOMPLETE)` about one workspace run in five). This test creates devices in parallel so
//! that a regression shows up here, in seconds, instead of as a flake in somebody else's binary.

use std::thread;

use mosh_render::{Gpu, GpuPreference};

/// How many threads create devices at the same time, and how many each creates. Measured on the
/// reference machine: 19 s alone, 19-20 s inside `cargo test --workspace`.
const THREADS: usize = 8;
const PER_THREAD: usize = 20;

/// Eight threads, twenty `Gpu::new` each, one device live per thread at a time. Every creation
/// must succeed: `Vulkan(INCOMPLETE)` is the failure this guards against.
#[test]
fn concurrent_gpu_creation_never_reports_incomplete() {
    let handles: Vec<_> = (0..THREADS)
        .map(|thread| {
            thread::spawn(move || {
                for creation in 0..PER_THREAD {
                    let gpu = Gpu::new(GpuPreference::Integrated).unwrap_or_else(|error| {
                        panic!("thread {thread}, creation {creation}: {error} ({error:?})")
                    });
                    assert!(
                        !gpu.adapter_name().is_empty(),
                        "thread {thread}, creation {creation}: the device has no adapter name"
                    );
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().expect("a creation thread panicked");
    }
}
