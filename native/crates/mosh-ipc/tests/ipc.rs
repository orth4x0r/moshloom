//! Task 1.1 tests: the shared-memory rings between the plugins and the app.
//!
//! Each test owns one `ShowRole`, so the tests never share a mapping and can run in parallel.

use mosh_core::{AudioBlock, MidiEvent, ShowRole, Transport};
use mosh_ipc::{IpcReader, IpcWriter};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

/// Tests share six role names, so they take turns. (Roles are the mapping names; two tests using
/// the same role at once would read each other's blocks.)
static TURN: Mutex<()> = Mutex::new(());

fn turn() -> MutexGuard<'static, ()> {
    TURN.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

// ---------------------------------------------------------------------------
// Counting global allocator, for `push_block_does_not_allocate`.
//
// The counters are thread-local: `cargo test` runs tests on several threads in
// one process, and only the allocations made by the calling thread are ours.
// ---------------------------------------------------------------------------

thread_local! {
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

struct CountingAllocator;

impl CountingAllocator {
    fn note<N>(f: impl FnOnce() -> N) -> N {
        if COUNTING.with(Cell::get) {
            ALLOCATIONS.with(|c| c.set(c.get() + 1));
        }
        f()
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        Self::note(|| unsafe { System.alloc(layout) })
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        Self::note(|| unsafe { System.alloc_zeroed(layout) })
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        Self::note(|| unsafe { System.realloc(ptr, layout, new_size) })
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn count_allocations<T>(f: impl FnOnce() -> T) -> (T, u64) {
    ALLOCATIONS.with(|c| c.set(0));
    COUNTING.with(|c| c.set(true));
    let out = f();
    COUNTING.with(|c| c.set(false));
    let n = ALLOCATIONS.with(Cell::get);
    (out, n)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn transport_at(ppq: f64) -> Transport {
    Transport {
        tempo_bpm: 93.5,
        ppq,
        bar_start_ppq: ppq - 2.5,
        sig_num: 4,
        sig_den: 4,
        playing: true,
        sample_pos: 48_000 * 3,
    }
}

/// 256 frames of stereo, every sample equal to `index`, so a torn read is visible as a mix.
fn constant_block(index: usize) -> Vec<f32> {
    vec![index as f32; 512]
}

// ---------------------------------------------------------------------------
// Tests (plan Task 1.1, Step 1)
// ---------------------------------------------------------------------------

#[test]
fn roundtrip_one_block() {
    let _guard = turn();
    let mut writer = IpcWriter::create(ShowRole::Master, 48_000, 2).expect("create");

    let t = transport_at(6.5);
    let samples: Vec<f32> = (0..512).map(|i| i as f32 * 0.001 - 0.25).collect();
    writer.push_block(Some(&t), &samples, 256);

    let events = [
        MidiEvent {
            ppq: 6.5,
            status: 0x90,
            data1: 36,
            data2: 100,
        },
        MidiEvent {
            ppq: 6.75,
            status: 0xB0,
            data1: 48,
            data2: 64,
        },
    ];
    for e in &events {
        writer.push_midi(e);
    }

    let mut reader = IpcReader::open(ShowRole::Master).expect("open");
    let (mut blocks, mut midi) = (Vec::new(), Vec::new());
    let stats = reader.poll(&mut blocks, &mut midi);

    assert_eq!(stats.new_blocks, 1, "one block pushed, one block read");
    assert_eq!(stats.dropped, 0);
    assert!(stats.writer_alive, "the writer just wrote a block");

    let block: &AudioBlock = &blocks[0];
    assert_eq!(block.transport, Some(t));
    assert_eq!(block.sample_rate, 48_000);
    assert_eq!(block.channels, 2);
    assert_eq!(block.frames(), 256);
    assert_eq!(block.samples, samples);

    assert_eq!(midi, events);
}

#[test]
fn overwrite_counts_drops() {
    let _guard = turn();
    let mut writer = IpcWriter::create(ShowRole::Drums, 44_100, 2).expect("create");
    let mut samples = constant_block(0);
    for index in 0..300usize {
        samples.fill(index as f32);
        writer.push_block(Some(&transport_at(index as f64)), &samples, 256);
    }

    let mut reader = IpcReader::open(ShowRole::Drums).expect("open");
    let (mut blocks, mut midi) = (Vec::new(), Vec::new());
    let stats = reader.poll(&mut blocks, &mut midi);

    assert_eq!(stats.new_blocks, 256, "the ring holds 256 blocks");
    assert_eq!(stats.dropped, 44, "300 pushed into 256 slots, 44 lost");
    assert_eq!(blocks.len(), 256);
    for (offset, block) in blocks.iter().enumerate() {
        let expected = (44 + offset) as f32;
        let t = block.transport.expect("transport");
        assert_eq!(t.ppq as f32, expected, "block {offset} out of order");
        assert!(
            block.samples.iter().all(|s| *s == expected),
            "block {offset} does not carry its own index"
        );
    }
}

#[test]
fn torn_read_is_skipped() {
    let _guard = turn();
    let mut writer = IpcWriter::create(ShowRole::Bass, 48_000, 2).expect("create");
    writer.push_block(None, &constant_block(0), 256);

    let mut reader = IpcReader::open(ShowRole::Bass).expect("open");
    let (mut blocks, mut midi) = (Vec::new(), Vec::new());
    reader.poll(&mut blocks, &mut midi);
    blocks.clear();

    // Hammer the ring from a second thread while this thread reads: every block that
    // comes back must be one whole write, never a splice of two.
    let stop = Arc::new(AtomicBool::new(false));
    let writer_thread = {
        let stop = Arc::clone(&stop);
        thread::spawn(move || {
            let mut samples = constant_block(0);
            let mut written = 0usize;
            while !stop.load(Ordering::Relaxed) {
                written += 1;
                samples.fill((written % 100_000) as f32);
                writer.push_block(None, &samples, 256);
            }
            written
        })
    };

    let deadline = Instant::now() + Duration::from_millis(200);
    let mut polls = 0u64;
    let mut checked = 0u64;
    while Instant::now() < deadline {
        blocks.clear();
        reader.poll(&mut blocks, &mut midi);
        polls += 1;
        for block in &blocks {
            checked += 1;
            assert_eq!(block.frames(), 256, "a torn read changed the frame count");
            let first = block.samples[0];
            assert!(
                block.samples.iter().all(|s| *s == first),
                "torn block: mixed samples in one block"
            );
        }
    }
    stop.store(true, Ordering::Relaxed);
    let written = writer_thread.join().expect("writer thread");

    assert!(checked > 0, "no block arrived in 200 ms ({polls} polls)");
    eprintln!("torn_read_is_skipped: {written} writes, {polls} polls, {checked} blocks checked");
}

#[test]
fn reader_attaches_after_writer_and_survives_writer_exit() {
    let _guard = turn();
    let mut writer = IpcWriter::create(ShowRole::Vocal, 48_000, 1).expect("create");
    writer.push_block(Some(&transport_at(4.0)), &vec![0.25f32; 128], 128);
    thread::sleep(Duration::from_millis(20));

    let mut reader = IpcReader::open(ShowRole::Vocal).expect("a reader may attach late");
    let (mut blocks, mut midi) = (Vec::new(), Vec::new());
    let stats = reader.poll(&mut blocks, &mut midi);
    assert!(stats.writer_alive, "the writer is alive");
    assert_eq!(stats.new_blocks, 1);
    assert_eq!(blocks[0].samples, vec![0.25f32; 128]);

    drop(writer);
    thread::sleep(Duration::from_millis(600));

    let stats = reader.poll(&mut blocks, &mut midi);
    assert!(!stats.writer_alive, "the heartbeat is older than 500 ms");
    assert_eq!(stats.new_blocks, 0, "a dead writer adds no blocks");

    // Polling a region whose writer is gone must not panic, now or later.
    let stats = reader.poll(&mut blocks, &mut midi);
    assert!(!stats.writer_alive);
    assert_eq!(stats.new_blocks, 0);
}

#[test]
fn writer_restart_rebinds_reader() {
    let _guard = turn();
    let mut writer = IpcWriter::create(ShowRole::Master, 48_000, 2).expect("create");
    let mut samples = constant_block(0);
    for index in 0..300usize {
        samples.fill(index as f32);
        writer.push_block(Some(&transport_at(index as f64)), &samples, 256);
    }

    let mut reader = IpcReader::open(ShowRole::Master).expect("open");
    let (mut blocks, mut midi) = (Vec::new(), Vec::new());
    let stats = reader.poll(&mut blocks, &mut midi);
    assert_eq!(stats.new_blocks, 256);
    assert_eq!(stats.dropped, 44);

    // The plugin is unloaded and loaded again mid-set: a new writer re-creates the region, so
    // its counters start at zero. The reader must follow the new ring, not wait for the old
    // count to be reached again.
    drop(writer);
    let mut writer = IpcWriter::create(ShowRole::Master, 48_000, 2).expect("re-create");
    for index in 0..3usize {
        samples.fill(1000.0 + index as f32);
        writer.push_block(None, &samples, 256);
    }

    blocks.clear();
    let stats = reader.poll(&mut blocks, &mut midi);
    assert_eq!(
        stats.new_blocks, 3,
        "a restarted writer is read from its own ring"
    );
    assert_eq!(stats.dropped, 0, "a restart is not a drop");
    assert_eq!(blocks[0].samples[0], 1000.0);
    assert_eq!(blocks[2].samples[0], 1002.0);
    assert!(stats.writer_alive);
}

#[test]
fn push_block_does_not_allocate() {
    let _guard = turn();
    let mut writer = IpcWriter::create(ShowRole::Other, 48_000, 2).expect("create");
    let mut samples = constant_block(1);
    let t = transport_at(1.0);

    let (_, allocations) = count_allocations(|| {
        for index in 0..10_000usize {
            samples[0] = index as f32;
            writer.push_block(Some(&t), black_box(&samples), 256);
        }
    });

    assert_eq!(
        allocations, 0,
        "push_block allocated {allocations} times in 10 000 calls"
    );
}

#[test]
fn push_block_256_frames_under_5us() {
    const RUNS: usize = 10_000;
    let _guard = turn();
    let mut writer = IpcWriter::create(ShowRole::Control, 48_000, 2).expect("create");
    let samples = constant_block(2);
    let t = transport_at(2.0);

    // Warm the mapping up first (first touch of the 2 MB region faults pages in).
    for _ in 0..RING_WARMUP {
        writer.push_block(Some(&t), &samples, 256);
    }

    let mut times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let start = Instant::now();
        writer.push_block(Some(&t), black_box(&samples), 256);
        times.push(start.elapsed());
    }
    times.sort_unstable();
    let median = times[RUNS / 2];
    let p99 = times[RUNS * 99 / 100];
    eprintln!("push_block(256 frames stereo): median {median:?}, p99 {p99:?} over {RUNS} pushes");

    assert!(
        median < Duration::from_micros(5),
        "median {median:?} is over the 5 us budget for a 256-frame stereo push"
    );
}

const RING_WARMUP: usize = 512;
