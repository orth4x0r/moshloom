//! Task 1.3 (N-features): causal features, short-term loudness and bar aggregation.
//!
//! Every test drives the analyzer the way the plugin does: 256-frame blocks at 48 kHz with a
//! transport snapshot on each block, and the musical position always derived from the sample
//! position (120 BPM: 24 000 frames per beat).

use std::f32::consts::PI;
use std::time::Instant;

use mosh_analysis::{AnalysisConfig, Analyzer};
use mosh_core::{AudioBlock, EventFlags, Transport};

/// A per-thread allocation counter, so the other tests in this binary cannot pollute the count
/// of `push_does_not_allocate_after_warmup`.
mod alloc {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static ALLOCS: Cell<usize> = const { Cell::new(0) };
    }

    pub struct Counting;

    impl Counting {
        fn bump() {
            let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        }
    }

    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            Self::bump();
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            Self::bump();
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            Self::bump();
            unsafe { System.realloc(ptr, layout, new_size) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    /// Allocations made on this thread since it started.
    pub fn count() -> usize {
        ALLOCS.with(|c| c.get())
    }
}

#[global_allocator]
static ALLOC: alloc::Counting = alloc::Counting;

const SR: u32 = 48_000;
const HOP: usize = 256;
const BPM: f64 = 120.0;
/// 120 BPM at 48 kHz.
const FRAMES_PER_BEAT: i64 = 24_000;

/// Ableton's transport for `sample`, on a 4/4 grid whose downbeat is ppq 0.
fn transport_for(sample: i64) -> Transport {
    let ppq = sample as f64 / SR as f64 * BPM / 60.0;
    Transport {
        tempo_bpm: BPM,
        ppq,
        bar_start_ppq: (ppq / 4.0).floor() * 4.0,
        sig_num: 4,
        sig_den: 4,
        playing: true,
        sample_pos: sample,
    }
}

fn block(samples: &[f32], sample: i64) -> AudioBlock {
    AudioBlock {
        transport: Some(transport_for(sample)),
        sample_rate: SR,
        channels: 1,
        samples: samples.to_vec(),
    }
}

fn block_at(samples: Vec<f32>, transport: Transport) -> AudioBlock {
    AudioBlock {
        transport: Some(transport),
        sample_rate: SR,
        channels: 1,
        samples,
    }
}

fn sine(frames: usize, hz: f32, amp: f32) -> Vec<f32> {
    (0..frames)
        .map(|i| amp * (2.0 * PI * hz * i as f32 / SR as f32).sin())
        .collect()
}

/// A decaying 50 Hz thump: the low end of a kick drum.
fn kick(frames: usize, amp: f32) -> Vec<f32> {
    (0..frames)
        .map(|i| {
            amp * (-(i as f32) / 1000.0).exp() * (2.0 * PI * 50.0 * i as f32 / SR as f32).sin()
        })
        .collect()
}

/// A kick on every beat, over `total` frames.
fn kick_train(total: usize, beats: usize, amp: f32) -> Vec<f32> {
    let mut audio = vec![0.0f32; total];
    for b in 0..beats {
        let start = b * FRAMES_PER_BEAT as usize;
        for (i, v) in kick(4000, amp).into_iter().enumerate() {
            if start + i < total {
                audio[start + i] = v;
            }
        }
    }
    audio
}

#[test]
fn sine_1khz_lands_in_mid_band() {
    let mut a = Analyzer::new(AnalysisConfig::default());
    let audio = sine(SR as usize * 4, 1000.0, 0.5);
    let mut frame = None;
    for (n, chunk) in audio.chunks(HOP).enumerate() {
        a.push(&block(chunk, (n * HOP) as i64));
        frame = Some(a.take_frame());
    }
    let f = frame.expect("one frame");

    assert!(
        f.bands[1] > f.bands[0] + 0.05,
        "mid must dominate low: bands {:?}",
        f.bands
    );
    assert!(
        f.bands[1] > f.bands[2] + 0.05,
        "mid must dominate high: bands {:?}",
        f.bands
    );
    for (i, v) in f.bands.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(v),
            "band {i} out of 0..1: {}",
            f.bands[i]
        );
    }

    // The log-spaced 30 Hz .. 16 kHz mel axis puts 1 kHz around band 13.
    let (peak, value) =
        f.mel.iter().enumerate().fold(
            (0usize, f32::MIN),
            |acc, (i, v)| {
                if *v > acc.1 { (i, *v) } else { acc }
            },
        );
    assert!(
        (11..=15).contains(&peak),
        "mel peak at band {peak} (value {value}), expected 1 kHz near band 13"
    );

    // A steady tone has no onsets once the analyzer is warm.
    assert!(
        f.onset < 0.5,
        "steady tone onset should be near zero, got {}",
        f.onset
    );
    assert!(
        f.loudness_db > -30.0 && f.loudness_db < 3.0,
        "0.5 amplitude 1 kHz sine should meter near -9 dBFS, got {} dB",
        f.loudness_db
    );
}

#[test]
fn kick_train_120bpm_onsets_on_beats() {
    let mut a = Analyzer::new(AnalysisConfig::default());
    let lead = FRAMES_PER_BEAT; // one beat of silence before the first kick
    let beats = 4usize;
    let total = lead as usize + beats * FRAMES_PER_BEAT as usize;
    let mut audio = vec![0.0f32; total];
    audio[lead as usize..].copy_from_slice(&kick_train(total - lead as usize, beats, 0.8));

    let mut onsets = Vec::new();
    for (n, chunk) in audio.chunks(HOP).enumerate() {
        a.push(&block(chunk, (n * HOP) as i64));
        onsets.push(a.take_frame().onset);
    }

    // The strongest local maxima of the onset envelope are the kicks. A frame produced from block
    // i covers samples [i*hop, (i+1)*hop), so "within one hop of the beat" means the peak lands in
    // the block nearest the beat or its immediate neighbour.
    let mut peaks: Vec<(usize, f32)> = (1..onsets.len() - 1)
        .filter(|&i| onsets[i] > 0.0 && onsets[i] > onsets[i - 1] && onsets[i] >= onsets[i + 1])
        .map(|i| (i, onsets[i]))
        .collect();
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("no NaN onset"));
    peaks.truncate(beats);
    assert_eq!(peaks.len(), beats, "expected {beats} onset peaks");

    let mut matched = [false; 4];
    for (i, value) in &peaks {
        let (best, dist) = (0..beats)
            .map(|b| {
                let beat_block =
                    ((lead + b as i64 * FRAMES_PER_BEAT) as f64 / HOP as f64).round() as i64;
                (b, (*i as i64 - beat_block).abs())
            })
            .fold(
                (0usize, i64::MAX),
                |acc, x| if x.1 < acc.1 { x } else { acc },
            );
        assert!(
            dist <= 1,
            "onset peak {value} in block {i} is {dist} hops from beat {best}"
        );
        assert!(!matched[best], "two onset peaks reported on beat {best}");
        matched[best] = true;
    }
    assert!(matched.iter().all(|m| *m), "every beat got an onset");
}

#[test]
fn events_accumulate_until_take() {
    let mut a = Analyzer::new(AnalysisConfig::default());
    let audio = sine(HOP, 440.0, 0.2);

    // First block: the grid starts at beat 0, which is not a crossing.
    a.push(&block(&audio, 0));
    assert!(!a.take_frame().events.contains(EventFlags::BEAT));

    // Just past ppq 1.0: a beat crossing, but not a bar line.
    let t = transport_for(FRAMES_PER_BEAT + 100);
    a.push(&block_at(audio.clone(), t));
    let f = a.take_frame();
    assert!(
        f.events.contains(EventFlags::BEAT),
        "beat crossing must set BEAT, got {:?}",
        f.events
    );
    assert!(
        !f.events.contains(EventFlags::BAR),
        "ppq 1 is not a bar line"
    );
    assert!(
        !a.take_frame().events.contains(EventFlags::BEAT),
        "events are cleared by take_frame"
    );
}

#[test]
fn bar_stats_aggregate_one_bar() {
    let mut a = Analyzer::new(AnalysisConfig::default());
    // Four beats of kicks: the bar closes when the transport reports ppq 4.
    let audio = kick_train(FRAMES_PER_BEAT as usize * 4 + HOP, 4, 0.8);
    for (n, chunk) in audio.chunks(HOP).enumerate() {
        a.push(&block(chunk, (n * HOP) as i64));
        let _ = a.take_frame();
    }
    let bars = a.drain_bars();
    assert_eq!(
        bars.len(),
        1,
        "expected exactly one closed bar, got {bars:?}"
    );
    let bar = bars[0];
    assert_eq!(bar.start_ppq, 0.0);
    assert_eq!(bar.bar_index, 0);
    assert_eq!(bar.beats, 4);
    assert!(
        bar.loudness_db > -60.0,
        "kicks are audible: {} dB",
        bar.loudness_db
    );
    // Onsets per beat, raw (the intensity model normalises it): four kicks on four beats is
    // about one, and a double-detected transient can push it a little higher.
    assert!(
        bar.onset_density > 0.5 && bar.onset_density <= 4.0,
        "onset density {} out of range",
        bar.onset_density
    );
    assert!(
        (0.0..=1.0).contains(&bar.low),
        "low {} out of 0..1",
        bar.low
    );
    assert!(
        (0.0..=1.0).contains(&bar.loudness),
        "loudness {} out of 0..1",
        bar.loudness
    );
    assert!(
        (0.0..=1.0).contains(&bar.brightness),
        "brightness {} out of 0..1",
        bar.brightness
    );
    assert!(
        a.drain_bars().is_empty(),
        "drain_bars consumes the closed bars"
    );
}

/// Push one block whose transport says `ppq` on a 4/4 grid. The audio clock and the musical clock
/// are independent, exactly like a playhead that is moved while the audio keeps flowing.
fn push_at_ppq(a: &mut Analyzer, samples: Vec<f32>, ppq: f64) {
    let mut t = transport_for(0);
    t.ppq = ppq;
    t.bar_start_ppq = (ppq / 4.0).floor() * 4.0;
    a.push(&block_at(samples, t));
    let _ = a.take_frame();
}

#[test]
fn ppq_jump_resets_bar_accumulator() {
    let mut a = Analyzer::new(AnalysisConfig::default());
    // Seven beats of kicks: unmistakable onsets, so a bar that inherited them would be obvious.
    let audio = kick_train(FRAMES_PER_BEAT as usize * 7, 7, 0.8);
    let step = HOP as f64 / FRAMES_PER_BEAT as f64; // beats per block at 120 BPM
    let blocks_per_bar = FRAMES_PER_BEAT as usize * 4 / HOP;
    let chunk = |k: usize| -> Vec<f32> {
        let at = (k * HOP).min(audio.len() - HOP);
        audio[at..at + HOP].to_vec()
    };

    // Bar 15 (ppq 60 .. 64) is loud and closes normally. The block that starts exactly on ppq 64
    // is the one that carries the bar line, so it is pushed too.
    for k in 0..=blocks_per_bar {
        push_at_ppq(&mut a, chunk(k), 60.0 + k as f64 * step);
    }
    let closed = a.drain_bars();
    assert_eq!(closed.len(), 1, "bar 15 closes at ppq 64");
    assert_eq!(closed[0].start_ppq, 60.0);
    assert!(
        closed[0].onset_density > 0.0,
        "four kicks are onsets: {}",
        closed[0].onset_density
    );

    // Half a bar more, still loud: no bar line, then Live jumps back to ppq 16 (bar 4).
    for k in 0..(blocks_per_bar / 2) {
        push_at_ppq(
            &mut a,
            chunk(blocks_per_bar + 1 + k),
            64.0 + k as f64 * step,
        );
    }
    assert!(
        a.drain_bars().is_empty(),
        "a partial bar must not be reported"
    );

    push_at_ppq(&mut a, vec![0.0f32; HOP], 16.0);
    assert!(
        a.drain_bars().is_empty(),
        "the jump back must not emit BarStats for the partial bar"
    );

    // Bar 4 (ppq 16 .. 20) is silent. Stopping the kicks dead is itself one broadband click, so
    // the bar may carry a single onset (0.25 per beat); a bar of leaked kicks would be >= 3.
    let silence = vec![0.0f32; HOP];
    let mut bars = Vec::new();
    for k in 0..=blocks_per_bar {
        push_at_ppq(&mut a, silence.clone(), 16.0 + k as f64 * step);
        bars.append(&mut a.drain_bars());
    }
    assert_eq!(bars.len(), 1, "bar 4 closes at ppq 20, got {bars:?}");
    assert_eq!(bars[0].start_ppq, 16.0);
    assert_eq!(bars[0].beats, 4);
    assert!(
        bars[0].onset_density < 1.0,
        "the pre-jump onsets leaked into the new bar: {bars:?}"
    );
}

#[test]
fn push_does_not_allocate_after_warmup() {
    let mut a = Analyzer::new(AnalysisConfig::default());
    let audio = kick_train(FRAMES_PER_BEAT as usize * 7, 7, 0.8);
    // Build the blocks up front: the measured loop must not allocate for the test's own reasons.
    let blocks: Vec<AudioBlock> = audio
        .chunks(HOP)
        .enumerate()
        .map(|(n, chunk)| block(chunk, (n * HOP) as i64))
        .collect();

    // Warm up: sample rate known, mel edges built, the 3 s loudness ring allocated.
    for b in &blocks {
        a.push(b);
        let _ = a.take_frame();
        let _ = a.drain_bars();
    }

    let before = alloc::count();
    for b in &blocks {
        a.push(b);
    }
    let allocated = alloc::count() - before;
    assert_eq!(
        allocated, 0,
        "push allocated {allocated} times after warm-up: the FFT plan and every buffer must be \
         preallocated"
    );

    // One bar closed in that pass, so a bar push did not allocate either.
    assert_eq!(a.drain_bars().len(), 1);
}

#[test]
fn the_allocation_counter_sees_allocations() {
    let before = alloc::count();
    let v: Vec<u8> = Vec::with_capacity(64);
    assert!(alloc::count() > before, "the counter is not counting");
    drop(v);
}

#[test]
fn push_256_frames_under_200us() {
    let mut a = Analyzer::new(AnalysisConfig::default());
    let audio = sine(SR as usize * 2, 1000.0, 0.3);

    // Warm up: the FFT plan, the mel edges and the loudness ring are preallocated by now.
    for (n, chunk) in audio[..SR as usize].chunks(HOP).enumerate() {
        a.push(&block(chunk, (n * HOP) as i64));
        let _ = a.take_frame();
    }

    let mut micros = Vec::with_capacity(2000);
    for n in 0..2000usize {
        let start = (n * HOP) % (audio.len() - HOP);
        let chunk = &audio[start..start + HOP];
        let b = block(chunk, (n * HOP) as i64);
        let t = Instant::now();
        a.push(&b);
        micros.push(t.elapsed().as_nanos() as f64 / 1000.0);
        let _ = a.take_frame();
    }
    micros.sort_by(|x, y| x.partial_cmp(y).expect("no NaN timing"));
    let median = micros[micros.len() / 2];
    let p95 = micros[micros.len() * 95 / 100];
    println!("push(256 frames): median {median:.1} us, p95 {p95:.1} us over 2000 blocks");

    // The plan's 200 us budget is a release number. The debug profile compiles this crate at
    // opt-level 0 and its dependencies at opt-level 1, where rustfft's 2048-point transform alone
    // costs ~216 us against ~2.1 us in release, so a debug run enforces a loose ceiling that still
    // catches a per-block allocation or a second transform. The gate runs both:
    // `cargo test -p mosh-analysis` (debug ceiling) and the release number quoted in the PR.
    let budget_us = if cfg!(debug_assertions) {
        2000.0
    } else {
        200.0
    };
    assert!(
        median < budget_us,
        "push median {median:.1} us exceeds the {budget_us:.0} us budget"
    );
}
