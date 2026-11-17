//! Task 2.5 (N-tempo): the causal tempo and beat-phase tracker for loopback mode.
//!
//! The tracker sees the analyzer's onset envelope, one value per audio block, and has to recover
//! the tempo and the beat phase from it without ever looking ahead: a rolling 12 s window, no
//! transport, no playhead. The synthetic click tracks are exact, so the numbers here are the
//! tempo in BPM and the phase error in milliseconds.

use std::f32::consts::PI;
use std::path::Path;
use std::time::Instant;

use mosh_analysis::features::FeatureExtractor;
use mosh_analysis::offline;
use mosh_analysis::tempo::TempoTracker;
use mosh_analysis::{AnalysisConfig, Analyzer};
use mosh_core::{AudioBlock, EventFlags, TransportSource};

/// A per-thread allocation counter, so the other tests in this binary cannot pollute the count of
/// `push_does_not_allocate_after_warmup` (same pattern as `tests/features.rs`).
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

    pub fn count() -> usize {
        ALLOCS.with(|c| c.get())
    }
}

#[global_allocator]
static ALLOC: alloc::Counting = alloc::Counting;

const SR: u32 = 48_000;
const HOP: usize = 256;
/// Block hop in seconds: 5.333 ms, the analyzer's onset cadence.
const HOP_S: f64 = HOP as f64 / SR as f64;

/// Push `seconds` of a click track's onset envelope into the tracker: one impulse of 1.0 in the
/// block that contains each click, zero everywhere else. This is the input the analyzer gives the
/// tracker for a synthetic click track, without the spectral smearing.
fn push_click_onsets(tracker: &mut TempoTracker, bpm: f64, seconds: f64) {
    let blocks = (seconds / HOP_S) as usize;
    let period = 60.0 / bpm;
    for j in 0..blocks {
        let t0 = j as f64 * HOP_S;
        let t1 = t0 + HOP_S;
        // The first click at or after the start of this block, if it lands inside the block.
        let next = (t0 / period).ceil() * period;
        let click = next < t1 - 1e-12;
        tracker.push_onset(if click { 1.0 } else { 0.0 }, HOP as u32);
    }
}

#[test]
fn locks_on_click_tracks() {
    for bpm in [93.0, 120.0, 140.0] {
        let mut tracker = TempoTracker::new(SR);
        push_click_onsets(&mut tracker, bpm, 8.0);
        let est = tracker
            .estimate()
            .unwrap_or_else(|| panic!("{bpm} BPM click track never locked"));
        assert!(
            (est.bpm - bpm).abs() <= 0.5,
            "{bpm} BPM locked to {:.2} BPM (confidence {:.2})",
            est.bpm,
            est.confidence
        );
        // Half and double are the trap: a click train is also a click train at twice and half the
        // rate, so neither may win on the synthetic tracks where the answer is known.
        assert!(
            (est.bpm - bpm / 2.0).abs() > 0.5,
            "{bpm} BPM was reported as half ({}): half tempo must not win",
            est.bpm
        );
        assert!(
            (est.bpm - bpm * 2.0).abs() > 0.5,
            "{bpm} BPM was reported as double ({}): double tempo must not win",
            est.bpm
        );
        assert!(
            (0.0..=1.0).contains(&est.confidence) && est.confidence > 0.0,
            "confidence {} out of range",
            est.confidence
        );
        assert!(
            (0.0..1.0).contains(&est.beat_phase),
            "beat_phase {} out of 0..1",
            est.beat_phase
        );
    }
}

#[test]
fn phase_matches_clicks() {
    // 16 s at 120 BPM: lock by 8 s, then every click after that must sit on beat phase 0.
    let bpm = 120.0;
    let period = 60.0 / bpm;
    let mut tracker = TempoTracker::new(SR);
    let blocks = (16.0 / HOP_S) as usize;
    let mut worst_s = 0.0f64;
    let mut checked = 0usize;

    for j in 0..blocks {
        let t0 = j as f64 * HOP_S;
        let t1 = t0 + HOP_S;
        let next = (t0 / period).ceil() * period;
        let click = next < t1 - 1e-12;
        tracker.push_onset(if click { 1.0 } else { 0.0 }, HOP as u32);
        if !click || t0 < 8.0 {
            continue;
        }
        let est = tracker.estimate().expect("locked by 8 s");
        // Distance from the click to the nearest beat boundary, in seconds.
        let phase = est.beat_phase;
        let distance_s = phase.min(1.0 - phase) * 60.0 / est.bpm;
        worst_s = worst_s.max(distance_s);
        checked += 1;
    }

    assert!(checked >= 10, "only {checked} clicks checked");
    println!(
        "phase at {checked} clicks after 8 s: worst {:.1} ms",
        worst_s * 1000.0
    );
    assert!(
        worst_s <= 0.030,
        "worst click phase error {:.1} ms over {checked} clicks",
        worst_s * 1000.0
    );
}

#[test]
fn no_lock_on_noise() {
    // Uniform random onset values at the analyzer's cadence: no period, so no lock. 24 s is two
    // full windows, i.e. three times the 8 s the click tracks are allowed.
    let mut tracker = TempoTracker::new(SR);
    let blocks = (24.0 / HOP_S) as usize;
    let mut seed = 0x1234_5678u32;
    for _ in 0..blocks {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let v = (seed >> 8) as f32 / (1u32 << 24) as f32;
        tracker.push_onset(v, HOP as u32);
        assert!(
            tracker.estimate().is_none(),
            "uniform noise locked at {:.2} BPM",
            tracker.estimate().map(|e| e.bpm).unwrap_or(0.0)
        );
    }
    assert!(tracker.estimate().is_none());

    // The same through the analyzer, on the real onset envelope of white noise: the spectral flux
    // of noise fluctuates every block, so coverage on any grid stays near 1/period and no grid is
    // "supported" enough to lock.
    let mut a = Analyzer::new(AnalysisConfig::default());
    let mut seed = 0x9e37_79b9u32;
    let mut state = 0x2545_f491u32;
    let mut tracked = 0usize;
    for _ in 0..(20.0 / HOP_S) as usize {
        let mut samples = Vec::with_capacity(HOP);
        for _ in 0..HOP {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let white = (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
            samples.push(white * 0.3);
        }
        a.push(&AudioBlock {
            transport: None,
            sample_rate: SR,
            channels: 1,
            samples,
        });
        if a.take_frame().source == TransportSource::Tracked {
            tracked += 1;
        }
    }
    assert_eq!(
        tracked, 0,
        "the analyzer reported {tracked} tracked frames on 20 s of white noise"
    );
}

/// A short broadband click: three tones and a fast decay, so the spectral-flux onset detector sees
/// a transient without needing noise.
const CLICK_FRAMES: usize = 600;
fn click_sample(i: usize) -> f32 {
    let n = i as f32;
    let env = (-n / 150.0).exp();
    let at = |hz: f32| (2.0 * PI * hz * n / SR as f32).sin();
    env * (0.5 * at(300.0) + 0.3 * at(3000.0) + 0.2 * at(80.0))
}

fn click_track(frames: usize, bpm: f64) -> Vec<f32> {
    let mut audio = vec![0.0f32; frames];
    let step = SR as f64 * 60.0 / bpm;
    let mut k = 0usize;
    loop {
        let start = (k as f64 * step).round() as usize;
        if start + CLICK_FRAMES >= frames {
            break;
        }
        for i in 0..CLICK_FRAMES {
            audio[start + i] += click_sample(i);
        }
        k += 1;
    }
    audio
}

fn click_times(seconds: f64, bpm: f64) -> Vec<f64> {
    let period = 60.0 / bpm;
    let mut out = Vec::new();
    let mut k = 0usize;
    loop {
        let t = k as f64 * period;
        if t > seconds {
            return out;
        }
        out.push(t);
        k += 1;
    }
}

fn nearest(t: f64, times: &[f64]) -> f64 {
    times
        .iter()
        .map(|x| (x - t).abs())
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn analyzer_marks_tracked_source() {
    // Loopback: audio but no transport. The analyzer has to track the tempo itself, synthesise a
    // transport from it, and still produce beats on the grid.
    let bpm = 120.0;
    let seconds = 16.0;
    let audio = click_track(SR as usize * seconds as usize, bpm);
    let clicks = click_times(seconds, bpm);
    let mut a = Analyzer::new(AnalysisConfig::default());
    let mut beats: Vec<f64> = Vec::new();

    for (n, chunk) in audio.chunks(HOP).enumerate() {
        let block = AudioBlock {
            transport: None,
            sample_rate: SR,
            channels: 1,
            samples: chunk.to_vec(),
        };
        a.push(&block);
        let f = a.take_frame();
        let t = ((n + 1) * HOP) as f64 / SR as f64;
        if f.events.contains(EventFlags::BEAT) {
            beats.push(t);
        }
        if n >= 1 && t >= 9.0 {
            assert_eq!(
                f.source,
                TransportSource::Tracked,
                "transport source at {t:.2} s is {:?}, expected Tracked",
                f.source
            );
            assert!(f.playing, "tracked frames must report playing at {t:.2} s");
            assert!(
                (f.tempo_bpm as f64 - bpm).abs() <= 0.5,
                "frame tempo {} at {t:.2} s",
                f.tempo_bpm
            );
        }
    }

    let late: Vec<f64> = beats.iter().copied().filter(|t| *t >= 9.0).collect();
    assert!(
        late.len() >= 12,
        "expected >= 12 tracked beats after 9 s, got {}",
        late.len()
    );
    let mut worst = 0.0f64;
    for t in &late {
        let d = nearest(*t, &clicks);
        worst = worst.max(d);
        assert!(
            d <= 0.030,
            "beat at {t:.3} s is {:.1} ms from the nearest click",
            d * 1000.0
        );
    }
    // Bars follow the same grid: 4/4 at 120 BPM is one bar per 2 s, so 9..16 s holds three bars.
    println!(
        "tracked beats after 9 s: {}, worst click error {:.1} ms",
        late.len(),
        worst * 1000.0
    );
}

#[test]
fn analyzer_tracks_at_44k1() {
    // The tuning tracks are 44.1 kHz. The analyzer starts its tracker at 48 kHz, so the first block
    // exercises `set_sample_rate`: the window is re-allocated for the new hop and the tracker still
    // locks at the right tempo.
    const RATE: u32 = 44_100;
    let bpm = 120.0;
    let hop = 256usize;
    let hop_s = hop as f64 / RATE as f64;
    let frames = (8.0 / hop_s) as usize * hop;
    let step = RATE as f64 * 60.0 / bpm;
    let mut audio = vec![0.0f32; frames];
    let mut k = 0usize;
    while (k as f64 * step) as usize + CLICK_FRAMES < frames {
        let start = (k as f64 * step).round() as usize;
        for i in 0..CLICK_FRAMES {
            let n = i as f32;
            let env = (-n / 150.0).exp();
            let at = |hz: f32| (2.0 * PI * hz * n / RATE as f32).sin();
            audio[start + i] += env * (0.5 * at(300.0) + 0.3 * at(3000.0) + 0.2 * at(80.0));
        }
        k += 1;
    }

    let mut a = Analyzer::new(AnalysisConfig::default());
    let mut frame = None;
    for chunk in audio.chunks(hop) {
        a.push(&AudioBlock {
            transport: None,
            sample_rate: RATE,
            channels: 1,
            samples: chunk.to_vec(),
        });
        frame = Some(a.take_frame());
    }
    let f = frame.expect("frames");
    assert_eq!(f.source, TransportSource::Tracked);
    assert!(
        (f.tempo_bpm as f64 - bpm).abs() <= 0.5,
        "44.1 kHz click track reported {} BPM",
        f.tempo_bpm
    );
}

/// The analyzer calls [`TempoTracker::push_onset`] from `push`, which must stay allocation-free
/// after warm-up: the window buffers are allocated on the first push, and the search allocates
/// nothing (the percentile uses its scratch buffer, the candidate array is on the stack).
#[test]
fn push_does_not_allocate_after_warmup() {
    let mut tracker = TempoTracker::new(SR);
    push_click_onsets(&mut tracker, 120.0, 3.0); // warm-up: first allocation, one analysis

    let before = alloc::count();
    push_click_onsets(&mut tracker, 120.0, 20.0);
    let allocated = alloc::count() - before;
    assert_eq!(
        allocated, 0,
        "push_onset allocated {allocated} times after warm-up: the window, the smoothing scratch \
         and the percentile sort must all be preallocated"
    );
    assert!(tracker.estimate().is_some(), "the tracker must be locked");
}

/// `push` with the tracker on (loopback: no transport) stays inside the plan's per-block budget on
/// the median and p95 block. The search itself runs once per `ANALYSIS_PERIOD_S` (0.5 s), so one
/// block in ~94 carries it; the median and p95 describe the blocks that do not, and the printed max
/// is the search's own cost.
#[test]
fn push_256_frames_with_the_tracker_under_200us() {
    let audio = click_track(SR as usize * 12, 120.0);
    let mut a = Analyzer::new(AnalysisConfig::default());

    // Warm up: the FFT plan, the loudness ring and the tracker's window exist by now.
    for chunk in audio[..SR as usize].chunks(HOP) {
        let block = AudioBlock {
            transport: None,
            sample_rate: SR,
            channels: 1,
            samples: chunk.to_vec(),
        };
        a.push(&block);
        let _ = a.take_frame();
    }

    let mut micros = Vec::with_capacity(2000);
    for n in 0..2000usize {
        let start = (n * HOP) % (audio.len() - HOP);
        let chunk = &audio[start..start + HOP];
        let block = AudioBlock {
            transport: None,
            sample_rate: SR,
            channels: 1,
            samples: chunk.to_vec(),
        };
        let t = Instant::now();
        a.push(&block);
        micros.push(t.elapsed().as_nanos() as f64 / 1000.0);
        let _ = a.take_frame();
    }
    micros.sort_by(|x, y| x.partial_cmp(y).expect("no NaN timing"));
    let median = micros[micros.len() / 2];
    let p95 = micros[micros.len() * 95 / 100];
    let max = micros[micros.len() - 1];
    println!(
        "push(256 frames) with the tracker: median {median:.1} us, p95 {p95:.1} us, \
         max {max:.1} us over 2000 blocks"
    );

    // Same reasoning as `tests/features.rs`: the plan's 200 us is a release number and the debug
    // profile runs rustfft at opt-level 0 (~216 us per transform), so the debug ceiling is loose.
    let budget_us = if cfg!(debug_assertions) {
        2000.0
    } else {
        200.0
    };
    assert!(
        median < budget_us,
        "push median {median:.1} us exceeds the {budget_us:.0} us budget"
    );
    assert!(
        p95 < budget_us,
        "push p95 {p95:.1} us exceeds the {budget_us:.0} us budget"
    );
}

#[test]
#[ignore = "needs the human's tuning tracks in C:\\Users\\ilanf\\moshloom-tracks (never committed)"]
fn real_tracks_with_tempo_in_name() {
    // Four tracks whose file names carry the tempo. The report is the tracker's own estimate after
    // 90 s of audio per track; the assertion accepts the exact double or half, because 12 s of
    // onsets cannot always tell a beat from its subdivision.
    const DIR: &str = r"C:\Users\ilanf\moshloom-tracks";
    const TRACKS: [(&str, f64); 4] = [
        ("reza pedia dmin 93.wav", 93.0),
        ("biolsong 140ebmaj.wav", 140.0),
        ("la suerte v3 live 87 amin.wav", 87.0),
        ("vampira v2_edit 137bpm - metamor.wav", 137.0),
    ];
    const SECONDS: f64 = 90.0;

    let mut failures = Vec::new();
    for (name, expected) in TRACKS {
        let path = Path::new(DIR).join(name);
        let (samples, rate, channels) = offline::read_audio(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let channels = usize::from(channels.max(1));
        let frames = samples.len() / channels;
        let limit = frames.min((SECONDS * f64::from(rate)) as usize);

        let mut fx = FeatureExtractor::new(2048);
        let mut tracker = TempoTracker::new(rate);
        let mut mono = Vec::with_capacity(HOP);
        let mut estimate = None;
        let mut locked_at = None;
        let mut start = 0usize;
        while start + HOP <= limit {
            mono.clear();
            let inv = 1.0 / channels as f32;
            for f in 0..HOP {
                let mut sum = 0.0f32;
                for c in 0..channels {
                    sum += samples[(start + f) * channels + c];
                }
                mono.push(sum * inv);
            }
            let fv = fx.process(&mono, rate);
            tracker.push_onset(fv.onset, HOP as u32);
            estimate = tracker.estimate();
            if locked_at.is_none() && estimate.is_some() {
                locked_at = Some(start as f64 / f64::from(rate));
            }
            start += HOP;
        }

        match estimate {
            Some(est) => {
                let exact = (est.bpm - expected).abs() <= 1.0;
                let double = (est.bpm - expected * 2.0).abs() <= 1.0;
                let half = (est.bpm - expected / 2.0).abs() <= 1.0;
                println!(
                    "{name}: name says {expected:.0} BPM, tracker says {:.2} BPM at {SECONDS:.0} s \
                     (locked at {:.1} s, confidence {:.2}) exact={exact} double={double} half={half}",
                    est.bpm,
                    locked_at.unwrap_or(f64::NAN),
                    est.confidence
                );
                if !(exact || double || half) {
                    failures.push(format!(
                        "{name}: expected {expected:.0} BPM (or double/half), got {:.2} BPM",
                        est.bpm
                    ));
                }
            }
            None => {
                println!("{name}: name says {expected:.0} BPM, tracker never locked");
                failures.push(format!("{name}: never locked"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "tempo tracker missed {} of 4 tracks:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
