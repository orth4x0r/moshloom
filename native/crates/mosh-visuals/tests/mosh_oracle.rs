//! An independent oracle for the native datamosh (Task 2.2, V-mosh).
//!
//! The native pass has its own tests (`tests/mosh.rs`), and they can only fail in the ways its
//! author thought of. These tests answer a different question: *does it do what the web app does?*
//! The answer comes from `tests/oracle/reference.rs`, a CPU port of the web's P-frame written from
//! `gpu.js` (`WG_ME`, the P-frame branch of `WG_FB`) and `engine.c` (`decide()`), and from the
//! synthetic clips in `tests/oracle/clips.rs`.
//!
//! # The four numbers
//!
//! | case | how it is measured |
//! | --- | --- |
//! | **smear strength** | `m = 0.8`, one I-frame and 20 P-frames of a square translating 3 grid pixels per frame: the mean absolute difference between the output and the fresh picture, 0..1 |
//! | **carry-over** | a Datamosh clip change: how long the old clip's pixels survive, measured as the difference against the new clip's fresh frames at +1, +4 and +8, and the first frame under 0.05 |
//! | **leak floor** | `m = 1.0` (leak clamped at 0.004): frames until the difference against the new source is under 0.05 |
//! | **I-frame on cut** | a Cut clip change: an exact copy of the new clip's frame |
//!
//! The first four tests are the oracle alone; they pin the web's behaviour and need no GPU. The
//! rest drive `MoshPass::apply` with the same pictures and compare the summary numbers against the
//! oracle's within [`TOLERANCE`]. Pixel-exact is not the bar: the native pass samples its vector
//! field with a linear sampler where the web fetches one vector per 8x8 block, and every GPU
//! number here has been through an 8-bit read-back.
//!
//! On the current main the comparison tests are `#[ignore]`d and **fail**: the native pass uses
//! `amount` as the mix weight (so it converges to the source at 20% a frame instead of carrying the
//! picture), applies no `1 + 2.2 m` vector gain, and treats every clip change as an I-frame. Run
//! `cargo test -p mosh-visuals --test mosh_oracle -- --ignored` to see the gap; the numbers are
//! printed by each test and quoted in the task report.

mod common;
mod oracle;

use mosh_core::AnalysisFrame;
use mosh_render::{FrameRecorder, Gpu, LayerTarget, Pipeline};
use mosh_shaders::compile_wgsl;
use mosh_visuals::mosh::MoshPass;

use oracle::clips::{self, Clip};
use oracle::reference::{self, Cue, Transition, Web};
use oracle::{Frame, HEIGHT, WIDTH, mean_abs_diff};

/// How far a GPU summary number may sit from the oracle's, as a fraction of the oracle's.
///
/// ±25%: the two implementations differ in *how* they read the same field (one `textureLoad` per
/// 8x8 block against a bilinear sample per pixel), in the rounding of the vector to whole pixels,
/// and in an 8-bit read-back. The point of these tests is a factor-of-ten error — a pass that does
/// not carry the picture at all — not the last few percent.
const TOLERANCE: f32 = 0.25;

/// The difference below which a picture counts as "the new one": 0.05 of full scale, ~13/255.
const CONVERGED: f32 = 0.05;

/// The mosh amount the smear and carry-over cases run at. At 0.8 the web's leak, `(1-0.8)^5`, is
/// under its 0.004 floor, so the buffer carries almost everything: that is the smear.
const AMOUNT: f32 = 0.8;

// ---------------------------------------------------------------------------------------------
// The run scripts. The oracle and the GPU tests run the same frames through their own engine, so
// each case below has one script with two runners.
// ---------------------------------------------------------------------------------------------

/// The smear case: one I-frame plus this many P-frames of clip A, which translates 3 grid pixels
/// a frame.
const SMEAR_P_FRAMES: i32 = 20;

/// Frames of clip A before a clip change, so the buffer is full of clip A.
const WARM_FRAMES: i32 = 10;

/// How many frames the carry-over case follows the new clip for. The web's leak at `amount = 0.8`
/// is its 0.004 floor, so the old picture outlives this cap by hundreds of frames; the cap keeps
/// the test quick and the interesting numbers are the differences at +1, +4 and +8.
const CARRY_CAP: usize = 60;

/// Cap on the leak case's run, not its answer: the measured answer is a little under 200 frames
/// (the `[-0.15, 1.15]` clamp stretches the pure 0.996 decay), and this only bounds the loop.
const LEAK_CAP: usize = 700;

fn plain_cue(amount: f32) -> Cue {
    Cue {
        amount,
        entered: None,
        jump: false,
    }
}

/// What the carry-over case reports.
#[derive(Clone, Copy, Debug)]
struct Carry {
    /// Difference against the new clip's fresh frame, 1, 4 and 8 frames after the change.
    at: [f32; 3],
    /// First frame (1-based) under [`CONVERGED`], or `None` when the cap was reached first.
    under: Option<usize>,
    cap: usize,
}

/// What the leak case reports.
#[derive(Clone, Debug)]
struct Leak {
    /// Difference against the new source on the first frame after the switch.
    start: f32,
    /// First frame (1-based) under [`CONVERGED`], or `None` when the cap was reached first.
    under: Option<usize>,
    cap: usize,
    /// The decay curve: `trace[i]` is the difference on frame `i + 1`.
    trace: Vec<f32>,
}

/// The square's left edge on frame `t`: 32 + 3 t (see `clips::SHADER`).
fn square_left(t: i32) -> usize {
    (32 + 3 * t) as usize
}

/// The disagreement between two pictures, split at column `split`.
fn mass_split(out: &Frame, fresh: &Frame, split: usize) -> (f32, f32) {
    let (mut left, mut right) = (0.0f64, 0.0f64);
    for y in 0..out.height {
        for x in 0..out.width {
            let (a, b) = (out.at(x, y), fresh.at(x, y));
            let mass: f64 = (0..3).map(|c| f64::from((a[c] - b[c]).abs())).sum();
            if x < split {
                left += mass;
            } else {
                right += mass;
            }
        }
    }
    (left as f32, right as f32)
}

/// Mean absolute difference over a rectangle: where the motion is, the mosh should be at its
/// strongest, and the whole-frame metric dilutes it by the size of the picture.
fn rect_diff(out: &Frame, fresh: &Frame, x0: usize, y0: usize, width: usize, height: usize) -> f32 {
    let mut total = 0.0f64;
    let mut channels = 0usize;
    for y in y0..(y0 + height).min(out.height) {
        for x in x0..(x0 + width).min(out.width) {
            let (a, b) = (out.at(x, y), fresh.at(x, y));
            for c in 0..3 {
                total += f64::from((a[c] - b[c]).abs());
                channels += 1;
            }
        }
    }
    (total / channels.max(1) as f64) as f32
}

/// The smear case on the CPU reference: `SMEAR_P_FRAMES + 1` frames of clip A at [`AMOUNT`],
/// returned as the output and the fresh picture to measure it against.
fn oracle_smear() -> (Frame, Frame) {
    let mut web = Web::new();
    let mut out = Frame::new(WIDTH, HEIGHT);
    for t in 0..=SMEAR_P_FRAMES {
        let plan = reference::decide(&plain_cue(AMOUNT), t == 0);
        out = web.step(&clips::frame(Clip::A, t), &plan);
    }
    (out, clips::frame(Clip::A, SMEAR_P_FRAMES))
}

/// The carry-over case on the CPU reference: [`WARM_FRAMES`] frames of clip A, then a change to
/// clip B through the Datamosh transition (which sets `skipres`), then [`CARRY_CAP`] P-frames.
fn oracle_carry() -> Carry {
    let mut web = Web::new();
    for t in 0..WARM_FRAMES {
        let plan = reference::decide(&plain_cue(AMOUNT), t == 0);
        web.step(&clips::frame(Clip::A, t), &plan);
    }
    let mut diffs = Vec::with_capacity(CARRY_CAP);
    for k in 0..CARRY_CAP {
        let t = WARM_FRAMES + k as i32;
        let entered = if k == 0 {
            Some(Transition::Datamosh)
        } else {
            None
        };
        let plan = reference::decide(
            &Cue {
                amount: AMOUNT,
                entered,
                jump: false,
            },
            false,
        );
        let out = web.step(&clips::frame(Clip::B, t), &plan);
        diffs.push(mean_abs_diff(&out, &clips::frame(Clip::B, t)));
    }
    summarise_carry(&diffs)
}

fn summarise_carry(diffs: &[f32]) -> Carry {
    Carry {
        at: [diffs[0], diffs[3], diffs[7]],
        under: first_under(diffs),
        cap: diffs.len(),
    }
}

/// The leak case on the CPU reference: [`WARM_FRAMES`] still frames of clip A at `amount`, then a
/// switch to a still frame of clip B with no clip entry, so only `leak` moves the buffer.
fn oracle_leak(amount: f32) -> Leak {
    let mut web = Web::new();
    for t in 0..WARM_FRAMES {
        let plan = reference::decide(&plain_cue(amount), t == 0);
        web.step(&clips::still(Clip::A), &plan);
    }
    let source = clips::still(Clip::B);
    let mut diffs = Vec::new();
    for _ in 0..LEAK_CAP {
        let plan = reference::decide(&plain_cue(amount), false);
        let out = web.step(&source, &plan);
        diffs.push(mean_abs_diff(&out, &source));
        // The decay is geometric from here; stop as soon as it has crossed, like the GPU run.
        if diffs.last().is_some_and(|d| *d < CONVERGED) {
            break;
        }
    }
    Leak {
        start: diffs[0],
        under: first_under(&diffs),
        cap: LEAK_CAP,
        trace: diffs,
    }
}

/// The I-frame case on the CPU reference: a Cut clip change, which `decide()` turns into an
/// I-frame, so the output is the new clip's frame bit for bit.
fn oracle_cut() -> (Frame, Frame) {
    let mut web = Web::new();
    for t in 0..5 {
        let plan = reference::decide(&plain_cue(AMOUNT), t == 0);
        web.step(&clips::frame(Clip::A, t), &plan);
    }
    let plan = reference::decide(
        &Cue {
            amount: AMOUNT,
            entered: Some(Transition::Cut),
            jump: false,
        },
        false,
    );
    let fresh = clips::frame(Clip::B, 5);
    (web.step(&fresh, &plan), fresh)
}

/// First frame (1-based) at or under [`CONVERGED`].
fn first_under(diffs: &[f32]) -> Option<usize> {
    diffs.iter().position(|d| *d < CONVERGED).map(|i| i + 1)
}

// ---------------------------------------------------------------------------------------------
// The GPU side: the same scripts through `MoshPass::apply`.
// ---------------------------------------------------------------------------------------------

/// A 320x180 offscreen stage: a clip generator, the layer's `src` and `dst`, a scratch target for
/// the "fresh picture" of a frame, and the pass under test.
struct Rig<'a> {
    gpu: &'a Gpu,
    clip: Pipeline,
    src: LayerTarget,
    dst: LayerTarget,
    scratch: LayerTarget,
    mosh: MoshPass,
}

impl<'a> Rig<'a> {
    fn new(gpu: &'a Gpu) -> Rig<'a> {
        let words = compile_wgsl(clips::SHADER).expect("the clip shader compiles");
        Rig {
            gpu,
            clip: Pipeline::fullscreen(gpu, &words, 16, 0)
                .expect("clip pipeline")
                .with_name("oracle clip"),
            src: LayerTarget::new(gpu, WIDTH as u32, HEIGHT as u32),
            dst: LayerTarget::new(gpu, WIDTH as u32, HEIGHT as u32),
            scratch: LayerTarget::new(gpu, WIDTH as u32, HEIGHT as u32),
            mosh: MoshPass::new(gpu, WIDTH as u32, HEIGHT as u32).expect("mosh pass"),
        }
    }

    /// A mosh frame's `AnalysisFrame`: 60 fps, bar 1 (not a multiple of four, so no bar I-frame),
    /// no events.
    fn analysis(t: i32) -> AnalysisFrame {
        AnalysisFrame {
            time_s: f64::from(t) / 60.0,
            bar_index: 1,
            ..AnalysisFrame::default()
        }
    }

    /// Draws `clip` at frame `t` into `dst` and returns the read-back.
    fn draw(&self, clip: Clip, t: i32, target: &LayerTarget) -> Frame {
        let mut rec = FrameRecorder::new(self.gpu);
        rec.draw(&self.clip, target, &clips::uniform(clip, t), &[])
            .expect("clip pass");
        self.gpu.wait_frame(rec.submit());
        Frame::from_rgba8(&self.gpu.read_back(target), WIDTH, HEIGHT)
    }

    /// Draws `clip` at frame `t` into `src`, moshes it into `dst` with `amount`, and returns the
    /// output's read-back. `invalidate` is the facade's clip-change path (`Visuals::draw_layer`
    /// invalidates on a clip change), which makes this frame an I-frame.
    fn step(&mut self, clip: Clip, t: i32, amount: f32, invalidate: bool) -> Frame {
        if invalidate {
            self.mosh.invalidate(0);
        }
        let mut rec = FrameRecorder::new(self.gpu);
        rec.draw(&self.clip, &self.src, &clips::uniform(clip, t), &[])
            .expect("clip pass");
        self.mosh
            .apply(
                &mut rec,
                0,
                &self.src,
                &self.dst,
                amount,
                &Rig::analysis(t),
                None,
            )
            .expect("mosh apply");
        self.gpu.wait_frame(rec.submit());
        Frame::from_rgba8(&self.gpu.read_back(&self.dst), WIDTH, HEIGHT)
    }

    /// The fresh picture of a frame, drawn on its own so the mosh state is untouched.
    fn fresh(&self, clip: Clip, t: i32) -> Frame {
        self.draw(clip, t, &self.scratch)
    }

    /// The smear case, GPU side: the output and the fresh picture it is measured against.
    fn smear(&mut self) -> (Frame, Frame) {
        let mut out = Frame::new(WIDTH, HEIGHT);
        for t in 0..=SMEAR_P_FRAMES {
            out = self.step(Clip::A, t, AMOUNT, false);
        }
        (out, self.fresh(Clip::A, SMEAR_P_FRAMES))
    }

    /// The carry-over case, GPU side.
    ///
    /// The native API has no `skipres`, so the only way to express the web's Datamosh entry — keep
    /// the buffer, skip the residual for one frame — is to change the clip *without* calling
    /// `invalidate`. Calling it is the web's Cut rule and is what `gpu_cut_clip_change_copies`
    /// checks separately.
    fn carry(&mut self) -> Carry {
        for t in 0..WARM_FRAMES {
            self.step(Clip::A, t, AMOUNT, false);
        }
        let mut diffs = Vec::with_capacity(CARRY_CAP);
        for k in 0..CARRY_CAP {
            let t = WARM_FRAMES + k as i32;
            let out = self.step(Clip::B, t, AMOUNT, false);
            diffs.push(mean_abs_diff(&out, &self.fresh(Clip::B, t)));
        }
        summarise_carry(&diffs)
    }

    /// The leak case, GPU side: a still clip A, then a still clip B with no clip entry.
    fn leak(&mut self, amount: f32, frames: usize) -> Leak {
        for _ in 0..WARM_FRAMES {
            self.step(Clip::A, 0, amount, false);
        }
        // The GPU's own still of clip B, so the metric does not also measure the generator.
        let source = self.fresh(Clip::B, 0);
        let mut diffs = Vec::new();
        for _ in 0..frames {
            let out = self.step(Clip::B, 0, amount, false);
            diffs.push(mean_abs_diff(&out, &source));
            if diffs.last().is_some_and(|d| *d < CONVERGED) {
                break;
            }
        }
        Leak {
            start: diffs[0],
            under: first_under(&diffs),
            cap: frames,
            trace: diffs,
        }
    }
}

/// Prints one metric against the oracle's and says whether it is inside [`TOLERANCE`].
fn close(what: &str, gpu: f32, oracle: f32) -> bool {
    let scale = oracle.abs().max(1e-6);
    let error = (gpu - oracle).abs() / scale;
    let ok = error <= TOLERANCE;
    println!(
        "  {what}: gpu {gpu:.4}, oracle {oracle:.4}, error {:.1}% of the oracle (limit {:.0}%) {}",
        error * 100.0,
        TOLERANCE * 100.0,
        if ok { "ok" } else { "MISMATCH" }
    );
    ok
}

/// Compares two frame counts. `None` means the run hit its cap, which agrees only with another
/// `None`; two counts agree within 25% of the oracle's, or two frames.
fn close_frames(what: &str, gpu: Option<usize>, oracle: Option<usize>) -> bool {
    let ok = match (gpu, oracle) {
        (Some(gpu), Some(oracle)) => {
            let limit = ((oracle as f32) * TOLERANCE).max(2.0);
            (gpu as f32 - oracle as f32).abs() <= limit
        }
        (None, None) => true,
        _ => false,
    };
    println!(
        "  {what}: gpu {}, oracle {} {}",
        describe(gpu),
        describe(oracle),
        if ok { "ok" } else { "MISMATCH" }
    );
    ok
}

fn describe(frames: Option<usize>) -> String {
    match frames {
        Some(frames) => format!("{frames} frame(s)"),
        None => "not within its cap".to_string(),
    }
}

// ---------------------------------------------------------------------------------------------
// The oracle alone: these pin the web's behaviour and need no GPU.
// ---------------------------------------------------------------------------------------------

/// A 3-pixel-a-frame translation at `m = 0.8` leaves the picture far from the fresh frame: the
/// buffer carries the old content, and the `1 + 2.2 m` vector gain carries it *past* where the
/// motion points, so the goalposts run away from the square.
///
/// The absolute number looks small because it is a whole-frame mean: the mosh disagreement lives
/// in one 64x64 square plus its trail, about 7% of the frame. The scale to read it against is the
/// picture's own motion, one frame of which changes it by ~0.018 — the smear is worth more than
/// that, and a pass that does not carry anything lands below it.
#[test]
fn oracle_smear_strength() {
    let (out, fresh) = oracle_smear();
    let strength = mean_abs_diff(&out, &fresh);
    // The picture's own frame-to-frame change, for scale: one frame of 3-pixel motion.
    let per_frame = mean_abs_diff(
        &clips::frame(Clip::A, SMEAR_P_FRAMES - 1),
        &clips::frame(Clip::A, SMEAR_P_FRAMES),
    );
    let (left, right) = mass_split(&out, &fresh, square_left(SMEAR_P_FRAMES) + 32);
    // Where the square is now: the whole-frame mean above is diluted by the 93% of the picture the
    // square does not cover, so this is the number that says the mosh is total inside the motion.
    let inside = rect_diff(&out, &fresh, square_left(SMEAR_P_FRAMES), 48, 64, 64);
    println!(
        "oracle: smear strength after {SMEAR_P_FRAMES} P-frames at amount {AMOUNT}: mean absolute \
         difference to the fresh frame {strength:.4} (0..1, {:.1}/255); one frame of motion alone \
         changes the picture by {per_frame:.4}, so the smear is {:.1} frames of motion's worth",
        strength * 255.0,
        strength / per_frame.max(1e-6)
    );
    println!(
        "  inside the square now: {inside:.4}; direction: disagreement mass left of its centre \
         {left:.1}, right of it {right:.1} (the trail it has already passed through is the larger \
         one)"
    );
    assert!(
        strength > 1.5 * per_frame,
        "the reference barely smears: {strength:.4} against the fresh frame, one frame of motion \
         is {per_frame:.4}"
    );
    // The direction of the carry is pinned exactly, on the vector itself, by
    // `reference::tests::block_matching_finds_the_translation_and_its_sign`: a block that moved
    // 3 pixels right answers -3, which `uv + d` turns into "take the picture from the left".
}

/// A Datamosh clip change carries the old clip's pixels into the new clip: `skipres` suppresses
/// the residual for the one entry frame, and `leak = 0.004` then holds them.
#[test]
fn oracle_carry_over_after_a_datamosh_clip_change() {
    let carry = oracle_carry();
    println!(
        "oracle: Datamosh clip change at amount {AMOUNT}: difference to the new clip's fresh \
         frame {:.4} at +1 frame, {:.4} at +4, {:.4} at +8; under {CONVERGED} {} (cap {} \
         frames)",
        carry.at[0],
        carry.at[1],
        carry.at[2],
        describe(carry.under),
        carry.cap
    );
    assert!(
        carry.at[0] > 0.1,
        "the old clip is gone on the entry frame: {:.4}",
        carry.at[0]
    );
    assert!(
        carry.at[2] > 0.05,
        "the old clip is mostly gone 8 frames later: {:.4}",
        carry.at[2]
    );
    assert!(
        carry.at[2] > 0.7 * carry.at[0],
        "the carried picture decayed from {:.4} to {:.4} in 8 frames: too fast for a 0.004 leak",
        carry.at[0],
        carry.at[2]
    );
}

/// `m = 1.0` puts the leak on its 0.004 floor: the buffer still converges to the source, but at
/// 0.4% a frame, which is hundreds of frames.
///
/// Two things make the measured count larger than `ln(0.05/d0)/ln(1-leak)`: the feedback clamps at
/// `[-0.15, 1.15]`, so pixels the clip change drove out of range sit on the clamp until the mix
/// brings them back; and the metric is a whole-frame mean over a picture whose disagreement is
/// concentrated in one square. The *rate* is the leak itself, so the test pins that: over a late
/// window, with the clamps long cleared, the difference falls by exactly `(1-0.004)^40`.
#[test]
fn oracle_leak_floor() {
    let leak = oracle_leak(1.0);
    let scale = |frame: usize| {
        leak.trace
            .get(frame - 1)
            .map_or(f32::NAN, |difference| *difference)
    };
    println!(
        "oracle: amount 1.0 (leak {:.4}): difference to the new source {:.4} on frame +1, {:.4} at \
         +50, {:.4} at +100, {:.4} at +150; under {CONVERGED} {} (cap {} frames)",
        reference::LEAK_FLOOR,
        leak.start,
        scale(50),
        scale(100),
        scale(150),
        describe(leak.under),
        leak.cap
    );
    let frames = leak
        .under
        .expect("the reference must converge: a 0.004 leak is not a freeze");
    assert!(
        frames > 100,
        "the leak floor converged in {frames} frames, far too fast for 0.004 a frame"
    );
    // The rate, once the clamps have cleared: 40 frames of a 0.996 leak is a factor of 0.8518.
    let (from, to) = (110, 150);
    let measured = scale(to) / scale(from);
    let expected = (1.0 - reference::LEAK_FLOOR).powi((to - from) as i32);
    println!(
        "  decay +{from} to +{to}: {measured:.5}, the leak's own rate is {expected:.5} \
         ({:.1}% apart)",
        (measured / expected - 1.0) * 100.0
    );
    assert!(
        (measured / expected - 1.0).abs() < 0.05,
        "the decay from +{from} to +{to} is {measured:.5}, not the leak floor's {expected:.5}"
    );
}

/// A Cut clip change is an I-frame: the output is the new clip's frame exactly.
#[test]
fn oracle_cut_clip_change_copies_exactly() {
    let (out, fresh) = oracle_cut();
    let worst = oracle::max_abs_diff(&out, &fresh);
    let differing = out
        .rgb
        .iter()
        .zip(&fresh.rgb)
        .filter(|(left, right)| *left != *right)
        .count();
    println!(
        "oracle: Cut clip change: {differing} of {} channels differ from the new clip's frame, \
         worst {worst:.6}",
        out.rgb.len()
    );
    assert_eq!(differing, 0, "a Cut clip change must copy the frame");
}

// ---------------------------------------------------------------------------------------------
// GPU against oracle.
// ---------------------------------------------------------------------------------------------

/// The harness's own check: the GPU draws the same pictures the reference is fed. Without this
/// every comparison below would be comparing two things *and* two input pictures.
#[test]
fn synthetic_clips_match_on_the_gpu() {
    let gpu = common::gpu();
    let rig = Rig::new(&gpu);
    let mut worst = 0.0f32;
    for (clip, t) in [
        (Clip::A, 0),
        (Clip::A, 1),
        (Clip::A, SMEAR_P_FRAMES),
        (Clip::B, 0),
        (Clip::B, WARM_FRAMES + 8),
    ] {
        let drawn = rig.draw(clip, t, &rig.scratch);
        let difference = oracle::max_abs_diff(&drawn, &clips::frame(clip, t));
        worst = worst.max(difference);
    }
    println!("inputs: the GPU's clip frames match the CPU twin to within {worst:.4} (0..1)");
    assert!(
        worst <= 2.0 / 255.0,
        "the GPU and CPU clip generators disagree by {worst:.4}"
    );
    common::assert_no_validation(&gpu);
}

/// Smear strength, GPU against oracle.
#[test]
#[ignore = "pending V-mosh-2: the native pass mixes at `amount` and has no vector gain"]
fn gpu_smear_strength_matches_the_oracle() {
    let gpu = common::gpu();
    let mut rig = Rig::new(&gpu);
    let (oracle_out, oracle_fresh) = oracle_smear();
    let (gpu_out, gpu_fresh) = rig.smear();
    println!(
        "smear after {SMEAR_P_FRAMES} P-frames at amount {AMOUNT} (mean absolute difference to the \
         fresh frame, 0..1):"
    );
    let ok = close(
        "smear strength",
        mean_abs_diff(&gpu_out, &gpu_fresh),
        mean_abs_diff(&oracle_out, &oracle_fresh),
    );
    common::assert_no_validation(&gpu);
    assert!(ok, "the native pass does not smear like the web app");
}

/// Carry-over survival after a Datamosh clip change, GPU against oracle.
#[test]
#[ignore = "pending V-mosh-2: the native pass has no skipres and its leak is `amount`"]
fn gpu_carry_over_matches_the_oracle() {
    let gpu = common::gpu();
    let mut rig = Rig::new(&gpu);
    let oracle = oracle_carry();
    let gpu_carry = rig.carry();
    println!(
        "Datamosh clip change at amount {AMOUNT}, {CARRY_CAP} frames followed (difference to the \
         new clip's fresh frame):"
    );
    let mut ok = true;
    for (index, at) in ["+1 frame", "+4 frames", "+8 frames"].iter().enumerate() {
        ok &= close(at, gpu_carry.at[index], oracle.at[index]);
    }
    ok &= close_frames("frames under 0.05", gpu_carry.under, oracle.under);
    common::assert_no_validation(&gpu);
    assert!(
        ok,
        "the native pass does not carry the old clip like the web app"
    );
}

/// The leak floor, GPU against oracle.
#[test]
#[ignore = "pending V-mosh-2: the native pass never refreshes at amount 1.0"]
fn gpu_leak_floor_matches_the_oracle() {
    let gpu = common::gpu();
    let mut rig = Rig::new(&gpu);
    let oracle = oracle_leak(1.0);
    // Run the GPU side for as long as the oracle needed, so a converging pass has the room to
    // converge and a frozen one is measured on the same span.
    let frames = oracle.under.map_or(LEAK_CAP, |frames| frames + 5);
    let gpu_leak = rig.leak(1.0, frames);
    println!("amount 1.0, still source, {frames} frames run (difference to the new source, 0..1):");
    let ok = close(
        "difference on the first frame",
        gpu_leak.start,
        oracle.start,
    ) & close_frames("frames under 0.05", gpu_leak.under, oracle.under);
    common::assert_no_validation(&gpu);
    assert!(ok, "the native pass does not leak like the web app");
}

/// A Cut clip change is an I-frame in the web app; `Visuals::draw_layer` invalidates on a clip
/// change, so this one holds today and is the harness's end-to-end check.
#[test]
fn gpu_cut_clip_change_copies_exactly() {
    let gpu = common::gpu();
    let mut rig = Rig::new(&gpu);
    for t in 0..5 {
        rig.step(Clip::A, t, AMOUNT, false);
    }
    let out = rig.step(Clip::B, 5, AMOUNT, true);
    let fresh = rig.fresh(Clip::B, 5);
    let (oracle_out, oracle_fresh) = oracle_cut();
    let worst = oracle::max_abs_diff(&out, &fresh);
    println!(
        "Cut clip change: the output differs from the new clip's fresh frame by at most \
         {worst:.4} (0..1, {:.1}/255); the oracle's copy differs by {:.6}",
        worst * 255.0,
        oracle::max_abs_diff(&oracle_out, &oracle_fresh)
    );
    assert!(
        worst <= 2.0 / 255.0,
        "a Cut clip change did not copy the frame: worst difference {worst:.4}"
    );
    common::assert_no_validation(&gpu);
}
