//! An independent oracle for the native datamosh (Task 2.2, V-mosh).
//!
//! The native pass has its own tests (`tests/mosh.rs`), and they can only fail in the ways its
//! author thought of. These tests answer a different question: *does it do what the web app does?*
//! The answer comes from `tests/oracle/reference.rs`, a CPU port of the web's P-frame written from
//! `gpu.js` (`WG_ME`, the P-frame branch of `WG_FB`) and `engine.c` (`decide()`), and from the
//! synthetic clips in `tests/oracle/clips.rs`.
//!
//! # The numbers
//!
//! | case | how it is measured | driven through |
//! | --- | --- | --- |
//! | **smear strength** | `m = 0.8`, one I-frame and 20 P-frames of a square translating 3 grid pixels per frame: the mean absolute difference between the output and the fresh picture, 0..1, whole frame and inside the square | `MoshPass::apply` |
//! | **carry-over** | a `Datamosh` clip change: the difference against the new clip's fresh frames at +1, +4 and +8, and the first frame under 0.05 | `Visuals::draw_layer` |
//! | **leak floor** | `m = 1.0` (leak clamped at 0.004): frames until the difference against the new source is under 0.05, and the decay rate | `MoshPass::apply` |
//! | **I-frame on cut** | a `Cut` clip change: an exact copy of the new clip's frame | `Visuals::draw_layer` |
//! | **matcher** | a 1-pixel checker moving 1 pixel a frame into the bottom block row, at `m = 0.9` where `coarse` is live | `MoshPass::apply` |
//!
//! The clip changes go through the facade because that is where the web's `entered` rule lives: the
//! web keeps the mosh buffer on a `Datamosh` entry and sets `skipres` for that frame
//! (`engine.c:337`), while a `Cut` is an I-frame. `MoshPass::apply` alone cannot tell the two
//! apart — it has no `skipres` parameter and no transition — so a comparison that skips the facade
//! would be testing something no caller can do.
//!
//! # The gate is 320x180 only
//!
//! Every number here is measured on a 320x180 layer, where the web's fixed 320x180 analysis grid
//! (`GS`, gpu.js:296) and the canvas size `F.res` coincide, so neither shader interpolates
//! anything. At another internal size the web rounds the carried displacement to *canvas* pixels
//! (`d = round(mv * F.res) / F.res`, gpu.js:327) and samples the buffer bilinearly where the field
//! is fractional; none of that is exercised here. A port that is right at 320x180 and wrong
//! somewhere else will pass this file.
//!
//! # What each summary number catches
//!
//! The point of a gating test is to fail on a wrong port. The mutations below were applied to
//! `tests/oracle/reference.rs` one at a time; every one is caught, by the test named.
//!
//! | bug in the reference | caught by |
//! | --- | --- |
//! | no residual (`c = carried` only) | `reference::tests::feedback_adds_the_residual_unless_skipres`, and the leak case's first frame |
//! | `leak = 1 - m` (no fifth power, no floor) | `reference::tests::decide_carries_the_web_apps_numbers`, the carry case's decay |
//! | no `skipres` on a clip entry | `reference::tests::decide_carries_the_web_apps_numbers`, the carry case's +1 frame |
//! | no `1 + 2.2 m` vector gain | `reference::tests::block_matching_finds_the_translation_and_its_sign` |
//! | the warp taking `uv - d` | `reference::tests::feedback_carries_the_picture_from_the_left` |
//! | no `bc & -2` when coarse | `reference::tests::feedback_pairs_the_blocks_when_coarse` |
//! | the match one frame stale (`N` against `N-2`) | `reference::tests::block_matching_compares_this_frame_with_the_last_one` |
//! | the I-frame clamp inside the P-frame branch | `reference::tests::an_iframe_stores_the_frame_unchanged` |
//!
//! The one the suite cannot see is the native pass's per-pixel *linear* sampling of the 40x23
//! vector field where the web does one `textureLoad` per block: that is a documented divergence in
//! `mosh_feedback.wgsl`, and the ±25% tolerance absorbs it.
//!
//! # The numbers the oracle measures
//!
//! `-- --nocapture` prints them; they are repeated here so a reader has them without a GPU.
//!
//! * smear, `m = 0.8`, 20 P-frames of clip A: **0.0458** of full scale (11.7/255) over the whole
//!   frame, **0.4776** (121.8/255) inside the moving square — the mosh is total where the motion
//!   is, and the whole-frame mean is diluted by the 93% of the picture that is still background;
//! * Datamosh clip change, `m = 0.8`: **0.2954** at +1 frame, **0.2916** at +4, **0.2919** at +8,
//!   never under 0.05 within 60 frames;
//! * leak floor, `m = 1.0`: under 0.05 at **frame 72**, and the +32..+72 decay is **0.871**
//!   against the leak's own 0.8519 — 2.2% apart;
//! * Cut clip change: **0 of 172800** channels differ, an exact copy;
//! * matcher case, clip C at `m = 0.9`: **0.0135** whole frame, **0.0534** over the swept
//!   rectangle.
//!
//! The comparison tests against the native pass are `#[ignore = "pending V-mosh-2"]`: on the
//! current main they fail, with the numbers in the task report.

mod common;
mod oracle;

use mosh_core::{
    ActiveTransition, AnalysisFrame, BlendMode, ClipParams, EffectKind, EffectSlot, LayerState,
    TransitionKind,
};
use mosh_render::{FrameRecorder, Gpu, LayerTarget, Pipeline};
use mosh_shaders::compile_wgsl;
use mosh_visuals::mosh::MoshPass;
use mosh_visuals::{LayerSlot, Stage, Visuals};

use oracle::clips::{self, Clip};
use oracle::reference::{self, Cue, Transition, Web};
use oracle::{Frame, HEIGHT, WIDTH, max_abs_diff, mean_abs_diff};

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

/// The matcher case's amount: above 0.85, so `coarse` (`bc & -2`) is live.
const MATCHER_AMOUNT: f32 = 0.9;

// ---------------------------------------------------------------------------------------------
// The run scripts. The oracle and the GPU tests run the same frames through their own engine, so
// each case below has one script with two runners.
// ---------------------------------------------------------------------------------------------

/// The smear case: one I-frame plus this many P-frames of clip A, which translates 3 grid pixels
/// a frame.
const SMEAR_P_FRAMES: i32 = 20;

/// The matcher case: one I-frame plus this many P-frames of clip C, which translates 1 pixel a
/// frame and reaches the bottom block row.
const MATCHER_P_FRAMES: i32 = 12;

/// Frames of clip A before a clip change, so the buffer is full of clip A.
const WARM_FRAMES: i32 = 10;

/// How many frames the carry-over case follows the new clip for. The web's leak at `amount = 0.8`
/// is its 0.004 floor, so the old picture outlives this cap; the cap keeps the test quick and the
/// interesting numbers are the differences at +1, +4 and +8.
const CARRY_CAP: usize = 60;

/// Cap on the leak case's run, not its answer: the measured answer is around 70 frames, and this
/// only bounds the loop for a port that never converges.
const LEAK_CAP: usize = 700;

fn plain_cue(amount: f32) -> Cue {
    Cue {
        amount,
        entered: None,
        jump: false,
    }
}

/// What the carry-over case reports.
#[derive(Clone, Debug)]
struct Carry {
    /// Difference against the new clip's fresh frame, 1, 4 and 8 frames after the change.
    at: [f32; 3],
    /// First frame (1-based) under [`CONVERGED`], or `None` when the cap was reached first.
    under: Option<usize>,
    /// The last frame's difference.
    last: f32,
    cap: usize,
}

/// What the leak case reports.
#[derive(Clone, Debug)]
struct Leak {
    /// Difference against the new source on the first frame after the switch.
    start: f32,
    /// First frame (1-based) under [`CONVERGED`], or `None` when the cap was reached first.
    under: Option<usize>,
    /// The last frame's difference.
    last: f32,
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

/// Mean absolute difference over clip C's swept rectangle — where the matcher case's mosh
/// disagreement lives. The rectangle starts at x = 48, is 96 wide, and moves 1 pixel a frame for
/// `MATCHER_P_FRAMES`, so the swept area is 108 x 104.
fn matcher_rect(out: &Frame, fresh: &Frame) -> f32 {
    rect_diff(out, fresh, 48, 76, 96 + MATCHER_P_FRAMES as usize, 104)
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

/// The matcher case on the CPU reference: clip C's 1-pixel checker at `m = 0.9`, so `coarse` is on
/// and the carried picture runs at `round(1 * 2.98) = 3` pixels a frame against the motion's 1.
fn oracle_matcher() -> (Frame, Frame) {
    let mut web = Web::new();
    let mut out = Frame::new(WIDTH, HEIGHT);
    for t in 0..=MATCHER_P_FRAMES {
        let plan = reference::decide(&plain_cue(MATCHER_AMOUNT), t == 0);
        out = web.step(&clips::frame(Clip::C, t), &plan);
    }
    (out, clips::frame(Clip::C, MATCHER_P_FRAMES))
}

/// The carry-over case on the CPU reference: [`WARM_FRAMES`] frames of clip A, then a change to
/// clip B through the Datamosh transition (which sets `skipres`), then [`CARRY_CAP`] P-frames.
///
/// Fed the 8-bit pictures the facade's overlay path carries (`clips::quantized`), so the oracle
/// and the GPU are looking at the same bytes.
fn oracle_carry() -> Carry {
    let mut web = Web::new();
    for t in 0..WARM_FRAMES {
        let plan = reference::decide(&plain_cue(AMOUNT), t == 0);
        web.step(&clips::quantized(Clip::A, t), &plan);
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
        let out = web.step(&clips::quantized(Clip::B, t), &plan);
        diffs.push(mean_abs_diff(&out, &clips::quantized(Clip::B, t)));
    }
    summarise_carry(&diffs)
}

fn summarise_carry(diffs: &[f32]) -> Carry {
    Carry {
        at: [diffs[0], diffs[3], diffs[7]],
        under: first_under(diffs),
        last: *diffs.last().expect("the cap is not zero"),
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
        last: *diffs.last().expect("the cap is not zero"),
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
// The `MoshPass::apply` side: the synthetic clips drawn by the WGSL generator.
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
    /// output's read-back.
    fn step(&mut self, clip: Clip, t: i32, amount: f32) -> Frame {
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

    /// `p_frames + 1` frames of `clip` (one I-frame, then P-frames), as the output and the fresh
    /// picture it is measured against.
    fn case(&mut self, clip: Clip, p_frames: i32, amount: f32) -> (Frame, Frame) {
        let mut out = Frame::new(WIDTH, HEIGHT);
        for t in 0..=p_frames {
            out = self.step(clip, t, amount);
        }
        (out, self.fresh(clip, p_frames))
    }

    /// The leak case, GPU side: a still clip A, then a still clip B with no clip entry.
    fn leak(&mut self, amount: f32, frames: usize) -> Leak {
        for _ in 0..WARM_FRAMES {
            self.step(Clip::A, 0, amount);
        }
        // The GPU's own still of clip B, so the metric does not also measure the generator.
        let source = self.fresh(Clip::B, 0);
        let mut diffs = Vec::new();
        for _ in 0..frames {
            let out = self.step(Clip::B, 0, amount);
            diffs.push(mean_abs_diff(&out, &source));
            if diffs.last().is_some_and(|d| *d < CONVERGED) {
                break;
            }
        }
        Leak {
            start: diffs[0],
            under: first_under(&diffs),
            last: *diffs.last().expect("at least one frame"),
            cap: frames,
            trace: diffs,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The `Visuals::draw_layer` side: the clip changes, which only the facade can express.
// ---------------------------------------------------------------------------------------------

/// The same picture with its rows reversed.
///
/// A layer target is V-mirrored relative to the stage (`tests/orientation.rs`), so a layer
/// read-back is upside down relative to the source picture the oracle's reference works in. This
/// turns one into the other; the mosh itself is equivariant under the flip, so an output compared
/// after this is the output the oracle computes from the source-order picture.
fn flip_rows(frame: Frame) -> Frame {
    let (width, height) = (frame.width, frame.height);
    Frame::from_fn(width, height, |x, y| {
        let at = ((height - 1 - y) * width + x) * 3;
        [frame.rgb[at], frame.rgb[at + 1], frame.rgb[at + 2]]
    })
}

/// One layer through the real facade, fed with the synthetic clips.
///
/// `Visuals::draw_layer`'s only way to put arbitrary pixels into a layer is the overlay path, so
/// these cases upload each frame as 8-bit BGRA (`clips::bgra`) and the reference is fed
/// `clips::quantized` — the same pictures. `the_overlay_path_delivers_the_oracle_pictures` pins
/// that the upload and the cover pass come out as the pictures the oracle expects, through
/// [`flip_rows`] (a layer target is V-mirrored relative to the stage, and the oracle's pictures
/// are in source order).
struct Facade<'a> {
    gpu: &'a Gpu,
    visuals: Visuals,
    slot: LayerSlot,
}

impl<'a> Facade<'a> {
    fn new(gpu: &'a Gpu) -> Facade<'a> {
        Facade {
            gpu,
            visuals: Visuals::new(gpu, WIDTH as u32, HEIGHT as u32).expect("visuals"),
            slot: LayerSlot::new(gpu, 0, WIDTH as u32, HEIGHT as u32),
        }
    }

    /// One frame: hand `clip` at frame `t` to the overlay path, draw the layer, and return the
    /// layer's output. `clip_id` is what `LayerState::clip` says; a change of it is a clip change.
    ///
    /// `transition` is the clip's transition kind on the frame the clip changes, or `None`. The
    /// `Datamosh` and `Cut` cases pass it with `progress = 1.0`, so `TransitionPass::apply` has no
    /// blend left to draw: the web's `Datamosh` entry draws no blend either (`engine.c` sets `xf`
    /// for `T_XFADE` only), and its carry-over *is* the transition. The stage guard below makes
    /// sure the number read back is the mosh output and not a blend.
    fn step(
        &mut self,
        clip: Clip,
        t: i32,
        clip_id: u32,
        transition: Option<TransitionKind>,
        amount: f32,
    ) -> Frame {
        self.visuals
            .overlay
            .set_frame(
                self.gpu,
                0,
                WIDTH as u32,
                HEIGHT as u32,
                &clips::bgra(clip, t),
            )
            .expect("overlay frame");
        let state = LayerState {
            layer: 0,
            clip: Some(clip_id),
            visualizer: Some("overlay".to_string()),
            params: ClipParams::default(),
            effects: vec![EffectSlot {
                kind: EffectKind::Mosh,
                amount,
                param: 0.0,
            }],
            transition: transition.map(|kind| ActiveTransition {
                kind,
                progress: 1.0,
                from: Some(1),
            }),
            blend: BlendMode::Normal,
            opacity: 1.0,
            seed: 0,
        };
        let frame = AnalysisFrame {
            time_s: f64::from(t) / 60.0,
            bar_index: 1,
            ..AnalysisFrame::default()
        };
        let mut rec = FrameRecorder::new(self.gpu);
        self.visuals
            .draw_layer(&mut rec, &mut self.slot, &state, &frame, 0.0)
            .expect("draw layer");
        self.gpu.wait_frame(rec.submit());
        if amount > 0.0 {
            // The mosh output, or (for a Cut) the transition pass's copy of it: either is the mosh
            // stage's picture. A blend, an effect or the scene would mean this comparison is
            // reading the wrong stage.
            let expected = match transition {
                Some(TransitionKind::Cut) => &[Stage::Mosh, Stage::Transition][..],
                _ => &[Stage::Mosh][..],
            };
            assert!(
                expected.contains(&self.slot.output_stage),
                "the layer's output is {:?}, not the mosh stage: a blend or an effect on top would \
                 hide what this comparison is about",
                self.slot.output_stage
            );
        }
        // A layer target is V-mirrored relative to the stage: its row 0 is the audience's bottom
        // row, while the oracle's pictures are in source order (row 0 is the top). `flip_rows`
        // turns the layer back into that order. The mosh is NOT exactly equivariant under a row
        // flip (180 rows / 8 = 22.5 blocks, so flipped blocks straddle two; the `lo.y + 8 > 184`
        // bound only binds the last block row): the input and Cut checks stay exact, and the
        // metric checks stay within tolerance because clips A-C have no vertical structure within
        // ~4 rows of the top or bottom edge. A clip that does may need the frame fed flipped.
        flip_rows(Frame::from_rgba8(
            &self.gpu.read_back(self.slot.output()),
            WIDTH,
            HEIGHT,
        ))
    }

    /// The carry-over case, GPU side: [`WARM_FRAMES`] frames of clip A, then a `Datamosh` clip
    /// change to clip B, then [`CARRY_CAP`] P-frames.
    fn carry(&mut self) -> Carry {
        for t in 0..WARM_FRAMES {
            self.step(Clip::A, t, 1, None, AMOUNT);
        }
        let mut diffs = Vec::with_capacity(CARRY_CAP);
        for k in 0..CARRY_CAP {
            let t = WARM_FRAMES + k as i32;
            let transition = if k == 0 {
                Some(TransitionKind::Datamosh)
            } else {
                None
            };
            let out = self.step(Clip::B, t, 2, transition, AMOUNT);
            diffs.push(mean_abs_diff(&out, &clips::quantized(Clip::B, t)));
        }
        summarise_carry(&diffs)
    }
}

// ---------------------------------------------------------------------------------------------
// Comparison helpers.
// ---------------------------------------------------------------------------------------------

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

/// Prints a metric against the number measured from the reference and says whether it is within
/// 30% of it. Used where the point is that the reference's own answer is pinned exactly enough to
/// catch a mutation, not that it matches a GPU.
fn band(what: &str, value: f32, expected: f32) -> bool {
    let error = (value - expected).abs() / expected.abs().max(1e-6);
    let ok = error <= 0.30;
    println!(
        "  {what}: {value:.4}, pinned {expected:.4}, {:.1}% apart (limit 30%) {}",
        error * 100.0,
        if ok { "ok" } else { "MISMATCH" }
    );
    ok
}

/// Compares two frame counts. `None` means the run hit its cap, which agrees with another `None`
/// only when the pictures had also converged to within [`TOLERANCE`] of each other: a pass that
/// freezes forever reaches no frame count, and must not read as agreement.
fn close_frames(
    what: &str,
    gpu: Option<usize>,
    oracle: Option<usize>,
    gpu_last: f32,
    oracle_last: f32,
) -> bool {
    let counts = match (gpu, oracle) {
        (Some(gpu), Some(oracle)) => {
            let limit = ((oracle as f32) * TOLERANCE).max(2.0);
            (gpu as f32 - oracle as f32).abs() <= limit
        }
        (None, None) => close("difference at the cap", gpu_last, oracle_last),
        _ => false,
    };
    println!(
        "  {what}: gpu {}, oracle {}{}",
        describe(gpu),
        describe(oracle),
        if counts { "" } else { " MISMATCH" }
    );
    counts
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
/// motion points.
///
/// The whole-frame number looks small because it is a mean: the mosh disagreement lives in one
/// 64x64 square plus the trail it has passed through, about 7% of the picture. `inside the square`
/// is the number that says the mosh is total where the motion is.
#[test]
fn oracle_smear_strength() {
    let (out, fresh) = oracle_smear();
    let strength = mean_abs_diff(&out, &fresh);
    // The picture's own frame-to-frame change, for scale: one frame of 3-pixel motion.
    let per_frame = mean_abs_diff(
        &clips::frame(Clip::A, SMEAR_P_FRAMES - 1),
        &clips::frame(Clip::A, SMEAR_P_FRAMES),
    );
    let inside = rect_diff(&out, &fresh, square_left(SMEAR_P_FRAMES), 48, 64, 64);
    let (left, right) = mass_split(&out, &fresh, square_left(SMEAR_P_FRAMES) + 32);
    println!(
        "oracle: smear strength after {SMEAR_P_FRAMES} P-frames at amount {AMOUNT}: mean absolute \
         difference to the fresh frame {strength:.4} (0..1, {:.1}/255), inside the square \
         {inside:.4} ({:.1}/255); one frame of motion alone changes the picture by {:.4}, so the \
         smear is {:.1} frames of motion's worth",
        strength * 255.0,
        inside * 255.0,
        per_frame,
        strength / per_frame.max(1e-6)
    );
    println!(
        "  direction: disagreement mass left of the square's centre {left:.1}, right of it \
         {right:.1} (the trail the square has already passed through is the larger one)"
    );
    assert!(
        inside > 0.25,
        "the reference does not smear inside the moving square: {inside:.4} against the fresh \
         picture"
    );
    assert!(
        strength > 1.5 * per_frame,
        "the reference barely smears: {strength:.4} against the fresh frame, one frame of motion \
         is {per_frame:.4}"
    );
    // The direction of the carry is pinned exactly, on the vector itself, by
    // `reference::tests::block_matching_finds_the_translation_and_its_sign` and
    // `reference::tests::feedback_carries_the_picture_from_the_left`.
}

/// The matcher on the picture it is worst at: 1-pixel detail, 1 pixel a frame, down to the last
/// block row, at `m = 0.9` where `coarse` pairs the blocks.
///
/// A 1-pixel checker cannot tell a one-pixel horizontal move from a one-pixel vertical one, so the
/// web's answer here is decided by its search order: `dy` outermost from -6, `dx` from -6, first
/// strictly-cheaper offset wins (`gpu.js:304-310`). An interior block answers `(0, -1)`; the last
/// block row answers `(-1, 0)`, because the rows below y = 179 are clamp-to-edge copies and break
/// the vertical symmetry. A port that scans in another order, or that drops the web's 184 bound,
/// lands on another vector and smears this clip in another direction.
#[test]
fn oracle_matcher_case() {
    let (out, fresh) = oracle_matcher();
    let strength = mean_abs_diff(&out, &fresh);
    let inside = matcher_rect(&out, &fresh);
    let plan = reference::decide(&plain_cue(MATCHER_AMOUNT), false);
    println!(
        "oracle: clip C ({MATCHER_P_FRAMES} P-frames, amount {MATCHER_AMOUNT}, coarse {}): mean \
         absolute difference {strength:.4} (0..1, {:.1}/255), over the swept rectangle \
         {inside:.4} ({:.1}/255)",
        plan.coarse,
        strength * 255.0,
        inside * 255.0
    );
    assert!(plan.coarse, "amount 0.9 must pair the blocks");
    // The number is pinned because it is what a mutation moves: the matcher's answer, the
    // `1 + 2.2 m` gain and the block pairing all feed it, and a 1-pixel checker turns any of them
    // being wrong into a different smear. (Clip C's own frame-to-frame change is no scale here: a
    // 1-pixel checker shift rewrites every pixel, 0.156 of the picture.)
    assert!(
        band("inside the swept rectangle", inside, 0.0534),
        "the reference's clip C answer moved: the matcher, the gain or the block pairing changed"
    );
    // The vector for the last block row, which only the web's 184 bound lets search at all.
    let mut web = Web::new();
    for t in 0..=MATCHER_P_FRAMES {
        let plan = reference::decide(&plain_cue(MATCHER_AMOUNT), t == 0);
        web.step(&clips::frame(Clip::C, t), &plan);
    }
    let bottom = web.field()[22 * reference::MOTION_WIDTH + 8];
    let middle = web.field()[10 * reference::MOTION_WIDTH + 8];
    println!(
        "  interior block vector: {middle:?}, last block row vector: {bottom:?} (uv), coarse {}",
        plan.coarse
    );
    assert_ne!(
        bottom,
        [0.0, 0.0],
        "the reference found no motion in the last block row"
    );
    assert!(
        bottom[1] <= 0.0,
        "the reference searched the last block row downwards past the 184 bound: {bottom:?}"
    );
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
    // `skipres` is the whole difference between a Datamosh entry and the residual's own cleanup:
    // with the residual on, the new clip's residual cancels the old picture down to ~0.07 on the
    // entry frame.
    assert!(
        carry.at[0] > 0.2,
        "the old clip is gone on the entry frame: {:.4} — this is what dropping skipres looks \
         like",
        carry.at[0]
    );
    assert!(
        carry.at[2] > 0.2,
        "the old clip is mostly gone 8 frames later: {:.4}",
        carry.at[2]
    );
    // A leak of `1 - m` (or a floor of nothing) would drain 0.8 a frame instead of holding.
    assert!(
        carry.at[2] > 0.7 * carry.at[0],
        "the carried picture decayed from {:.4} to {:.4} in 8 frames: too fast for a 0.004 leak",
        carry.at[0],
        carry.at[2]
    );
}

/// `m = 1.0` puts the leak on its 0.004 floor: the buffer still converges to the source, but at
/// 0.4% a frame, which is tens of frames.
///
/// The measured count is larger than `ln(0.05/d0)/ln(1-leak)` because the feedback clamps at
/// `[-0.15, 1.15]`: pixels the clip change drove out of range sit on the clamp until the mix brings
/// them back. The *rate* is the leak itself, so the test pins that, over a late window where the
/// clamps have long cleared.
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
         +20, {:.4} at +40, {:.4} at +60; under {CONVERGED} {} (cap {} frames)",
        reference::LEAK_FLOOR,
        leak.start,
        scale(20),
        scale(40),
        scale(60),
        describe(leak.under),
        leak.cap
    );
    // The residual cancels the old clip on the entry frame. With the residual dropped, the entry
    // frame would still show the whole old picture: ~0.31.
    assert!(
        leak.start < 0.15,
        "the first frame after the switch is still {:.4} away from the new source: the residual \
         is missing",
        leak.start
    );
    let frames = leak
        .under
        .expect("the reference must converge: a 0.004 leak is not a freeze");
    assert!(
        frames > 40,
        "the leak floor converged in {frames} frames, far too fast for 0.004 a frame"
    );
    assert!(
        frames < 200,
        "{frames} frames: the clamp-heavy slow path of a missing residual, not a 0.004 leak"
    );
    // The rate, once the clamps have cleared: 40 frames of a 0.996 leak is a factor of 0.8518.
    let (from, to) = (frames.saturating_sub(40), frames);
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
    let worst = max_abs_diff(&out, &fresh);
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

/// The `MoshPass::apply` rig's own check: the GPU draws the same pictures the reference is fed.
/// Without this every comparison below would be comparing two things *and* two input pictures.
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
        (Clip::C, 0),
        (Clip::C, MATCHER_P_FRAMES),
    ] {
        let drawn = rig.draw(clip, t, &rig.scratch);
        let difference = max_abs_diff(&drawn, &clips::frame(clip, t));
        worst = worst.max(difference);
    }
    println!("inputs: the GPU's clip frames match the CPU twin to within {worst:.5} (0..1)");
    assert!(
        worst <= 1.0 / 255.0,
        "the GPU and CPU clip generators disagree by {worst:.5}"
    );
    common::assert_no_validation(&gpu);
}

/// The facade rig's own check: the overlay path delivers the pictures the oracle is fed, so the
/// clip-change comparisons below are not measuring the upload, the cover pass or a row flip.
#[test]
fn the_overlay_path_delivers_the_oracle_pictures() {
    let gpu = common::gpu();
    let mut facade = Facade::new(&gpu);
    let mut worst = 0.0f32;
    // No mosh (`amount` 0), so the layer's output is the scene the overlay path drew.
    for (clip, t) in [
        (Clip::A, 0),
        (Clip::A, SMEAR_P_FRAMES),
        (Clip::B, WARM_FRAMES),
        (Clip::C, MATCHER_P_FRAMES),
    ] {
        let drawn = facade.step(clip, t, 1, None, 0.0);
        let difference = max_abs_diff(&drawn, &clips::quantized(clip, t));
        worst = worst.max(difference);
    }
    println!("inputs: the overlay path matches the 8-bit oracle pictures to within {worst:.5}");
    assert!(
        worst <= 1.0 / 255.0,
        "the overlay path does not deliver the pictures the oracle is fed ({worst:.5} apart)"
    );
    common::assert_no_validation(&gpu);
}

/// Smear strength, GPU against oracle.
#[test]
fn gpu_smear_strength_matches_the_oracle() {
    let gpu = common::gpu();
    let mut rig = Rig::new(&gpu);
    let (oracle_out, oracle_fresh) = oracle_smear();
    let (gpu_out, gpu_fresh) = rig.case(Clip::A, SMEAR_P_FRAMES, AMOUNT);
    println!(
        "smear after {SMEAR_P_FRAMES} P-frames at amount {AMOUNT} (mean absolute difference to the \
         fresh frame, 0..1):"
    );
    let strength = close(
        "whole frame",
        mean_abs_diff(&gpu_out, &gpu_fresh),
        mean_abs_diff(&oracle_out, &oracle_fresh),
    );
    // The number that matters: a pass that carries at 20% a frame looks almost right over the
    // whole frame and wrong inside the motion.
    let inside = close(
        "inside the square",
        rect_diff(
            &gpu_out,
            &gpu_fresh,
            square_left(SMEAR_P_FRAMES),
            48,
            64,
            64,
        ),
        rect_diff(
            &oracle_out,
            &oracle_fresh,
            square_left(SMEAR_P_FRAMES),
            48,
            64,
            64,
        ),
    );
    common::assert_no_validation(&gpu);
    assert!(
        strength && inside,
        "the native pass does not smear like the web app"
    );
}

/// The matcher case, GPU against oracle: clip C's 1-pixel detail, 1 pixel a frame, into the bottom
/// block row, at `m = 0.9` where `coarse` is live.
#[test]
fn gpu_matcher_case_matches_the_oracle() {
    let gpu = common::gpu();
    let mut rig = Rig::new(&gpu);
    let (oracle_out, oracle_fresh) = oracle_matcher();
    let (gpu_out, gpu_fresh) = rig.case(Clip::C, MATCHER_P_FRAMES, MATCHER_AMOUNT);
    println!(
        "clip C, {MATCHER_P_FRAMES} P-frames at amount {MATCHER_AMOUNT} (mean absolute difference \
         to the fresh frame, 0..1):"
    );
    let strength = close(
        "whole frame",
        mean_abs_diff(&gpu_out, &gpu_fresh),
        mean_abs_diff(&oracle_out, &oracle_fresh),
    );
    let inside = close(
        "inside the rectangle",
        matcher_rect(&gpu_out, &gpu_fresh),
        matcher_rect(&oracle_out, &oracle_fresh),
    );
    common::assert_no_validation(&gpu);
    assert!(
        strength && inside,
        "the native matcher does not carry clip C like the web app"
    );
}

/// Carry-over survival after a Datamosh clip change, GPU against oracle, through the facade.
#[test]
fn gpu_carry_over_matches_the_oracle() {
    let gpu = common::gpu();
    let mut facade = Facade::new(&gpu);
    let oracle = oracle_carry();
    let gpu_carry = facade.carry();
    println!(
        "Datamosh clip change at amount {AMOUNT}, {CARRY_CAP} frames followed (difference to the \
         new clip's fresh frame):"
    );
    let mut ok = true;
    for (index, at) in ["+1 frame", "+4 frames", "+8 frames"].iter().enumerate() {
        ok &= close(at, gpu_carry.at[index], oracle.at[index]);
    }
    ok &= close_frames(
        "frames under 0.05",
        gpu_carry.under,
        oracle.under,
        gpu_carry.last,
        oracle.last,
    );
    common::assert_no_validation(&gpu);
    assert!(
        ok,
        "the native facade does not carry the old clip like the web app"
    );
}

/// The leak floor, GPU against oracle.
#[test]
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
    ) & close_frames(
        "frames under 0.05",
        gpu_leak.under,
        oracle.under,
        gpu_leak.last,
        oracle.last,
    );
    common::assert_no_validation(&gpu);
    assert!(ok, "the native pass does not leak like the web app");
}

/// A Cut clip change is an I-frame in the web app, and the facade invalidates the mosh on a clip
/// change, so this one holds today and is the harness's end-to-end check.
#[test]
fn gpu_cut_clip_change_copies_exactly() {
    let gpu = common::gpu();
    let mut facade = Facade::new(&gpu);
    for t in 0..5 {
        facade.step(Clip::A, t, 1, None, AMOUNT);
    }
    let out = facade.step(Clip::B, 5, 2, Some(TransitionKind::Cut), AMOUNT);
    let fresh = clips::quantized(Clip::B, 5);
    let (oracle_out, oracle_fresh) = oracle_cut();
    let worst = max_abs_diff(&out, &fresh);
    println!(
        "Cut clip change through the facade: the output differs from the new clip's 8-bit frame by \
         at most {worst:.5} (0..1, {:.2}/255); the oracle's copy differs by {:.6}",
        worst * 255.0,
        max_abs_diff(&oracle_out, &oracle_fresh)
    );
    assert!(
        worst <= 2.0 / 255.0,
        "a Cut clip change did not copy the frame: worst difference {worst:.5}"
    );
    common::assert_no_validation(&gpu);
}
