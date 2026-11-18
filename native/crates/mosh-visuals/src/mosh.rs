//! Datamosh: block-matching motion estimation and the feedback pass, per layer.
//! Owner: V-mosh (Task 2.2). The public signatures are frozen by the coordinator.
//!
//! Ported from `gpu.js`: [`MoshPass::apply`] runs the web app's motion pass (`WG_ME`: 8x8 blocks
//! on a fixed 320x180 luminance grid, a +-6 grid-pixel search against the previous frame, the same
//! SAD cost with its 3-per-pixel motion penalty) and then the P-frame half of its feedback pass
//! (`WG_FB`), as written:
//!
//! ```text
//! bc = clamp(floor(uv * (40, 22.5)), 0, (39, 22))            one vector per 8x8 block
//! d  = round(motion * gain * mvScale * res / grid)            whole output pixels
//! c  = buf(pixel - d + flow)                                  nearest, mirror-repeat
//! c  = c + src(pixel) - prv(pixel)                            the carried picture + residual
//! c  = mix(c, src(pixel), leak)                               leak = (1 - amount)^5, >= 0.004
//! ```
//!
//! with `gain = 1 + 2.2 * amount`. The residual is the difference between this frame and last
//! frame *before* the mosh, so a moving picture leaves its motion on the buffer instead of being
//! faded towards the new frame; the leak is a fifth power, so at amount 0.8 it is at its 0.004
//! floor and the picture survives hundreds of frames. A crossfade (`mix(fresh, carried, amount)`)
//! is not a datamosh: it dies in a handful of frames and never accumulates the residual.
//!
//! The block lookup is the web's, `textureLoad(mvT, bc)` with `bc & -2` above amount 0.85, and so
//! is the sampling: the displacement is rounded to whole output pixels and read through the web's
//! nearest, mirror-repeat sampler (`G.nm`), which is a `textureLoad` at a mirrored texel index
//! here. The vector's own clamp in `WG_ME` (`clamp(bd / GS * gain, +-56/640, +-56/360)`) is the
//! one web line not ported: it cannot bind while the search is +-6 grid pixels and the gain is at
//! most 3.2, so it is dead code rather than a behaviour.
//!
//! The motion search itself is the web's search *at a quarter of its samples*: the pass is
//! `mosh_motion_fast` (V-budget), which compares a 4x4 subsample of each block instead of all 64
//! pixels, because the full search measured 1.6 ms per layer per frame at 1080p and is the frame
//! budget's biggest single pass. The offsets, the +-6 range, the 3-per-grid-pixel penalty, the
//! search's 184-row bound (`gpu.js:307`) and the early exit are unchanged; what it can lose is
//! discrimination between offsets that differ only in a block's odd rows or columns.
//! `tests/mosh.rs` (`motion_subsample_matches_the_full_search`) reports how far that moves the
//! vectors: 236 of 240 identical on its moving-bar frames.
//!
//! The shaders are `mosh_motion` (the full search, kept as the reference the subsampled one is
//! checked against), `mosh_motion_fast`, `mosh_grid`, `mosh_feedback` and `mosh_copy`.
//!
//! # Per-layer state
//!
//! State is keyed by the `layer` index `apply` is given and created lazily, at the internal size
//! passed to [`MoshPass::new`]:
//!
//! * `grid`: the analysis-grid copy of the picture, two `LayerTarget`s alternating, so one holds
//!   this frame and the other last frame. Block matching runs on the web's fixed 320x180 grid
//!   whatever the layer's internal size, so a 320x180 copy is all it needs: one cheap pass per
//!   layer per frame (`mosh_grid.wgsl`) and then cache-resident reads, instead of sampling a
//!   4-8 MB picture at grid coordinates and keeping a full-size previous frame (V-budget: the
//!   motion pass measured 4.1 ms per frame at 1080p, and the previous-frame copy 0.46 ms).
//! * `motion`: 40x23 vectors, one per 8x8 block of the analysis grid (the web's `mvT`), so it
//!   costs the same at every internal size.
//! * `prev_src`: last frame's source picture at the internal size, one image, written by a
//!   `mosh_copy` pass after the feedback pass. The residual needs `src - prv` at the *layer's*
//!   resolution: the 320x180 grid pair cannot stand in for it, because a static picture would
//!   then have a non-zero residual wherever the grid copy lost detail, and every edge would ring.
//!   Cost: one full-size copy pass per layer per frame, measured in `--release` on the reference
//!   machine at **0.221 ms at 960x540 and 0.378 ms at 1280x720** (the whole mosh apply is 0.640
//!   and 1.079 ms), against the ~6 ms 1080p frame budget. That is the price of the residual: a
//!   second source target (a ping-pong `LayerSlot::scene`, which would let the *renderer* hand
//!   over last frame's picture with no pass at all) is the obvious follow-up, but `LayerSlot`'s
//!   public fields are frozen and it would move the copy out of this pass's lane.
//! * the feedback buffer is `dst`'s own previous image: the mosh stage is the only writer of
//!   `dst`, so its readable image *is* last frame's mosh output and no extra full-size target is
//!   allocated. `valid` tracks whether that image may be used.
//!
//! # `amount`
//!
//! `amount` is the mosh strength, the same knob as the web app's `mosh`: `leak` and `gain` are
//! derived from it in [`feedback_uniform`], and `amount > 0.85` makes the block lookup coarse
//! (`bc & -2`, every other block) exactly as `decide()` does. `amount <= 0.001` is the web app's
//! "no mosh" branch (`if (m <= 0.001) iframe = 1`): nothing is written and the buffer is
//! forgotten, so the next non-zero amount starts on an I-frame. `leak` is floored at 0.004, so
//! `amount = 1` cannot freeze a layer forever.
//!
//! Freeze/bloom (`mvScale *= 1.02` per frame before a Melt) is *not* ported: nothing in the native
//! render plan asks for it yet (`LayerState.transition` is the *entering* transition, not the
//! upcoming one). `mv_scale` is a uniform field and is always 1.0 until then.
//!
//! # I-frames
//!
//! There is no per-frame plan here; `engine.c`'s `decide()` belongs to the web app. A layer takes
//! an I-frame (a copy of the new frame into the output, no motion pass, no residual) when:
//!
//! * it has no valid buffer (first use, or the first use after `amount` was 0), or
//! * the frame's `time_s` is not within `IFRAME_GAP_S` of the previous frame's: playback jumped
//!   backwards, or a hitch left a hole. `decide()` refreshes after eight 30 fps frames for the
//!   same reason; at 60 fps this is 15 frames.
//!
//! There is deliberately **no** I-frame on a bar boundary, on `EventFlags::DROP`, or every four
//! bars: `decide()` has no such rule, and an I-frame refreshes the picture, which is the opposite
//! of the mosh. The old four-bar rule was one of the reasons the native datamosh read as a ghost
//! trail that kept being reset.
//!
//! A *clip change* follows `decide()`'s `entered` rule exactly (`engine.c:337`):
//!
//! ```text
//! if (lastK<0 || tr==T_CUT) iframe=1;
//! else if (tr==T_STROBE) { iframe=1; flash=1; }
//! else if (tr!=T_XFADE) P.skipres=1;
//! ```
//!
//! So `Visuals::draw_layer` calls [`MoshPass::invalidate`] for a Cut or a Strobe entry (an I-frame)
//! and for the first clip, which has no transition, and [`MoshPass::carry_over`] for a Datamosh or
//! a Melt entry (`skipres`: keep the buffer, skip the residual for the entry frame, let the new
//! clip's motion push the old picture - the classic datamosh cut). A Crossfade entry touches
//! nothing: it is an ordinary P-frame, residual and all, because the two pictures are blended by
//! the transition instead.
//!
//! # `extra_flow`
//!
//! The fluid visualizer's velocity field (RG, pixels per frame at the internal size) is added to
//! the sample offset, so a layer above a fluid layer is stirred by it. The web app adds its own
//! velocity to the offset (`WG_FB`: `d += vel ...; sample(buf, uv + d)`), which moves the carried
//! picture *against* the field; this does the same, with `1.0` where the web uses `0.03` because
//! this field is already a per-frame displacement rather than uv per second. Without a fluid layer
//! the slot is bound to a 1x1 zero texture, which costs one texel per fragment.

use mosh_core::AnalysisFrame;
use mosh_render::{FrameRecorder, Gpu, LayerTarget, Pipeline, RenderError};
use mosh_shaders::spirv;

/// The web app's analysis grid (`GS` in `gpu.js`): block matching always runs here, whatever the
/// layer's internal size, so a mosh layer costs the same at 720p as at 360p.
const GRID: [f32; 2] = [320.0, 180.0];

/// One motion vector per [`MOTION_BLOCK`] grid pixels: the web's 40x23 `mvT`. The last row is half
/// covered, exactly as in `gpu.js` (180 / 8 = 22.5).
const MOTION_BLOCK: u32 = 8;
const MOTION_WIDTH: u32 = GRID[0] as u32 / MOTION_BLOCK;
const MOTION_HEIGHT: u32 = (GRID[1] as u32).div_ceil(MOTION_BLOCK);

/// A playback gap this long (or any backwards jump) refreshes a layer's buffer.
const IFRAME_GAP_S: f64 = 0.25;

/// The web app's `if (m <= 0.001f) iframe = 1`: below this, a layer has no mosh.
const MIN_AMOUNT: f32 = 0.001;

/// The floor on the leak, so `amount = 1` cannot freeze a layer forever (`decide()` clamps it the
/// same way).
const MIN_LEAK: f32 = 0.004;

/// `decide()`'s `gain = 1 + 2.2 * m`: the carried picture is displaced slightly ahead of the
/// motion each frame, which is the overshoot that stretches the smear.
const GAIN_PER_AMOUNT: f32 = 2.2;

/// `decide()`'s `coarse = m > 0.85`: at full strength the vector lookup rounds to every other
/// block, which makes the smear's blocks 16 grid pixels wide.
const COARSE_ABOVE: f32 = 0.85;

/// Frames of residual skipped after a Datamosh clip change. The web app sets `skipres` for the
/// entry frame only (`entered` is true on exactly one frame), so this is one.
const SKIPRES_FRAMES: u32 = 1;

const MOTION_UNIFORM_BYTES: usize = 16;
const FEEDBACK_UNIFORM_BYTES: usize = 40;
const COPY_UNIFORM_BYTES: usize = 16;
const GRID_UNIFORM_BYTES: usize = 16;

pub struct MoshPass {
    width: u32,
    height: u32,
    copy: Pipeline,
    grid: Pipeline,
    motion: Pipeline,
    feedback: Pipeline,
    /// Bound in the feedback pass's fifth texture slot when no fluid layer is running.
    no_flow: LayerTarget,
    layers: Vec<Option<LayerState>>,
}

/// One layer's feedback state. Created on the first `apply` for that layer.
struct LayerState {
    /// The analysis-grid picture, this frame and last frame (see `grid_front`).
    grids: [LayerTarget; 2],
    /// Which of `grids` holds last frame's picture.
    grid_front: usize,
    /// 40x23 displacement field, RG = grid pixels per frame, forward.
    motion: LayerTarget,
    /// Last frame's source picture, at the layer's internal size, for the residual.
    prev_src: LayerTarget,
    last_time_s: f64,
    /// True once the buffer in the caller's `dst` holds this layer's last output.
    valid: bool,
    /// Frames that must still run without the residual (`skipres`), set by `carry_over`.
    skipres_frames: u32,
}

impl LayerState {
    fn new(gpu: &Gpu, width: u32, height: u32) -> LayerState {
        LayerState {
            grids: [
                LayerTarget::new(gpu, GRID[0] as u32, GRID[1] as u32),
                LayerTarget::new(gpu, GRID[0] as u32, GRID[1] as u32),
            ],
            grid_front: 0,
            motion: LayerTarget::new(gpu, MOTION_WIDTH, MOTION_HEIGHT),
            prev_src: LayerTarget::new_single(gpu, width, height),
            last_time_s: f64::NAN,
            valid: false,
            skipres_frames: 0,
        }
    }
}

/// The I-frame rule documented on the module: no valid buffer, or a discontinuity in playback
/// time. Nothing musical refreshes a layer; `decide()` never does either.
fn iframe_due(valid: bool, last_time_s: f64, frame: &AnalysisFrame) -> bool {
    if !valid {
        return true;
    }
    let gap = frame.time_s - last_time_s;
    !(0.0..=IFRAME_GAP_S).contains(&gap)
}


impl MoshPass {
    /// `width`/`height`: the layers' internal size. Per-layer state (feedback, motion field) is
    /// keyed by the layer index passed to `apply`.
    pub fn new(gpu: &Gpu, width: u32, height: u32) -> Result<MoshPass, RenderError> {
        assert!(width > 0 && height > 0, "mosh pass needs a non-zero size");
        // `mosh_motion_fast` is `mosh_motion` with a 4x4 block subsample (see the shader): the
        // exhaustive +-6 search stays, the sample count drops 4x.
        let motion_shader = spirv("mosh_motion_fast").ok_or(RenderError::EmptyShader)?;
        let feedback_shader = spirv("mosh_feedback").ok_or(RenderError::EmptyShader)?;
        let copy_shader = spirv("mosh_copy").ok_or(RenderError::EmptyShader)?;
        let grid_shader = spirv("mosh_grid").ok_or(RenderError::EmptyShader)?;
        Ok(MoshPass {
            width,
            height,
            copy: Pipeline::fullscreen(gpu, copy_shader, COPY_UNIFORM_BYTES, 1)?
                .with_name("mosh copy"),
            grid: Pipeline::fullscreen(gpu, grid_shader, GRID_UNIFORM_BYTES, 1)?
                .with_name("mosh grid"),
            motion: Pipeline::fullscreen(gpu, motion_shader, MOTION_UNIFORM_BYTES, 2)?
                .with_name("mosh motion"),
            feedback: Pipeline::fullscreen(gpu, feedback_shader, FEEDBACK_UNIFORM_BYTES, 5)?
                .with_name("mosh feedback"),
            no_flow: LayerTarget::new(gpu, 1, 1),
            layers: Vec::new(),
        })
    }

    /// Mosh `src` into `dst` with `amount` 0..1. `extra_flow` is the fluid visualizer's velocity
    /// field (RG = velocity in pixels/frame) when a fluid layer is running; add it to the motion.
    /// Returns `Ok(true)` when `dst` was written.
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &mut self,
        rec: &mut FrameRecorder<'_>,
        layer: usize,
        src: &LayerTarget,
        dst: &LayerTarget,
        amount: f32,
        frame: &AnalysisFrame,
        extra_flow: Option<&LayerTarget>,
    ) -> Result<bool, RenderError> {
        if amount <= MIN_AMOUNT {
            self.invalidate(layer);
            return Ok(false);
        }
        if self.layers.len() <= layer {
            self.layers.resize_with(layer + 1, || None);
        }
        let (width, height) = (self.width, self.height);
        if self.layers[layer].is_none() {
            self.layers[layer] = Some(LayerState::new(rec.gpu(), width, height));
        }
        debug_assert_eq!(
            (dst.width(), dst.height()),
            (width, height),
            "mosh output must be at the internal size"
        );

        // Fields are borrowed one by one: the pipelines are immutable and the layer state is
        // mutable, and the recorder needs `&mut` for the duration.
        let (copy, grid, motion, feedback, no_flow) =
            (&self.copy, &self.grid, &self.motion, &self.feedback, &self.no_flow);
        let state = self.layers[layer]
            .as_mut()
            .expect("layer state just created");
        let res = [width as f32, height as f32];
        let iframe = iframe_due(state.valid, state.last_time_s, frame);
        let copy_uniform = copy_uniform(res);
        let grid_uniform = motion_uniform(GRID);

        // This frame's analysis grid, then block matching against last frame's grid: the copy is
        // 320x180 whatever the layer size, so the motion pass reads a cache-resident pair. It runs
        // on an I-frame too, because the next P-frame's block match differs against *this* frame.
        let next_grid = 1 - state.grid_front;
        rec.draw(grid, &state.grids[next_grid], &grid_uniform, &[src])?;

        if iframe {
            rec.draw(copy, dst, &copy_uniform, &[src])?;
        } else {
            let motion_uniform = motion_uniform(GRID);
            rec.draw(
                motion,
                &state.motion,
                &motion_uniform,
                &[&state.grids[next_grid], &state.grids[state.grid_front]],
            )?;
            // `skipres` is consumed by a P-frame, not by an I-frame: an I-frame has already
            // replaced the picture, so it must not eat the carry-over it interrupts.
            let skipres = state.skipres_frames > 0;
            if skipres {
                state.skipres_frames -= 1;
            }
            let feedback_uniform = feedback_uniform(res, amount, GRID, skipres);
            let flow = extra_flow.unwrap_or(no_flow);
            rec.draw(
                feedback,
                dst,
                &feedback_uniform,
                &[src, dst, &state.prev_src, &state.motion, flow],
            )?;
        }
        // The residual needs last frame's source, so every frame - I-frame or not - this frame's
        // source becomes it. (The web app keeps the same picture in its `G.src` ping-pong.)
        rec.draw(copy, &state.prev_src, &copy_uniform, &[src])?;
        state.grid_front = next_grid;

        state.valid = true;
        state.last_time_s = frame.time_s;
        Ok(true)
    }

    /// Forgets a layer's buffer, so the next `apply` refreshes it (an I-frame). Called when the
    /// clip has no mosh to apply, and by `Visuals::draw_layer` on a Cut or Strobe clip entry (and
    /// on the first clip): the web app's `entered -> iframe` rule.
    pub fn invalidate(&mut self, layer: usize) {
        if let Some(Some(state)) = self.layers.get_mut(layer) {
            state.valid = false;
        }
    }

    /// A Datamosh or Melt clip change: keep the layer's buffer, and run the next P-frame without
    /// the residual. The old picture is then pushed along by whatever motion the new clip makes -
    /// the web app's `entered -> P.skipres = 1`, and the whole point of a datamosh cut.
    ///
    /// A layer with no buffer yet (the previous clip never moshed) has nothing to carry: its
    /// first frame is an ordinary I-frame. Calling this twice before the next `apply` is the same
    /// as calling it once; the skip covers one P-frame.
    ///
    /// `Visuals::draw_layer` calls this for a layer whose new clip enters on a Datamosh or Melt
    /// transition. It is the only way to express that entry: `invalidate` is the Cut/Strobe rule
    /// (an I-frame), and a plain `apply` applies the residual, which is the web's *Crossfade*
    /// entry.
    pub fn carry_over(&mut self, layer: usize) {
        if let Some(Some(state)) = self.layers.get_mut(layer) {
            state.skipres_frames = SKIPRES_FRAMES;
        }
    }
}

fn write_f32(bytes: &mut [u8], offset: usize, value: f32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn copy_uniform(res: [f32; 2]) -> [u8; COPY_UNIFORM_BYTES] {
    let mut bytes = [0u8; COPY_UNIFORM_BYTES];
    write_f32(&mut bytes, 0, res[0]);
    write_f32(&mut bytes, 4, res[1]);
    bytes
}

fn motion_uniform(grid: [f32; 2]) -> [u8; MOTION_UNIFORM_BYTES] {
    let mut bytes = [0u8; MOTION_UNIFORM_BYTES];
    write_f32(&mut bytes, 0, grid[0]);
    write_f32(&mut bytes, 4, grid[1]);
    bytes
}

fn feedback_uniform(res: [f32; 2], amount: f32, grid: [f32; 2], skipres: bool) -> [u8; FEEDBACK_UNIFORM_BYTES] {
    let mut bytes = [0u8; FEEDBACK_UNIFORM_BYTES];
    let amount = amount.clamp(0.0, 1.0);
    write_f32(&mut bytes, 0, res[0]);
    write_f32(&mut bytes, 4, res[1]);
    // `decide()`: leak = (1 - m)^5, floored; gain = 1 + 2.2 * m; coarse = m > 0.85.
    write_f32(&mut bytes, 8, leak_for(amount));
    // No freeze/bloom yet: the web multiplies this by 1.02 per frozen frame (see the module).
    write_f32(&mut bytes, 12, 1.0);
    write_f32(&mut bytes, 16, 1.0 + GAIN_PER_AMOUNT * amount);
    write_f32(&mut bytes, 20, if skipres { 1.0 } else { 0.0 });
    write_f32(&mut bytes, 24, if amount > COARSE_ABOVE { 1.0 } else { 0.0 });
    write_f32(&mut bytes, 32, grid[0]);
    write_f32(&mut bytes, 36, grid[1]);
    bytes
}

/// The web app's `leak`: a fifth power of the remaining freshness, floored so that the picture
/// can never freeze completely.
fn leak_for(amount: f32) -> f32 {
    let fresh = 1.0 - amount;
    (fresh * fresh * fresh * fresh * fresh).max(MIN_LEAK)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mosh_core::EventFlags;

    #[test]
    fn grid_and_field_shapes_match_the_web_app() {
        // The web's GS and mvT: a 320x180 luminance grid, 8x8 blocks, 40x23 vectors.
        assert_eq!(GRID, [320.0, 180.0]);
        assert_eq!(MOTION_BLOCK, 8);
        assert_eq!(MOTION_WIDTH, 40);
        assert_eq!(MOTION_HEIGHT, 23);
        assert_eq!(MOTION_WIDTH * MOTION_BLOCK, GRID[0] as u32);
        // 180 / 8 = 22.5: the vector texture has one more, half-covered row.
        assert!((MOTION_HEIGHT - 1) * MOTION_BLOCK < GRID[1] as u32);
        assert!(MOTION_HEIGHT * MOTION_BLOCK > GRID[1] as u32);
    }

    #[test]
    fn uniforms_are_laid_out_like_the_shaders_read_them() {
        let at = |bytes: &[u8], index: usize| {
            f32::from_le_bytes([bytes[index], bytes[index + 1], bytes[index + 2], bytes[index + 3]])
        };
        let copy = copy_uniform([960.0, 540.0]);
        assert_eq!(at(&copy, 0), 960.0);
        assert_eq!(at(&copy, 4), 540.0);
        let motion = motion_uniform(GRID);
        assert_eq!(at(&motion, 0), 320.0);
        assert_eq!(at(&motion, 4), 180.0);

        // `struct FU` in mosh_feedback.wgsl: res, leak, mv_scale, gain, skipres, coarse, pad, grid.
        let feedback = feedback_uniform([960.0, 540.0], 0.8, GRID, false);
        assert_eq!(FEEDBACK_UNIFORM_BYTES, 40);
        assert_eq!(at(&feedback, 0), 960.0);
        assert_eq!(at(&feedback, 4), 540.0);
        // amount 0.8: (1 - 0.8)^5 = 0.00032, so the floor takes over.
        assert_eq!(at(&feedback, 8), MIN_LEAK);
        assert_eq!(at(&feedback, 12), 1.0, "mvScale: no freeze/bloom yet");
        assert_eq!(at(&feedback, 16), 1.0 + 2.2 * 0.8);
        assert_eq!(at(&feedback, 20), 0.0, "skipres off");
        assert_eq!(at(&feedback, 24), 0.0, "0.8 is not coarse");
        assert_eq!(at(&feedback, 28), 0.0, "the pad stays zero");
        assert_eq!(at(&feedback, 32), 320.0);
        assert_eq!(at(&feedback, 36), 180.0);

        // The fifth power and its floor, and the two thresholds that come with it.
        let low = feedback_uniform([1.0, 1.0], 0.6, GRID, false);
        assert!((at(&low, 8) - 0.4f32.powi(5)).abs() < 1e-7);
        assert_eq!(at(&low, 24), 0.0);
        let high = feedback_uniform([1.0, 1.0], 0.9, GRID, true);
        assert_eq!(at(&high, 8), MIN_LEAK);
        assert_eq!(at(&high, 20), 1.0, "skipres on");
        assert_eq!(at(&high, 24), 1.0, "0.9 is coarse");

        // Out-of-range amounts are clamped rather than fed to the shader.
        let over = feedback_uniform([1.0, 1.0], 4.0, GRID, false);
        assert_eq!(at(&over, 8), MIN_LEAK);
        assert_eq!(at(&over, 16), 3.2);
        let under = feedback_uniform([1.0, 1.0], -2.0, GRID, false);
        assert_eq!(at(&under, 8), 1.0, "amount 0 leaks everything");
        assert_eq!(at(&under, 16), 1.0, "and gains nothing");
    }

    #[test]
    fn iframe_rule_fires_only_on_a_missing_buffer_or_a_time_jump() {
        let rule = |valid: bool, last: f64, frame: &AnalysisFrame| iframe_due(valid, last, frame);
        let plain = AnalysisFrame {
            time_s: 1.0,
            bar_index: 1,
            ..AnalysisFrame::default()
        };
        assert!(rule(false, 0.0, &plain), "no buffer yet");
        assert!(!rule(true, 0.9, &plain), "ordinary frame");
        // Nothing musical starts an I-frame: `decide()` has no bar rule and no drop rule, and an
        // I-frame refreshes the picture, which is what the mosh is not.
        for events in [
            EventFlags::BAR,
            EventFlags::DROP,
            EventFlags::BEAT | EventFlags::PHRASE | EventFlags::BUILD_START,
        ] {
            for bar_index in [0, 4, 8, 5] {
                assert!(
                    !rule(
                        true,
                        0.9,
                        &AnalysisFrame {
                            events,
                            bar_index,
                            time_s: 1.0,
                            ..plain
                        }
                    ),
                    "{events:?} on bar {bar_index} must not refresh the buffer"
                );
            }
        }
        assert!(
            rule(
                true,
                1.0,
                &AnalysisFrame {
                    time_s: 0.5,
                    ..plain
                }
            ),
            "a backwards jump"
        );
        assert!(
            rule(
                true,
                1.0,
                &AnalysisFrame {
                    time_s: 1.3,
                    ..plain
                }
            ),
            "a playback hitch"
        );
        assert!(
            !rule(
                true,
                1.0,
                &AnalysisFrame {
                    time_s: 1.2,
                    ..plain
                }
            ),
            "a normal frame interval"
        );
    }
}
