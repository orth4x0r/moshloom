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
//! * `prev_src`: last frame's source picture at the internal size for the residual (`src - prv`
//!   must run at the *layer's* resolution, not the 320x180 grid, so a static picture has zero
//!   residual and never rings). When `src` is a ping-pong target written once per frame (the
//!   normal case: `LayerSlot::scene` is a ping-pong `LayerTarget::new`), its back image *is*
//!   last frame's picture and `mosh_feedback` samples it directly via `TargetSample::Previous`
//!   with zero copy passes and zero extra memory. Only a single-image `src`
//!   (`LayerTarget::new_single`) lazily allocates `prev_src` and runs `mosh_copy`.
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
//! # Freeze / Bloom (`mv_scale`)
//!
//! Ported from `engine.c:342,349,370,372,391` and `gpu.js:893,902-903,912`: when a layer is frozen
//! (via [`MoshPass::set_freeze`], driven by an active `TransitionKind::Melt` or by `EffectKind::Mosh`'s
//! `param` as `bloom` in the last `bloom * 0.5` of each 4-beat bar via [`MoshPass::bloom_freeze`]):
//!
//! * the motion estimation pass is **skipped** (once at least one P-frame has populated `motion`),
//!   reusing the frozen motion vectors from the frame before the freeze;
//! * `mv_scale` grows multiplicatively every frozen frame (`mv_scale *= 1.02`, capped at `64.0` to
//!   prevent `f32` overflow on a paused transport), stretching the carried smear exponentially;
//! * `skipres` is forced on (`PL[2] = P.skipres || P.freeze`), so the residual and leak are
//!   suppressed for the duration of the freeze.
//!
//! On the first non-frozen P-frame (or on an I-frame / `invalidate`), `mv_scale` resets to `1.0`.
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
use mosh_render::{FrameRecorder, Gpu, LayerTarget, Pipeline, RenderError, TargetSample};
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

/// Per-frame multiplier on the frozen motion vector scale (`gpu.js:902`: `G.mvScale *= 1.02`,
/// `engine.c:370`: `mvx[i] *= 1.02f; mvy[i] *= 1.02f`).
const FREEZE_MV_GROWTH: f32 = 1.02;

/// Safety cap on `mv_scale` (~210 consecutive frozen frames = 3.5 s at 60 fps) so an indefinitely
/// held freeze (e.g. a paused transport inside a bloom window) never overflows `f32`.
const MAX_MV_SCALE: f32 = 64.0;

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
    /// True once `motion` has been written at least once since creation / `invalidate`, so a
    /// freeze pass never reads an uninitialized motion texture.
    has_motion: bool,
    /// Fallback copy of last frame's source picture, allocated lazily only when `src` is a
    /// single-image target (`LayerTarget::new_single`). A ping-pong `src` (such as
    /// `LayerSlot::scene`) already keeps last frame's image in its back half, so no copy pass
    /// or extra target is needed.
    prev_src: Option<LayerTarget>,
    /// Identity and write counter of `src` on the last `apply`, to verify its back image is
    /// still last frame's picture.
    last_src_id: u64,
    last_src_writes: u64,
    last_time_s: f64,
    /// True once the buffer in the caller's `dst` holds this layer's last output.
    valid: bool,
    /// Frames that must still run without the residual (`skipres`), set by `carry_over`.
    skipres_frames: u32,
    /// Multiplicative scale on the frozen motion field (`G.mvScale` in `gpu.js`).
    mv_scale: f32,
    /// One-shot freeze request for the next `apply`, set by `set_freeze`.
    freeze_next: bool,
}

impl LayerState {
    fn new(gpu: &Gpu) -> LayerState {
        LayerState {
            grids: [
                LayerTarget::new(gpu, GRID[0] as u32, GRID[1] as u32),
                LayerTarget::new(gpu, GRID[0] as u32, GRID[1] as u32),
            ],
            grid_front: 0,
            motion: LayerTarget::new(gpu, MOTION_WIDTH, MOTION_HEIGHT),
            has_motion: false,
            prev_src: None,
            last_src_id: 0,
            last_src_writes: 0,
            last_time_s: f64::NAN,
            valid: false,
            skipres_frames: 0,
            mv_scale: 1.0,
            freeze_next: false,
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
            self.layers[layer] = Some(LayerState::new(rec.gpu()));
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

        let src_id = src.id();
        let src_writes = src.writes();
        let freeze = std::mem::take(&mut state.freeze_next);
        if iframe {
            rec.draw(copy, dst, &copy_uniform, &[src])?;
            state.mv_scale = 1.0;
        } else {
            // `gpu.js:902,912`: while frozen, skip motion re-estimation (reusing the previous
            // frame's vectors) and stretch `mv_scale *= 1.02` each frame; otherwise re-estimate
            // motion and reset `mv_scale = 1.0`. If `motion` has never been written yet (the first
            // P-frame after an I-frame), run it once so the texture is initialized before freezing.
            if freeze && state.has_motion {
                state.mv_scale = (state.mv_scale * FREEZE_MV_GROWTH).min(MAX_MV_SCALE);
            } else {
                let motion_uniform = motion_uniform(GRID);
                rec.draw(
                    motion,
                    &state.motion,
                    &motion_uniform,
                    &[&state.grids[next_grid], &state.grids[state.grid_front]],
                )?;
                state.has_motion = true;
                if freeze {
                    state.mv_scale = (state.mv_scale * FREEZE_MV_GROWTH).min(MAX_MV_SCALE);
                } else {
                    state.mv_scale = 1.0;
                }
            }
            // `skipres` is consumed by a P-frame, not by an I-frame: an I-frame has already
            // replaced the picture, so it must not eat the carry-over it interrupts.
            // `engine.c:391`: `PL[2] = P.skipres || P.freeze` — a frozen frame also suppresses the
            // residual and leak so the carried picture blooms cleanly along the frozen vectors.
            let carry_skip = state.skipres_frames > 0;
            if carry_skip {
                state.skipres_frames -= 1;
            }
            let skipres = carry_skip || freeze;
            let feedback_uniform =
                feedback_uniform(res, amount, state.mv_scale, GRID, skipres);
            let flow = extra_flow.unwrap_or(no_flow);
            // When `src` is a ping-pong target written once since our last call, its back image
            // *is* last frame's source (zero copy). When `src` was not redrawn since last call,
            // its current image is unchanged. Single-image targets fall back to `state.prev_src`.
            let previous = if !src.is_single()
                && src_id == state.last_src_id
                && src_writes == state.last_src_writes.wrapping_add(1)
            {
                TargetSample::Previous(src)
            } else if src_id == state.last_src_id && src_writes == state.last_src_writes {
                TargetSample::Current(src)
            } else if let Some(prev) = state.prev_src.as_ref() {
                TargetSample::Current(prev)
            } else {
                TargetSample::Current(src)
            };
            rec.draw_samples(
                feedback,
                dst,
                &feedback_uniform,
                &[
                    TargetSample::Current(src),
                    TargetSample::Current(dst),
                    previous,
                    TargetSample::Current(&state.motion),
                    TargetSample::Current(flow),
                ],
            )?;
        }
        // Only single-image sources need an explicit `prev_src` copy: ping-pong sources keep last
        // frame's picture in their other half for free.
        if src.is_single() {
            let prev = state
                .prev_src
                .get_or_insert_with(|| LayerTarget::new_single(rec.gpu(), width, height));
            rec.draw(copy, prev, &copy_uniform, &[src])?;
        }
        state.grid_front = next_grid;
        state.last_src_id = src_id;
        state.last_src_writes = src_writes;

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
            state.has_motion = false;
            state.mv_scale = 1.0;
            state.freeze_next = false;
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

    /// Requests freeze/bloom on the next [`MoshPass::apply`] for `layer` (one-shot per frame).
    ///
    /// When `freeze` is `true` on a P-frame, motion estimation is skipped (reusing the previous
    /// frame's motion vectors), `mv_scale` is multiplied by `1.02`, and `skipres` is forced on.
    /// Calling `set_freeze(layer, false)` (or not calling it before the next P-frame) resumes
    /// normal motion estimation and resets `mv_scale` to `1.0`.
    pub fn set_freeze(&mut self, layer: usize, freeze: bool) {
        if let Some(Some(state)) = self.layers.get_mut(layer) {
            state.freeze_next = freeze;
        }
    }

    /// Current multiplicative scale applied to `layer`'s motion field (`1.0` when not frozen,
    /// growing by `1.02` per frozen frame up to `64.0`).
    pub fn mv_scale(&self, layer: usize) -> f32 {
        self.layers
            .get(layer)
            .and_then(|slot| slot.as_ref())
            .map_or(1.0, |state| state.mv_scale)
    }

    /// The web app's periodic bar-end bloom rule (`engine.c:349`):
    /// `if (bloom > 0 && fract(bar_phase) > 1.0 - bloom * 0.5) freeze = true`.
    pub fn bloom_freeze(bloom: f32, bar_phase: f32) -> bool {
        let bloom = bloom.clamp(0.0, 1.0);
        if bloom <= 0.0 || !bar_phase.is_finite() {
            return false;
        }
        let phase = bar_phase.rem_euclid(1.0);
        phase > 1.0 - bloom * 0.5
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

fn feedback_uniform(
    res: [f32; 2],
    amount: f32,
    mv_scale: f32,
    grid: [f32; 2],
    skipres: bool,
) -> [u8; FEEDBACK_UNIFORM_BYTES] {
    let mut bytes = [0u8; FEEDBACK_UNIFORM_BYTES];
    let amount = amount.clamp(0.0, 1.0);
    write_f32(&mut bytes, 0, res[0]);
    write_f32(&mut bytes, 4, res[1]);
    // `decide()`: leak = (1 - m)^5, floored; gain = 1 + 2.2 * m; coarse = m > 0.85.
    write_f32(&mut bytes, 8, leak_for(amount));
    write_f32(&mut bytes, 12, mv_scale.clamp(1.0, MAX_MV_SCALE));
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
        let feedback = feedback_uniform([960.0, 540.0], 0.8, 1.0, GRID, false);
        assert_eq!(FEEDBACK_UNIFORM_BYTES, 40);
        assert_eq!(at(&feedback, 0), 960.0);
        assert_eq!(at(&feedback, 4), 540.0);
        // amount 0.8: (1 - 0.8)^5 = 0.00032, so the floor takes over.
        assert_eq!(at(&feedback, 8), MIN_LEAK);
        assert_eq!(at(&feedback, 12), 1.0, "mvScale when not frozen");
        assert_eq!(at(&feedback, 16), 1.0 + 2.2 * 0.8);
        assert_eq!(at(&feedback, 20), 0.0, "skipres off");
        assert_eq!(at(&feedback, 24), 0.0, "0.8 is not coarse");
        assert_eq!(at(&feedback, 28), 0.0, "the pad stays zero");
        assert_eq!(at(&feedback, 32), 320.0);
        assert_eq!(at(&feedback, 36), 180.0);

        // Frozen mv_scale flows through to offset 12 and is clamped to [1.0, MAX_MV_SCALE].
        let frozen = feedback_uniform([960.0, 540.0], 0.8, 1.02f32.powi(10), GRID, true);
        assert!((at(&frozen, 12) - 1.02f32.powi(10)).abs() < 1e-6);
        assert_eq!(at(&frozen, 20), 1.0, "skipres on while frozen");
        let capped = feedback_uniform([960.0, 540.0], 0.8, 999.0, GRID, true);
        assert_eq!(at(&capped, 12), MAX_MV_SCALE);

        // The fifth power and its floor, and the two thresholds that come with it.
        let low = feedback_uniform([1.0, 1.0], 0.6, 1.0, GRID, false);
        assert!((at(&low, 8) - 0.4f32.powi(5)).abs() < 1e-7);
        assert_eq!(at(&low, 24), 0.0);
        let high = feedback_uniform([1.0, 1.0], 0.9, 1.0, GRID, true);
        assert_eq!(at(&high, 8), MIN_LEAK);
        assert_eq!(at(&high, 20), 1.0, "skipres on");
        assert_eq!(at(&high, 24), 1.0, "0.9 is coarse");

        // Out-of-range amounts are clamped rather than fed to the shader.
        let over = feedback_uniform([1.0, 1.0], 4.0, 1.0, GRID, false);
        assert_eq!(at(&over, 8), MIN_LEAK);
        assert_eq!(at(&over, 16), 3.2);
        let under = feedback_uniform([1.0, 1.0], -2.0, 0.25, GRID, false);
        assert_eq!(at(&under, 8), 1.0, "amount 0 leaks everything");
        assert_eq!(at(&under, 12), 1.0, "mv_scale never drops below 1.0");
        assert_eq!(at(&under, 16), 1.0, "and gains nothing");
    }

    #[test]
    fn bloom_freeze_matches_engine_c_bar_window() {
        // engine.c:349: if (bloom > 0) { if (fr > 1 - bloom * 0.5) freeze = 1; }
        assert!(!MoshPass::bloom_freeze(0.0, 0.99), "bloom 0 never freezes");
        assert!(!MoshPass::bloom_freeze(0.5, 0.74), "before last quarter of bar");
        assert!(!MoshPass::bloom_freeze(0.5, 0.75), "strict > matches engine.c");
        assert!(MoshPass::bloom_freeze(0.5, 0.76), "inside last quarter of bar");
        assert!(MoshPass::bloom_freeze(1.0, 0.51), "bloom 1.0 covers second half of bar");
        assert!(!MoshPass::bloom_freeze(1.0, 0.49), "first half of bar stays unfrozen");
        assert!(!MoshPass::bloom_freeze(0.5, f32::NAN), "non-finite phase never freezes");
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
