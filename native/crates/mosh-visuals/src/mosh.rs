//! Datamosh: block-matching motion estimation and the feedback pass, per layer.
//! Owner: V-mosh (Task 2.2). The public signatures are frozen by the coordinator.
//!
//! Ported from `gpu.js`: [`MoshPass::apply`] runs the web app's motion pass (`WG_ME`: 8x8 blocks
//! on a fixed 320x180 luminance grid, a +-6 grid-pixel search against the previous frame, the same
//! SAD cost with its 3-per-pixel motion penalty) and then the P-frame half of its feedback pass
//! (`WG_FB`): the previous mosh buffer displaced along the motion field, mixed with the new frame.
//! The shaders are `mosh_motion`, `mosh_feedback` and `mosh_copy` in `mosh-shaders`.
//!
//! # Per-layer state
//!
//! State is keyed by the `layer` index `apply` is given and created lazily, at the internal size
//! passed to [`MoshPass::new`]:
//!
//! * `previous`: a copy of last frame's input. A `LayerTarget` is a ping-pong pair whose readable
//!   image is this frame's, so block matching needs its own previous frame.
//! * `motion`: 40x23 vectors, one per 8x8 block of the analysis grid (the web's `mvT`), so it
//!   costs the same at every internal size.
//! * the feedback buffer is `dst`'s own previous image: the mosh stage is the only writer of
//!   `dst`, so its readable image *is* last frame's mosh output and no extra full-size target is
//!   allocated. `valid` tracks whether that image may be used.
//!
//! # `amount`
//!
//! `amount` is the P-frame strength: `out = mix(new_frame, carried_buffer, amount)`. The web
//! derives `leak = (1 - amount)^5` (clamped to 0.004) and `gain = 1 + 2.2 * amount` from the same
//! parameter; this pass uses `1 - amount` as the refresh weight and no vector gain, because
//! `amount` is already the strength and a gain would carry the picture ahead of its motion. At
//! `amount = 1` the refresh is clamped to 0.004, so a layer keeps a trickle of new pixels (the
//! web does the same) instead of freezing forever. `amount == 0` records nothing and invalidates
//! the buffer, so the next non-zero amount starts on an I-frame.
//!
//! # I-frames
//!
//! There is no per-frame plan here; `engine.c`'s `decide()` belongs to the web app. A layer takes
//! an I-frame (a copy of the new frame into the output and into `previous`, no motion pass) when:
//!
//! * it has no valid buffer (first use, or the first use after `amount` was 0), or
//! * `EventFlags::DROP` is set, or
//! * `EventFlags::BAR` is set and `bar_index % 4 == 0`, i.e. every four bars, or
//! * the frame's `time_s` is not within `IFRAME_GAP_S` of the previous frame's: playback jumped
//!   backwards, or a hitch left a hole. `decide()` refreshes after eight 30 fps frames for the
//!   same reason; at 60 fps this is 15 frames.
//!
//! A *clip change* is not in that list: `apply` is never told the clip, and the frozen signature
//! has no force-refresh flag, so the web app's `entered -> iframe` rule cannot be reproduced from
//! this side. Reported to the coordinator; a `Datamosh` transition (Task 2.3) that needs the
//! buffer refreshed on a clip change needs an addition to the contract, not a workaround here.
//!
//! # `extra_flow`
//!
//! The fluid visualizer's velocity field (RG, pixels per frame at the internal size) is added to
//! the frame's displacement, so a layer above a fluid layer is stirred by it. Without a fluid
//! layer the slot is bound to a 1x1 zero texture, which costs one texel per fragment.

use mosh_core::{AnalysisFrame, EventFlags};
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

/// The smallest refresh the feedback keeps, so `amount = 1` cannot freeze a layer forever. The
/// web clamps its leak the same way.
const MIN_REFRESH: f32 = 0.004;

const MOTION_UNIFORM_BYTES: usize = 16;
const FEEDBACK_UNIFORM_BYTES: usize = 32;
const COPY_UNIFORM_BYTES: usize = 16;

pub struct MoshPass {
    width: u32,
    height: u32,
    copy: Pipeline,
    motion: Pipeline,
    feedback: Pipeline,
    /// Bound in the feedback pass's fourth texture slot when no fluid layer is running.
    no_flow: LayerTarget,
    layers: Vec<Option<LayerState>>,
}

/// One layer's feedback state. Created on the first `apply` for that layer.
struct LayerState {
    /// Last frame's input, for block matching (`previous.front`).
    previous: LayerTarget,
    /// 40x23 displacement field, RG = grid pixels per frame, forward.
    motion: LayerTarget,
    last_time_s: f64,
    /// True once the buffer in the caller's `dst` holds this layer's last output.
    valid: bool,
}

impl LayerState {
    fn new(gpu: &Gpu, width: u32, height: u32) -> LayerState {
        LayerState {
            previous: LayerTarget::new(gpu, width, height),
            motion: LayerTarget::new(gpu, MOTION_WIDTH, MOTION_HEIGHT),
            last_time_s: f64::NAN,
            valid: false,
        }
    }
}

/// The I-frame rule documented on the module: no valid buffer, a DROP, every fourth bar, or a
/// discontinuity in playback time.
fn iframe_due(valid: bool, last_time_s: f64, frame: &AnalysisFrame) -> bool {
    if !valid {
        return true;
    }
    if frame.events.contains(EventFlags::DROP) {
        return true;
    }
    if frame.events.contains(EventFlags::BAR) && frame.bar_index.rem_euclid(4) == 0 {
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
        let motion_shader = spirv("mosh_motion").ok_or(RenderError::EmptyShader)?;
        let feedback_shader = spirv("mosh_feedback").ok_or(RenderError::EmptyShader)?;
        let copy_shader = spirv("mosh_copy").ok_or(RenderError::EmptyShader)?;
        Ok(MoshPass {
            width,
            height,
            copy: Pipeline::fullscreen(gpu, copy_shader, COPY_UNIFORM_BYTES, 1)?
                .with_name("mosh copy"),
            motion: Pipeline::fullscreen(gpu, motion_shader, MOTION_UNIFORM_BYTES, 2)?
                .with_name("mosh motion"),
            feedback: Pipeline::fullscreen(gpu, feedback_shader, FEEDBACK_UNIFORM_BYTES, 4)?
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
        if amount <= 0.0 {
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
        let (copy, motion, feedback, no_flow) =
            (&self.copy, &self.motion, &self.feedback, &self.no_flow);
        let state = self.layers[layer]
            .as_mut()
            .expect("layer state just created");
        let res = [width as f32, height as f32];
        let iframe = iframe_due(state.valid, state.last_time_s, frame);
        let copy_uniform = copy_uniform(res);

        if iframe {
            rec.draw(copy, dst, &copy_uniform, &[src])?;
        } else {
            let motion_uniform = motion_uniform(GRID);
            rec.draw(
                motion,
                &state.motion,
                &motion_uniform,
                &[src, &state.previous],
            )?;
            let feedback_uniform = feedback_uniform(res, amount, GRID);
            let flow = extra_flow.unwrap_or(no_flow);
            rec.draw(
                feedback,
                dst,
                &feedback_uniform,
                &[src, dst, &state.motion, flow],
            )?;
        }
        // Last frame's input for the next call, after the motion pass has read the old one.
        rec.draw(copy, &state.previous, &copy_uniform, &[src])?;

        state.valid = true;
        state.last_time_s = frame.time_s;
        Ok(true)
    }

    /// Forgets a layer's buffer, so the next `apply` refreshes it. Called when the clip has no
    /// mosh to apply: the output target is then whatever the chain left there.
    fn invalidate(&mut self, layer: usize) {
        if let Some(Some(state)) = self.layers.get_mut(layer) {
            state.valid = false;
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

fn feedback_uniform(res: [f32; 2], amount: f32, grid: [f32; 2]) -> [u8; FEEDBACK_UNIFORM_BYTES] {
    let mut bytes = [0u8; FEEDBACK_UNIFORM_BYTES];
    let amount = amount.clamp(0.0, 1.0);
    write_f32(&mut bytes, 0, res[0]);
    write_f32(&mut bytes, 4, res[1]);
    write_f32(&mut bytes, 8, amount);
    write_f32(&mut bytes, 12, MIN_REFRESH);
    write_f32(&mut bytes, 16, grid[0]);
    write_f32(&mut bytes, 20, grid[1]);
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let copy = copy_uniform([960.0, 540.0]);
        assert_eq!(f32::from_le_bytes(copy[0..4].try_into().unwrap()), 960.0);
        assert_eq!(f32::from_le_bytes(copy[4..8].try_into().unwrap()), 540.0);
        let motion = motion_uniform(GRID);
        assert_eq!(f32::from_le_bytes(motion[0..4].try_into().unwrap()), 320.0);
        assert_eq!(f32::from_le_bytes(motion[4..8].try_into().unwrap()), 180.0);
        let feedback = feedback_uniform([960.0, 540.0], 0.8, GRID);
        assert_eq!(
            f32::from_le_bytes(feedback[0..4].try_into().unwrap()),
            960.0
        );
        assert_eq!(
            f32::from_le_bytes(feedback[4..8].try_into().unwrap()),
            540.0
        );
        assert_eq!(f32::from_le_bytes(feedback[8..12].try_into().unwrap()), 0.8);
        assert_eq!(
            f32::from_le_bytes(feedback[12..16].try_into().unwrap()),
            MIN_REFRESH
        );
        assert_eq!(
            f32::from_le_bytes(feedback[16..20].try_into().unwrap()),
            320.0
        );
        assert_eq!(
            f32::from_le_bytes(feedback[20..24].try_into().unwrap()),
            180.0
        );
        // Out-of-range amounts are clamped rather than fed to `mix()`.
        let over = feedback_uniform([1.0, 1.0], 4.0, GRID);
        assert_eq!(f32::from_le_bytes(over[8..12].try_into().unwrap()), 1.0);
        let under = feedback_uniform([1.0, 1.0], -2.0, GRID);
        assert_eq!(f32::from_le_bytes(under[8..12].try_into().unwrap()), 0.0);
    }

    #[test]
    fn iframe_rule_fires_on_drop_every_fourth_bar_and_on_a_jump() {
        let rule = |valid: bool, last: f64, frame: &AnalysisFrame| iframe_due(valid, last, frame);
        let plain = AnalysisFrame {
            time_s: 1.0,
            bar_index: 1,
            ..AnalysisFrame::default()
        };
        assert!(rule(false, 0.0, &plain), "no buffer yet");
        assert!(!rule(true, 0.9, &plain), "ordinary frame");
        assert!(
            !rule(
                true,
                0.9,
                &AnalysisFrame {
                    events: EventFlags::BAR,
                    bar_index: 5,
                    time_s: 1.0,
                    ..plain
                }
            ),
            "bar 5 is not a multiple of four"
        );
        assert!(
            rule(
                true,
                0.9,
                &AnalysisFrame {
                    events: EventFlags::BAR,
                    bar_index: 4,
                    time_s: 1.0,
                    ..plain
                }
            ),
            "bar 4"
        );
        assert!(
            rule(
                true,
                0.9,
                &AnalysisFrame {
                    events: EventFlags::DROP,
                    time_s: 1.0,
                    ..plain
                }
            ),
            "a drop"
        );
        assert!(
            !rule(
                true,
                0.9,
                &AnalysisFrame {
                    events: EventFlags::BEAT | EventFlags::PHRASE,
                    time_s: 1.0,
                    ..plain
                }
            ),
            "a beat or a phrase boundary is not an I-frame"
        );
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
