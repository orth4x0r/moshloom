//! Fluid as its own visualizer: stable fluids on a 256x144 grid, driven by the music.
//! Owner: V-fluid (Task 2.4). Signatures are frozen.
//!
//! # What runs each `draw`
//!
//! One `draw` is one step of the solver plus the two picture passes, all recorded into the
//! caller's `FrameRecorder` (nothing is submitted here):
//!
//! ```text
//! velocity (256x144, grid uv/s)       dye (256x144, rgb)
//!   advect      -> velocity_b           dye advect + inject + fade -> dye
//!   forces      -> velocity_a           present      -> dst          (layer internal size)
//!   curl        -> curl                 flow         -> flow         (velocity in pixels/draw,
//!   vorticity   -> velocity_b                                         at the internal size,
//!   divergence  -> divergence                                         for a mosh layer)
//!   jacobi x N  -> pressure_a/b
//!   gradient    -> velocity_a           (the final field, in `velocity_a` front)
//! ```
//!
//! The seven velocity passes are the web app's own (`gpu.js`, extracted by N-shaders into
//! `mosh-shaders/wgsl/fluid_{advect,force,curl,vorticity,divergence,jacobi,gradient}.wgsl`), run
//! unchanged at 256x144 with dt = 1/30 s, exactly as `fluidStep` in `gpu.js` does. Two additions
//! are this crate's: the **dye** field (the web app has no dye, it only pushes pixels with the
//! velocity) and the **flow** conversion the mosh pass samples.
//!
//! Cost: 9 + N passes per fluid layer, N = pressure iterations (8..30). At N = 8 that is 17
//! draws; `mosh-render` allows 64 draws per frame, shared with every other layer's mosh, effect
//! and transition passes, so two fluid layers at N = 30 (39 + 39 = 78) would not fit. The web
//! app's 27 passes per frame measured +0.17 ms/frame on the discrete GPU; the number for this
//! port is in the PR body (offscreen, integrated GPU).
//!
//! # `p0..p7` — the `VISUALIZERS` list V-scenes needs
//!
//! All eight are 0..1 like every other clip parameter (`ParamRange::default() == 0..1`); each
//! has a baseline so `ClipParams::default()` (all zeros) still draws a living fluid:
//!
//! | # | name | meaning |
//! | --- | --- | --- |
//! | `p0` | viscosity | solver damping, 0..1 |
//! | `p1` | dye amount | injector gain, `0.25 + 0.75 * p1` |
//! | `p2` | force scale | kick and swirl gain, `0.5 + 1.5 * p2` |
//! | `p3` | swirl | extra rotation from `beat_phase`, `0.2 + 0.8 * p3` |
//! | `p4` | colour cycle | adds to the palette's hue cycle (and drifts with `t`) |
//! | `p5` | vorticity | confinement strength, `0.2 + 1.2 * p5` |
//! | `p6` | jacobi iterations | `round(8 + 22 * p6)`, clamped 8..30 |
//! | `p7` | dye fade | extra fade per step, `0.005 + 0.020 * p7` of the dye per draw |
//!
//! `mosh-shaders::VISUALIZERS` (owned by N-shaders/V-scenes) still lists the older names
//! `step (dt), viscosity, hue, vorticity, kick impulse, dye gain, beat swirl, reserved` for
//! `fluid`; that list and this table disagree and V-scenes has to reconcile them (reported in
//! the PR, not silently fixed from this lane).
//!
//! # Music mapping
//!
//! * **kick** = `frame.onset > 0.5 && frame.bands[0] > 0.5`: a radial velocity impulse at a
//!   centre hashed from `frame.beat_index` (the web app's hash), plus a dye splash there.
//! * **bands** 0/1/2 (low/mid/high) inject dye at three positions (lower-left, centre,
//!   upper-right), drifting slowly with `t`; the colours come from a palette chosen by
//!   `params.palette % 8` and rotated by `p4`.
//! * **`beat_phase`** sets the swirl: strongest at the downbeat, gone by the next beat.
//! * **`frame.level`** (1..4) raises both force and dye (0.75..1.0 on the kick, 0.4..1.0 on the
//!   injectors, and the present gain).
//!
//! # `velocity()`
//!
//! The target holds RG = velocity in **pixels per draw** of the layers' internal size
//! (`velocity_uv_per_s * dt * size`), so a mosh shader can use it as a UV displacement almost
//! directly (`d += v * amount / size`). It is written at the solver's 256x144 grid rather than the
//! layer's size (V-budget): the field is smooth, every consumer samples it with normalised
//! coordinates, and the pass measured 0.15 ms per frame at 1080p plus 8 MB of targets at the
//! layer size, against 0.006 ms and 0.9 MB here. The units stay the internal size's pixels.
//! `None` until the first `draw`.
//!
//! # Known integration gaps this lane had to bridge (reported, not hidden)
//!
//! 1. `mosh-render`'s `Pipeline` names its fragment entry point `fs_main` and lays bind group 0
//!    out as `0 uniform, 1 image, 2 sampler, 3 image, ...`, while `mosh-shaders`' extracted
//!    modules (from the web app) name the entry point `fs` and lay it out as
//!    `0 uniform, 1 sampler, 2..5 images`. [`adapt_module`] rewrites those two things in the
//!    SPIR-V words, so the extracted WGSL stays the single source of truth. If N-render ever
//!    takes the entry-point name from the module, this adapter shrinks to nothing.
//! 2. `mosh-shaders::VISUALIZERS`'s `fluid` parameter names disagree with the table above.
//! 3. `mosh-render`'s 64-draw-per-frame budget is shared by every layer (see the cost note).

use mosh_core::{AnalysisFrame, ClipParams};
use mosh_render::{FrameRecorder, Gpu, LayerTarget, Pipeline, RenderError};

/// The solver grid: fixed, like `gpu.js` (`FL.w = 256, FL.h = 144`), independent of the layer
/// size so the cost of a fluid layer does not grow with the output resolution.
pub const GRID_WIDTH: u32 = 256;
pub const GRID_HEIGHT: u32 = 144;

/// Seconds per simulation step, as `fluidStep` in `gpu.js` writes it.
const SIM_DT: f32 = 1.0 / 30.0;

/// Pressure-solve iterations before clamping, `p6`.
const JACOBI_MIN: f32 = 8.0;
const JACOBI_MAX: f32 = 30.0;
const JACOBI_SPAN: f32 = JACOBI_MAX - JACOBI_MIN;

/// A kick is an onset with the low band up (`gpu.js`: `pulse > 0.7 && bass > 0.3`).
const KICK_ONSET: f32 = 0.5;
const KICK_LOW: f32 = 0.5;

/// Dye injected per step at band level 1, before `p1` and the level scale. Tuned against the
/// `p7 = 0` fade of 0.995/draw, so injection / (1 - fade) settles near 1 (the dye pass clamps at
/// 4). A quarter of this injected dye that a level-4 held note read back at mean luminance 0.007,
/// which is too sparse to see; this value reads 0.024 after one second.
const INJECT: f32 = 0.020;

/// The body force the forces pass applies when it is fed a zero motion field (`Q.gain`), as in
/// `gpu.js`.
const MOTION_GAIN: f32 = 1.6;

/// The flow pass clamps a component to this many pixels per draw, so a pathological field can
/// never turn into a giant displacement in a mosh shader.
const FLOW_LIMIT: f32 = 4096.0;

/// `FUB` in the seven extracted passes: 64 bytes (vec2 texel, dt, visc, vec4 impulse, then the
/// scalars). `tests` pin this against `mosh_shaders::uniform_bytes`.
const FUB_BYTES: usize = 64;
const DYE_BYTES: usize = 128;
const PRESENT_BYTES: usize = 48;
const FLOW_BYTES: usize = 48;

/// Hue triples per palette row (`params.palette % 8`), rotated by `p4` and `t`.
const PALETTES: [[f32; 3]; 8] = [
    [0.55, 0.62, 0.85],
    [0.02, 0.10, 0.55],
    [0.30, 0.45, 0.08],
    [0.80, 0.92, 0.55],
    [0.12, 0.55, 0.75],
    [0.47, 0.58, 0.66],
    [0.00, 0.33, 0.66],
    [0.58, 0.05, 0.35],
];

/// The seven velocity passes plus this lane's dye, present and flow passes.
struct Passes {
    advect: Pipeline,
    forces: Pipeline,
    curl: Pipeline,
    vorticity: Pipeline,
    divergence: Pipeline,
    jacobi: Pipeline,
    gradient: Pipeline,
    dye: Pipeline,
    present: Pipeline,
    flow: Pipeline,
}

impl Passes {
    fn new(gpu: &Gpu) -> Result<Passes, RenderError> {
        Ok(Passes {
            advect: fluid_pipeline(gpu, "fluid_advect", "fluid advect")?,
            forces: fluid_pipeline(gpu, "fluid_force", "fluid forces")?,
            curl: fluid_pipeline(gpu, "fluid_curl", "fluid curl")?,
            vorticity: fluid_pipeline(gpu, "fluid_vorticity", "fluid vorticity")?,
            divergence: fluid_pipeline(gpu, "fluid_divergence", "fluid divergence")?,
            jacobi: fluid_pipeline(gpu, "fluid_jacobi", "fluid jacobi")?,
            gradient: fluid_pipeline(gpu, "fluid_gradient", "fluid gradient")?,
            dye: pipeline(gpu, "fluidvis_dye", "fluid dye", DYE_BYTES, 2)?,
            present: pipeline(gpu, "fluidvis_present", "fluid present", PRESENT_BYTES, 1)?,
            flow: pipeline(gpu, "fluidvis_flow", "fluid flow", FLOW_BYTES, 1)?,
        })
    }
}

/// A pipeline from one of this lane's own `fluidvis_*` modules, which are written for
/// `mosh-render`'s bind group 0.
fn pipeline(
    gpu: &Gpu,
    module: &str,
    name: &str,
    uniform_bytes: usize,
    textures: u32,
) -> Result<Pipeline, RenderError> {
    let words = mosh_shaders::spirv(module).ok_or(RenderError::EmptyShader)?;
    Pipeline::fullscreen(gpu, words, uniform_bytes, textures).map(|built| built.with_name(name))
}

/// A pipeline from one of the seven extracted `fluid_*` modules, adapted to `mosh-render`'s
/// entry-point name and bind group 0 (see the module docs).
fn fluid_pipeline(gpu: &Gpu, module: &str, name: &str) -> Result<Pipeline, RenderError> {
    let words = mosh_shaders::spirv(module).ok_or(RenderError::EmptyShader)?;
    let adapted = adapt_module(words, &FUB_BINDINGS, FRAGMENT_ENTRY);
    Pipeline::fullscreen(gpu, &adapted, FUB_BYTES, 4).map(|built| built.with_name(name))
}

/// The fragment entry point `mosh-render` asks every fragment module for.
const FRAGMENT_ENTRY: &str = "fs_main";

/// Web bind group 0 -> `mosh-render` bind group 0: the web app's sampler sits at binding 1 and
/// its textures at 2..5; `mosh-render` samples images at 1, 3, 5, 7 and puts its immutable
/// samplers at 2, 4, 6, 8.
const FUB_BINDINGS: [(u32, u32); 5] = [(1, 2), (2, 1), (3, 3), (4, 5), (5, 7)];

/// Replaces the fragment entry point's name and remaps `Binding` decorations in a SPIR-V module.
///
/// SPIR-V instructions are `word_count << 16 | opcode` followed by their operands, and nothing
/// refers to an instruction by offset, so an `OpEntryPoint` may grow in place. Bindings are
/// `OpDecorate %x Binding <n>` literals, so those just change value.
///
/// `OpEntryPoint` is `ExecutionModel, EntryPoint, Name, Interface...`: the name is a null-
/// terminated string packed into words, and the interface ids after it must survive the rewrite.
fn adapt_module(words: &[u32], bindings: &[(u32, u32)], entry: &str) -> Vec<u32> {
    const OP_ENTRY_POINT: u16 = 15;
    const OP_DECORATE: u16 = 71;
    const DECORATION_BINDING: u32 = 33;
    const EXECUTION_MODEL_FRAGMENT: u32 = 4;

    let mut output = Vec::with_capacity(words.len() + 2);
    // The five-word SPIR-V header is not an instruction.
    let header = words.len().min(5);
    output.extend_from_slice(&words[..header]);
    let mut index = header;
    while index < words.len() {
        let count = (words[index] >> 16) as usize;
        let opcode = (words[index] & 0xffff) as u16;
        if count == 0 || index + count > words.len() {
            // A truncated stream is not something to guess about: keep the tail as it is and
            // let pipeline creation report it.
            output.extend_from_slice(&words[index..]);
            break;
        }
        let instruction = &words[index..index + count];
        match opcode {
            OP_ENTRY_POINT
                if count > 3 && instruction[1] == EXECUTION_MODEL_FRAGMENT =>
            {
                // The name ends at the first word that carries a null byte; everything after
                // that belongs to the interface list.
                let mut end = 3;
                while end < count && !instruction[end].to_le_bytes().contains(&0) {
                    end += 1;
                }
                let interface = if end < count {
                    &instruction[end + 1..count]
                } else {
                    &instruction[count..count]
                };

                let mut name = entry.as_bytes().to_vec();
                name.push(0);
                while !name.len().is_multiple_of(4) {
                    name.push(0);
                }
                let name_words = name.len() / 4;
                let total = 3 + name_words + interface.len();
                output.push(((total as u32) << 16) | u32::from(OP_ENTRY_POINT));
                output.push(instruction[1]);
                output.push(instruction[2]);
                for chunk in name.as_chunks::<4>().0 {
                    output.push(u32::from_le_bytes(*chunk));
                }
                output.extend_from_slice(interface);
            }
            OP_DECORATE if count == 4 && instruction[2] == DECORATION_BINDING => {
                let old = instruction[3];
                let new = bindings
                    .iter()
                    .find_map(|(from, to)| (*from == old).then_some(*to))
                    .unwrap_or(old);
                output.extend_from_slice(&[words[index], instruction[1], instruction[2], new]);
            }
            _ => output.extend_from_slice(instruction),
        }
        index += count;
    }
    output
}

pub struct FluidVisualizer {
    /// The velocity ping-pong: the front of `velocity_a` is the field after the pressure
    /// projection, which is the one the dye and the flow pass read.
    velocity_a: LayerTarget,
    velocity_b: LayerTarget,
    pressure_a: LayerTarget,
    pressure_b: LayerTarget,
    divergence: LayerTarget,
    curl: LayerTarget,
    /// The dye field, ping-ponged by its own feedback pass.
    dye: LayerTarget,
    /// A 1x1 transparent black image for the inputs a pass must bind but does not sample.
    zero: LayerTarget,
    /// The velocity field in pixels per draw at the layers' internal size.
    flow: LayerTarget,
    passes: Passes,
    width: u32,
    height: u32,
    started: bool,
}

impl FluidVisualizer {
    /// `width`/`height`: the layers' internal size. The dye is drawn into `dst` at this size and
    /// [`FluidVisualizer::velocity`] is expressed in its pixels; the solver itself stays at
    /// 256x144 whatever this is.
    pub fn new(gpu: &Gpu, width: u32, height: u32) -> Result<FluidVisualizer, RenderError> {
        Ok(FluidVisualizer {
            velocity_a: LayerTarget::new(gpu, GRID_WIDTH, GRID_HEIGHT),
            velocity_b: LayerTarget::new(gpu, GRID_WIDTH, GRID_HEIGHT),
            pressure_a: LayerTarget::new(gpu, GRID_WIDTH, GRID_HEIGHT),
            pressure_b: LayerTarget::new(gpu, GRID_WIDTH, GRID_HEIGHT),
            divergence: LayerTarget::new(gpu, GRID_WIDTH, GRID_HEIGHT),
            curl: LayerTarget::new(gpu, GRID_WIDTH, GRID_HEIGHT),
            dye: LayerTarget::new(gpu, GRID_WIDTH, GRID_HEIGHT),
            zero: LayerTarget::new(gpu, 1, 1),
            // The flow field is smooth and only ever sampled as a displacement, so it lives at the
            // solver's grid instead of the layer's internal size (V-budget: 0.15 ms per frame at
            // 1080p and 7.9 MB of targets became 0.02 ms and 0.9 MB).
            flow: LayerTarget::new(gpu, GRID_WIDTH, GRID_HEIGHT),
            passes: Passes::new(gpu)?,
            width,
            height,
            started: false,
        })
    }

    /// Step the simulation once and draw the dye into `dst`.
    pub fn draw(
        &mut self,
        rec: &mut FrameRecorder<'_>,
        dst: &LayerTarget,
        frame: &AnalysisFrame,
        params: &ClipParams,
        t: f32,
    ) -> Result<(), RenderError> {
        let plan = Plan::build(
            frame,
            params,
            t,
            [dst.width() as f32, dst.height() as f32],
            [self.width as f32, self.height as f32],
        );

        // 1. advect the velocity through itself (gpu.js: gAdv = velA -> velB).
        rec.draw(
            &self.passes.advect,
            &self.velocity_b,
            &plan.fluid,
            &[&self.velocity_a, &self.zero, &self.zero, &self.zero],
        )?;
        // 2. forces: kick impulse, swirl, viscosity (gFor = velB -> velA). The web app also feeds
        //    the block-motion field here; a fluid layer has none, so it gets the zero image.
        rec.draw(
            &self.passes.forces,
            &self.velocity_a,
            &plan.fluid,
            &[&self.velocity_b, &self.zero, &self.zero, &self.zero],
        )?;
        // 3. curl of the forced field.
        rec.draw(
            &self.passes.curl,
            &self.curl,
            &plan.fluid,
            &[&self.velocity_a, &self.zero, &self.zero, &self.zero],
        )?;
        // 4. vorticity confinement.
        rec.draw(
            &self.passes.vorticity,
            &self.velocity_b,
            &plan.fluid,
            &[&self.velocity_a, &self.curl, &self.zero, &self.zero],
        )?;
        // 5. divergence.
        rec.draw(
            &self.passes.divergence,
            &self.divergence,
            &plan.fluid,
            &[&self.velocity_b, &self.zero, &self.zero, &self.zero],
        )?;
        // 6. Jacobi pressure solve, alternating the two pressure images like `fluidStep` does.
        for iteration in 0..plan.iterations {
            let (read, written) = if iteration % 2 == 0 {
                (&self.pressure_a, &self.pressure_b)
            } else {
                (&self.pressure_b, &self.pressure_a)
            };
            rec.draw(
                &self.passes.jacobi,
                written,
                &plan.fluid,
                &[read, &self.divergence, &self.zero, &self.zero],
            )?;
        }
        // An even count ends on an odd-numbered iteration, which writes `pressure_a`; an odd
        // count ends on an even one, which writes `pressure_b`.
        let pressure = if plan.iterations.is_multiple_of(2) {
            &self.pressure_a
        } else {
            &self.pressure_b
        };
        // 7. subtract the pressure gradient: the final velocity is in `velocity_a`'s front.
        rec.draw(
            &self.passes.gradient,
            &self.velocity_a,
            &plan.fluid,
            &[&self.velocity_b, pressure, &self.zero, &self.zero],
        )?;
        // 8. the dye: advected by that velocity, injected by the music, faded. The second input is
        //    the dye target itself: a feedback pass reading what it wrote last step.
        rec.draw(
            &self.passes.dye,
            &self.dye,
            &plan.dye,
            &[&self.velocity_a, &self.dye],
        )?;
        // 9. the picture, and the field a mosh layer stirs with.
        rec.draw(&self.passes.present, dst, &plan.present, &[&self.dye])?;
        rec.draw(&self.passes.flow, &self.flow, &plan.flow, &[&self.velocity_a])?;

        self.started = true;
        Ok(())
    }

    /// The velocity field (RG, pixels per frame at the layers' internal size) after the last
    /// `draw`, for a mosh layer to stir with. `None` until the simulation has run.
    pub fn velocity(&self) -> Option<&LayerTarget> {
        self.started.then_some(&self.flow)
    }
}

/// One frame's uniform blocks and iteration count, computed on the CPU.
struct Plan {
    fluid: [u8; FUB_BYTES],
    dye: [u8; DYE_BYTES],
    present: [u8; PRESENT_BYTES],
    flow: [u8; FLOW_BYTES],
    iterations: u32,
}

impl Plan {
    /// `size`: the size of `dst` (the picture). `flow_size`: the layers' internal size, which
    /// `velocity()`'s field is expressed in.
    fn build(
        frame: &AnalysisFrame,
        params: &ClipParams,
        t: f32,
        size: [f32; 2],
        flow_size: [f32; 2],
    ) -> Plan {
        let p = &params.p;
        let param = |index: usize| p[index].clamp(0.0, 1.0);
        let level = frame.level.clamp(1, 4);
        let level01 = f32::from(level - 1) / 3.0;
        let low = frame.bands[0].clamp(0.0, 1.0);
        let mid = frame.bands[1].clamp(0.0, 1.0);
        let high = frame.bands[2].clamp(0.0, 1.0);
        let playing = if frame.playing { 1.0 } else { 0.0 };
        let phase = if frame.beat_phase.is_finite() {
            frame.beat_phase.clamp(0.0, 1.0)
        } else {
            0.0
        };

        let force = 0.5 + 1.5 * param(2);
        // The swirl is strongest on the downbeat and gone by the next one, and it is the same
        // term the forces pass uses for the bass (`Q.bass`), like the web app.
        let swirl = (1.0 - phase) * playing * (0.2 + 0.8 * param(3));
        let bass = low + swirl;

        let kick = frame.onset > KICK_ONSET && low > KICK_LOW;
        let (kick_x, kick_y, kick_force) = if kick {
            let beat = frame.beat_index as f32;
            (
                0.15 + 0.7 * hash01(beat * 12.9898),
                0.15 + 0.7 * hash01(beat * 78.233),
                1.6 * force * (0.5 + 0.5 * low) * (0.75 + 0.25 * level01),
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        // `FUB`, the extracted passes' uniform, in their field order: texel.xy, dt, visc,
        // impulse.xyzw (kick centre + strength), beat, bass (= low band + swirl), pulse, gain,
        // vort, then the three pad floats.
        let fluid = pack::<16, FUB_BYTES>([
            1.0 / GRID_WIDTH as f32,
            1.0 / GRID_HEIGHT as f32,
            SIM_DT,
            param(0),
            kick_x,
            kick_y,
            kick_force,
            0.0,
            frame.ppq as f32,
            bass,
            frame.onset.clamp(0.0, 1.0),
            MOTION_GAIN,
            0.2 + 1.2 * param(5),
            0.0,
            0.0,
            0.0,
        ]);

        // Dye: three injectors, one per band, drifting slowly so a held note still moves.
        let drift = (t * 0.11).sin() * 0.06;
        let position = |x: f32, y: f32| [x.clamp(0.08, 0.92), y.clamp(0.08, 0.92)];
        let [x0, y0] = position(0.28 + drift, 0.66 - drift * 0.5);
        let [x1, y1] = position(0.50 - drift * 0.7, 0.50 + drift * 0.6);
        let [x2, y2] = position(0.72 + drift * 0.4, 0.34 + drift * 0.8);

        let dye_amount = (0.25 + 0.75 * param(1)) * (0.4 + 0.6 * level01);
        let palette = PALETTES[usize::from(params.palette % 8)];
        let cycle = param(4) * 0.5 + t * 0.02;
        let value = 0.55 + 0.45 * level01;
        let colour = |hue: f32| hsv(hue + cycle, 0.85, value);
        let [r0, g0, b0] = colour(palette[0]);
        let [r1, g1, b1] = colour(palette[1]);
        let [r2, g2, b2] = colour(palette[2]);
        let kick_dye = if kick { 3.0 * INJECT * (0.5 + 0.5 * level01) } else { 0.0 };

        // `DU` for `fluidvis_dye`, in its field order: col0/col1/col2 (rgb + amount), kick
        // (xy centre, z amount), pos01, pos2 (xy position, z fade, w dt), timing (x swirl,
        // y cycle, z level, w t), texel.
        let dye = pack::<32, DYE_BYTES>([
            r0,
            g0,
            b0,
            low * dye_amount * INJECT,
            r1,
            g1,
            b1,
            mid * dye_amount * INJECT,
            r2,
            g2,
            b2,
            high * dye_amount * INJECT,
            kick_x,
            kick_y,
            kick_dye,
            0.0,
            x0,
            y0,
            x1,
            y1,
            x2,
            y2,
            1.0 - (0.005 + 0.020 * param(7)),
            SIM_DT,
            0.0,
            cycle,
            level01,
            t,
            1.0 / GRID_WIDTH as f32,
            1.0 / GRID_HEIGHT as f32,
            0.0,
            0.0,
        ]);

        let present = pack::<12, PRESENT_BYTES>([
            0.92,
            0.96,
            1.0,
            0.0,
            0.45,
            0.55,
            1.0 + 0.5 * level01,
            0.0,
            1.0 / size[0].max(1.0),
            1.0 / size[1].max(1.0),
            0.0,
            0.0,
        ]);

        let flow = pack::<12, FLOW_BYTES>([
            SIM_DT * flow_size[0].max(1.0),
            SIM_DT * flow_size[1].max(1.0),
            0.0,
            0.0,
            1.0 / flow_size[0].max(1.0),
            1.0 / flow_size[1].max(1.0),
            0.0,
            0.0,
            FLOW_LIMIT,
            0.0,
            0.0,
            0.0,
        ]);

        let iterations = (JACOBI_MIN + JACOBI_SPAN * param(6))
            .round()
            .clamp(JACOBI_MIN, JACOBI_MAX) as u32;

        Plan {
            fluid,
            dye,
            present,
            flow,
            iterations,
        }
    }
}

/// Serialises a uniform block: little-endian f32 in field order, `B` bytes exactly.
fn pack<const N: usize, const B: usize>(values: [f32; N]) -> [u8; B] {
    assert_eq!(B, N * 4, "uniform block size in bytes");
    let mut output = [0u8; B];
    for (index, value) in values.iter().enumerate() {
        output[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    output
}

/// The web app's beat hash (`(sin(b * k) * 43758.5453 mod 1 + 1) mod 1`), 0.5 for a NaN beat.
fn hash01(x: f32) -> f32 {
    if !x.is_finite() {
        return 0.5;
    }
    let value = x.sin() * 43_758.547;
    (value % 1.0 + 1.0) % 1.0
}

/// Hue 0..1, saturation/value 0..1 -> rgb, the same HSB the web app's accents use.
fn hsv(hue: f32, saturation: f32, value: f32) -> [f32; 3] {
    let h = hue.rem_euclid(1.0) * 6.0;
    let sector = h.floor();
    let f = h - sector;
    let p = value * (1.0 - saturation);
    let q = value * (1.0 - saturation * f);
    let u = value * (1.0 - saturation * (1.0 - f));
    match sector as i32 % 6 {
        0 => [value, u, p],
        1 => [q, value, p],
        2 => [p, value, u],
        3 => [p, q, value],
        4 => [u, p, value],
        _ => [value, p, q],
    }
}

#[cfg(test)]
mod tests;
