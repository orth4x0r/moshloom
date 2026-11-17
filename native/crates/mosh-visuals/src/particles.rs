//! Particle layer: a GPU particle field driven by the music (kicks burst outward, bands steer and
//! colour, intensity sets density and speed, swirl on the beat, trails). Owner: L-particles.
//! Signatures are frozen.
//!
//! # What runs each `draw`
//!
//! `mosh-render` records fullscreen fragment passes only: no compute, no instancing, one 3-vertex
//! draw per pass, and a pass writes a *cleared* target while every input is read-only. So the
//! particle state cannot be a buffer an instance shader reads; it is a texture, and a particle is
//! one texel of it. One `draw` is five passes, all recorded into the caller's `FrameRecorder`:
//!
//! ```text
//! state (512 x 2 texels)              table (512 x 3 texels)             tiles (144 x 25)
//!   step      -> state  (feedback)      per-particle -> table              144 tile columns,
//!                                       (box, ends, radius+energy+colour)  candidate lists
//!                                                                              |
//!   trail (half the layer size, ping-pong):  gather every pixel's candidates, add their glow,
//!                                            decay last frame's trail  -> trail
//!   present: trail -> dst (up-scaled, graded, vignetted)
//! ```
//!
//! The state target keeps two rows: row 0 is position and velocity in **screen units** (centre at
//! 0, `x * aspect`, `y` in -0.5..0.5, y down), row 1 is life, age, hue phase and spawn phase. The
//! step pass reads the previous frame's rows and writes the next, so the simulation is one
//! feedback pass; both rows recompute the same deterministic spawn from (index, time), which is
//! why the two halves of the state can be written without reading each other's output.
//!
//! # The gather, and why it is bucketed
//!
//! A fullscreen pass cannot splat particles. It gathers instead: each pixel asks which particles
//! could reach it and adds their glow. Asking all 512 per pixel is what made the web app's swarm
//! cost 1.6-4.5 ms at 960x540 in its first generation (`scenes/swarm.wgsl`), so the same cure is
//! used here (`scenes/swarm_fast.wgsl` + `scenes/swarm_buckets.wgsl`): a 16x9 tile grid, and for
//! each tile the list of particles whose padded box overlaps it.
//!
//! Two departures from the swarm's version, both measured (see `particles/buckets.wgsl`):
//!
//! * each tile's scan is split into four *shards* of the particle range, one fragment per shard,
//!   because one fragment scanning all 158 particles took 0.27 ms on its own and sharding it took
//!   that to 0.08 ms;
//! * four indices are packed into each texel, so a shard's list is two texels rather than eight.
//!
//! A shard whose list overflows is not a lost particle: the gather scans that shard's slice of the
//! table with the box test instead, so a saturated tile still draws everything it holds.
//!
//! The picture the gather writes is half the layer size (`TRAIL_DIVISOR`): the glow is soft, the
//! present pass samples it bilinearly, and it costs a quarter of the pixels and a quarter of the
//! memory a full-size trail would (2.1 MB of trail targets per layer at 960x540, against 8.3 MB).
//!
//! # Trails
//!
//! The trail target is a ping-pong pair the draw pass both samples (last frame) and writes (this
//! frame): `trail = trail * decay + glow`, clamped, so a spark leaves a fading tail that is the
//! layer's own previous frame rather than a per-particle line. `p5` sets the decay
//! (0.84..0.96 per 1/60 s, converted to the real frame delta) and the light is scaled by
//! `1 - decay`, so a longer trail smears the field instead of saturating it. The table pass also
//! draws each particle as a short capsule from its position to where it was `1.6 * dt` ago: a spark
//! moves 20-50 pixels a frame in a busy clip, and a round point alone would leave a row of dots.
//!
//! # `p0..p7`
//!
//! All eight are 0..1 like every other clip parameter, and every one has a baseline so
//! `ClipParams::default()` (all zeros) draws a living field, not a black layer:
//!
//! | # | name | meaning |
//! | --- | --- | --- |
//! | `p0` | count | particles: `512 * (0.26 + 0.74 * p0) * (0.7 + 0.3 * intensity)` |
//! | `p1` | size | glow radius: `0.006 + 0.020 * p1` of the screen height, times a per-particle hash |
//! | `p2` | speed | drift and force scale: `0.6 + 1.6 * p2` |
//! | `p3` | burst | kick impulse and brightness: `0.5 + 1.5 * p3` |
//! | `p4` | swirl | vortex around the centre: `0.25 + 1.1 * p4`, strongest on the downbeat |
//! | `p5` | trail | trail decay per 1/60 s: `0.84 + 0.12 * p5` |
//! | `p6` | colour | hue rotation on top of the clip's palette |
//! | `p7` | gravity | downward pull: `0.05 + 0.30 * p7` |
//!
//! `params.palette % 8` picks the base hue (the same 8 rows the fluid layer's palette uses), and
//! `params.seed` perturbs every particle's hashes, so two clips of the same visualizer differ.
//!
//! # Music mapping
//!
//! * **kick** = `frame.onset > 0.5 && frame.bands[0] > 0.5`: a shockwave from a point hashed from
//!   `frame.beat_index` pushes every particle away from it, and a brightness burst rides on top
//!   (gated by [`FLASH_INTERVAL`]: see "Flash" below).
//! * **bands** 0/1/2 (low/mid/high): low pushes outward, mid twists, high lifts and swells the
//!   glow plus the per-particle twinkle (glitter). Band values also raise the colour's value.
//! * **`beat_phase`** makes the swirl strongest at the downbeat and gone by the next beat.
//! * **`frame.level`** (1..4) raises the particles' brightness and the present gain.
//! * **`frame.intensity`** scales the count and the speed, per the brief.
//!
//! # Flash
//!
//! The burst is a sparse field of glowing points, never a full-frame flash: the background stays
//! black and only the sparkles lift. It is still gated to one brightness burst per
//! [`FLASH_INTERVAL`] per layer ([`FlashGate`]), the same 400 ms floor the renderer's
//! `FlashLimiter` enforces on a full-frame flash, so a 16th-note kick pattern cannot strobe.
//!
//! # Cost
//!
//! Per layer per frame: 5 passes (the renderer allows 64 recorded passes per frame, shared with
//! every other layer). The state, table and tile passes are 1 024 / 1 536 / 1 728 texels; the
//! gather runs at half the layer size. At 960x540 with the default clip (130 particles) the layer
//! measured, offscreen on the integrated GPU:
//!
//! | pass | ms |
//! | --- | --- |
//! | sim | 0.040 |
//! | table | 0.027 |
//! | buckets | 0.070 |
//! | draw | 0.310 |
//! | present | 0.370 |
//! | **layer** | **0.88 median** |
//!
//! The present pass is a full-resolution picture pass and costs what any visualizer's does (the
//! fluid layer's present at 1280x720 is 0.70 ms); the gather is what is left of the budget. The
//! numbers are back-to-back runs of `tests/particles_budget.rs::gpu_ms_per_draw`, which asserts the
//! median under 1 ms in release: the default clip read 0.86-0.91 ms median over four debug runs
//! (best frame 0.81) and 0.83 ms in a release run straight after a cold start (best 0.41, GPU warm
//! from the run before). One earlier run with this iGPU in a different power state read 0.36 ms
//! for the same code, so the absolute numbers are machine-state dependent; the shape (present plus
//! gather, the rest is noise) is not.
//!
//! The budget is quoted at the default count, as the brief asks. `p0` past it spends the headroom:
//! a full-density clip (`p0 = 1`, 435 particles) measured 1.2-2.0 ms over the same runs, of which
//! the gather is 1.0-1.7. The ceiling is the state's capacity rather than the budget on purpose (an
//! artist can ask for a heavier field — it is the same shape, only denser); the default is a fifth
//! of it because that is what fits.
//!
//! # What this lane did not do (reported, not hidden)
//!
//! 1. One texel per particle is a fine simulation but a poor *splat*: a fragment pass cannot add
//!    to a target at a scattered position, so the picture is a gather over a candidate list. With
//!    vertex instancing (one quad per particle, additive blending) the same field would need no
//!    tile buckets and no 512-particle loop in any pixel, and the count could go to tens of
//!    thousands for the same cost. That is a `mosh-render` change (instanced draws with a storage
//!    or vertex buffer), not one this lane may make.
//! 2. Compute shaders would let the simulation run at a fixed 60 Hz independent of the frame rate
//!    instead of using the measured delta, and would make a real bucket sort (count, prefix sum,
//!    scatter) possible instead of the sharded scan and its overflow fallback.
//! 3. The layer has no image export of its own: a still frame is the present pass's output, which
//!    is what the tests write to `native/target/style-previews/particles/`.
//! 4. The five modules are compiled with naga when [`Particles::new`] runs: 177 ms in a debug build
//!    and 150 ms in release, on the render thread at start-up (the app already compiles its scenes
//!    the same way there). Moving them into `mosh-shaders/wgsl/` would make them build-time SPIR-V,
//!    and that tree is N-shaders'.
//!
//! # The shaders
//!
//! They live in `src/particles/*.wgsl` (not `mosh-shaders/wgsl/`, which is N-shaders' tree and is
//! generated from `gpu.js`): a shared prelude with the uniform block and the helpers, then one
//! body per pass, compiled once in [`Particles::new`] with `mosh_shaders::compile_wgsl`. A module
//! that does not compile is a defect in this crate and panics at construction with naga's message,
//! the way `Scenes::new` does, never mid-set.

use mosh_core::{AnalysisFrame, ClipParams};
use mosh_render::{FLASH_INTERVAL, FrameRecorder, Gpu, LayerTarget, Pipeline, RenderError};

/// What `p0..p7` mean for `particles` (shown in the clip editor).
pub const PARAMS: [&str; 8] = [
    "count", "size", "speed", "burst", "swirl", "trail", "colour", "gravity",
];

/// The particle capacity of the state: the most a clip's `p0` can ask for. The table and the
/// gather are sized from it, and the count an individual frame draws is `clock.z` in the uniform
/// block, so the unused tail costs three dead texels and nothing per pixel.
pub const MAX_PARTICLES: u32 = 512;

/// The candidate-list tile grid: 16 x 9 tiles over the half-size trail (30x30 pixels at 960x540),
/// and each tile's scan is split into `SHARDS` slices of the particle range so one fragment does
/// not scan every particle. A shard lists `PER_SHARD` indices, four per texel (a half float
/// carries an integer up to 2048 exactly), so a tile is `SHARDS * (1 + PER_SHARD / 4)` texels.
const TILES_X: u32 = 16;
const TILES_Y: u32 = 9;
const SHARDS: u32 = 4;
const PER_SHARD: u32 = 8;
/// Rows per tile: a count and a list per shard.
const TILE_ROWS: u32 = SHARDS * (1 + PER_SHARD / 4);

/// The trail, and the gather that writes it, run at the layer size divided by this. The glow is
/// soft and the present pass samples it linearly, so half size is invisible in the picture and a
/// quarter of the pixels and memory.
const TRAIL_DIVISOR: u32 = 2;

/// The uniform block every particle pass shares: twelve `vec4<f32>` rows, in the field order
/// [`Plan::build`] writes. Rows instead of scalars so the layout has no padding to get wrong.
const UNIFORM_ROWS: usize = 12;
const UNIFORM_BYTES: usize = UNIFORM_ROWS * 16;

/// Step length when a caller gives no usable delta (the first frame of a layer).
const DEFAULT_DT: f32 = 1.0 / 60.0;
/// A long frame (a stall, a debugger) must not teleport the field.
const DT_MIN: f32 = 1.0 / 240.0;
const DT_MAX: f32 = 1.0 / 20.0;

/// A kick is an onset with the low band up, the same test the fluid layer uses.
const KICK_ONSET: f32 = 0.5;
const KICK_LOW: f32 = 0.5;

/// The eight base hues, one per palette row, as the fluid layer's `PALETTES` uses eight rows.
const PALETTE_HUES: [f32; 8] = [0.55, 0.02, 0.30, 0.80, 0.12, 0.47, 0.00, 0.58];

/// The shared prelude every particle pass is compiled on top of.
const PRELUDE: &str = include_str!("particles/prelude.wgsl");
const SIM: &str = include_str!("particles/sim.wgsl");
const TABLE: &str = include_str!("particles/table.wgsl");
const BUCKETS: &str = include_str!("particles/buckets.wgsl");
const DRAW: &str = include_str!("particles/draw.wgsl");
const PRESENT: &str = include_str!("particles/present.wgsl");

/// The five passes of a particle layer.
struct Passes {
    sim: Pipeline,
    table: Pipeline,
    buckets: Pipeline,
    draw: Pipeline,
    present: Pipeline,
}

impl Passes {
    fn new(gpu: &Gpu) -> Result<Passes, RenderError> {
        Ok(Passes {
            sim: build(gpu, SIM, "particles.sim", 1)?,
            table: build(gpu, TABLE, "particles.table", 1)?,
            buckets: build(gpu, BUCKETS, "particles.buckets", 1)?,
            draw: build(gpu, DRAW, "particles.draw", 3)?,
            present: build(gpu, PRESENT, "particles.present", 1)?,
        })
    }
}

/// Compiles the prelude plus one pass body and builds its pipeline.
fn build(gpu: &Gpu, body: &str, name: &str, textures: u32) -> Result<Pipeline, RenderError> {
    let source = format!("{PRELUDE}\n{body}");
    let words = mosh_shaders::compile_wgsl(&source)
        .unwrap_or_else(|error| panic!("particle pass {name} does not compile: {error}"));
    Pipeline::fullscreen(gpu, &words, UNIFORM_BYTES, textures).map(|built| built.with_name(name))
}

/// One layer's GPU state and its clock.
struct LayerState {
    /// `MAX_PARTICLES` x 2: row 0 position + velocity, row 1 life + age + hue + spawn phase.
    state: LayerTarget,
    /// `MAX_PARTICLES` x 3: row 0 the capsule's box, row 1 head + tail, row 2 radius + energy + hue.
    table: LayerTarget,
    /// `TILES_X * TILES_Y` x `TILE_ROWS`: for each of the tile's shards, a count row and then four
    /// candidate indices per texel.
    tiles: LayerTarget,
    /// The picture before the present pass, at `TRAIL_DIVISOR` of the layer size.
    trail: LayerTarget,
    /// The `t` of the previous step, for the step length.
    last_t: Option<f32>,
    /// The 400 ms floor between two brightness bursts.
    flash: FlashGate,
}

impl LayerState {
    fn new(gpu: &Gpu, size: (u32, u32)) -> LayerState {
        let trail = (
            (size.0 / TRAIL_DIVISOR).max(1),
            (size.1 / TRAIL_DIVISOR).max(1),
        );
        LayerState {
            state: LayerTarget::new(gpu, MAX_PARTICLES, 2),
            table: LayerTarget::new(gpu, MAX_PARTICLES, 3),
            tiles: LayerTarget::new(gpu, TILES_X * TILES_Y, TILE_ROWS),
            trail: LayerTarget::new(gpu, trail.0, trail.1),
            last_t: None,
            flash: FlashGate::default(),
        }
    }
}

/// The 400 ms floor between two brightness bursts, per layer.
///
/// It is the renderer's own limit for a full-frame flash ([`FLASH_INTERVAL`]) applied to the
/// particle field's burst: a kick half a beat apart at 180 BPM is 333 ms, which would strobe.
#[derive(Default)]
struct FlashGate {
    last: Option<f32>,
}

impl FlashGate {
    /// True when a burst may start at `t`, and records it. A clock that steps backwards does not
    /// unlock a burst.
    fn allow(&mut self, t: f32) -> bool {
        if !t.is_finite() {
            return false;
        }
        let floor = FLASH_INTERVAL.as_secs_f32();
        match self.last {
            Some(last) if t - last < floor => false,
            _ => {
                self.last = Some(t);
                true
            }
        }
    }
}

pub struct Particles {
    /// Per-layer state, created on a layer's first `draw` and kept for the show's lifetime.
    layers: Vec<LayerState>,
    passes: Passes,
    /// The layers' internal size, from [`Particles::new`]: the trail's size and the aspect every
    /// pass uses. `dst` may be a different size (a thumbnail); the present pass up-scales.
    width: u32,
    height: u32,
}

impl Particles {
    /// `width`/`height`: the layers' internal size.
    pub fn new(gpu: &Gpu, width: u32, height: u32) -> Result<Particles, RenderError> {
        Ok(Particles {
            layers: Vec::new(),
            passes: Passes::new(gpu)?,
            width: width.max(1),
            height: height.max(1),
        })
    }

    /// Step `layer`'s particles once and draw them into `dst`.
    pub fn draw(
        &mut self,
        rec: &mut FrameRecorder<'_>,
        layer: usize,
        dst: &LayerTarget,
        frame: &AnalysisFrame,
        params: &ClipParams,
        t: f32,
    ) -> Result<(), RenderError> {
        let gpu = rec.gpu();
        let size = (self.width, self.height);
        while self.layers.len() <= layer {
            self.layers.push(LayerState::new(gpu, size));
        }

        let state = &mut self.layers[layer];
        let dt = step_dt(state.last_t, t);
        state.last_t = Some(t);
        let kick = is_kick(frame);
        let flash = kick && state.flash.allow(t);
        let uniforms = Plan::build(frame, params, t, dt, kick, flash, size);

        // 1. the step: one feedback pass over the state's two rows.
        rec.draw(&self.passes.sim, &state.state, &uniforms, &[&state.state])?;
        // 2. the per-particle table: box, ends, radius + energy + colour.
        rec.draw(&self.passes.table, &state.table, &uniforms, &[&state.state])?;
        // 3. which particles can reach which tile.
        rec.draw(
            &self.passes.buckets,
            &state.tiles,
            &uniforms,
            &[&state.table],
        )?;
        // 4. the glow: every pixel gathers its tile's candidates and the trail decays. The inputs
        //    are the table (binding 1), the buckets (3) and the trail being written (5).
        rec.draw(
            &self.passes.draw,
            &state.trail,
            &uniforms,
            &[&state.table, &state.tiles, &state.trail],
        )?;
        // 5. the picture.
        rec.draw(&self.passes.present, dst, &uniforms, &[&state.trail])
    }
}

/// The step length for a layer: the difference between this frame's `t` and the last one, clamped.
fn step_dt(last: Option<f32>, t: f32) -> f32 {
    match last {
        Some(last) if t.is_finite() && last.is_finite() && t > last => {
            (t - last).clamp(DT_MIN, DT_MAX)
        }
        _ => DEFAULT_DT,
    }
}

/// A kick is an onset with the low band up.
fn is_kick(frame: &AnalysisFrame) -> bool {
    unit(frame.onset) > KICK_ONSET && unit(frame.bands[0]) > KICK_LOW
}

/// 0..1 from an analysis value that may be outside the range (or not a number).
fn unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// One frame's uniform block, built on the CPU.
struct Plan;

impl Plan {
    /// The twelve rows `particles/prelude.wgsl` declares, in its field order.
    ///
    /// `kick` is the shockwave and brightness burst of a kick frame; `flash` says the burst
    /// passed the 400 ms gate this frame. The two are separate on purpose: the field always
    /// reacts to a kick, only the *brightness* lift is rate-limited.
    fn build(
        frame: &AnalysisFrame,
        params: &ClipParams,
        t: f32,
        dt: f32,
        kick: bool,
        flash: bool,
        size: (u32, u32),
    ) -> [u8; UNIFORM_BYTES] {
        let p = &params.p;
        let param = |index: usize| unit(p[index]);
        let level = frame.level.clamp(1, 4);
        let level01 = f32::from(level - 1) / 3.0;
        let low = unit(frame.bands[0]);
        let mid = unit(frame.bands[1]);
        let high = unit(frame.bands[2]);
        let onset = unit(frame.onset);
        let intensity = unit(frame.intensity);
        let playing = if frame.playing { 1.0 } else { 0.0 };
        let phase = unit(frame.beat_phase);
        let t = if t.is_finite() { t } else { 0.0 };

        let width = size.0.max(1) as f32;
        let height = size.1.max(1) as f32;
        let trail = (
            (size.0 / TRAIL_DIVISOR).max(1) as f32,
            (size.1 / TRAIL_DIVISOR).max(1) as f32,
        );

        // Count: p0, damped by intensity, never more than the state's capacity. The baseline is
        // what the 1 ms budget allows at 960x540 (see `tests/particles_budget.rs`).
        let wanted = MAX_PARTICLES as f32 * (0.26 + 0.74 * param(0)) * (0.7 + 0.3 * intensity);
        let count = (wanted.round() as u32).clamp(1, MAX_PARTICLES) as f32;

        let size_r = 0.006 + 0.020 * param(1);
        let speed = (0.6 + 1.6 * param(2)) * (0.75 + 0.35 * intensity) * (0.55 + 0.45 * playing);
        let burst = 0.5 + 1.5 * param(3);
        // The swirl is strongest on the downbeat and gone by the next beat.
        let swirl = (0.25 + 1.1 * param(4)) * (0.45 + 0.55 * (1.0 - phase)) * (0.4 + 0.6 * playing);
        // Trail decay is authored per 1/60 s, then converted to this frame's delta. The baseline
        // is long enough that consecutive frames' sparks overlap into a smear rather than a row
        // of dots at the default speed.
        let decay60 = 0.84 + 0.12 * param(5);
        let decay = decay60.powf(dt * 60.0).clamp(0.0, 0.995);
        let palette = PALETTE_HUES[usize::from(params.palette % 8)];
        let hue = (palette + param(6) + 0.03 * unit(frame.brightness)).rem_euclid(1.0);
        let gravity = 0.05 + 0.30 * param(7);

        // The kick's origin, in screen units, hashed from the beat so two kicks differ.
        let beat = frame.beat_index as f32;
        let kick_origin = [
            (hash01(beat * 12.9898) * 2.0 - 1.0) * 0.55 * (width / height),
            (hash01(beat * 78.233) * 2.0 - 1.0) * 0.45,
        ];
        let kick_gain = if kick {
            burst * (0.6 + 0.4 * low) * (0.8 + 0.2 * level01)
        } else {
            0.0
        };

        // Light flux: a denser or larger field must not turn into a white disc just because more
        // particles overlap. The per-particle energy is scaled so `count * radius^2` stays near the
        // default clip's, keeping the total light in the same range across `p0` and `p1`.
        let reference = 40.0;
        let flux = count * (size_r * 1.15 / 0.012).powi(2);
        let energy_scale = (reference / flux.max(1.0)).sqrt().clamp(0.12, 1.5);

        // `grade.y` is the glitter flicker, `grade.z` the second hue's offset, `grade.w` the
        // vignette; `forces.x` the home spring and `forces.y` the drag.
        pack([
            width,
            height,
            trail.0,
            trail.1,
            dt,
            t,
            count,
            MAX_PARTICLES as f32,
            size_r,
            speed,
            burst,
            swirl,
            decay,
            hue,
            gravity,
            intensity,
            low,
            mid,
            high,
            onset,
            level01,
            playing,
            phase,
            beat,
            kick_origin[0],
            kick_origin[1],
            kick_gain,
            if flash { 1.0 } else { 0.0 },
            1.0 + 0.5 * level01,
            0.35 + 1.1 * high,
            0.35,
            0.40 + 0.25 * intensity,
            f32::from(TILES_X as u16),
            f32::from(TILES_Y as u16),
            f32::from(PER_SHARD as u16),
            0.0,
            1.0 / width,
            1.0 / height,
            1.0 / trail.0,
            1.0 / trail.1,
            params.seed as f32,
            width / height,
            12.0,
            energy_scale,
            1.05,
            1.30,
            1.4 + 1.6 * low,
            0.9 + 1.1 * high,
        ])
    }
}

/// Serialises the uniform block: little-endian f32 in field order, 16 bytes per row.
fn pack(values: [f32; UNIFORM_ROWS * 4]) -> [u8; UNIFORM_BYTES] {
    let mut output = [0u8; UNIFORM_BYTES];
    for (index, value) in values.iter().enumerate() {
        let value = if value.is_finite() { *value } else { 0.0 };
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

#[cfg(test)]
mod preview;
#[cfg(test)]
mod tests;
