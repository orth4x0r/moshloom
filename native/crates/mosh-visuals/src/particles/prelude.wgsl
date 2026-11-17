// The particle layer's shared prelude: one uniform block and the helpers every particle pass
// uses. `particles.rs` compiles `this text + one pass body` per pipeline.
//
// The block is a flat row of `vec4<f32>`s, so its byte layout is exactly the field order
// `Plan::build` writes (twelve rows, 192 bytes, no padding to get wrong).

struct P {
  // layer width, height, trail width, height
  res: vec4f,
  // seconds this step, seconds since start, particles this frame, the state's capacity
  clock: vec4f,
  // glow radius (screen heights), speed scale, burst gain, swirl
  gains: vec4f,
  // trail decay per step, base hue, gravity, intensity
  look: vec4f,
  // low, mid, high, onset
  bands: vec4f,
  // level 0..1, playing, beat phase, beat index
  state: vec4f,
  // kick origin in screen units, kick gain, gated brightness burst
  kick: vec4f,
  // exposure, glitter flicker, second hue offset, vignette
  grade: vec4f,
  // tile columns, tile rows, candidate capacity, unused
  tiles: vec4f,
  // 1/layer w, 1/layer h, 1/trail w, 1/trail h
  texel: vec4f,
  // seed, aspect, flicker rate, the CPU's flux normalisation
  motion: vec4f,
  // home spring, drag, low-band push, high-band lift
  forces: vec4f,
};

@group(0) @binding(0) var<uniform> U: P;

const TAU: f32 = 6.2831853;
const MAX_COUNT: i32 = 512;

fn h1(n: f32) -> f32 { return fract(sin(n * 127.1) * 43758.5453); }

/// NaN -> `fallback` (`v == v` is false for a NaN), so no arithmetic on a bad input reaches a
/// target. Callers clamp to their own range on top.
fn safe(v: f32, fallback: f32) -> f32 { return select(fallback, v, v == v); }

/// A particle's own constants, deterministic in its index: the same values the step pass, the
/// table pass and the previews all derive, so no per-particle constant needs storing.
struct Seed {
  angle: f32,
  radius: f32,
  hue: f32,
  size: f32,
  life: f32,
};

fn seed_of(i: f32) -> Seed {
  return Seed(
    h1(i + 1.0) * TAU,
    h1(i + 7.0),
    h1(i + 13.0),
    h1(i + 19.0),
    2.5 + 5.5 * h1(i + 29.0),
  );
}

/// Where a particle rests: a disk around the centre, in screen units (centre 0, `x * aspect`,
/// `y` in -0.5..0.5, y down).
fn home_of(seed: Seed) -> vec2f {
  return vec2f(cos(seed.angle), sin(seed.angle)) * (0.02 + 0.22 * seed.radius);
}
