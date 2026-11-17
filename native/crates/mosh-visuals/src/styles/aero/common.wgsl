// Shared helpers for the Y2K + Frutiger Aero pack. Prepended to every body with `concat!` in
// `styles/aero.rs`, so each scene still compiles as its own module.
//
// Everything here is written down to the cost model `tests/styles_aero.rs` prints, because this
// pack's budget is under 0.5 ms per scene at 960x540 on an integrated GPU and the empty pass
// already costs part of it. Measured on the reference machine (AMD Radeon iGPU), per full frame at
// 960x540: one sin/cos 0.013 ms, one exp or non-constant pow 0.012 ms, one `h2` hash 0.023 ms, one
// `fwidth` 0.02 ms, and a `0.5 + 0.5 * cos(TAU * (vec3f(0, .33, .67) + v))` palette 0.049 ms. So:
//
// * the palette is one sin and one cos, and the other two harmonies are component rotations of it
//   (the three cosine taps are exactly 1/3 turn apart), which is free;
// * glows are rational falloffs, not `exp`;
// * powers are multiplications, not `pow`;
// * anti-aliasing comes from the analytic gradient of the shape, not from `fwidth`;
// * a hash is spent only where glitter or cloud texture is the point.

// The cosine palette from one angle: `c` and `s` are cos and sin of the hue angle. The second and
// third taps are the same angle 120 and 240 degrees on, so they are linear combinations with
// constants, not two more cos calls.
fn pal(c: f32, s: f32) -> vec3f {
  return 0.5 + 0.5 * vec3f(c, -0.5 * c - 0.8660254 * s, -0.5 * c + 0.8660254 * s);
}

// The same palette one and two thirds of a turn round: a permutation of the channels, so a scene
// gets three harmonised colours for the price of the one sin/cos pair.
fn pal_next(v: vec3f) -> vec3f {
  return vec3f(v.z, v.x, v.y);
}

fn next2(v: vec3f) -> vec3f {
  return vec3f(v.y, v.z, v.x);
}

// A glow that falls off without an `exp`: 1 at 0, half at 1/sqrt(k).
fn falloff(d: f32, k: f32) -> f32 {
  return 1.0 / (1.0 + k * d * d);
}

// The same shape without the reciprocal: 1 at 0, 0 at 1/k. For a soft band or a halo it reads the
// same, and a division is the most expensive thing a body can do per pixel here (the scenes whose
// cost did not move when the GPU clock halved were the ones full of them).
fn glow(d: f32, k: f32) -> f32 {
  let x = max(0.0, 1.0 - k * abs(d));
  return x * x;
}

fn sq(x: f32) -> f32 {
  return x * x;
}

// A line one pixel wide, from a distance in the shape's own units and that distance's screen
// gradient (computed analytically by the caller: see `aa` in each body).
fn hairline(d: f32, width: f32) -> f32 {
  return 1.0 - smoothstep(0.0, max(width, 1e-5), abs(d));
}

// Sparse four-point glints on a hashed grid: the Y2K glitter and the Aero fireflies. One `h2` call.
fn glints(p: vec2f, t: f32, scale: f32, threshold: f32) -> f32 {
  let grid = p * scale;
  let cell = floor(grid);
  let local = fract(grid) - 0.5;
  let h = h2(cell);
  let on = step(threshold, h) * (0.45 + 0.55 * sin(t * 2.1 + h * TAU));
  let core = smoothstep(0.26, 0.0, length(local));
  let arms = smoothstep(0.05, 0.0, abs(local.x)) * smoothstep(0.5, 0.0, abs(local.y))
    + smoothstep(0.05, 0.0, abs(local.y)) * smoothstep(0.5, 0.0, abs(local.x));
  return on * max(core, arms * 0.55);
}

// One out-of-focus highlight: a soft disc with a slightly brighter rim, brighter when the bar is
// loud. Broad and low, because a bokeh that reads as a ring looks like a drawn circle instead.
fn bokeh(q: vec2f, c: vec2f, r: f32, shade: vec3f, nrg: f32) -> vec3f {
  let d = length(q - c) / r;
  let soft = smoothstep(1.0, 0.45, d);
  let rim = smoothstep(0.55, 0.98, d) * smoothstep(1.06, 0.96, d);
  return shade * (soft * 0.07 + rim * 0.09) * (0.55 + 0.45 * nrg);
}

// The energy: bands and pulse in one number, 0..1.3. The bands are smoothed by the analyser, so
// anything that multiplies a phase by this moves smoothly; `pulse` only decays between beats.
fn energy() -> f32 {
  return clamp(0.5 * U.bass + 0.4 * U.mid + 0.25 * U.high + 0.2 * U.pulse, 0.0, 1.3);
}

// Three waves instead of a noise lattice: at cloud scale it reads the same, for three sines
// rather than four hashes per octave.
fn clouds(p: vec2f, t: f32) -> f32 {
  let a = sin(p.x * 1.7 + t * 0.05) * sin(p.y * 2.2 - t * 0.03);
  let b = sin(p.x * 3.9 - t * 0.08 + 1.3) * sin(p.y * 5.3 + 0.7);
  let c = sin((p.x + p.y) * 7.1 + t * 0.04);
  return 0.5 + 0.5 * (a * 0.62 + b * 0.28 + c * 0.10);
}
