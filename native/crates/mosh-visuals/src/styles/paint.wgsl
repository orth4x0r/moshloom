// The style packs' shared paint: prepended to every body in `styles/cyber` and `styles/aero` with
// `concat!` in `styles/cyber.rs` and `styles/aero.rs`, so each scene still compiles as its own
// module. It began as the Y2K + Aero pack's `common.wgsl` and grew the chrome and grain the abstract
// packs need.
//
// Everything here is written down to the cost model `tests/styles_aero.rs` prints, because the
// budget is under 0.5 ms per scene at 960x540 on an integrated GPU and the empty pass already costs
// part of it. Measured on the reference machine (AMD Radeon iGPU), per full frame at 960x540: one
// sin/cos 0.013 ms, one exp or non-constant pow 0.012 ms, one `h2` hash 0.023 ms, one `fwidth`
// 0.02 ms, and a `0.5 + 0.5 * cos(TAU * (vec3f(0, .33, .67) + v))` palette 0.049 ms. So:
//
// * the palette is one sin and one cos, and the other two harmonies are component rotations of it
//   (the three cosine taps are exactly 1/3 turn apart), which is free;
// * glows are rational falloffs, not `exp`;
// * powers are multiplications, not `pow`;
// * anti-aliasing comes from the analytic gradient of the shape, not from `fwidth`;
// * a hash is spent only where glitter or film grain is the point.

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

// Three waves instead of a noise lattice: at cloud scale it reads the same, for three sines rather
// than four hashes per octave.
fn clouds(p: vec2f, t: f32) -> f32 {
  let a = sin(p.x * 1.7 + t * 0.05) * sin(p.y * 2.2 - t * 0.03);
  let b = sin(p.x * 3.9 - t * 0.08 + 1.3) * sin(p.y * 5.3 + 0.7);
  let c = sin((p.x + p.y) * 7.1 + t * 0.04);
  return 0.5 + 0.5 * (a * 0.62 + b * 0.28 + c * 0.10);
}

// --- the abstract packs -----------------------------------------------------

// Film grain as a multiplier around 1. One hash per call, no transcendental. The sample point is
// fixed per output pixel, so the grain sits on the sensor rather than crawling with the picture.
fn grain(pos: vec2f, amount: f32) -> f32 {
  return 1.0 + (h2(pos.xy + 0.37) - 0.5) * amount;
}

// A holographic ramp indexed by `v` turns: three channels a third of a turn apart, which is one
// vector cosine (three scalar cosines) and is what makes chrome read as iridescent foil rather than
// as tinted metal. Never reaches 1 on all three channels, so it cannot clip to white on its own.
fn irid(v: f32) -> vec3f {
  return 0.5 + 0.5 * cos(TAU * (vec3f(0.0, 0.3333, 0.6667) + v));
}

// The lit colour of a thick stroke. `d` is the distance to the stroke's centreline (so a body keeps the
// *centreline* field and thickens only for coverage: `1 - smoothstep(w - aa, w + aa, d)`), `g` that
// field's screen gradient, `w` the stroke's half width. The cross-section is a rod with a key light
// from above-left: the gradient's own sign says which side of the centreline a pixel is on, so `s` runs
// -1 (shadow side) to +1 (lit side) and the rod gets a lit shoulder, a specular band, a shadow side, a
// dark rim and one thin accent sliver. That asymmetry is what makes a line read as polished metal with
// weight instead of as a drawn stripe; the profile is glows and smoothsteps, no transcendental and no
// division, so width costs nothing.
fn chrome_ridge(d: f32, g: vec2f, w: f32, tint: vec3f, accent: vec3f, gloss: f32) -> vec3f {
  let t = clamp(d / max(w, 1e-5), 0.0, 1.0);
  let u = t * t * (3.0 - 2.0 * t);
  let gd = g / max(length(g), 1e-6);
  var sgn = -1.0;
  if (gd.x * -0.42 + gd.y * 0.78 > 0.0) {
    sgn = 1.0;
  }
  let s = u * sgn;
  let gl = clamp(gloss, 0.0, 1.0);
  // The body: dark metal, lifted on the lit side, driven down on the shadow side.
  var m = tint * 0.24;
  m = m + tint * 0.86 * smoothstep(-0.05, 0.80, s);
  m = m * (1.0 - 0.52 * smoothstep(-0.10, -0.90, s));
  // The specular band on the lit shoulder, its tightness the clip's gloss.
  m = m + vec3f(1.0, 0.99, 0.98) * mix(0.95, 0.50, gl) * glow(s - 0.58, mix(4.0, 9.0, gl));
  // The rim: a thin bright sliver on the edge, then the accent outside it.
  m = m + vec3f(0.92, 0.96, 1.0) * 0.35 * glow(abs(s) - 0.94, 16.0);
  m = m + accent * glow(abs(s) - 1.0, 26.0) * (0.55 + 0.45 * U.pulse);
  return m;
}

// The dark plate these packs sit on: a charcoal-to-plum gradient with two soft aura pools, so the
// chrome has something to reflect and the frame never reads as flat black. One sine, two rational
// falloffs.
fn plate(q: vec2f, tint: vec3f, t: f32, seed: f32) -> vec3f {
  let rise = smoothstep(-1.25, 1.25, -q.y);
  var c = mix(vec3f(0.020, 0.018, 0.030), vec3f(0.060, 0.036, 0.078), rise);
  let drift = 0.10 * sin(t * 0.13 + seed);
  // `glow`, not `falloff`: a pool this broad reads the same and a pass pays no division for it.
  c = c + tint * 0.16 * glow(length(q - vec2f(-0.58 + drift, 0.44)), 0.62);
  c = c + vec3f(0.055, 0.120, 0.170) * glow(length(q - vec2f(0.64, -0.52 - drift)), 0.85);
  c = c + vec3f(0.130, 0.095, 0.050) * glow(length(q - vec2f(0.30, 0.82)), 1.10);
  return c;
}

// A smooth three-sine field, 0..1: the melt every smear, ribbon and contour body drags around. Three
// sines, no hash.
fn melt_field(p: vec2f, t: f32) -> f32 {
  let a = sin(p.x * 1.6 + p.y * 1.1 + t * 0.31);
  let b = sin(p.x * 2.7 - p.y * 1.9 - t * 0.23 + 1.7);
  let c = sin((p.x + p.y) * 0.9 + t * 0.17 + 4.1);
  return 0.5 + 0.5 * (a * 0.5 + b * 0.30 + c * 0.20);
}

// A soft highlight shoulder: linear below 0.55, compressive above, asymptotic to 1 but never
// reaching it, so a specular can be as hot as it likes and still roll off instead of clipping to a
// flat white patch. One division per channel, and only the bright bodies call it.
fn shoulder(c: vec3f) -> vec3f {
  let k = vec3f(0.55);
  let x = max(c, vec3f(0.0));
  let over = max(x - k, vec3f(0.0));
  return min(x, k) + 0.45 * (over / (over + 0.45));
}

// A satin profile across a ribbon: 0 at both edges, a broad bright roll at 0.35 of the width and a
// darker fold on the other side. Used by the silk bodies; three smoothsteps, no transcendental.
fn satin(u: f32) -> f32 {
  let edge = smoothstep(-1.0, -0.55, u) * smoothstep(1.0, 0.60, u);
  let roll = smoothstep(-0.75, 0.10, u) * smoothstep(0.95, 0.30, u);
  let fold = smoothstep(0.55, -0.05, u);
  return edge * (0.28 + 0.85 * roll * roll) * (0.62 + 0.38 * fold);
}
