// Sigil Glyph: a rune column that redraws itself once a bar, in chrome ink on black.
//
// The rune is implicit: two wavy strokes run down the figure at different frequencies and phases,
// a stem, a diamond gate and a column of stipple dots finish it, and a dashed ring and a red
// sunburst behind it are the tattoo flash's paper. Every bar the waves change frequency, phase and
// offset from the bar's hash, crossfaded across the bar, so the glyph visibly rewrites itself
// instead of cutting.
//
// Audio: the waves' amplitude is one mel band each, the stem swells with `U.bass`, `U.pulse` flares
// the sunburst and shifts the stipple. `U.dens` adds a stroke, `U.vari` shifts which bands feed
// which stroke.
//
// P(0) clock rate, P(1) density, P(2) hue, P(3) curl, P(4) stroke width, P(5) flash, P(6) ring,
// P(7) grain.

const RED = vec3f(0.74, 0.035, 0.060);
const ICE = vec3f(0.62, 0.82, 1.0);
const STEEL = vec3f(0.58, 0.64, 0.74);

/// One wavy stroke of the rune: a vertical wave whose amplitude and phase the bar's hash picks,
/// crossfaded from the bar's start to the next bar.
fn rune_stroke(
  p: vec2f, y0: f32, phase0: f32, phase1: f32, f: f32, freq: f32, amp: f32, w: f32
) -> f32 {
  let ph = mix(phase0, phase1, f);
  let x = 0.045 + amp * sin(p.y * freq + ph) * (0.35 + 0.65 * smoothstep(-0.95, 0.95, p.y)) + y0;
  return abs(p.x - x) - w;
}

/// Chrome from the field's screen normal: a bright ceiling, a dark floor, one strip light, a rim.
fn chrome(nrm: vec3f, pulse: f32) -> vec3f {
  let up = clamp(nrm.y * 0.5 + 0.5, 0.0, 1.0);
  let up2 = up * up * (3.0 - 2.0 * up);
  var env = mix(vec3f(0.16, 0.19, 0.26), vec3f(0.94, 0.98, 1.0), up2);
  let strip = clamp(1.0 - abs(nrm.x) * 1.8, 0.0, 1.0) * clamp(nrm.y * 1.4 + 0.35, 0.0, 1.0);
  env += vec3f(0.95, 0.98, 1.0) * strip * 0.85;
  let rim = 1.0 - clamp(nrm.z, 0.0, 1.0);
  let rim2 = rim * rim;
  var metal = STEEL * (0.20 + env * 1.40);
  metal += ICE * rim2 * 0.45 + RED * rim2 * (0.12 + 0.60 * pulse);
  return metal;
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let px = 2.0 / U.res.y;
  let aa = px * 1.15;
  let e = px * 2.2;
  let dens = clamp(U.dens, 0.0, 1.0);
  let p = vec2f(abs(q.x), q.y);
  let r = length(q);
  let a = atan2(q.y, q.x);

  // The bar the rune is rewriting itself for, and where in that rewrite we are.
  let bar = U.beat / 4.0;
  let k = floor(bar);
  let f = smoothstep(0.0, 1.0, fract(bar));
  let grow = (0.85 + 0.35 * dens + 0.30 * P(4)) * (1.0 + 0.22 * clamp(U.high, 0.0, 1.0) + 0.10 * U.pulse);
  let w = (0.0075 + 0.006 * P(4)) * grow;

  var d = 1e6;
  // The stem: a plain tapered line, the rune's spine.
  d = min(d, abs(p.x - 0.020) - (0.005 + 0.006 * grow) * (1.0 + 0.25 * clamp(U.bass, 0.0, 1.0)));
  // Two wavy strokes, each answering to its own mel band and its own bar-to-bar rewrite.
  let band0 = mel(i32(3.0 + U.vari * 2.0));
  let band1 = mel(i32(11.0 + U.vari * 2.0));
  d = min(d, rune_stroke(
    p, 0.10, h1(k * 7.31 + 1.0) * 6.0, h1((k + 1.0) * 7.31 + 1.0) * 6.0, f,
    14.0 + 9.0 * h1(k * 3.17), 0.022 + 0.052 * band0, w
  ));
  d = min(d, rune_stroke(
    p, 0.30, h1(k * 5.13 + 2.0) * 6.0, h1((k + 1.0) * 5.13 + 2.0) * 6.0, f,
    19.0 + 11.0 * h1(k * 9.71), 0.016 + 0.040 * band1, w * 0.75
  ));
  if (dens > 0.55) {
    d = min(d, rune_stroke(
      p, 0.52, h1(k * 2.71 + 3.0) * 6.0, h1((k + 1.0) * 2.71 + 3.0) * 6.0, f,
    16.0 + 10.0 * h1(k * 4.43), 0.012 + 0.034 * band0, w * 0.6
    ));
  }

  // The rungs: two short crossbars, so the rune reads as a character and not a ribbon.
  d = min(d, max(abs(p.y - 0.24) - 0.006, (abs(p.x) - 0.34) * 0.5));
  d = min(d, max(abs(p.y + 0.18) - 0.006, (abs(p.x) - 0.28) * 0.5));
  // The gate: a diamond ring around the rune's waist.
  d = min(d, abs(abs(p.x) * 1.25 + abs(p.y - 0.04) * 0.85 - 0.19) - 0.0050);
  // The ring: a dashed circle, the flash sheet's frame.
  let dash = step(0.32, fract(a * 9.0 / TAU + 0.10));
  d = min(d, max(abs(r - 1.16) - 0.0030, (1.0 - dash) * 0.05));
  // Stipple: a column of dots inside the ring.
  let row = (p.y + 1.1) * 16.0;
  let dot = vec2f(
    p.x - (0.44 + 0.14 * h1(floor(row) * 5.1 + floor(U.T * 0.4))),
    (fract(row) - 0.5) / 16.0
  );
  d = min(d, length(dot) - 0.0028);
  let g = vec2f(dpdx(d), dpdy(d)) * (e / px);

  // --- the flash sheet: black paper, a red sunburst behind the glyph, grain, a vignette.
  let rays = smoothstep(0.42, 0.0, abs(fract((a + U.T * (0.05 + 0.10 * P(0))) * 9.0 / TAU + 0.5) - 0.5) * 2.0);
  var c = vec3f(0.008, 0.008, 0.012) + vec3f(0.022, 0.026, 0.038) * exp(-r * 1.6);
  c += RED * (0.020 + 0.055 * P(5)) * rays * smoothstep(1.25, 0.05, r) * (0.35 + 1.00 * U.pulse * U.pulse);
  c += RED * 0.045 * exp(-r * 2.2) * (0.30 + 1.00 * U.pulse * U.pulse);
  c += ICE * 0.012 * exp(-r * 2.6) * (0.30 + 0.70 * U.high);
  c *= 0.92 + 0.16 * h2(pos.xy * 0.75 + U.frame * 0.7) * (0.3 + 0.7 * P(7));

  // --- the ink. The chrome is only sampled near a stroke, which is where it is used.
  if (d < px * 9.0) {
    let cov = 1.0 - smoothstep(-aa, aa, d);
    let nrm = normalize(vec3f(-g, e * 0.60));
    var metal = chrome(nrm, U.pulse);
    metal *= 0.76 + 0.24 * smoothstep(0.0, -0.010, d);
    c = mix(c, metal, cov);
    c += STEEL * 0.10 * exp(-max(d, 0.0) * 70.0) * (1.0 - cov);
  }
  c += RED * 0.10 * exp(-abs(d) * 120.0) * U.pulse * U.pulse;
  // Hue: a channel rotation on the finished frame, so P(2) shifts the whole plate from blood red
  // toward violet without touching the chrome.
  c *= mix(vec3f(1.0), vec3f(0.88, 0.58, 1.28), U.hue);
  return fin(c);
}
