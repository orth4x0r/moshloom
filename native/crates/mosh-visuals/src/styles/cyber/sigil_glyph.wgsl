// Sigil Glyph: a rune column that rewrites itself once a bar, cast in chrome on a grainy plate.
//
// The rune is implicit: three wavy strokes run down the figure at different frequencies and phases, a
// stem, two rungs, a diamond gate and a column of stipple beads finish it, and a dashed rectangular
// frame is the flash sheet's border. Every bar the waves change frequency and phase from one hash,
// crossfaded across the bar, so the glyph visibly rewrites itself instead of cutting. Every stroke is lit
// as a chrome rod the full width of its line (`chrome_ridge` in `styles/paint.wgsl`): a bright shoulder,
// a specular band, a shadowed side and an iridescent rim.
//
// The frame is a box and the light is a diagonal shaft, not a sunburst: an angle would cost an `atan2`
// per pixel, and this scene's budget goes on the rune's strokes instead.
//
// Audio: the waves' amplitude is one mel band each, the stem swells with `U.bass`, `U.pulse` widens the
// stroke, swings the iridescent rim and lifts the shaft. Nothing strobes inside the 400 ms limit.
//
// P(0) clock rate, P(1) density, P(2) hue, P(3) curl, P(4) stroke weight, P(5) flash, P(6) frame,
// P(7) grain.

const STEEL = vec3f(0.88, 0.91, 0.97);
const RED = vec3f(0.82, 0.045, 0.070);

/// One wavy stroke of the rune: a vertical wave whose amplitude and phase the bar's hash picks,
/// crossfaded from the bar's start to the next bar.
fn rune_stroke(p: vec2f, y0: f32, phase0: f32, phase1: f32, f: f32, freq: f32, amp: f32) -> f32 {
  let ph = mix(phase0, phase1, f);
  let x = 0.040 + amp * sin(p.y * freq + ph) * (0.32 + 0.68 * smoothstep(-0.95, 0.95, p.y)) + y0;
  return abs(p.x - x);
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let px = 2.0 / U.res.y;
  let aa = px * 1.10;
  let dens = clamp(U.dens, 0.0, 1.0);
  let p = vec2f(abs(q.x), q.y);
  let t = U.T;

  // The bar the rune is rewriting itself for, and where in that rewrite we are.
  let bar = U.beat / 4.0;
  let k = floor(bar);
  let f = smoothstep(0.0, 1.0, fract(bar));
  let grow = 0.85 + 0.30 * dens + 0.22 * P(4);
  // One hash for the bar, its other values derived by multiplication: a hash is a `sin`, and the rune
  // needs six of them otherwise.
  let ks = h1(k * 7.31 + 1.0);
  let k1 = fract(ks * 17.13);
  let k2 = fract(ks * 41.71);
  let band0 = mel(i32(3.0 + U.vari * 2.0));
  let band1 = mel(i32(11.0 + U.vari * 2.0));

  var d = 1e6;
  // The stem: the rune's spine, and the heaviest line in the figure.
  d = min(d, abs(p.x - 0.018));
  // Three wavy strokes, each answering to its own mel band and its own bar-to-bar rewrite.
  d = min(d, rune_stroke(
    p, 0.10, ks * 6.0, fract(ks * 11.7) * 6.0, f,
    12.0 + 8.0 * k1, 0.020 + 0.050 * band0
  ));
  d = min(d, rune_stroke(
    p, 0.28, k1 * 6.0, fract(k1 * 9.3) * 6.0, f,
    17.0 + 10.0 * k2, 0.014 + 0.038 * band1
  ));
  if (dens > 0.5) {
    d = min(d, rune_stroke(
      p, 0.50, k2 * 6.0, fract(k2 * 7.9) * 6.0, f,
      15.0 + 9.0 * (k1 * 0.5 + k2 * 0.5), 0.010 + 0.032 * band0
    ));
  }
  // The rungs: two short crossbars, so the rune reads as a character and not a ribbon.
  d = min(d, max(abs(p.y - 0.22) - 0.006, (abs(p.x) - 0.30) * 0.5));
  d = min(d, max(abs(p.y + 0.16) - 0.006, (abs(p.x) - 0.25) * 0.5));
  // The gate: a diamond at the rune's waist.
  d = min(d, abs(abs(p.x) * 1.25 + abs(p.y - 0.03) * 0.85 - 0.17));
  // The frame: a dashed box around the sheet. `Frame` walks it out past the glyph's feet and back in
  // around the stem, and the dashes run along the edge each pixel is nearest.
  let inset = 0.98 - 0.26 * P(6);
  let ex = inset * 1.34;
  let frame = max(abs(q.x) - ex, abs(q.y) - inset);
  let near_side = step(abs(q.y) - inset, abs(q.x) - ex);
  let dash = mix(
    step(0.34, fract(q.x * 4.2 + 0.35)),
    step(0.34, fract(q.y * 4.2 + 0.10)),
    near_side
  );
  d = min(d, max(abs(frame), (1.0 - dash) * 0.06));
  let g = vec2f(dpdx(d), dpdy(d));
  // The stroke weight breathes with the low end: the glyph gains weight on a heavy bar.
  let w = (0.0190 + 0.0055 * dens + 0.0090 * P(4))
    * (1.0 + 0.10 * U.pulse + 0.08 * clamp(U.bass, 0.0, 1.0));

  // --- the flash sheet: a grainy plate, a red shaft of light behind the glyph, a cold top light.
  let shaft = glow(q.x * 0.72 - q.y * 0.62 - (0.25 + 0.55 * P(0) * t) * 0.10, 1.6);
  let hue_ang = TAU * P(2);
  let hue_tint = pal(cos(hue_ang), sin(hue_ang));
  var c = plate(q, hue_tint * 0.36 + vec3f(0.09, 0.07, 0.10), t, 2.6);
  c += RED * (0.045 + 0.095 * P(5)) * shaft * (0.35 + 1.05 * U.pulse * U.pulse);
  c += vec3f(0.10, 0.16, 0.24) * 0.14 * glow(q.y - 0.90, 1.5) * (0.4 + 0.6 * U.high);

  // --- the metal.
  let cov = 1.0 - smoothstep(w - aa, w + aa, d);
  if (d < w + px * 2.5) {
    let accent = mix(RED, hue_tint, 0.40) * (0.75 + 0.55 * P(5));
    let metal = chrome_ridge(d, g, w, STEEL, accent, 0.45 + 0.45 * clamp(U.high, 0.0, 1.0));
    c = mix(c, metal, cov);
  }
  // The stipple column: chrome beads beside the rune, on their own thin field so they stay beads. A bead
  // is one sphere, so it is shaded with two smoothsteps rather than the full rod shader.
  let row = (p.y + 1.1) * 17.0;
  let bead = length(vec2f(
    p.x - (0.42 + 0.13 * h1(floor(row) * 5.1 + floor(t * 0.4))),
    (fract(row) - 0.5) / 17.0
  ));
  let bw = 0.0042 + 0.0030 * P(4);
  let bcov = 1.0 - smoothstep(bw - aa, bw + aa, bead);
  if (bead < bw + px * 3.0) {
    var bm = STEEL * (0.16 + 0.95 * smoothstep(bw, 0.0, bead));
    bm = bm + vec3f(1.0) * smoothstep(bw * 0.45, 0.0, bead) * 0.55;
    c = mix(c, bm + RED * 0.30 * smoothstep(bw * 0.7, 0.0, bead), bcov);
  }
  c *= 1.0 - 0.24 * glow(d - w, 30.0);
  c += RED * 0.09 * glow(d - w, 48.0) * U.pulse * U.pulse;
  c *= grain(pos.xy, 0.10 + 0.20 * P(7));
  c *= mix(vec3f(1.0), vec3f(0.96, 0.80, 1.10), P(2) * 0.55);
  return fin(shoulder(c));
}
