// Sigil Thorn: a bilateral thorn crest in cold chrome on a near-black plate.
//
// Cybersigilism as tattoo flash: one side is authored, the other is its mirror. The figure is built
// from implicit fields — a wavy stem whose thickness pulses into barbs, one horn arc, a pointed
// gate and a star outline whose cusps are the thorns — so a dense crest costs a few dozen
// instructions instead of a few dozen square roots, which is what keeps this pass inside its budget
// on an integrated GPU.
//
// Lighting is a fake chrome studio built straight from the field's screen normal, so a stroke reads
// as a polished thorn: a dark blackwork core, a bright bevel and an ice-blue rim, over a blood-red
// bloom that breathes on the beat. A dark plate fills the star, so the crest has mass.
//
// Audio: the star's points are the crest's silhouette and lengthen with a mel band, the barbs pulse
// with `U.bass`, `U.pulse` widens the stem and drives the red glow. `U.dens` adds points and `U.vari`
// rotates the star.
//
// P(0) clock rate, P(1) density, P(2) hue, P(3) sway, P(4) thorn length, P(5) bloom, P(6) gate,
// P(7) grain.

const RED = vec3f(0.70, 0.030, 0.055);
const ICE = vec3f(0.62, 0.82, 1.0);
const STEEL = vec3f(0.56, 0.62, 0.72);

/// The star the crest is drawn around: a polygon whose cusps are the thorns. `1 - sqrt(|sin|)`
/// makes the points sharp and the valleys round for two instructions.
fn star_radius(a: f32, t: f32, dens: f32) -> f32 {
  let points = 5.0 + floor(dens * 5.0);
  let phase = a * points * 0.5 + t * 0.15 + U.vari * 0.5;
  return 0.74 + (0.22 + 0.30 * mel(i32(4.0 + dens * 12.0))) * (1.0 - sqrt(abs(sin(phase))));
}

/// The crest in mirrored half-space (`p.x >= 0`).
fn crest(p: vec2f, a: f32, r: f32, t: f32, grow: f32, dens: f32, gate: f32, sway: f32) -> f32 {
  var d = 1e6;

  // The stem: a wavy branch whose thickness pulses into barbs.
  let sx = 0.02 + 0.11 * sin(p.y * 2.30 + t * 0.45 + sway) * (0.35 + 0.65 * smoothstep(-0.95, 0.85, p.y));
  let thick = 0.0075 + 0.022 * max(0.0, sin(p.y * 15.0 + t * 1.10)) * grow;
  d = min(d, abs(p.x - sx) - thick);

  // The gate: a diamond around the figure's waist, drawn only when the clip asks for it.
  let diamond = abs(p.x) * 1.15 + abs(p.y - 0.02) * 0.90 - 0.27;
  d = min(d, max(abs(diamond) - 0.0055, (1.0 - gate) * 0.20));

  // The thorn ring: the star's outline.
  d = min(d, abs(r - star_radius(a, t, dens)) - 0.0080);

  // Stipple: a column of dots beside the stem, tattoo flash's negative space.
  let row = (p.y + 1.1) * 18.0;
  let dot = vec2f(
    p.x - (0.36 + 0.14 * h1(floor(row) * 3.1 + floor(t * 0.35))),
    (fract(row) - 0.5) / 18.0
  );
  d = min(d, length(dot) - 0.0030);
  return d;
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
  var metal = STEEL * (0.18 + env * 1.35);
  metal += ICE * rim2 * 0.42 + RED * rim2 * (0.15 + 0.65 * pulse);
  return metal;
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let px = 2.0 / U.res.y;
  let aa = px * 1.15;
  let e = px * 2.2;
  let dens = clamp(U.dens, 0.0, 1.0);
  let p = vec2f(abs(q.x), q.y);
  let a = atan2(p.y, p.x);
  let r = length(p);

  let grow = 0.65 + 0.70 * dens + 0.45 * P(4) + 0.30 * U.pulse;
  let t = U.T * (0.35 + 0.35 * P(0));
  let sway = 0.20 * sin(t * 0.7 + P(3) * 2.0);
  let d = crest(p, a, r, t, grow, dens, 0.55 + 0.45 * P(6), sway);
  let g = vec2f(dpdx(d), dpdy(d)) * (e / px);

  // --- background: a cold plate, a red bloom behind the figure, grain, a vignette.
  let b = length(vec2f(q.x * 0.80, q.y - 0.02));
  var c = mix(vec3f(0.030, 0.034, 0.046), vec3f(0.004, 0.004, 0.007), smoothstep(0.20, 1.30, b));
  c += RED * (0.030 + 0.045 * P(5)) * exp(-b * 2.4) * (0.35 + 1.00 * U.pulse * U.pulse);
  c += ICE * 0.016 * exp(-b * 2.8) * (0.35 + 0.65 * U.high);
  c *= 0.92 + 0.16 * h2(pos.xy * 0.75 + U.frame * 0.7) * (0.3 + 0.7 * P(7));

  // --- the blackwork fill: the inside of the star is a dark plate with a red heart, so the crest
  // has mass instead of being a wireframe. Its edge is the same star the outline is drawn on.
  let dfill = r - star_radius(a, t, dens);
  c = mix(c, vec3f(0.030, 0.012, 0.018) + RED * (0.10 + 0.55 * U.pulse * U.pulse) * smoothstep(0.55, 0.0, r),
    1.0 - smoothstep(-aa, aa, dfill));
  c += ICE * 0.30 * exp(-abs(dfill) * 110.0);

  // --- the ink. The chrome is only sampled near a stroke, which is where it is used.
  if (d < px * 9.0) {
    let cov = 1.0 - smoothstep(-aa, aa, d);
    let nrm = normalize(vec3f(-g, e * 0.55));
    var metal = chrome(nrm, U.pulse);
    // Ink: a dark blackwork core, so the chrome reads as a bevel around ink.
    metal *= 0.72 + 0.28 * smoothstep(0.0, -0.012, d);
    c = mix(c, metal, cov);
    c += STEEL * 0.10 * exp(-max(d, 0.0) * 60.0) * (1.0 - cov);
  }
  c += RED * 0.16 * exp(-abs(d) * 110.0) * U.pulse * U.pulse;
  // Hue: a channel rotation on the finished frame, so P(2) shifts the whole plate from blood red
  // toward violet without touching the chrome.
  c *= mix(vec3f(1.0), vec3f(0.88, 0.58, 1.28), U.hue);
  return fin(c);
}
