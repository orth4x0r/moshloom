// Sigil Thorn: a bilateral thorn crest in polished chrome on a grainy plum plate.
//
// Cybersigilism as a tattoo flash sheet, re-lit as metal: one side of the figure is authored and the
// other is its mirror, the silhouette is a star whose cusps are the thorns, a second contour runs a
// hair inside it, a barbed stem grows down the spine and a diamond gate sits at the waist. Every
// element is a distance to a centreline, so the whole crest costs a few dozen instructions instead
// of a few dozen square roots, and the metal is a rod across the full width of each stroke (see
// `chrome_ridge` in `styles/paint.wgsl`): a bright spine, a specular band, a shadowed edge, a
// blood-red rim and an iridescent shift. Behind it the plate is a grainy gradient with an aura pool,
// never flat black, and the stroke's shoulder rolls the specular off instead of clipping to white.
//
// Audio: the star's cusps lengthen with a mel band, the stem with `U.bass`, `U.pulse` widens the
// stroke, swings the iridescence and swells the red heart. Brightness moves at the beat's rate, so
// nothing here flashes inside the 400 ms limit.
//
// P(0) clock rate, P(1) arms, P(2) hue, P(3) sway, P(4) thorn length, P(5) bloom, P(6) gate,
// P(7) grain.

const STEEL = vec3f(0.88, 0.91, 0.97);
const RED = vec3f(0.82, 0.045, 0.070);

/// The star the crest is drawn around: a polygon whose cusps are the thorns. `1 - sqrt(|sin|)`
/// makes the points sharp and the valleys round for two instructions.
fn star_radius(a: f32, t: f32, arms: f32, grow: f32, sway: f32, band: f32) -> f32 {
  let points = 4.0 + floor(arms * 6.0);
  let phase = a * points * 0.5 + t * 0.15 + U.vari * 0.5 + sway;
  return 0.66 + (0.16 + 0.30 * band) * grow * (1.0 - sqrt(abs(sin(phase))));
}

/// The crest in mirrored half-space (`p.x >= 0`): the silhouette and its inner contour, the gate and the
/// barbed stem, as distances to their centrelines. The silhouette `big` is computed once by the caller,
/// because the blackwork fill needs the same curve. Small numbers, one `sin` for the stem.
fn crest(p: vec2f, r: f32, t: f32, gate: f32, sway: f32, big: f32, grow: f32) -> f32 {
  var d = abs(r - big);
  // The second contour, a hair inside the silhouette: the double line the style is drawn with.
  d = min(d, abs(r - (big - 0.085 - 0.030 * grow)));
  // The stem: a wavy branch down the spine, the crest's axis of symmetry.
  let sx = 0.018 + 0.095 * sin(p.y * 2.20 + t * 0.45 + sway) * (0.30 + 0.70 * smoothstep(-0.95, 0.85, p.y));
  d = min(d, abs(p.x - sx));
  // The gate: a diamond at the waist, present only when the clip asks for it (`gate` 0 parks it far
  // from every pixel, which costs one `max` rather than a branch).
  let diamond = abs(p.x) * 1.15 + abs(p.y - 0.02) * 0.90 - (0.26 + 0.14 * gate);
  d = min(d, abs(diamond) + (1.0 - gate) * 0.45);
  return d;
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let px = 2.0 / U.res.y;
  let aa = px * 1.10;
  let dens = clamp(U.dens, 0.0, 1.0);
  let p = vec2f(abs(q.x), q.y);
  let a = atan2(p.y, p.x);
  let r = length(p);

  let t = U.T * (0.35 + 0.35 * P(0));
  let grow = 0.85 + 0.50 * dens + 0.40 * P(4) + 0.28 * U.pulse;
  let sway = 0.18 * sin(t * 0.7 + P(3) * 2.0);
  let gate = clamp(P(6), 0.0, 1.0);
  let band = mel(i32(4.0 + dens * 12.0)) * 1.2;
  // The silhouette, computed once: the crest's lines and its blackwork fill are the same curve.
  let big = star_radius(a, t, dens, grow, sway, band);

  let d = crest(p, r, t, gate, sway, big, grow);
  let g = vec2f(dpdx(d), dpdy(d));
  // The stroke's half width: thick enough to carry a bevel at preview size, and it breathes with the
  // beat so the crest gains weight on a downbeat instead of only getting brighter.
  let w = (0.0255 + 0.0070 * dens + 0.0100 * P(4)) * (1.0 + 0.12 * U.pulse);

  // --- the plate: a grainy gradient with a plum pool upper-left and a red heart behind the crest.
  let hue_ang = TAU * P(2);
  let hue_tint = pal(cos(hue_ang), sin(hue_ang));
  var c = plate(q, hue_tint * 0.42 + vec3f(0.10, 0.06, 0.12), t, 1.7);
  c += RED * (0.035 + 0.055 * P(5)) * falloff(r, 2.1) * (0.40 + 1.10 * U.pulse * U.pulse);
  c += vec3f(0.10, 0.17, 0.26) * 0.16 * glow(q.y - 0.92, 1.6) * (0.4 + 0.6 * U.high);

  // --- the blackwork fill: the inside of the silhouette is a dark mass with a red heart, so the
  // crest has body and the chrome line has something to sit on.
  let dfill = r - big;
  let fill = 1.0 - smoothstep(-aa, aa, dfill);
  let mass = mix(vec3f(0.050, 0.024, 0.040), vec3f(0.013, 0.011, 0.018), smoothstep(0.0, 0.75, r));
  c = mix(c, mass + RED * 0.07 * falloff(r, 3.0) * (0.5 + 0.9 * U.pulse), fill);

  // --- the metal. The iridescence is sampled only where the stroke is, so the ramp costs the stroke's
  // pixels rather than the frame's.
  let cov = 1.0 - smoothstep(w - aa, w + aa, d);
  if (d < w + px * 2.5) {
    let accent = mix(RED, hue_tint, 0.40) * (0.75 + 0.55 * P(5));
    let gloss = 0.40 + 0.50 * clamp(U.high, 0.0, 1.0);
    c = mix(c, chrome_ridge(d, g, w, STEEL, accent, gloss), cov);
  }
  // The contact shadow the metal throws on the plate, then the plate's own grain.
  c *= 1.0 - 0.30 * glow(d - w, 28.0) * (1.0 - fill);
  c += RED * 0.10 * glow(d - w, 46.0) * U.pulse * U.pulse;
  c *= grain(pos.xy, 0.10 + 0.20 * P(7));
  // Hue: a gentle rotation on the finished plate, so P(2) walks the red heart toward ember or violet
  // while the metal keeps its own iridescent accent.
  c *= mix(vec3f(1.0), vec3f(0.96, 0.80, 1.10), P(2) * 0.55);
  return fin(shoulder(c));
}
