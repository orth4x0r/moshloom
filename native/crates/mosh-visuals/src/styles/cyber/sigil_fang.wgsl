// Sigil Fangs: a curtain of tapered chrome fangs hung from the top and bottom edges, mirrored, beak
// to beak, in front of a blood-red backlight on a grainy plate.
//
// Each fang is one tapered rod: a segment distance, and a half width that falls to nothing at the tip,
// so the whole curtain is a dozen cheap evaluations rather than a dozen triangles. The rod is lit
// across its full width (`chrome_ridge` in `styles/paint.wgsl`), which is what makes it read as a
// polished fang with a specular streak down one flank, a shadowed edge, a blood-red backlight rim and
// an iridescent turn — where the old build had a flat chrome fill. The columns are mirrored, the fangs
// hang at different lengths per column, and a barb on each flank gives the curtain its bite. `Depth`
// thins the fangs and pushes the gum away from the frame edge instead of paying for a second row. A
// shoulder on the highlights keeps the specular from clipping to white.
//
// Audio: each column's fang length is one mel band, `U.bass` opens the gap and drops the curtain,
// `U.pulse` makes the fangs lunge and the backlight flare. The backlight moves at the beat's rate or
// slower, never inside the 400 ms flash limit.
//
// P(0) clock rate, P(1) density, P(2) hue, P(3) shear, P(4) fang length, P(5) backlight,
// P(6) curtain depth, P(7) grain.

const STEEL = vec3f(0.88, 0.91, 0.97);
const RED = vec3f(0.86, 0.050, 0.075);

/// A rod: its centreline distance and its local half width, plus where along the rod the pixel is.
fn rod(p: vec2f, root: vec2f, tip: vec2f, w0: f32) -> vec3f {
  let pa = p - root;
  let ba = tip - root;
  let hl = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-7), 0.0, 1.0);
  let d = length(pa - ba * hl);
  return vec3f(d, max(w0 * (1.0 - hl * hl * 0.88), 0.0030), hl);
}

/// One row of the curtain: fangs from the top edge and from the bottom edge in every column.
/// Returns the winning rod's (distance, half width) as `xy`, plus the tooth's own mask in `z`.
fn curtain(p: vec2f, cw: f32, shear: f32, grow: f32, back: f32) -> vec3f {
  var best = 1e6;
  var bw = 0.01;
  var tooth = 0.0;

  let ix = floor(p.x / cw);
  let cx = (ix + 0.5) * cw;
  // One hash per column, with its other values derived by multiplication: a hash is a `sin`, and this
  // scene's whole budget is spent on how few of them it needs.
  let seed = h1(ix * 3.71 + 1.30);
  let seed2 = fract(seed * 37.13);
  let seed3 = fract(seed * 91.70);
  let band0 = mel(i32(ix * 2.0 + 1.0));
  let band1 = mel(i32(ix * 2.0 + 8.0));

  let top_len = (0.42 + 0.32 * seed + 0.24 * band0 + 0.20 * U.bass) * grow;
  let bot_len = (0.36 + 0.30 * seed2 + 0.24 * band1) * grow;
  // Alternating columns bite deeper, so the curtain has a rhythm instead of a fringe.
  let parity = select(0.80, 1.22, fract(ix * 0.5) > 0.25);
  let lean = shear * (0.5 - seed) * 0.26;

  let top_root = vec2f(cx + lean * 0.25, 1.10);
  let top_tip = top_root + vec2f(lean, -top_len * parity);
  let bot_root = vec2f(cx - lean * 0.25, -1.10);
  let bot_tip = bot_root + vec2f(lean, bot_len / parity);

  // Front row: the long fang from the top and from the bottom of this column.
  let f0 = rod(p, top_root, top_tip, cw * 0.23);
  best = f0.x - f0.y;
  bw = f0.y;
  tooth = 1.0;
  let f1 = rod(p, bot_root, bot_tip, cw * 0.21);
  if (f1.x - f1.y < best) {
    best = f1.x - f1.y;
    bw = f1.y;
  }

  // A barb on the flank of the top fang: the thorn that makes a fang a fang. `Depth` shifts it and the
  // gum down, so the curtain reads deeper without a second row of rods (two more square roots per pixel,
  // which this budget cannot afford).
  let along = normalize(top_tip - top_root);
  let flank = vec2f(-along.y, along.x);
  let at = mix(top_root, top_tip, 0.50);
  let f2 = rod(p, at, at + flank * (0.050 + 0.050 * seed2 + 0.030 * back) * parity, 0.0075);
  if (f2.x - f2.y < best) {
    best = f2.x - f2.y;
    bw = f2.y;
  }
  bw = bw * (1.0 - 0.22 * clamp(back, 0.0, 1.0));
  tooth = tooth * (0.75 + 0.25 * clamp(back, 0.0, 1.0));

  // The gum: the heavy chain along the top and bottom edges the fangs hang from.
  let gum = min(abs(p.y - (1.05 - 0.04 * back)), abs(p.y + 1.05));
  if (gum - 0.012 < best) {
    best = gum - 0.012;
    bw = 0.012;
    tooth = 0.0;
  }
  return vec3f(best, bw, tooth);
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let px = 2.0 / U.res.y;
  let aa = px * 1.10;
  let dens = clamp(U.dens, 0.0, 1.0);
  let p = vec2f(abs(q.x), q.y);

  let cols = 5.0 + floor(dens * 5.0);
  let cw = (U.res.x / U.res.y) / cols;
  let shear = 1.0 + 0.6 * P(3) * sin(U.T * (0.20 + 0.25 * P(0)));
  let grow = 0.78 + 0.40 * P(4) + 0.20 * clamp(U.mid, 0.0, 1.0) + 0.22 * U.pulse;
  let back = 0.55 + 0.45 * P(6);
  let hit = curtain(p, cw, shear, grow, back);
  let g = vec2f(dpdx(hit.x), dpdy(hit.x));

  // --- the backlight: a hot red band between the two rows of fangs, a cold light from above, grain.
  let band = glow(q.y, 1.1);
  let hue_ang = TAU * P(2);
  let hue_tint = pal(cos(hue_ang), sin(hue_ang));
  var c = plate(q, hue_tint * 0.30 + vec3f(0.10, 0.06, 0.08), U.T, 3.3);
  c += RED * (0.075 + 0.150 * P(5)) * band * (0.30 + 1.00 * U.pulse * U.pulse);
  c += RED * 0.045 * falloff(length(vec2f(q.x * 0.55, q.y)), 1.5) * (0.35 + 0.9 * U.mid);
  c += vec3f(0.10, 0.15, 0.22) * 0.20 * glow(q.y - 1.0, 1.8) * (0.4 + 0.6 * U.high);

  // --- the fangs, lit across the rod; the back row sits darker so the curtain has depth.
  let cov = 1.0 - smoothstep(-aa, aa, hit.x);
  if (hit.x < px * 2.5) {
    let accent = mix(RED, hue_tint, 0.34) * (0.80 + 0.45 * P(5));
    var metal = chrome_ridge(
      max(hit.x, 0.0), g, hit.y, STEEL, accent, 0.40 + 0.50 * clamp(U.high, 0.0, 1.0)
    );
    metal = metal * mix(0.55, 1.0, hit.z);
    metal = metal + RED * glow(hit.x + hit.y * 0.4, 22.0) * (0.10 + 0.45 * U.pulse);
    c = mix(c, metal, cov);
  }
  c *= 1.0 - 0.30 * glow(hit.x, 30.0);
  c *= grain(pos.xy, 0.10 + 0.20 * P(7));
  c *= mix(vec3f(1.0), vec3f(0.96, 0.80, 1.10), P(2) * 0.55);
  return fin(shoulder(c));
}
