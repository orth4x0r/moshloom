// Sigil Fangs: a curtain of chrome fangs hung from the top and bottom edges, mirrored, beak to
// beak, in front of a blood-red backlight.
//
// Each fang is a tapered capsule (a capsule whose radius falls to zero at the tip), so the whole
// curtain is 24 cheap distance evaluations rather than 24 triangles, and the field's gradient
// gives each fang a rounded chrome facet with a black core and an ice rim. The columns are
// mirrored, the fangs hang at different lengths per column, and the row behind them is shorter and
// darker, so the curtain has depth.
//
// Audio: each column's fang length is one mel band, `U.bass` drops the whole curtain and widens the
// gap, `U.pulse` makes the fangs lunge and the backlight flare. The backlight moves at the beat's
// rate or slower, never inside the 400 ms flash limit.
//
// P(0) clock rate, P(1) density, P(2) hue, P(3) shear, P(4) fang length, P(5) backlight,
// P(6) curtain depth, P(7) grain.

const RED = vec3f(0.76, 0.045, 0.070);
const ICE = vec3f(0.62, 0.82, 1.0);
const STEEL = vec3f(0.60, 0.66, 0.76);

fn seg(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-7), 0.0, 1.0);
  return length(pa - ba * h);
}

/// A fang: a capsule tapering from `w` at `root` to a point at `tip`.
fn fang(p: vec2f, root: vec2f, tip: vec2f, w: f32) -> f32 {
  let pa = p - root;
  let ba = tip - root;
  let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-7), 0.0, 1.0);
  return length(pa - ba * h) - w * (1.0 - h * h * 0.9);
}

/// The curtain: fangs from the top edge and from the bottom edge, in mirrored columns.
fn curtain(p: vec2f, cols: f32, cw: f32, shear: f32, grow: f32, back: f32) -> f32 {
  var d = 1e6;
  let ix = floor(p.x / cw);
  let cx = (ix + 0.5) * cw;
  let seed = h1(ix * 3.71 + 1.30);
  let seed2 = h1(ix * 9.13 + 4.70);

  // The front row: a long fang from the top and from the bottom in every column.
  let top_len = (0.40 + 0.34 * seed + 0.26 * mel(i32(ix * 2.0 + 1.0)) + 0.22 * U.bass) * grow;
  let bot_len = (0.34 + 0.32 * seed2 + 0.26 * mel(i32(ix * 2.0 + 8.0))) * grow;
  // Alternating columns bite deeper, so the curtain has a rhythm instead of a fringe.
  let parity = select(0.78, 1.25, fract(ix * 0.5) > 0.25);
  let lean = shear * (0.5 - seed) * 0.30 + P(3) * 0.02;
  let top_root = vec2f(cx + lean * 0.2, 1.12);
  let bot_root = vec2f(cx - lean * 0.2, -1.12);
  let top_tip = top_root + vec2f(lean, -top_len * parity);
  let bot_tip = bot_root + vec2f(lean, bot_len / parity);
  d = min(d, fang(p, top_root, top_tip, cw * 0.29));
  d = min(d, fang(p, bot_root, bot_tip, cw * 0.29));
  // A barb on the flank of the top fang: the thorn that makes a fang a fang.
  let along = normalize(top_tip - top_root);
  let flank = vec2f(-along.y, along.x);
  let at = mix(top_root, top_tip, 0.55);
  d = min(d, seg(p, at, at + flank * (0.045 + 0.045 * seed2) * parity) - 0.0042);

  // The row behind is drawn only when the clip asks for depth: each extra fang is a square root.
  if (back > 0.75) {
    let bx = cx + cw * 0.5;
    let seed3 = h1((ix + 0.5) * 5.53 + 2.10);
    let back_len = (0.18 + 0.30 * seed3 + 0.18 * mel(i32(ix * 2.0 + 4.0))) * grow * back;
    d = min(d, fang(p, vec2f(bx, 1.12), vec2f(bx + lean * 0.7, 1.12 - back_len), cw * 0.21));
    d = min(d, fang(p, vec2f(bx, -1.12), vec2f(bx + lean * 0.7, -1.12 + back_len * 0.8), cw * 0.21));
  }

  // The gum: two thin chains along the top and bottom edges that the fangs hang from.
  d = min(d, abs(p.y - 1.06) - 0.006);
  d = min(d, abs(p.y + 1.06) - 0.006);
  return d;
}

/// A soft cold studio for a chrome facet: bright ceiling, dark floor, one hard strip light.
fn chrome(n: vec3f, refl: vec3f) -> vec3f {
  let up = smoothstep(0.05, 0.92, refl.y);
  let down = smoothstep(0.0, -0.85, refl.y);
  var env = mix(vec3f(0.18, 0.21, 0.28), vec3f(0.95, 0.98, 1.0), up);
  env = mix(env, vec3f(0.03, 0.035, 0.055), down * 0.85);
  let strip = smoothstep(0.45, 0.05, abs(n.x)) * smoothstep(-0.35, 0.55, n.y);
  env += vec3f(0.95, 0.98, 1.0) * strip * 0.9;
  return env;
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let px = 2.0 / U.res.y;
  let aa = px * 1.15;
  let e = px * 2.2;
  let asp = U.res.x / U.res.y;
  let dens = clamp(U.dens, 0.0, 1.0);
  let p = vec2f(abs(q.x), q.y);

  let cols = 5.0 + floor(dens * 5.0);
  let cw = asp / cols;
  let shear = 1.0 + 0.6 * P(3) * sin(U.T * (0.20 + 0.25 * P(0)));
  let grow = 0.72 + 0.45 * P(4) + 0.22 * clamp(U.mid, 0.0, 1.0) + 0.20 * U.pulse;
  let back = 0.55 + 0.45 * P(6);
  let d = curtain(p, cols, cw, shear, grow, back);

  // --- the backlight: a hot red band between the two rows of fangs, a cold top light, grain.
  let band = exp(-abs(q.y) * 2.3);
  var c = vec3f(0.006, 0.006, 0.010);
  c += RED * (0.10 + 0.16 * P(5)) * band * (0.30 + 1.00 * U.pulse * U.pulse);
  c += RED * 0.035 * exp(-length(vec2f(q.x * 0.55, q.y)) * 1.4) * (0.30 + 0.9 * U.mid);
  c += ICE * 0.020 * smoothstep(0.6, 1.1, -q.y) * (0.3 + 0.7 * U.high);
  c *= 0.92 + 0.16 * h2(pos.xy * 0.75 + U.frame * 0.7) * (0.3 + 0.7 * P(7));

  // --- the fangs: a chrome facet with a black core, and a red rim from the backlight.
  let g = vec2f(dpdx(d), dpdy(d)) * (e / px);
  if (d < px * 12.0) {
    let cov = 1.0 - smoothstep(-aa, aa, d);
    let nrm = normalize(vec3f(-g, e * 0.85));
    let refl = reflect(vec3f(0.0, 0.0, -1.0), nrm);
    let env = chrome(nrm, refl);
    let rim = pow(1.0 - clamp(nrm.z, 0.0, 1.0), 1.9);
    let spec = pow(max(dot(refl, normalize(vec3f(-0.35, 0.75, 0.55))), 0.0), 30.0);
    var metal = STEEL * (0.20 + env * 1.40) + vec3f(1.0, 0.99, 0.97) * spec * 1.3;
    metal = mix(metal * 0.45, metal, smoothstep(0.0, 0.55, -d / 0.10));
    metal += ICE * rim * 0.45 * (0.4 + 0.6 * U.high);
    metal += RED * rim * (0.35 + 1.10 * U.pulse) * band;
    c = mix(c, metal, cov);
    c += STEEL * 0.08 * exp(-max(d, 0.0) * 60.0) * (1.0 - cov);
  }
  c += RED * 0.12 * exp(-abs(d) * 140.0) * U.pulse * U.pulse;
  // Hue: a channel rotation on the finished frame, so P(2) shifts the whole plate from blood red
  // toward violet without touching the chrome.
  c *= mix(vec3f(1.0), vec3f(0.88, 0.58, 1.28), U.hue);
  return fin(c);
}
