// Scene 02 (numogram), hand-written for this renderer. It replaces the extracted web body
// (`mosh-shaders/wgsl/scene02_numogram.wgsl`, left untouched: `node scripts/extract_wgsl.mjs
// --check` still pins it) and keeps that body's geometry, colours and clock: the same ZP zode
// table, the same three edge lists (the outer loop, the inner chords, the great triangles), the
// same ten mel discs, the same ring glow, applied in the same order.
//
// Only the arrangement of the maths changed, because the extracted body measured 3.4-9.5 ms per
// pass at 960x540 on the integrated GPU — more than the whole frame budget on its own:
//
//   * The four tables were copied into function-scope arrays (`var zp = ZP; var lp = LP; ...`)
//     and indexed with loop counters. A dynamically indexed function-scope array is scratch
//     memory on this GPU, so every one of ~105 segment endpoints per pixel was a scratch round
//     trip. Here every table entry is its own `const` and every index is a literal.
//   * Each ring is evaluated in its own frame: rotate q by -ang and divide by s once, then the 21
//     segments are the *unit* zode edges, so their direction and length are compile-time
//     constants. Distances are scaled back by s where the shader compares them with a pixel.
//   * A ring's rotation is `+-U.T * 0.11 + k` and its brightness is `sin(PI * (ph + k) / 5)`, so
//     both are angle sums of an angle computed once per pixel and `cos k` / `sin k` constants:
//     four transcendentals per pixel instead of 28. The constants are picked with `if` chains,
//     which a GPU resolves once per draw because `k` is uniform.
//   * The three distance fields take their minimum over *squared* distances and are rooted once
//     each: 3 square roots per ring instead of 21. `pow(x, 1.5)` is written `x * sqrt(x)`, the
//     same curve without two transcendentals.
//   * A ring is skipped for a pixel that cannot see it: every piece of a ring lies within its
//     reach of the centre (a stroke at most 0.79 s out, a mel disc at most (0.79 + 0.05 + 0.07 e)
//     s out), so a pixel beyond reach + half a stroke width gets nothing from it. Exact, and it
//     drops most of the screen for the small rings.

const ZP0 = vec2f(-0.25, 0.75);
const ZP1 = vec2f(-0.55, 0.0);
const ZP2 = vec2f(-0.3, 0.35);
const ZP3 = vec2f(-0.25, -0.75);
const ZP4 = vec2f(-0.3, -0.35);
const ZP5 = vec2f(0.3, -0.35);
const ZP6 = vec2f(0.25, -0.75);
const ZP7 = vec2f(0.3, 0.35);
const ZP8 = vec2f(0.55, 0.0);
const ZP9 = vec2f(0.25, 0.75);

// `cos k` and `sin k` for ring `k`, and `cos(k PI / 5)` / `sin(k PI / 5)` for its brightness
// phase: the web body's angles, as numbers. `k` is the same for every pixel of a draw, so these
// are uniform selections, not per-pixel work.
fn cos_k(k: i32) -> f32 {
  var v = -0.65364362;
  if (k == 3) { v = -0.98999250; }
  if (k == 2) { v = -0.41614684; }
  if (k == 1) { v = 0.54030231; }
  if (k == 0) { v = 1.0; }
  return v;
}

fn sin_k(k: i32) -> f32 {
  var v = -0.75680250;
  if (k == 3) { v = 0.14112001; }
  if (k == 2) { v = 0.90929743; }
  if (k == 1) { v = 0.84147098; }
  if (k == 0) { v = 0.0; }
  return v;
}

fn cos_b(k: i32) -> f32 {
  var v = -0.80901699;
  if (k == 3) { v = -0.30901699; }
  if (k == 2) { v = 0.30901699; }
  if (k == 1) { v = 0.80901699; }
  if (k == 0) { v = 1.0; }
  return v;
}

fn sin_b(k: i32) -> f32 {
  var v = 0.58778525;
  if (k == 3) { v = 0.95105652; }
  if (k == 2) { v = 0.95105652; }
  if (k == 1) { v = 0.58778525; }
  if (k == 0) { v = 0.0; }
  return v;
}

/// Squared distance from `p` to the segment `a -> a + ba`, where `inv_baba = 1 / dot(ba, ba)`
/// is a compile-time constant (zero divisions per segment).
fn seg2_c(p: vec2f, a: vec2f, ba: vec2f, inv_baba: f32) -> f32 {
  let pa = p - a;
  let h = clamp(dot(pa, ba) * inv_baba, 0., 1.);
  let d = pa - ba * h;
  return dot(d, d);
}

/// Squared distance from `p` to a horizontal segment `[-hx, +hx]` at `y = y0`.
fn hseg2(p: vec2f, hx: f32, y0: f32) -> f32 {
  let dx = max(abs(p.x) - hx, 0.);
  let dy = p.y - y0;
  return dx * dx + dy * dy;
}

// Precomputed segment vectors and inverse squared lengths for the non-horizontal segments.
// LP angled edge (after folding into the first quadrant): (0.55, 0.0) -> (0.30, 0.35).
const LP_A = vec2f(0.55, 0.0);
const LP_BA = vec2f(-0.25, 0.35);
const LP_INV = 5.40540541; // 1 / (0.25^2 + 0.35^2) = 1 / 0.185

// GT non-horizontal segments:
// 2->3: (-0.30, 0.35) -> (-0.25, -0.75)
const GT0_BA = vec2f(0.05, -1.10);
const GT0_INV = 0.82474227; // 1 / (0.0025 + 1.21)
// 4->1: (-0.30, -0.35) -> (-0.55, 0.0)
const GT1_BA = vec2f(-0.25, 0.35);
const GT1_INV = 5.40540541; // 1 / 0.185
// 5->6: (0.30, -0.35) -> (0.25, -0.75)
const GT2_BA = vec2f(-0.05, -0.40);
const GT2_INV = 6.15384615; // 1 / (0.0025 + 0.16)
// 7->1: (0.30, 0.35) -> (-0.55, 0.0)
const GT3_BA = vec2f(-0.85, -0.35);
const GT3_INV = 1.18343195; // 1 / (0.7225 + 0.1225)
// 8->9: (0.55, 0.0) -> (0.25, 0.75)
const GT4_BA = vec2f(-0.30, 0.75);
const GT4_INV = 1.53256705; // 1 / (0.09 + 0.5625)

// `exp2(5.5 / 5.0) = exp2(1.1)`: multiplicative scale step between consecutive rings.
const RING_SCALE_STEP = 2.143546925;

// One mel disc: a hole (`ain`) and, unless the band is loud, a thin ring around it. `ql`, `z` and
// the radii are in ring units (the ring's own rotated frame, divided by s); `margin` is the exact
// ring-unit radius beyond `rad` where both the fill and the ring stroke clamp to 0.
fn mel_disc(
  c: vec3f,
  ql: vec2f,
  z: vec2f,
  s_inv_px: f32,
  th_half_px: f32,
  margin: f32,
  b: f32,
  e: f32,
) -> vec3f {
  let rad = 0.05 + 0.07 * e;
  let dv = ql - z;
  let dd2 = dot(dv, dv);
  let reach = rad + margin;
  if (dd2 >= reach * reach) {
    return c;
  }
  let dd = sqrt(dd2);
  let ain = clamp((rad - dd) * s_inv_px + 0.5, 0., 1.);
  var out = c * (1. - ain);
  if (e > 0.75) {
    out = mix(out, vec3f(b), ain);
  } else {
    out = max(out, U.l0.rgb * b * clamp(th_half_px - abs(dd - rad) * s_inv_px, 0., 1.));
  }
  return out;
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let r = length(q) + 1e-4;
  let inv_px = U.res.y * 0.5;
  let px = 2. / U.res.y;
  let ss = U.res.x / 640.;
  let tw = f32(i32(U.vari) % 4);
  let rg = step(0.96, sin(log(r) * 14. - U.T * 1.5 + atan2(q.y, q.x) * tw));
  var c = rg * (0.10 * U.pulse * U.l1.rgb + 0.18 * U.l0.rgb);
  // The ten mel values the discs read, once: the widest disc bounds every ring's reach.
  let e0 = mel(3);
  let e1 = mel(5);
  let e2 = mel(7);
  let e3 = mel(9);
  let e4 = mel(11);
  let e5 = mel(13);
  let e6 = mel(15);
  let e7 = mel(17);
  let e8 = mel(19);
  let e9 = mel(21);
  let m0 = max(max(max(e0, e1), max(e2, e3)), max(max(e4, e5), max(e6, e7)));
  let emax = max(m0, max(max(e8, e9), 0.));
  // 0.7906 (the outermost zode) + 0.05 + 0.07 e (its disc).
  let max_disc = 0.05 + 0.07 * emax;
  let mel_reach = 0.7906 + max_disc;
  // The two angles every ring is built from, each once per pixel instead of once per ring: the
  // ring rotation `U.T * 0.11` and the brightness phase `PI * ph / 5`.
  let ph = fract(U.T * 0.13);
  let ct = cos(U.T * 0.11);
  let st = sin(U.T * 0.11);
  let cb = cos(PI * ph / 5.);
  let sb = sin(PI * ph / 5.);
  let ss_half_px = ss * 0.5 + 0.5;
  var s = 0.12 * exp2(ph * 1.1) * (0.5 + U.dens);
  for (var k = 0; k < 5; k = k + 1) {
    if (k > 0) {
      s = s * RING_SCALE_STEP;
    }
    let th = select(2.2, 1.2, s < 1.2) * ss * px;
    // Nothing of this ring is further out than mel_reach * s: skip before computing brightness
    // or rotation for pixels outside the ring's outer envelope.
    if (r > mel_reach * s + th + px) {
      continue;
    }
    // sin(PI * p) = sin(PI k / 5) cos(PI ph / 5) + cos(PI k / 5) sin(PI ph / 5).
    let bs = sin_b(k) * cb + cos_b(k) * sb;
    let b = bs * sqrt(max(bs, 0.));
    // ang = +-U.T * 0.11 + k: the web body's `select(-1, 1, (k & 1) == 1)` takes the sign off
    // the odd rings, so the even rings run backwards.
    let sign = select(-1., 1., (k & 1) == 1);
    let ck = cos_k(k);
    let sk = sin_k(k);
    let cs = ck * ct - sign * sk * st;
    let sn = sign * st * ck + ct * sk;
    // The ring's own frame: q rotated by -ang, divided by s.
    let inv = 1. / s;
    let ql = vec2f(q.x * cs + q.y * sn, q.y * cs - q.x * sn) * inv;
    let qabs = abs(ql);
    let th_half_px = th * 0.5 * inv_px + 0.5;
    let margin = th_half_px * px * inv;
    // Axis-aligned bounding box in ring space is tighter than the bounding circle (0.55 vs 0.7906
    // on X): skip the whole ring when outside its exact box.
    if (qabs.x > 0.55 + max_disc + margin || qabs.y > 0.75 + max_disc + margin) {
      continue;
    }

    let s_inv_px = s * inv_px;
    let stroke_r2 = margin * margin;

    // LP (outer loop 1-2-7-8-5-4-1) is symmetric in both X and Y, so folding into the first
    // quadrant collapses all 6 segments to 1 angled edge + 1 horizontal edge.
    let h_mid = hseg2(qabs, 0.30, 0.35);
    let d0 = min(seg2_c(qabs, LP_A, LP_BA, LP_INV), h_mid);
    if (d0 < stroke_r2) {
      c = max(c, U.l0.rgb * b * clamp(th_half_px - sqrt(d0) * s_inv_px, 0., 1.));
    }

    // GT (six great-triangle chords): 5 angled edges + 1 horizontal edge at y = -0.75.
    let gt_r = ss_half_px * px * inv;
    var d2 = seg2_c(ql, ZP2, GT0_BA, GT0_INV);
    d2 = min(d2, seg2_c(ql, ZP4, GT1_BA, GT1_INV));
    d2 = min(d2, seg2_c(ql, ZP5, GT2_BA, GT2_INV));
    d2 = min(d2, seg2_c(ql, ZP7, GT3_BA, GT3_INV));
    d2 = min(d2, seg2_c(ql, ZP8, GT4_BA, GT4_INV));
    d2 = min(d2, hseg2(ql, 0.25, -0.75));
    if (d2 < gt_r * gt_r) {
      c = max(c, U.l2.rgb * b * 0.8 * clamp(ss_half_px - sqrt(d2) * s_inv_px, 0., 1.));
    }

    // SZ (five horizontal inner chords) is also X/Y-symmetric: reuse `h_mid` (y = +-0.35) and add
    // y = +-0.75 and y = 0.0.
    let d1 = min(h_mid, min(hseg2(qabs, 0.25, 0.75), hseg2(qabs, 0.55, 0.0)));
    if (d1 < stroke_r2) {
      c = max(c, U.l1.rgb * b * clamp(th_half_px - sqrt(d1) * s_inv_px, 0., 1.));
    }

    // The ten mel discs, in zode order, with the band each one reads in the web body.
    c = mel_disc(c, ql, ZP0, s_inv_px, th_half_px, margin, b, e0);
    c = mel_disc(c, ql, ZP1, s_inv_px, th_half_px, margin, b, e1);
    c = mel_disc(c, ql, ZP2, s_inv_px, th_half_px, margin, b, e2);
    c = mel_disc(c, ql, ZP3, s_inv_px, th_half_px, margin, b, e3);
    c = mel_disc(c, ql, ZP4, s_inv_px, th_half_px, margin, b, e4);
    c = mel_disc(c, ql, ZP5, s_inv_px, th_half_px, margin, b, e5);
    c = mel_disc(c, ql, ZP6, s_inv_px, th_half_px, margin, b, e6);
    c = mel_disc(c, ql, ZP7, s_inv_px, th_half_px, margin, b, e7);
    c = mel_disc(c, ql, ZP8, s_inv_px, th_half_px, margin, b, e8);
    c = mel_disc(c, ql, ZP9, s_inv_px, th_half_px, margin, b, e9);
  }
  return fin(c);
}
