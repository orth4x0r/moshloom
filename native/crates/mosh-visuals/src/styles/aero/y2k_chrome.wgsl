// Y2K · Liquid Chrome.
//
// A mass of chrome metaballs, lit like a studio product shot: an inverse-square field makes the
// blobs merge and split, the field's own gradient is the normal, and the normal's reflection of a
// two-tone sky is what reads as chrome. The iridescent holographic tint rides the fresnel term, so
// the rainbow sits on the rim and in the reflected floor, the way Y2K metal does.
//
// p0 Speed, p1 Blob size, p2 Hue, p3 Chrome (how hard the darks are), p4 Iridescence, p5 Flow
// (how fast the cluster swirls), p6 Sparkle, p7 Gloss (specular tightness).
//
// Cost: four lobes on one shared orbit angle, so the whole field is one reciprocal per lobe and no
// transcendental at all; the normal comes from the same loop the field uses. Anti-aliasing uses the
// field's analytic gradient, not `fwidth`; the shading is one gradient, one softbox, one fresnel
// and two integer powers, because on this GPU the shading chain — not the geometry — is what costs.

// Orbit offset xy, base radius z, pulse amplitude w: a ring of four, uneven on purpose so the
// merged mass never reads as a flower.
const L0 = vec4f(0.60, 0.10, 1.00, 0.8);
const L1 = vec4f(-0.46, 0.34, 0.88, 0.5);
const L2 = vec4f(0.04, -0.56, 1.14, 0.9);
const L3 = vec4f(-0.32, -0.30, 0.72, 0.6);

// One lobe's contribution: (field, d(field)/dp.x, d(field)/dp.y). The group turns by the shared
// (ca, sa), so the whole cluster moves for one sine and one cosine per frame.
fn lobe(p: vec2f, base: vec4f, ca: f32, sa: f32, r: f32) -> vec3f {
  let c = vec2f(base.x * ca - base.y * sa, base.x * sa + base.y * ca);
  let d = p - c;
  let inv = 1.0 / (dot(d, d) + 0.0006);
  let w = r * r * inv;
  return vec3f(w, -2.0 * w * d.x * inv, -2.0 * w * d.y * inv);
}

// The field, accumulated lobe by lobe so only one lobe is live at a time.
fn chrome_field(p: vec2f, ca: f32, sa: f32, r: f32, breath: f32) -> vec3f {
  var f = 0.0;
  var g = vec2f(0.0);
  let a0 = lobe(p, L0, ca, sa, r * (1.0 + 0.18 * breath * L0.w));
  f = f + a0.x;
  g = g + a0.yz;
  let a1 = lobe(p, L1, ca, sa, r * (1.0 + 0.18 * breath * L1.w));
  f = f + a1.x;
  g = g + a1.yz;
  let a2 = lobe(p, L2, ca, sa, r * (1.0 + 0.18 * breath * L2.w));
  f = f + a2.x;
  g = g + a2.yz;
  let a3 = lobe(p, L3, ca, sa, r * (1.0 + 0.18 * breath * L3.w));
  f = f + a3.x;
  g = g + a3.yz;
  return vec3f(f, g.x, g.y);
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let t = U.T;
  let px = 2.0 / U.res.y;
  let size = P(1);
  let hue = P(2);
  let chrome = P(3);
  let irid_amt = P(4);
  let flow = P(5);
  let sparkle = P(6);
  let gloss = P(7);
  let nrg = energy();
  let grow = 1.0 + 0.12 * U.pulse + 0.10 * nrg;
  let ang = t * (0.22 + 0.5 * flow);
  let ca = cos(ang);
  let sa = sin(ang);
  // One shared breath for the radii, each lobe taking a literal share of it.
  let breath = sin(t * 0.5);
  let r = (0.19 + 0.13 * size) * grow;

  let fg = chrome_field(q, ca, sa, r, breath);
  let f = fg.x;
  let g = fg.yz;
  // d(field)/dp times one pixel is how far the field moves across a pixel: that is the exact
  // anti-aliasing width, for the price of one length.
  let aa = max(length(g) * px * 1.6, 1e-4);
  let inside = smoothstep(1.0 - aa, 1.0 + aa, f);
  // The halo: the field fades to 0 far outside, so `f` itself is the falloff towards the surface.
  let halo = (1.0 - inside) * clamp(f, 0.0, 1.0) * clamp(f, 0.0, 1.0);

  // Normal: outward (-gradient) plus a slow ripple so the metal reads as molten. `normalize` on the
  // 2D gradient is one rsqrt; the third component gives the fresnel term something to use.
  let gn = normalize(-g + vec2f(1e-6, 1e-6));
  let n2 = gn + vec2f(sin(q.y * 5.0 + t * 1.1), cos(q.x * 5.3 - t * 0.9)) * 0.06 * (0.25 + flow);
  let n = normalize(vec3f(n2.x, n2.y, 0.62));

  // Reflection, the cheap way: the vertical part of a reflected view ray is proportional to -n.y,
  // so `up` needs no reflect().
  let up = -n.y;
  // A studio environment read as one 1D ramp: lit sky high up, a dark band at the horizon, a dim
  // floor below, and one bright softbox. This is what gives chrome its grey mid-tones instead of a
  // white mask: the metal is mostly the ramp, and the highlights are on top of it.
  var tone = mix(0.28, 1.0, smoothstep(-0.10, 0.70, up));
  tone = tone * (1.0 - 0.72 * glow(up + 0.06, 3.9));
  tone = tone + 0.85 * glow(up - 0.58, 4.2) + 0.30 * glow(up + 0.52, 5.8);
  let env = mix(vec3f(0.05, 0.09, 0.17), vec3f(0.62, 0.80, 1.0), clamp(tone, 0.0, 1.0));
  let edge = 1.0 - n.z;
  let fres = edge * edge * edge;
  let hue_ang = TAU * (0.42 + hue);
  let tint = pal(cos(hue_ang), sin(hue_ang));

  var surf = env;
  // Holographic tint in the reflection, strongest on the rim.
  surf = mix(surf, surf * (0.45 + 1.05 * tint), 0.20 + 0.60 * irid_amt);
  surf = surf + tint * fres * (0.30 + 0.95 * irid_amt);
  surf = surf * (1.0 - 0.35 * smoothstep(0.20, -0.55, up) * chrome);
  // Specular: two integer powers blended by Gloss, so no `pow`.
  let ndl = max(dot(n, vec3f(-0.50, -0.65, 0.62)), 0.0);
  let p2 = ndl * ndl;
  let p16 = p2 * p2 * p2 * p2;
  surf = surf + vec3f(1.0, 0.98, 0.95) * mix(p2, p16 * p16, gloss) * (0.9 + 1.5 * U.pulse);

  // Background: deep indigo, a teal pool low left, a violet wash high right, a lens streak.
  let rise = smoothstep(-1.0, 1.0, -q.y);
  var col = mix(vec3f(0.012, 0.016, 0.045), vec3f(0.07, 0.026, 0.15), rise);
  col = col + vec3f(0.05, 0.16, 0.20) * glow(distance(q, vec2f(-0.8, 0.6)), 0.55)
    * (0.5 + 0.5 * nrg);
  col = col + tint * halo * 0.30;
  col = col + vec3f(0.40, 0.60, 1.00) * hairline(q.y - 0.12, px * 26.0) * (0.10 + 0.25 * U.pulse);
  col = mix(col, surf, inside);
  // The meniscus: a hairline of white where the metal ends.
  col = col + vec3f(0.65, 0.85, 1.0) * hairline(f - 1.0, aa * 2.0) * 0.5;
  col = col + vec3f(0.85, 0.9, 1.0) * glints(q, t, 27.0, 0.972)
    * (0.15 + 0.9 * sparkle) * (0.3 + 0.4 * nrg);
  return fin(col);
}
