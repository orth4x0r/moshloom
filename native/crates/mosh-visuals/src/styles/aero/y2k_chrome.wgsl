// Chrome · Liquid Mercury: a mass of chrome droplets merging and splitting, lit like a studio product
// shot.
//
// An inverse-square field makes the droplets merge and split; the field's own gradient is the surface
// normal, and the normal's vertical part indexes the reflection of a studio — a dim floor, a dark
// horizon band, a bright ceiling softbox and two vertical strips — so the metal is mostly mid-tones with
// one hot reflection running through it. That is what makes it read as chrome: a white mask reads as
// blown-out plastic. Two slow ripples on the normal keep the reflection liquid, the fresnel rim takes
// an iridescent tint so the edges of a droplet go magenta-to-teal, and the whole frame sits on a deep
// indigo gradient with a teal pool and a floor bounce. A soft shoulder rolls the specular off, so
// nothing clips to flat white (`tests/styles_aero.rs` measures the clipped share).
//
// Audio: `U.pulse` swells the droplets and the specular, `U.bass` sloshes the cluster and widens the
// ripple, `U.high` lifts the iridescent rim. Reflection and ripple move at the beat's rate or slower.
//
// p0 Speed, p1 Droplet size, p2 Hue, p3 Chrome (how hard the darks are), p4 Iridescence, p5 Flow (how
// fast the cluster swirls), p6 Sparkle, p7 Gloss (specular tightness).

// Orbit offset xy, base radius z, pulse amplitude w: a ring of four, uneven on purpose so the merged
// mass never reads as a flower.
const L0 = vec4f(0.60, 0.10, 1.00, 0.8);
const L1 = vec4f(-0.46, 0.34, 0.88, 0.5);
const L2 = vec4f(0.04, -0.56, 1.14, 0.9);
const L3 = vec4f(-0.32, -0.30, 0.72, 0.6);

// One droplet's contribution: (field, d(field)/dp.x, d(field)/dp.y). The group turns by the shared
// (ca, sa), so the whole cluster moves for one sine and one cosine per frame.
fn droplet(p: vec2f, base: vec4f, ca: f32, sa: f32, r: f32) -> vec3f {
  let c = vec2f(base.x * ca - base.y * sa, base.x * sa + base.y * ca);
  let d = p - c;
  let inv = 1.0 / (dot(d, d) + 0.0006);
  let w = r * r * inv;
  return vec3f(w, -2.0 * w * d.x * inv, -2.0 * w * d.y * inv);
}

// The field, accumulated droplet by droplet so only one is live at a time.
fn chrome_field(p: vec2f, ca: f32, sa: f32, r: f32, breath: f32) -> vec3f {
  var f = 0.0;
  var g = vec2f(0.0);
  let a0 = droplet(p, L0, ca, sa, r * (1.0 + 0.16 * breath * L0.w));
  f = f + a0.x;
  g = g + a0.yz;
  let a1 = droplet(p, L1, ca, sa, r * (1.0 + 0.16 * breath * L1.w));
  f = f + a1.x;
  g = g + a1.yz;
  let a2 = droplet(p, L2, ca, sa, r * (1.0 + 0.16 * breath * L2.w));
  f = f + a2.x;
  g = g + a2.yz;
  let a3 = droplet(p, L3, ca, sa, r * (1.0 + 0.16 * breath * L3.w));
  f = f + a3.x;
  g = g + a3.yz;
  return vec3f(f, g.x, g.y);
}

/// The reflection of a studio in a chrome surface, from the normal's tilt: a dim floor, a dark horizon
/// band, a bright ceiling softbox and two vertical strips. All rational falloffs, no transcendental.
fn reflect_env(n: vec3f) -> vec3f {
  let up = n.y;
  var e = mix(vec3f(0.030, 0.040, 0.062), vec3f(0.26, 0.32, 0.42), smoothstep(-0.60, 0.80, up));
  e = e + vec3f(0.50, 0.57, 0.70) * glow(up - 0.60, 5.5);
  e = e - vec3f(0.075, 0.065, 0.055) * glow(up + 0.04, 8.0);
  let strip = glow(n.x - 0.55, 9.0) + 0.65 * glow(n.x + 0.64, 13.0);
  e = e + vec3f(0.56, 0.62, 0.74) * strip * (0.45 + 0.55 * clamp(up + 0.5, 0.0, 1.0));
  e = e + vec3f(0.15, 0.11, 0.09) * glow(up + 0.66, 4.0);
  return e;
}@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
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
  let grow = 1.0 + 0.12 * U.pulse + 0.10 * clamp(nrg, 0.0, 1.0);
  let ang = t * (0.22 + 0.5 * flow);
  let ca = cos(ang);
  let sa = sin(ang);
  let breath = sin(t * 0.5);
  let r = (0.20 + 0.13 * size) * grow;

  let fg = chrome_field(q, ca, sa, r, breath);
  let f = fg.x;
  let g = fg.yz;
  // d(field)/dp times one pixel is how far the field moves across a pixel: the exact anti-aliasing
  // width, for the price of one length.
  let aa = max(length(g) * px * 1.6, 1e-4);
  let inside = smoothstep(1.0 - aa, 1.0 + aa, f);
  let halo = (1.0 - inside) * clamp(f, 0.0, 1.0) * clamp(f, 0.0, 1.0);

  // Normal: the outward field gradient, plus one slow ripple on each axis so the reflection stays liquid.
  let gn = normalize(-g + vec2f(1e-6, 1e-6));
  let ripple = 0.10 + 0.85 * clamp(U.bass, 0.0, 1.0) + 0.35 * U.pulse;
  let n2 = gn + vec2f(
    sin(q.y * 5.0 + t * 1.1),
    cos(q.x * 5.3 - t * 0.9)
  ) * 0.07 * ripple;
  let n = normalize(vec3f(n2.x, n2.y, 0.60));

  let env = reflect_env(n);
  let edge = 1.0 - n.z;
  let fres = edge * edge * edge;
  let tint = irid(TAU * (0.06 + hue) + 0.35 * U.pulse);

  // Chrome: silver where the environment is, darker where the clip asks for it, with a warm bounce from
  // the floor. `chrome` pushes the shadow side down, which is what gives the droplets their weight.
  var surf = vec3f(0.74, 0.78, 0.86) * env * (1.0 + 0.35 * chrome);
  surf = surf * (1.0 - 0.38 * smoothstep(0.25, -0.60, n.y) * (0.35 + 0.65 * chrome));
  surf = surf + tint * fres * (0.18 + 0.50 * irid_amt);
  // Specular: two integer powers blended by Gloss, and the shoulder downstream keeps it off the ceiling.
  let ndl = max(dot(n, normalize(vec3f(-0.50, -0.65, 0.62))), 0.0);
  let p2 = ndl * ndl;
  let p16 = p2 * p2 * p2 * p2;
  surf = surf + vec3f(1.0, 0.99, 0.96) * mix(p2, p16 * p16, clamp(gloss, 0.0, 1.0)) * (0.30 + 0.70 * U.pulse);

  // Background: a deep indigo gradient, a teal pool low left, a violet wash high right, a floor bounce.
  let rise = smoothstep(-1.0, 1.0, -q.y);
  var col = mix(vec3f(0.010, 0.014, 0.038), vec3f(0.055, 0.024, 0.120), rise);
  col = col + vec3f(0.04, 0.13, 0.17) * falloff(length(q - vec2f(-0.82, 0.58)), 1.1)
    * (0.45 + 0.45 * clamp(nrg, 0.0, 1.3));
  col = col + tint * halo * 0.22;
  // The droplet's reflection of itself on the floor: a soft vertical smear under the mass.
  col = col + vec3f(0.22, 0.30, 0.44) * glow(q.x * 1.2, 0.55) * glow(q.y + 1.05, 2.2) * 0.35;
  col = mix(col, surf, inside);
  // The meniscus: a hairline of light where the metal ends, and a dark contact shadow under it.
  col = col + vec3f(0.70, 0.82, 1.0) * hairline(f - 1.0, aa * 2.0) * 0.45;
  col = col * (1.0 - 0.30 * glow(f - 1.0 - px * 30.0, 0.020 / max(px, 1e-4) * 0.0011));
  col = col + vec3f(0.85, 0.92, 1.0) * glints(q, t, 27.0, 0.972) * (0.10 + 0.75 * sparkle)
    * (0.30 + 0.40 * clamp(nrg, 0.0, 1.3));
  col *= 0.94 + 0.10 * smoothstep(1.9, 0.0, length(vec2f(q.x * 0.62, q.y)));
  return fin(shoulder(col));
}
