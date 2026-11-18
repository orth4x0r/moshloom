// Glass · Caustic Net: the light net a still surface throws on the dark, with glass over it.
//
// No sky, no horizon: the frame is the deep end of a pool seen from just above it — a near-black teal
// plate, a net of caustics that is two pairs of travelling waves sharpened where each pair cancels, and
// bloom around the net where it is brightest. A tilted sheet of glass sits across the picture with one
// travelling sheen along its face, and seven droplets on the glass are little lenses: inverted image,
// dark inner ring, one hot dot. The net answers the spectrum, the beat swells it, and `shoulder` keeps
// the highlights off the ceiling — the previous version painted the water surface white, which is
// exactly the patch this pack must not have.
//
// p0 Speed, p1 Caustic scale, p2 Hue, p3 Droplets, p4 Glass, p5 Depth, p6 Glow, p7 Sparkle.

// The light net: two pairs of waves, sharpened where each pair cancels. Squared products keep the peaks
// thin without a `pow` of a possibly negative base.
fn caustics(p: vec2f, t: f32, scale: f32) -> f32 {
  let a = p * scale;
  let s1 = sin(a.x + t * 0.6) + sin(a.y * 1.3 - t * 0.5);
  let s2 = sin((a.x + a.y) * 0.9 + t * 0.4) + sin((a.x - a.y) * 1.15 - t * 0.7);
  let first = max(0.0, 1.0 - abs(s1 + s2) * 0.30);
  let sixth = first * first * first * first * first * first;
  let second = max(0.0, 1.0 - abs(s1 - s2) * 0.42);
  let fourth = second * second * second * second;
  return sixth + fourth * 0.55;
}

// Rounded box: the tilted sheet of glass the droplets sit on.
fn round_box(p: vec2f, half: vec2f, radius: f32) -> f32 {
  let d = abs(p) - half + radius;
  return length(max(d, vec2f(0.0, 0.0))) + min(max(d.x, d.y), 0.0) - radius;
}

// One droplet: the image inside a drop is inverted, so the top edge reads dark and the bottom bright,
// then a specular dot sits near the top left.
fn droplet(col: vec3f, q: vec2f, c: vec2f, r: f32, tint: vec3f, amount: f32) -> vec3f {
  let d = q - c;
  let r2 = dot(d, d) / (r * r);
  if (r2 > 1.12) {
    return col;
  }
  let dn = d / r;
  let z = sqrt(max(0.0, 1.0 - min(r2, 1.0)));
  let edge = 1.0 - z;
  let fres = edge * edge;
  var drop = mix(vec3f(0.03, 0.10, 0.14), tint * 0.75 + vec3f(0.20, 0.30, 0.34), fres * 0.9);
  drop = drop * (1.0 - 0.35 * smoothstep(0.0, -0.7, dn.y) * (1.0 - fres));
  drop = drop * (1.0 - 0.30 * smoothstep(0.80, 0.97, r2) * (1.0 - smoothstep(0.98, 1.03, r2)));
  drop = drop + tint * 0.18 * fres;
  let ndl = max(dn.x * -0.4 + dn.y * -0.7 + z * 0.6, 0.0);
  let s2 = ndl * ndl;
  drop = drop + vec3f(1.0) * (s2 * s2 * s2 * s2 * s2) * 1.1;
  let a = smoothstep(1.12, 0.96, r2) * clamp(0.20 + 0.8 * fres, 0.0, 1.0);
  return mix(col, drop, a * amount);
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let t = U.T;
  let px = 2.0 / U.res.y;
  let hue = P(2);
  let nrg = energy();
  let deep = P(5);
  let scale = 3.0 + 7.0 * P(1) + 1.5 * clamp(U.bass, 0.0, 1.0);
  let hue_ang = TAU * (0.30 + hue);
  let tint = pal(cos(hue_ang), sin(hue_ang));
  let tint2 = pal_next(tint);

  // --- the water column: a deep gradient, darker with `Depth`, with an aura pool in the water's own
  // colour so the plate is never flat.
  var col = mix(vec3f(0.020, 0.075, 0.095) + tint * 0.06, vec3f(0.004, 0.010, 0.026),
    smoothstep(-0.9, 1.0, q.y) * (0.45 + 0.75 * deep));
  col = col + (tint * 0.55 + vec3f(0.05, 0.16, 0.18)) * 0.22
    * falloff(length(q - vec2f(-0.35, 0.35)), 1.2) * (0.5 + 0.6 * U.bass);

  // --- the net on the floor: brightest where the waves cancel, flowering with the spectrum.
  let net = caustics(vec2f(q.x, q.y * 1.35 + 0.3), t, scale);
  let floor_amt = smoothstep(-0.95, 0.55, q.y) * (0.55 + 0.65 * (1.0 - deep));
  let glow_amt = 0.45 + 0.55 * P(6);
  col = col + (tint * 0.70 + vec3f(0.20, 0.45, 0.45)) * net * floor_amt
    * (0.40 + 0.55 * clamp(nrg, 0.0, 1.3)) * glow_amt;
  // Bloom around the net, not on it: the light spreads in the water.
  col = col + tint * 0.18 * net * net * floor_amt * glow_amt;

  // --- the sheet of glass across the frame: tilted, frosted at its edge, one travelling sheen.
  if (P(4) > 0.01) {
    // The tilt's cosine and sine are literal: the sheet only breathes a few degrees.
    let cs = 0.978;
    let sn = -0.208;
    let u = vec2f((q.x - 0.12) * cs - q.y * sn, (q.x - 0.12) * sn + q.y * cs);
    let d = round_box(u, vec2f(0.86, 0.52), 0.10);
    let edge = hairline(d, px * 2.0);
    let inside = smoothstep(0.01, -0.01, d) * P(4);
    let sweep = fract(t * 0.10 + 0.3) * 2.6 - 1.3;
    let sheen = falloff(u.x * 0.6 + u.y * 0.8 - sweep, 40.0);
    // Glass over water: the picture darkens a touch inside the sheet and gains a cold cast.
    col = mix(col, col * 0.92 + tint2 * 0.05, inside * 0.40);
    col = col + vec3f(0.55, 0.72, 0.85) * edge * 0.20 * P(4);
    col = col + vec3f(0.75, 0.92, 1.0) * sheen * inside * 0.30 * P(4);
  }

  // --- droplets on the glass, in front of everything: literal positions, one shared sway.
  let sway = sin(t * 0.25);
  let amount = P(3);
  col = droplet(col, q, vec2f(-0.62 + 0.02 * sway, -0.34), 0.055 * (0.4 + amount), tint, amount);
  col = droplet(col, q, vec2f(-0.18, 0.12 + 0.03 * sway), 0.035 * (0.4 + amount), tint2, amount);
  col = droplet(col, q, vec2f(0.34 + 0.02 * sway, -0.42), 0.070 * (0.4 + amount), tint, amount);
  col = droplet(col, q, vec2f(0.62, 0.28), 0.028 * (0.4 + amount), tint2, amount);

  // Sparkle where the net is brightest: the net already found the light, so the glitter costs no hash.
  col = col + vec3f(0.85, 0.98, 1.0) * net * net * 0.18 * P(7);
  col *= grain(pos.xy, 0.12 + 0.10 * deep);
  return fin(shoulder(col));
}
