// Frutiger Aero · Caustic Pool.
//
// Looking down into a pool: the light net of caustics on the floor, a watery surface band at the
// top with the sky's reflection broken up by it, and a sheet of glass across the frame with
// droplets on it. The caustics are two pairs of travelling waves, sharpened where each pair
// cancels, which is the cheap way to get the light net without marching; the droplets are little
// lenses with a rim, a dark inner edge and a specular dot.
//
// p0 Speed, p1 Caustic scale, p2 Hue, p3 Droplets, p4 Glass, p5 Depth, p6 Glow, p7 Sparkle.
//
// Cost: four sines for the caustics, two for the surface, one palette. The glass is one rounded
// box; the seven droplets are written out one call each, and none of them needs a transcendental.

// The light net: two pairs of waves, sharpened where each pair cancels. Squared products keep the
// peaks thin without a `pow` of a possibly negative base.
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

// Rounded box: the glass sheet and the panel behind the droplets.
fn round_box(p: vec2f, half: vec2f, radius: f32) -> f32 {
  let d = abs(p) - half + radius;
  return length(max(d, vec2f(0.0, 0.0))) + min(max(d.x, d.y), 0.0) - radius;
}

// One droplet: the image inside a drop is inverted, so the top edge reads dark and the bottom
// bright, then a specular dot sits near the top left.
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
  var drop = mix(vec3f(0.40, 0.80, 0.90), vec3f(0.98, 1.0, 1.0), fres * 0.9);
  drop = drop * (1.0 - 0.35 * smoothstep(0.0, -0.7, dn.y) * (1.0 - fres));
  // A darker ring just inside the silhouette: the drop's own shadow on the glass.
  drop = drop * (1.0 - 0.30 * smoothstep(0.80, 0.97, r2) * (1.0 - smoothstep(0.98, 1.03, r2)));
  drop = drop + tint * 0.15 * fres;
  let ndl = max(dn.x * -0.4 + dn.y * -0.7 + z * 0.6, 0.0);
  let s2 = ndl * ndl;
  drop = drop + vec3f(1.0) * (s2 * s2 * s2 * s2 * s2) * 2.2;
  let a = smoothstep(1.12, 0.96, r2) * clamp(0.22 + 0.8 * fres, 0.0, 1.0);
  return mix(col, drop, a * amount);
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let t = U.T;
  let px = 2.0 / U.res.y;
  let hue = P(2);
  let nrg = energy();
  let deep = P(5);
  let surface = -0.72 + 0.10 * sin(t * 0.3) + 0.05 * U.bass;
  let scale = 3.0 + 7.0 * P(1) + 1.5 * U.bass;
  let hue_ang = TAU * (0.46 + hue);
  let tint = pal(cos(hue_ang), sin(hue_ang));
  let tint2 = pal_next(tint);

  // The water column: bright aqua under the surface, deep blue-green at the bottom.
  var col = mix(vec3f(0.50, 0.88, 0.92) + tint * 0.14, vec3f(0.01, 0.12, 0.24),
    smoothstep(surface, 1.0, q.y) * (0.5 + 0.7 * deep));
  // The caustic net lies on the pool floor, so it fades with depth.
  let floor_amt = smoothstep(surface - 0.05, 0.35, q.y) * (0.6 + 0.6 * (1.0 - deep));
  let net = caustics(vec2f(q.x, q.y * 1.35 + 0.3), t, scale);
  col = col + (tint * 0.75 + vec3f(0.40, 0.65, 0.50)) * net * floor_amt
    * (0.45 + 0.5 * nrg) * (0.6 + 0.6 * P(6));

  // The surface seen from below: a wavy bright band and the sky in the sliver above it.
  if (q.y < surface + 0.02) {
    let wave = sin(q.x * 9.0 + t * 0.8) * 0.02;
    let line = q.y - (surface + wave);
    let above = mix(vec3f(0.62, 0.88, 0.97), vec3f(0.94, 0.99, 1.0),
      smoothstep(-1.0, surface, q.y));
    // The sky is broken into ripple bands, bright where the surface is steep.
    let ripple = 0.5 + 0.5 * sin(q.x * 22.0 + q.y * 40.0 - t * 2.0);
    let r2 = ripple * ripple;
    let bands = r2 * r2 * r2 * r2;
    col = mix(col, above + vec3f(1.0) * bands * 0.22, smoothstep(0.05, -0.02, line));
    col = col + vec3f(1.0) * hairline(line, px * 3.0) * 0.45;
  }

  // A sheet of glass across the frame: tilted, frosted at its edge, with one travelling sheen.
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
    col = mix(col, col * 1.10 + tint2 * 0.08, inside * 0.35);
    col = col + vec3f(0.85, 0.98, 1.0) * edge * 0.28 * P(4);
    col = col + vec3f(1.0) * sheen * inside * 0.14 * P(4);
  }

  // Droplets on the glass, in front of everything: literal positions, one shared sway.
  let sway = sin(t * 0.25);
  let amount = P(3);
  col = droplet(col, q, vec2f(-0.62 + 0.02 * sway, -0.34), 0.055 * (0.4 + amount), tint, amount);
  col = droplet(col, q, vec2f(-0.18, 0.12 + 0.03 * sway), 0.035 * (0.4 + amount), tint2, amount);
  col = droplet(col, q, vec2f(0.34 + 0.02 * sway, -0.42), 0.070 * (0.4 + amount), tint, amount);
  col = droplet(col, q, vec2f(0.62, 0.28), 0.028 * (0.4 + amount), tint2, amount);
  col = droplet(col, q, vec2f(0.06, -0.62), 0.045 * (0.4 + amount), tint, amount);
  col = droplet(col, q, vec2f(-0.40, 0.42), 0.030 * (0.4 + amount), tint2, amount);
  col = droplet(col, q, vec2f(0.74, -0.10), 0.038 * (0.4 + amount), tint, amount);

  // Sparkle where the caustics are brightest: the net already found the light, so the glitter
  // costs no hash at all.
  col = col + vec3f(0.9, 1.0, 1.0) * net * net * 0.25 * P(7);
  return fin(col);
}
