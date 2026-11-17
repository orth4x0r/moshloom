// Frutiger Aero · Aqua Orb.
//
// The pack's cover shot: a glossy aqua sphere over a clean sky gradient, lit like a product
// photograph. A broad soft key plus a hot specular dot, a white fresnel rim, caustic ribbons
// swimming inside the glass, and a soft reflection on the floor under it. Behind it, slow
// optimistic clouds; six bokeh discs and five small bubbles give the depth the style lives on.
//
// p0 Speed, p1 Orb size, p2 Hue, p3 Clouds, p4 Gloss, p5 Rim light, p6 Bubbles, p7 Floor sheen.
//
// Cost: one sphere, three sines for the clouds, two for the caustics, one shared sway for every
// drifting element. The orb's own shading is all multiplication: the normals are already unit
// length and the speculars are integer powers, so there is no `normalize` and no `pow`.

// A small bubble in front of the orb: rim, one hot dot, and the same tint as the big one.
fn bead(col: vec3f, q: vec2f, c: vec2f, r: f32, tint: vec3f, sun: vec2f) -> vec3f {
  let d = q - c;
  let r2 = dot(d, d) / (r * r);
  if (r2 > 1.08) {
    return col;
  }
  let dn = d / r;
  let z = sqrt(max(0.0, 1.0 - r2));
  let fres = (1.0 - z) * (1.0 - z);
  var b = mix(vec3f(0.30, 0.70, 0.82), vec3f(0.95, 1.0, 1.0), fres);
  let ndl = max(dn.x * -0.42 + dn.y * -0.62 + z * 0.66, 0.0);
  let s2 = ndl * ndl;
  b = b + vec3f(1.0) * (s2 * s2 * s2 * s2) * 2.0;
  // The key's own reflection of the light source, so the bead sits in the same world as the orb.
  b = b + vec3f(1.0, 0.98, 0.85) * falloff(length(q - sun) * 2.0, 6.0);
  return mix(col, b, smoothstep(1.08, 0.97, r2) * clamp(0.25 + 0.8 * fres, 0.0, 1.0));
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let t = U.T;
  let px = 2.0 / U.res.y;
  let hue = P(2);
  let nrg = energy();
  let sun = vec2f(0.72, -0.88);
  let hue_ang = TAU * (0.46 + hue);
  let tint = pal(cos(hue_ang), sin(hue_ang));
  let tint2 = pal_next(tint);
  let sway = sin(t * 0.22);
  let sway_c = cos(t * 0.22);

  // Sky: pale blue overhead, almost white at the horizon, light from the top right.
  let sky_t = smoothstep(-1.0, 1.0, q.y);
  var col = mix(vec3f(0.44, 0.72, 0.93), vec3f(0.83, 0.93, 0.98), sky_t);
  col = col + vec3f(1.0, 0.99, 0.90) * falloff(length(q - sun), 2.6) * (0.50 + 0.12 * nrg);
  col = col + tint * 0.04 * smoothstep(0.2, -0.6, q.y);

  // Clouds: the shared three-wave field, thresholded soft so no edge is ever sharp.
  let cover = 0.30 + 0.50 * P(3);
  let cl = clouds(vec2f(q.x * 1.1, q.y * 1.7), t);
  let cloud = smoothstep(0.66 - cover * 0.24, 0.90 - cover * 0.14, cl)
    * smoothstep(0.9, 0.05, q.y) * smoothstep(-1.0, -0.2, q.y);
  col = mix(col, vec3f(0.98, 1.0, 1.0), cloud * (0.5 + 0.4 * P(3)));
  col = mix(col, tint2 * 0.5 + vec3f(0.22, 0.34, 0.44), cloud * 0.10);

  // Bokeh behind the orb: soft discs on literal orbits, all drifting with one sway.
  col = col + bokeh(q, vec2f(-0.80, 0.10) + vec2f(0.06, 0.03) * sway, 0.13, tint, nrg);
  col = col + bokeh(q, vec2f(0.62, 0.42) + vec2f(-0.05, 0.04) * sway_c, 0.10, tint2, nrg);
  col = col + bokeh(q, vec2f(0.30, -0.55) + vec2f(0.04, -0.03) * sway, 0.15, tint, nrg);
  col = col + bokeh(q, vec2f(-0.45, -0.62) + vec2f(-0.03, 0.02) * sway_c, 0.09, tint2, nrg);

  // The orb.
  let orb_c = vec2f(-0.12, 0.12);
  let orb_r = (0.26 + 0.28 * P(1)) * (1.0 + 0.03 * U.pulse + 0.04 * nrg);
  let os = (q - orb_c) / orb_r;
  let orr = dot(os, os);
  let aa = 2.5 * px * max(length(os), 0.2) / orb_r;
  if (orr < 1.0 + aa) {
    let z = sqrt(max(0.0, 1.0 - min(orr, 1.0)));
    // Body: deep teal at the edge, glowing aqua in the middle, with the hue control's colour.
    let zt = sqrt(max(z, 0.0));
    var body = mix(vec3f(0.02, 0.10, 0.16) + tint * 0.10, tint * 0.95 + vec3f(0.0, 0.12, 0.05), zt);
    // Caustic ribbons: three travelling waves that sharpen where they cancel, so the light inside
    // the glass is a net of cells rather than a lattice of two sines.
    let ca = abs(sin(os.x * 7.0 + t * 0.6) + sin(os.y * 9.0 - t * 0.5)
      + sin((os.x + os.y) * 5.5 + t * 0.9));
    let c1 = max(0.0, 1.0 - ca * 0.55);
    let caustic = c1 * c1 * c1 * c1 * z;
    body = body + (tint * 0.55 + vec3f(0.30, 0.55, 0.35)) * caustic * (0.35 + 0.40 * U.mid);
    // Key light top left: a broad lobe and a hot dot, both integer powers of the same dot product.
    let ndl = max(os.x * -0.42 + os.y * -0.62 + z * 0.66, 0.0);
    let s2 = ndl * ndl;
    let s4 = s2 * s2;
    let s8 = s4 * s4;
    body = body + vec3f(0.92, 1.0, 1.0) * mix(s2, s4, P(4)) * 0.35;
    body = body + vec3f(1.0) * mix(s8 * s8, s8 * s8 * s8, P(4)) * (1.6 + 0.8 * U.pulse);
    // White fresnel rim and the darker inner edge the gloss needs.
    let edge = 1.0 - z;
    let fres = edge * edge * edge;
    body = body * (1.0 - 0.30 * smoothstep(0.75, 1.0, orr));
    body = body + vec3f(0.88, 1.0, 1.0) * fres * (0.35 + 1.1 * P(5)) * (0.75 + 0.25 * nrg);
    // The lower inner face catches a second, greener light: the Frutiger bounce.
    body = body + vec3f(0.55, 1.0, 0.55) * sq(max(os.y, 0.0)) * sq(max(os.y, 0.0)) * 0.25 * z;
    col = mix(col, body, smoothstep(1.0 + aa, 1.0 - aa, orr));
  }

  // Small bubbles in front, on literal orbits, drifting with the same sway.
  let bubbly = P(6);
  col = bead(col, q, vec2f(0.42, -0.10) + vec2f(0.05, 0.02) * sway, 0.05 * (0.4 + bubbly), tint, sun);
  col = bead(col, q, vec2f(0.18, 0.52) + vec2f(-0.04, 0.03) * sway_c, 0.035 * (0.4 + bubbly), tint2, sun);
  col = bead(col, q, vec2f(0.58, 0.30) + vec2f(0.03, -0.02) * sway, 0.028 * (0.4 + bubbly), tint, sun);
  col = bead(col, q, vec2f(-0.62, -0.30) + vec2f(-0.03, 0.02) * sway_c, 0.045 * (0.4 + bubbly), tint2, sun);
  col = bead(col, q, vec2f(0.30, -0.55) + vec2f(0.04, 0.0) * sway, 0.022 * (0.4 + bubbly), tint, sun);

  // Floor: a broad soft reflection under the orb, a bright band where the floor meets the sky, and
  // a dark glassy shadow so the orb is seated rather than floating.
  let floor_y = orb_c.y + orb_r * 0.98;
  if (q.y > floor_y) {
    let dx = (q.x - orb_c.x) / (orb_r * 1.25);
    let dy = (q.y - floor_y) / (orb_r * 0.9);
    let r2 = dot(vec2f(dx, dy), vec2f(dx, dy));
    col = mix(col, tint * 0.45 + vec3f(0.30, 0.50, 0.52), falloff(r2, 2.0) * (0.18 + 0.45 * P(7)));
    col = col * (1.0 - 0.18 * falloff(r2, 6.0) * P(7));
  }
  col = col + vec3f(0.92, 1.0, 1.0) * falloff(abs(q.y - 0.85), 500.0) * 0.12 * (0.4 + 0.6 * P(7));
  return fin(col);
}
