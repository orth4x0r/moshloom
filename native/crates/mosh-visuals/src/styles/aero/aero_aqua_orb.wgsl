// Glass · Iridescent Orb: a glass sphere lit from inside, floating on a grainy gradient.
//
// Not a landscape: a deep plum-to-teal plate with two aura pools and film grain, and one sphere in front
// of it. The sphere is glass rather than plastic — a dark refracted body, a caustic net swimming inside
// it (three travelling waves sharpened where they cancel), a fresnel rim that takes the clip's hue, a
// broad key and one hot specular dot. Small glass beads sit in the same light, and out-of-focus bokeh
// gives the plate depth. Everything bright ends in `shoulder`, so the specular is hot without a flat
// white patch, which is what the old sky-and-clouds version did.
//
// p0 Speed, p1 Orb size, p2 Hue, p3 Haze, p4 Gloss, p5 Rim light, p6 Beads, p7 Reflection.

// A small glass bead in front of the orb: rim, one hot dot, and the same light as the big one.
fn bead(col: vec3f, q: vec2f, c: vec2f, r: f32, tint: vec3f, sun: vec2f) -> vec3f {
  let d = q - c;
  let r2 = dot(d, d) / (r * r);
  if (r2 > 1.08) {
    return col;
  }
  let dn = d / r;
  let z = sqrt(max(0.0, 1.0 - r2));
  let fres = (1.0 - z) * (1.0 - z);
  var b = mix(vec3f(0.05, 0.09, 0.13), tint * 0.85 + vec3f(0.10, 0.14, 0.18), fres);
  let ndl = max(dn.x * -0.42 + dn.y * -0.62 + z * 0.66, 0.0);
  let s2 = ndl * ndl;
  b = b + vec3f(1.0) * (s2 * s2 * s2 * s2) * 1.2;
  b = b + vec3f(1.0, 0.98, 0.85) * falloff(length(q - sun) * 2.0, 6.0) * 0.6;
  return mix(col, b, smoothstep(1.08, 0.97, r2) * clamp(0.22 + 0.8 * fres, 0.0, 1.0));
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let t = U.T;
  let px = 2.0 / U.res.y;
  let hue = P(2);
  let nrg = energy();
  let sun = vec2f(0.72, -0.88);
  let hue_ang = TAU * (0.30 + hue);
  let tint = pal(cos(hue_ang), sin(hue_ang));
  let tint2 = pal_next(tint);
  let sway = sin(t * 0.22);
  let sway_c = cos(t * 0.22);

  // --- the plate: a grainy gradient with a warm pool and a cold one, plus the haze control's wash.
  var col = plate(q, tint * 0.8, t, 6.4);
  let haze = 0.30 + 0.55 * P(3);
  col = col + tint2 * 0.10 * falloff(length(q - vec2f(-0.55, 0.45)), 1.4) * haze;
  col = col + mix(tint, vec3f(0.30, 0.55, 0.60), 0.5) * 0.10
    * falloff(length(q - vec2f(0.60, -0.40)), 1.8) * haze;
  // A slow wash of colour, out of focus behind the glass.
  col = col + tint * 0.07 * smoothstep(0.5, -0.4, q.y) * haze;

  // Bokeh behind the orb: soft discs on literal orbits, all drifting with one sway.
  col = col + bokeh(q, vec2f(-0.80, 0.10) + vec2f(0.06, 0.03) * sway, 0.15, tint, nrg);
  col = col + bokeh(q, vec2f(0.62, 0.42) + vec2f(-0.05, 0.04) * sway_c, 0.12, tint2, nrg);
  col = col + bokeh(q, vec2f(0.30, -0.60) + vec2f(0.04, -0.03) * sway, 0.17, tint, nrg) * 0.7;
  col = col + bokeh(q, vec2f(-0.45, -0.62) + vec2f(-0.03, 0.02) * sway_c, 0.10, tint2, nrg);

  // --- the orb.
  let orb_c = vec2f(-0.10, 0.10);
  let orb_r = (0.26 + 0.28 * P(1)) * (1.0 + 0.03 * U.pulse + 0.04 * nrg);
  let os = (q - orb_c) / orb_r;
  let orr = dot(os, os);
  let aa = 2.5 * px * max(length(os), 0.2) / orb_r;
  if (orr < 1.0 + aa) {
    let z = sqrt(max(0.0, 1.0 - min(orr, 1.0)));
    let zt = sqrt(max(z, 0.0));
    // Body: dark glass at the rim, the hue's own light in the middle.
    var body = mix(vec3f(0.012, 0.018, 0.035), tint * 0.62 + vec3f(0.03, 0.08, 0.08), zt);
    // The caustic net inside the glass: three waves, sharpened where they cancel, coloured by the hue
    // and lifted with the spectrum so it swims instead of being drawn on.
    let ca = abs(sin(os.x * 7.0 + t * 0.6) + sin(os.y * 9.0 - t * 0.5)
      + sin((os.x + os.y) * 5.5 + t * 0.9));
    let c1 = max(0.0, 1.0 - ca * 0.50);
    let c4 = c1 * c1;
    let caustic = c4 * c4 * c4 * zt;
    body = body + (tint * 0.95 + vec3f(0.10, 0.30, 0.26)) * caustic
      * (0.30 + 0.45 * clamp(U.mid, 0.0, 1.0));
    // The key: a broad lobe and one hot dot, both integer powers of the same dot product.
    let ndl = max(os.x * -0.42 + os.y * -0.62 + z * 0.66, 0.0);
    let s2 = ndl * ndl;
    let s4 = s2 * s2;
    let s8 = s4 * s4;
    body = body + tint2 * mix(s2, s4, P(4)) * 0.26;
    body = body + vec3f(0.98, 1.0, 1.0) * mix(s8 * s8, s8 * s8 * s8, P(4)) * (0.45 + 0.45 * U.pulse);
    // The rim: the hue's light inside, a thin bright meniscus outside, so the sphere has an edge in
    // the dark.
    let edge = 1.0 - z;
    let fres = edge * edge * edge;
    body = body * (1.0 - 0.40 * smoothstep(0.68, 1.0, orr));
    body = body + tint * fres * (0.45 + 1.0 * P(5)) * (0.70 + 0.30 * nrg);
    body = body + vec3f(0.90, 0.96, 1.0) * fres * 0.18 * P(5);
    col = mix(col, body, smoothstep(1.0 + aa, 1.0 - aa, orr));
  }

  // Small beads in front, on literal orbits, drifting with the same sway.
  let bubbly = P(6);
  col = bead(col, q, vec2f(0.42, -0.10) + vec2f(0.05, 0.02) * sway, 0.05 * (0.4 + bubbly), tint, sun);
  col = bead(col, q, vec2f(0.18, 0.52) + vec2f(-0.04, 0.03) * sway_c, 0.035 * (0.4 + bubbly), tint2, sun);
  col = bead(col, q, vec2f(-0.62, -0.30) + vec2f(-0.03, 0.02) * sway_c, 0.045 * (0.4 + bubbly), tint2, sun);

  // --- the orb's reflection on the plate under it: a soft vertical smear in its own colour.
  let under = vec2f((q.x - orb_c.x) / (orb_r * 1.15), (q.y - orb_c.y - orb_r * 1.15) / (orb_r * 0.75));
  col = col + (tint * 0.55 + vec3f(0.10, 0.16, 0.20)) * falloff(length(under), 2.2)
    * (0.10 + 0.40 * P(7));
  col = col + tint * 0.06 * smoothstep(1.0, -0.4, q.y) * (0.4 + 0.6 * nrg);
  col *= grain(pos.xy, 0.13 + 0.10 * P(3));
  return fin(shoulder(col));
}
