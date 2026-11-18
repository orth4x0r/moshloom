// Glass · Liquid Lens: a mass of molten glass refracting a field of colour, with real dispersion.
//
// Three lobes make one liquid lens: the field is a sum of inverse-square blobs, so they merge and split
// as they orbit, and the field's own gradient is the surface normal. Inside the lens the backdrop is
// sampled through the normal, and the three colour channels are sampled at slightly different strengths
// of the same displacement — which is dispersion, light bending by wavelength, the reason glass shows an
// orange edge on one side and a blue one on the other. A fresnel rim, a specular roll, a darkened
// thickness at the silhouette and a caustic hotspot under the lens finish it, and the whole sheet sits
// on a grainy gradient with an aura pool. A shoulder keeps the specular from clipping.
//
// Audio: `U.pulse` swells the lobes and widens the dispersion, `U.bass` sloshes the cluster, `U.high`
// lifts the rim. The lens moves at the beat's rate or slower.
//
// p0 Speed, p1 Lens size, p2 Hue, p3 Dispersion, p4 Refraction, p5 Frost, p6 Glow, p7 Grain.

/// One lobe of the metaball: (field, d(field)/dp.x, d(field)/dp.y). One reciprocal.
fn lobe(p: vec2f, c: vec2f, r: f32) -> vec3f {
  let d = p - c;
  let inv = 1.0 / (dot(d, d) + 0.0008);
  let w = r * r * inv;
  return vec3f(w, -2.0 * w * d.x * inv, -2.0 * w * d.y * inv);
}

/// The backdrop the lens refracts: one palette, a melt warp and a soft vertical wash. Three sines.
fn backdrop(q: vec2f, t: f32, hue: f32) -> vec3f {
  let w = melt_field(q * 0.90 + vec2f(0.0, t * 0.05), t * 0.8);
  let v = clamp(0.40 + 0.55 * w - 0.18 * q.y, 0.0, 1.0);
  var c = lut(0.05 + 0.85 * v + hue * 0.12) * (0.28 + 0.98 * v);
  // Two soft colour pools so the refraction has structure to bend, not just a gradient.
  c += lut(0.20 + hue * 0.12 + 0.55) * 0.20 * falloff(length(q - vec2f(-0.55, 0.30)), 1.2);
  c += lut(0.60 + hue * 0.12) * 0.16 * falloff(length(q - vec2f(0.62, -0.42)), 1.6);
  return c;
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let px = 2.0 / U.res.y;
  let t = U.T * (0.22 + 0.36 * P(0));
  let nrg = energy();
  let hue = P(2);
  let grow = 1.0 + 0.10 * U.pulse + 0.08 * clamp(nrg, 0.0, 1.3);
  let r = (0.20 + 0.14 * P(1)) * grow;
  let ang = t * (0.16 + 0.30 * clamp(U.bass, 0.0, 1.0));
  let ca = cos(ang);
  let sa = sin(ang);
  let breath = 0.5 + 0.5 * sin(t * 0.7);

  // The three lobes on one shared orbit, uneven so the merged mass never reads as a flower.
  let a0 = lobe(q, vec2f(0.42 * ca - 0.06 * sa, 0.42 * sa + 0.06 * ca), r * (0.95 + 0.10 * breath));
  let a1 = lobe(q, vec2f(-0.34 * ca - 0.30 * sa, -0.34 * sa + 0.30 * ca), r * (0.82 + 0.14 * breath));
  let a2 = lobe(q, vec2f(0.04 * ca + 0.40 * sa, 0.04 * sa - 0.40 * ca), r * (1.06 - 0.10 * breath));
  let f = a0.x + a1.x + a2.x;
  let g = vec2f(a0.y + a1.y + a2.y, a0.z + a1.z + a2.z);
  let gm = max(length(g), 1e-5);

  // Anti-aliasing from the field's own gradient: how far the surface moves across one pixel.
  let aa = max(gm * px * 1.5, 1e-4);
  let inside = smoothstep(1.0 - aa, 1.0 + aa, f);
  let edge = clamp(f * 0.5, 0.0, 1.0);

  // --- the world behind the glass.
  var c = backdrop(q, t, hue);
  c *= 0.94 + 0.14 * smoothstep(1.8, 0.0, length(vec2f(q.x * 0.6, q.y)));

  // --- the lens: disp at three scales is dispersion, and the sample point is the surface normal.
  let gdir = g / gm;
  let bend = (0.05 + 0.16 * P(4)) * (0.35 + 0.65 * (1.0 - edge));
  let spread = 1.0 + 1.2 * P(3);
  let base = q - gdir * bend;
  let cr = backdrop(base - gdir * bend * 0.16 * spread, t, hue);
  let cg = backdrop(base, t, hue);
  let cb = backdrop(base + gdir * bend * 0.16 * spread, t, hue);
  var glass = vec3f(cr.r, cg.g, cb.b);
  // Frost: the glass scatters, so the refracted colour washes toward its own average.
  let avg = (cr + cg + cb) / 3.0;
  glass = mix(glass, avg * 1.05 + 0.05, P(5) * 0.75);

  // The glass's own body: a cold silver tint at the rim, a shadow of thickness toward the silhouette,
  // and a fresnel edge that catches the light.
  let nz = sqrt(max(1.0 - edge * edge, 0.0));
  let fres = (1.0 - nz) * (1.0 - nz);
  glass = glass * (0.62 + 0.38 * nz);
  glass = glass + pal_next(lut(0.04 + 0.14 * hue)) * fres * (0.22 + 0.55 * U.pulse);
  // One specular roll, up and to the left, and its soft companion from the right.
  let roll = max(-gdir.x * 0.55 + gdir.y * 0.55 + nz * 0.62, 0.0);
  let roll2 = roll * roll * roll * roll;
  glass = glass + vec3f(0.98, 0.99, 1.0) * roll2 * roll2 * roll2 * (0.55 + 0.55 * P(6));
  glass = glass + vec3f(0.90, 0.95, 1.0) * glow(length(q - vec2f(-0.30, 0.34)) - r * 0.42, 5.0)
    * 0.35 * P(6);

  let cov = inside;
  c = mix(c, glass, cov);

  // The meniscus: a hairline of light where the glass ends, and a caustic just under it.
  c += vec3f(0.80, 0.92, 1.0) * hairline(f - 1.0, aa * 1.8) * (0.30 + 0.45 * P(6));
  c += pal_next(lut(0.04 + 0.14 * hue)) * (0.05 + 0.10 * P(6)) * glow(f - 1.0 - px * 26.0, 20.0);
  // The shadow the lens throws, so it sits on the plate instead of floating on it.
  c *= 1.0 - 0.28 * glow(f - 1.0 - px * 34.0, 0.030 / max(px, 1e-4) * 0.001);

  c *= grain(pos.xy, 0.12 + 0.20 * P(7));
  return fin(shoulder(c));
}
