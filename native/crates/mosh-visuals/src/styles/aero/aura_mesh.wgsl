// Aura · Gradient Mesh: five colour pools bled into one grainy gradient, breathing with the track.
//
// Contemporary gradient-mesh art: no shapes, no edges, no light source. Five rational falloffs warp by
// one three-sine melt into a single field, and the field indexes the palette twice — one ramp for the
// pools that sit low and one for the pools that sit high — so the two ramps bleed through each other
// instead of averaging to grey. The corners fall back to a deep plate colour, which is what gives the
// mesh its shape in a dark room. Film grain is added last so the gradients never band on an 8-bit
// output, a bloom under the loud parts makes the pools read as light, and a shoulder keeps the hottest
// pool from clipping to flat white.
//
// Audio: `U.pulse` swells the pools and their bloom, `U.bass`/`U.mid` move the drift phase, `U.high`
// sharpens the bloom. Brightness follows the beat and slower, never faster than 400 ms.
//
// p0 Speed, p1 Blob scale, p2 Hue, p3 Spread, p4 Bloom, p5 Drift, p6 Grain, p7 Contrast.

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let t = U.T * (0.20 + 0.30 * P(0));
  let nrg = energy();
  let hue = P(2);

  // One three-sine melt warps the whole mesh, so the pools fold into one another.
  let warp = 0.18 + 0.42 * P(5);
  let wv = melt_field(q * 0.8, t * 1.4);
  let p = q + (wv - 0.5) * warp * 2.0;

  // Five pools: two big, three small, each on its own slow orbit. `Spread` pushes them apart.
  let spread = 0.55 + 0.75 * P(3);
  let scale = 0.55 + 0.75 * P(1);
  let sway = sin(t * 0.6);
  let sway_c = cos(t * 0.5);
  let w0 = falloff(length(p * vec2f(0.75, 1.10) - vec2f(-0.52, 0.30) * spread), 0.55 * scale)
    * (0.75 + 0.25 * sway);
  let w1 = falloff(length(p * vec2f(1.05, 0.85) - vec2f(0.58, -0.34) * spread), 0.60 * scale)
    * (0.75 + 0.25 * sway_c);
  let w2 = falloff(length(p - vec2f(0.05 + 0.10 * sway, 0.42 + 0.08 * sway_c)), 1.10 * scale);
  let w3 = falloff(length(p * vec2f(0.90, 0.90) + vec2f(0.30, 0.62)), 0.95 * scale);
  let w4 = falloff(length(p * vec2f(0.80, 0.70) - vec2f(0.10, -0.70)), 1.30 * scale);
  let total = w0 + w1 + w2 + w3 + w4;

  // Five colour sources, layered by dominance rather than averaged: averaging complementary hues returns
  // grey-white, which is exactly what a mesh must not do, so the strongest pool is laid down first and
  // the others are mixed over it in proportion to their weight against it. The hues come from the
  // palette's harmonised taps rotated by the melt field and by `Hue`, so the hue drifts across the mesh.
  let hue_ang = TAU * (0.03 + P(2));
  let tint = pal(cos(hue_ang), sin(hue_ang));
  let tint2 = pal_next(tint);
  let tint3 = next2(tint);
  let c1 = mix(tint, tint2, 0.60);
  let c3 = mix(tint2, tint3, 0.50);
  let c4 = mix(tint3, tint, 0.40);
  let peak = max(max(w0, w1), max(max(w2, w3), w4));
  let inv = 1.0 / max(peak, 1e-3);
  var col = mix(tint, c1, wv);
  col = mix(col, c1, clamp(w1 * inv, 0.0, 1.0) * 0.85);
  col = mix(col, tint3, clamp(w2 * inv, 0.0, 1.0) * 0.75);
  col = mix(col, c3, clamp(w3 * inv, 0.0, 1.0) * 0.85);
  col = mix(col, c4, clamp(w4 * inv, 0.0, 1.0) * 0.70);
  // The mesh's own light: where the pools overlap the mesh is lit, and at its edges it falls into the
  // plate, which is what gives a flat gradient its depth.
  let light = clamp(peak * 0.85, 0.0, 1.0);
  col = col * (0.30 + 1.00 * light) + vec3f(0.05, 0.045, 0.075) * light * light;

  // The plate: where no pool reaches, the mesh falls back to a deep colour, so the frame has shape.
  let presence = clamp(peak * 0.80, 0.0, 1.0);
  col = mix(vec3f(0.012, 0.010, 0.024) + lut(hue * 0.10) * 0.04, col, smoothstep(0.03, 0.90, presence));

  // The bloom: pool light under the loud parts, on the beat's clock, local to each pool.
  let bloom = (0.06 + 0.20 * P(4)) * (0.45 + 0.85 * U.pulse) * (0.6 + 0.4 * clamp(U.high, 0.0, 1.0));
  col = col + col * bloom * 1.2 * (w0 + w1 + w2) * inv;

  // Contrast around the mesh's own middle, then grain, then a soft vignette.
  let contrast = 0.95 + 0.45 * P(7);
  col = (col - 0.06) * contrast + 0.06;
  col = col * (0.92 + 0.10 * clamp(nrg, 0.0, 1.3));
  col *= grain(pos.xy, 0.06 + 0.10 * P(6));
  col *= 0.78 + 0.32 * smoothstep(1.9, 0.0, length(vec2f(q.x * 0.60, q.y * 0.92)));
  return fin(shoulder(col));
}
