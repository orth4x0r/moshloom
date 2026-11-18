// Aura · Contour Field: topographic iso-lines of a slow noise, drawn as light on a grainy plate.
//
// Three crossed sine lattices make the field, a three-sine melt warps it, and the contours are the
// field's own iso-lines: `fract(level)` numbers every line, and the line's width is divided by the
// analytic gradient magnitude, so a line is exactly a hairline wide everywhere — dense where the field
// steepens, never aliased, and no `fwidth` anywhere. Each contour takes its colour from its own level
// number, so the family walks through a rich ramp and a multiple-of-five contour is drawn thicker and
// brighter as a major line. Between them the plate keeps a deep grainy gradient with an aura pool, so
// the map has relief. It reads as a topographic survey of a surface that is not there.
//
// Audio: `U.mid` packs the contours tighter, `U.bass` deepens the plate, `U.pulse` sends one bright
// ring outward through the field and `U.high` lifts the line light. Nothing flashes globally.
//
// p0 Speed, p1 Contour density, p2 Hue, p3 Warp, p4 Line width, p5 Depth, p6 Bloom, p7 Grain.

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let px = 2.0 / U.res.y;
  let t = U.T * (0.25 + 0.35 * P(0));
  let nrg = energy();
  let warp = 0.12 + 0.55 * P(3) + 0.12 * U.pulse;

  // --- the field and its analytic gradient: three phases, each read as a sine and a cosine.
  let ph0 = q.x * 1.30 + q.y * 0.85 + t * 0.31 + warp * 2.0;
  let ph1 = q.x * 1.75 - q.y * 1.15 - t * 0.23 + 1.70 + warp * 3.1;
  let ph2 = (q.x + q.y) * 0.95 + t * 0.17 + 4.10 - warp * 1.4;
  let s0 = sin(ph0);
  let s1 = sin(ph1);
  let s2 = sin(ph2);
  let c0 = cos(ph0);
  let c1 = cos(ph1);
  let c2 = cos(ph2);
  let v = 0.50 * s0 + 0.32 * s1 + 0.18 * s2;
  let gx = 0.65 * c0 + 0.56 * c1 + 0.171 * c2;
  let gy = 0.425 * c0 - 0.368 * c1 + 0.171 * c2;
  let gm = max(length(vec2f(gx, gy)), 1e-3);

  // --- the contours: density packs them, and the line's screen width follows the field's own slope,
  // which is the exact anti-aliasing width rather than a guess.
  let density = 2.5 + 9.0 * clamp(P(1), 0.0, 1.0) + 4.0 * clamp(U.mid, 0.0, 1.0);
  let lv = v * density + t * 0.10;
  let dist = abs(fract(lv) - 0.5);
  let wline = clamp(px * gm * density * (0.7 + 2.0 * P(4)), 1e-4, 0.42);
  let line = smoothstep(0.5, 0.5 - wline, dist);
  let major = smoothstep(0.5, 0.5 - min(wline * 2.4, 0.42), dist)
    * step(fract(floor(lv + 0.5) * 0.2), 0.02);

  // --- colour: the plate's aura, then the lines coloured by their own level number.
  let hue_ang = TAU * (0.10 + P(2));
  let tint = pal(cos(hue_ang), sin(hue_ang));
  let tint2 = pal_next(tint);
  var c = plate(q, tint * 0.7, t, 2.2);
  c = c * (1.0 - 0.35 * P(5)) + tint2 * 0.055 * P(5) * smoothstep(1.1, -0.2, q.y);
  // A soft halo under the loud parts: the map glows where its lines are dense.
  c += tint * (0.05 + 0.16 * P(6)) * glow(dist - 0.5, 3.0) * (0.5 + 0.7 * U.pulse) * line;

  let idx = fract(floor(lv + 0.5) * 0.137 + P(2) * 0.5);
  let line_col = lut(idx) * (0.55 + 0.85 * P(6));
  c = mix(c, line_col, line * 0.80);
  c = mix(c, lut(idx + 0.08) * (1.10 + 0.55 * P(6)), major * 0.85);

  // The beat's ring: one expanding contour, once per beat, never faster.
  let shock = abs(length(q) - (0.24 + 1.10 * (1.0 - U.pulse)));
  c += tint * 0.30 * glow(shock, 30.0) * U.pulse;

  c *= 0.88 + 0.26 * clamp(nrg, 0.0, 1.0);
  c *= 0.90 + 0.20 * smoothstep(1.7, 0.15, length(vec2f(q.x * 0.66, q.y * 0.92)));
  c *= grain(pos.xy, 0.12 + 0.22 * P(7));
  return fin(shoulder(c));
}
