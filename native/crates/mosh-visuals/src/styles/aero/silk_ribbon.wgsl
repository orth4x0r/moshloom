// Fold · Silk: five ribbons of satin light flowing across a dark plate.
//
// Each ribbon is one warped sine curve and a satin profile across its width: `satin` in
// `styles/paint.wgsl` gives the broad bright roll where the cloth turns toward the light, a darker fold
// on the far side and soft edges, so a flat band reads as cloth with volume. Every ribbon is a different
// width, frequency and phase, they overlap back to front, and their colours come from the palette at
// five different offsets, so the frame is a still-life of moving silk rather than a stack of stripes.
// The plate behind is a grainy gradient with an aura pool and the ribbons throw a soft shadow onto it,
// which is what separates them in depth. A shoulder rolls the satin highlight off instead of clipping.
//
// Audio: `U.pulse` brightens the roll and thickens the ribbons, `U.bass` lifts their amplitude,
// `U.high` speeds the flow, and the ribbons' drift follows the beat and slower — never a strobe.
//
// p0 Speed, p1 Ribbons, p2 Hue, p3 Width, p4 Flow, p5 Sheen, p6 Weave, p7 Grain.

/// One ribbon at `y0`: its satin body and its coverage. One sine for the curve, no hash.
fn ribbon(
  q: vec2f, y0: f32, amp: f32, k: f32, w: f32, ph: f32, t: f32, tint: vec3f, sheen: f32
) -> vec4f {
  let curve = y0 + amp * sin(q.x * k + ph + t * 0.55);
  let u = (q.y - curve) * (1.0 / max(w, 1e-4));
  let cover = smoothstep(1.0, 0.78, abs(u));
  let body = satin(u);
  var col = tint * (0.20 + 1.00 * body);
  // The roll: the band of light where the cloth turns toward the viewer.
  col = col + vec3f(1.0, 0.98, 0.95) * glow(u + 0.40, 8.0) * (0.10 + 0.55 * sheen);
  // The fold on the far side stays in its own shadow.
  col = col * (1.0 - 0.38 * smoothstep(0.15, 0.95, u));
  return vec4f(col, cover);
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let t = U.T * (0.30 + 0.55 * P(0));
  let nrg = energy();
  let hue = P(2);
  let sheen = clamp(P(5), 0.0, 1.0);
  let width = 0.09 + 0.15 * P(3);
  let flow = 0.4 + 1.3 * P(4) + 0.4 * clamp(U.high, 0.0, 1.0);
  let tint = lut(0.04 + 0.14 * hue);

  // --- the plate, its aura pool, and the shadow the ribbons throw on it.
  var c = plate(q, tint * 0.8, t, 5.3);
  c += pal_next(tint) * 0.09 * glow(length(q - vec2f(0.42, 0.34)), 0.75) * (0.5 + 0.7 * U.pulse);
  c *= 1.0 - 0.18 * smoothstep(0.9, -0.1, q.y) * (0.4 + 0.6 * nrg);

  // --- the ribbons, back to front: two mid ribbons and the front one are always drawn, and `Ribbons`
  // adds the thin back ribbon. The branch is coherent (the control is a uniform), so a clip with three
  // ribbons pays for three.
  let lift = 0.10 * clamp(U.bass, 0.0, 1.0) + 0.06 * U.pulse;
  // Back layer: a thin ribbon, dim, out of focus in colour. It is the `Ribbons` control's high end, so a
  // clip that asks for three ribbons pays for three.
  if (P(1) > 0.75) {
    let r2 = ribbon(q, 0.80 + lift, 0.13, 1.45, width * 0.55, 4.4, t * 0.7,
      lut(0.05 + 0.10 * hue + 0.18), sheen * 0.5);
    c = mix(c, r2.rgb, r2.a * 0.60);
  }
  // Two mid ribbons, crossed, because a single pair reads as a stripe and three read as cloth.
  let mid = ribbon(q, 0.34 + lift, 0.24, 0.95 * flow, width * 0.82, 0.9, t,
    lut(0.05 + 0.10 * hue + 0.52), sheen * 0.8);
  c = mix(c, mid.rgb, mid.a * 0.82);
  let mid2 = ribbon(q, -0.18 - lift, 0.30, 1.30 * flow, width * 0.92, 3.2, t,
    lut(0.05 + 0.10 * hue + 0.68), sheen * 0.9);
  c = mix(c, mid2.rgb, mid2.a * 0.88);
  // Front ribbon: the glossiest and the widest, and it lays one highlight along its whole length.
  let front = ribbon(q, 0.08 + lift * 0.6, 0.36, 0.78 * flow, width * 1.15, 5.1, t,
    lut(0.05 + 0.10 * hue + 0.02), sheen * 1.2);
  // The weave: the front ribbon crosses in front of the plate with a soft shadow under it.
  c = c * (1.0 - 0.22 * front.a * (0.4 + 0.6 * P(6)));
  c = mix(c, front.rgb * (1.0 + 0.25 * P(6)), front.a);
  // The ribbons' own bloom: satin light spilling onto the plate, once per beat and slower.
  c = c + front.rgb * 0.16 * U.pulse * front.a;

  c *= 0.94 + 0.12 * clamp(nrg, 0.0, 1.0);
  c *= 0.90 + 0.20 * smoothstep(1.7, 0.15, length(vec2f(q.x * 0.66, q.y * 0.92)));
  c *= grain(pos.xy, 0.12 + 0.20 * P(7));
  return fin(shoulder(c));
}
