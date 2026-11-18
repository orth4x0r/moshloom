// Fold · Foil: two creased families crossing into a sheet of holographic foil.
//
// A folded sheet: two families of V-creases, one running steeply across the frame and one shallow,
// each `abs(fract(u) - 0.5)` of a warped coordinate, so every crease is a sharp fold and every panel
// between them tilts. The height field's analytic gradient is the surface normal, and the normal is what
// indexes the colour: a holographic ramp turns with the tilt, so the foil reads as thin-film metal —
// magenta into teal into gold across one panel, then back the other way across the next. A shadow term
// darkens the falling side of every fold, thin bright lines sit on the creases themselves, and a sheen
// travels across the whole sheet, so the frame has depth and a light direction instead of being a flat
// pattern. A shoulder keeps the crease highlights from clipping to white.
//
// Audio: `U.high` sharpens the crease light, `U.bass` deepens the fold shadow, `U.pulse` swings the
// holographic ramp half a turn and brightens the sheen. The ramp's swing is one value per frame, so
// nothing flashes faster than the beat.
//
// p0 Speed, p1 Fold scale, p2 Hue, p3 Fold depth, p4 Holo shift, p5 Sheen, p6 Crease, p7 Grain.

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let t = U.T * (0.22 + 0.34 * P(0));
  let nrg = energy();
  let scale = 0.55 + 0.85 * P(1);
  let depth = 0.35 + 1.05 * P(3);

  // --- the sheet: one warped coordinate per crease family, `warp` making the folds flow.
  let warp = (melt_field(q * 0.75, t * 1.2) - 0.5) * (0.5 + 0.9 * nrg);
  // Family one, steep.
  let u1 = (q.x * 1.05 + q.y * 0.30 + warp * 0.60 + t * 0.12) * 0.62 * scale;
  let f1 = abs(fract(u1) - 0.5) * 2.0;
  let s1 = select(-1.0, 1.0, fract(u1) > 0.5);
  // Family two, shallow and crossing.
  let u2 = (q.y * 0.85 - q.x * 0.25 + warp * 0.42 - t * 0.09) * 0.47 * scale;
  let f2 = abs(fract(u2) - 0.5) * 2.0;
  let s2 = select(-1.0, 1.0, fract(u2) > 0.5);

  // The height is the sum of the two folds (linear in each, so the crease stays a crease) plus two slow
  // waves, and the normal is its analytic gradient — no `dpdx`, so the shading is smooth inside a fold.
  // The waves are what keep a panel from being one flat colour: they tilt the normal across the sheet.
  let wave = 0.30 * sin(q.x * 1.30 + q.y * 0.70 + t * 0.20 + warp)
    + 0.22 * sin(q.y * 2.10 - q.x * 0.55 - t * 0.17);
  let wave_x = 0.30 * cos(q.x * 1.30 + q.y * 0.70 + t * 0.20 + warp) * 1.30
    - 0.22 * cos(q.y * 2.10 - q.x * 0.55 - t * 0.17) * 0.55;
  let wave_y = 0.30 * cos(q.x * 1.30 + q.y * 0.70 + t * 0.20 + warp) * 0.70
    + 0.22 * cos(q.y * 2.10 - q.x * 0.55 - t * 0.17) * 2.10;
  let dhx = s1 * 1.40 * 0.62 * 1.05 * scale * depth + s2 * 1.40 * 0.47 * -0.25 * scale * depth
    + wave_x * (0.6 + 0.8 * depth);
  let dhy = s1 * 1.40 * 0.62 * 0.30 * scale * depth + s2 * 1.40 * 0.47 * 0.85 * scale * depth
    + wave_y * (0.6 + 0.8 * depth);
  let n = normalize(vec3f(-dhx, -dhy, 1.0));

  // --- the foil: a holographic ramp indexed by the tilt and by the sheet's own drift, so a panel always
  // carries a gradient across it rather than one flat colour.
  let shift = P(4) * 0.9 + P(2) + 0.16 * t + 0.30 * U.pulse;
  let holo = irid(0.30 + 0.70 * n.x + 0.42 * n.y + 0.22 * wave + shift);
  let lit = 0.30 + 0.70 * (0.5 + 0.5 * n.y);
  var col = holo * (0.40 + 0.85 * lit);
  // A cold metal top end: the foil's silver, so the holo does not read as plastic.
  col = mix(col, vec3f(0.66, 0.70, 0.78) * (0.40 + 0.85 * lit), 0.20 + 0.18 * P(6));

  // The crease itself: a thin bright line on the fold, the way foil catches the edge of the light.
  let crease = glow(f1, 16.0) + 0.65 * glow(f2, 22.0);
  col = col + vec3f(1.0, 0.98, 0.94) * crease * (0.16 + 0.55 * P(6)) * (0.7 + 0.3 * clamp(U.high, 0.0, 1.0));

  // The falling side of each fold is in shadow, so the sheet has body.
  col = col * (0.52 + 0.48 * smoothstep(0.02, 0.62, f1)) * (0.72 + 0.28 * smoothstep(0.02, 0.70, f2));
  col = col * (1.0 - 0.22 * clamp(U.bass, 0.0, 1.0) * (1.0 - f1));

  // --- the sheen: one band travelling across the sheet, plus the foil's own bloom.
  let sweep = fract(t * 0.09) * 3.2 - 1.6;
  col = col + vec3f(1.0, 0.99, 0.96) * glow(q.x * 0.62 + q.y * 0.42 - sweep, 12.0)
    * (0.06 + 0.28 * P(5));
  col = col + holo * (0.05 + 0.12 * nrg) * glow(f1 - 0.5, 1.6);

  col *= 0.92 + 0.16 * smoothstep(1.8, 0.0, length(vec2f(q.x * 0.62, q.y * 0.95)));
  col *= grain(pos.xy, 0.08 + 0.14 * P(7));
  return fin(shoulder(col));
}
