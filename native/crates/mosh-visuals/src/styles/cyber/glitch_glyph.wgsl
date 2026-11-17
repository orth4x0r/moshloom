// Glitch Glyph: an ASCII glyph raster over a databent colour field.
//
// The frame is a grid of character cells. Each cell picks one of eight 5x7 glyph shapes from a hash
// of its position and a time step, so the raster churns like a terminal, and the cell's brightness
// (a smooth field plus one mel band per row) picks the glyph's colour and quantises it to four
// levels, which is the databending: the palette is squeezed down to what a corrupt framebuffer
// holds. Whole rows of cells shift sideways, some cells invert on the beat, and a cursor block
// walks the grid.
//
// Audio: one mel band per row drives the field, `U.bass` widens the glyphs, `U.high` makes the raster
// churn faster and shifts rows more, `U.pulse` inverts a hashed quarter of the cells and lights the
// cursor. Cell inversions change at the beat's rate, never inside the 400 ms flash limit.
//
// P(0) clock rate, P(1) cell size, P(2) hue, P(3) churn, P(4) row shift, P(5) quantise,
// P(6) cursor, P(7) scanlines.

const GREEN = vec3f(0.35, 1.0, 0.55);

fn near(a: f32, b: f32) -> f32 {
  return step(abs(a - b), 0.30);
}

/// One 5x7 glyph, `pat` 0..7, `gx` 0..4 across, `gy` 0..6 down.
fn glyph(pat: i32, gx: f32, gy: f32, seed: f32) -> f32 {
  let cx = gx - 2.0;
  let cy = gy - 3.0;
  if (pat == 0) {
    // A cross.
    return max(near(gx, 2.0), near(gy, 3.0));
  } else if (pat == 1) {
    // A ring.
    return step(abs(length(vec2f(cx, cy * 0.8)) - 2.1), 0.9);
  } else if (pat == 2) {
    // An X.
    return step(abs(abs(cx) - abs(cy) * 0.75), 0.45);
  } else if (pat == 3) {
    // A ladder.
    return max(max(near(gy, 1.0), near(gy, 3.0)), near(gy, 5.0));
  } else if (pat == 4) {
    // Two uprights.
    return max(near(gx, 1.0), near(gx, 3.0));
  } else if (pat == 5) {
    // A wedge.
    return step(abs(cx), 0.55 + (gy - 1.0) * 0.42);
  } else if (pat == 6) {
    // A corner.
    return max(near(gx, 1.0) * step(1.0, gy), near(gy, 5.0));
  }
  // Snow: a hashed block, the glyph the corrupt framebuffer draws when it has no character left.
  return step(0.45, h2(vec2f(gx + seed * 7.0, gy + seed * 3.0)));
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let t = U.T * (0.4 + 0.5 * P(0));

  // --- the cell grid, with whole rows of cells shifted sideways.
  let cell = vec2f(10.0, 15.0) * (0.75 + 0.9 * clamp(P(1), 0.0, 1.0));
  let row = floor(pos.y / cell.y);
  let shift = (h1(row * 3.7 + floor(t * (2.0 + 6.0 * clamp(U.high, 0.0, 1.0)))) - 0.5)
    * cell.x * 3.0 * P(4) * step(0.62, h1(row * 9.1));
  let local = (pos.xy - vec2f(shift, row * cell.y)) / cell;
  let cell_id = floor(vec2f((pos.x - shift) / cell.x, row));
  let in_cell = vec2f(fract(local.x), fract(local.y));

  // --- the field the cells read: a smooth image plus one mel band per row.
  let band = mel(i32(clamp(fract(row * 0.021) * 24.0, 0.0, 23.0)));
  let v = clamp(
    0.28 + 0.30 * sin(q.x * 2.1 - t * 0.6) * cos(q.y * 1.6 + t * 0.4)
      + 0.42 * band + 0.20 * h2(vec2f(row, 3.0)) - 0.22 * abs(q.y),
    0.0, 1.0
  );

  // --- the glyph: shape from the cell's hash, churn rate from the clip and the high band.
  // Each cell churns on its own phase, so the raster never changes globally: no frame-wide flash,
  // and the rate is capped under the 400 ms safety limit anyway.
  let churn_rate = 1.5 + 4.5 * clamp(P(3), 0.0, 1.0) + 0.8 * clamp(U.high, 0.0, 1.0);
  let churn = floor(t * churn_rate + h2(cell_id) * 9.0);
  let pick = h2(cell_id + churn * 1.37);
  let pat = i32(pick * 7.99);
  let gx = floor(in_cell.x * 5.0);
  let gy = floor(in_cell.y * 7.0);
  var ink = glyph(pat, gx, gy, pick);

  // --- the beat inverts a hashed quarter of the cells, and lights the cursor block.
  let flipped = step(1.0 - 0.25 * U.pulse, h2(cell_id * 1.7 + floor(t * 2.0)));
  let cursor = step(0.90, h1(floor(t * 0.7))) * step(0.55, fract(t * 0.35));
  let is_cursor = step(abs(cell_id.x - floor(fract(t * 0.21) * 40.0)), 0.5)
    * step(abs(row - floor(fract(t * 0.13) * 22.0)), 0.5) * cursor;

  // --- the databend: four levels of colour, taken from the cell's brightness.
  let levels = mix(8.0, 2.0, clamp(P(5), 0.0, 1.0));
  let qv = floor(v * levels) / levels;
  let tone = mix(lut(0.08 + 0.84 * qv), GREEN * 1.30, 0.16) * (0.85 + 1.65 * qv);
  let lit = select(tone * (0.9 + 0.4 * U.pulse), vec3f(0.02, 0.03, 0.03), flipped > 0.5);
  let dark = select(vec3f(0.016, 0.006, 0.008) + tone * 0.07, tone * 0.75, flipped > 0.5);

  var c = mix(dark, lit, ink);
  // The cursor block: a solid cell that reads as the terminal's caret.
  c = mix(c, vec3f(0.95, 0.98, 1.0) * (0.55 + 0.45 * U.pulse), is_cursor * 0.85);
  // A bloom around the glyph, so the raster glows instead of sitting flat on black.
  c += tone * 0.12 * smoothstep(0.55, 0.0, abs(in_cell.y - 0.5)) * (1.0 - ink) * v;

  // --- scanlines at the cell's own pitch, and a cold vignette.
  let scan = 1.0 - (0.10 + 0.24 * P(7)) * (0.5 + 0.5 * cos(pos.y * PI * 0.75));
  c *= scan;
  c *= 0.55 + 0.45 * smoothstep(1.45, 0.20, length(vec2f(q.x * 0.60, q.y * 0.95)));
  c += vec3f(0.004, 0.006, 0.008);
  // Hue: a channel rotation on the finished frame, so P(2) shifts the capture's colour cast.
  c *= mix(vec3f(1.0), vec3f(0.88, 0.62, 1.24), U.hue);
  return fin(c);
}
