// Smear · Melt Drag: datamosh as collage — ribbons of colour dragged sideways, then smeared.
//
// The picture is a field of two woven ribbons over a slow melt. Bands of a few pixels are displaced
// sideways by a hash that is interpolated between neighbouring bands, so the edges are soft and the
// whole frame warps like a moving image whose motion vectors have been run together, instead of
// tearing along a hard seam. A second sample is stretched vertically and mixed in by `Melt`, which is
// what turns the warp into a smear: the trails keep the ribbons' colour and pick up a luminous edge
// where the stretch is strongest. Nothing here is a cathode-ray artefact: no scanlines, no block
// corruption, no inverted macroblocks, and no channel rotation.
//
// Audio: `U.high` widens the warp and re-rolls it faster, `U.bass` pushes the drag, `U.pulse` opens
// the smear and lifts the trails. The bands re-roll at a few hertz at most, far slower than the
// 400 ms flash limit, and only local luminance moves.
//
// P(0) clock rate, P(1) band height, P(2) hue, P(3) warp, P(4) drag, P(5) bloom, P(6) melt,
// P(7) grain.

/// Two woven ribbons over a slow melt: the picture the drag is applied to.
fn ribbons(q: vec2f, t: f32, hue: f32) -> vec3f {
  let w = melt_field(q * vec2f(0.85, 1.0), t);
  let band1 = 0.5 + 0.5 * sin(q.y * 3.10 + 2.20 * w + t * 0.40);
  let band2 = 0.5 + 0.5 * sin(q.x * 2.30 - 1.70 * w - t * 0.33 + 1.90);
  let v = clamp(0.22 + 0.52 * band1 * band2 + 0.30 * w, 0.0, 1.0);
  var c = lut(0.05 + 0.88 * v + hue * 0.10) * (0.32 + 1.25 * v);
  // A thin hot thread down each ribbon edge: the light the smear will trail.
  let thread = 1.0 - smoothstep(0.0, 0.06, abs(fract(q.y * 2.1 + 1.4 * w + t * 0.2) - 0.5));
  c += lut(0.10 + 0.80 * clamp(v + 0.35, 0.0, 1.0)) * thread * 0.30;
  return c;
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let px = 2.0 / U.res.y;
  let q = XY(pos);
  let t = U.T * (0.35 + 0.50 * P(0));
  let hue = P(2);
  let nrg = clamp(U.high, 0.0, 1.0);

  // --- the bands: a displaced row is a motion vector, and neighbouring bands lean into each other so
  // the warp is soft rather than a tear.
  let bh = 4.0 + 34.0 * clamp(P(1), 0.0, 1.0) * (0.6 + 0.7 * clamp(U.dens, 0.0, 1.0));
  let row = pos.y / bh;
  let band = floor(row);
  let blend = smoothstep(0.20, 0.80, fract(row));
  let hA = h1(band * 1.31 + 2.0);
  let hB = h1((band + 1.0) * 1.31 + 2.0);
  let hh = mix(hA, hB, blend);
  // A slow roll re-rolls the warp a few times a second, faster on a bright, high-energy bar.
  let roll = floor(t * (0.6 + 1.6 * nrg + 1.2 * U.pulse));
  let rollh = h1(band * 7.7 + roll * 13.1);
  let amp = (26.0 + 150.0 * clamp(P(3), 0.0, 1.0) * (0.5 + 0.7 * nrg)) * (0.55 + 0.75 * hh);
  let disp = (hh - 0.5) * amp + (rollh - 0.5) * amp * 0.45;

  // --- the clean drag: the picture sampled through the warp, and the smear: the same picture sampled
  // further along and stretched vertically, so the ribbons trail instead of repeating.
  let drag = disp + (P(4) - 0.5) * 90.0 + 26.0 * U.pulse;
  let melt = clamp(0.10 + 0.75 * P(6), 0.0, 1.0);
  let stretch = 1.0 - 0.28 * melt;
  let sharp = XY(vec4f(pos.x + drag, pos.y, 0.0, 0.0));
  let smeared = XY(vec4f(pos.x + drag * 1.55, (pos.y - U.res.y * 0.5) * stretch + U.res.y * 0.5, 0.0, 0.0));
  let a = ribbons(sharp, t, hue);
  let b = ribbons(smeared, t, hue);
  var c = mix(a, b, melt);

  // The trail: the difference between the two samples is where the smear is, and it is lit from the
  // direction of the drag, so the frame reads as one image pulled sideways.
  let pull = clamp((b.r + b.g + b.b) - (a.r + a.g + a.b), -1.0, 1.0);
  c += lut(0.10 + 0.80 * clamp(0.5 + 0.5 * pull, 0.0, 1.0)) * max(pull, 0.0) * (0.25 + 0.35 * P(5));
  c *= 0.92 + 0.22 * P(5) * (0.4 + 0.6 * U.pulse);
  c *= 0.86 + 0.30 * smoothstep(1.55, 0.20, length(vec2f(q.x * 0.62, q.y * 0.95)));
  c *= grain(pos.xy, 0.10 + 0.20 * P(7));
  return fin(shoulder(c));
}
