// Frutiger Aero · Aqua Hills.
//
// Three layers of rolling green hills under an optimistic sky: the far ridges washed blue by the
// air, the near one saturated grass with a glossy crest where the light catches it. Soft cumulus
// behind them, the sun blooming top right, and a strip of water at the bottom carrying the light
// bands and the sun's column. The ridges breathe with the bass; `pulse` only sharpens the crest.
//
// p0 Speed, p1 Hill height, p2 Hue, p3 Clouds, p4 Grass gloss, p5 Sun glow, p6 Waterline,
// p7 Sparkle.
//
// Cost: five sines for the three ridges, one for the water, three for the clouds, one palette.
// The ridge line's anti-aliasing comes from its own slope (the caller knows the wave's analytic
// derivative), not from `fwidth`.

// A ridge: y of the crest at x. `dl` is the same wave's slope, which gives the pixel width of the
// crest's edge for free (one more sine, so each near ridge costs two).
fn ridge(x: f32, base: f32, amp: f32, freq: f32, phase: f32, second: f32) -> f32 {
  return base + amp * (0.64 * sin(x * freq + phase) + second * sin(x * freq * 2.31 + phase * 1.7));
}

// The slopes are bounded by amp * freq, which is all the ridge's edge needs: one pixel of y plus
// one pixel of x times the steepest slope. No cosine, no `fwidth`.

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let t = U.T;
  let px = 2.0 / U.res.y;
  let hue = P(2);
  let nrg = energy();
  let sun = vec2f(0.66, -0.86);
  let waterline = 0.42 + 0.33 * P(6);
  let hue_ang = TAU * (0.46 + hue);
  let tint = pal(cos(hue_ang), sin(hue_ang));
  let tint2 = pal_next(tint);
  // The ridges' one shared clock term: bands scale it, so a louder bar moves the hills faster, and
  // a smoothed band never makes it jump.
  let drift = t * (0.10 + 0.10 * U.mid);

  // Sky.
  var col = mix(vec3f(0.55, 0.82, 0.96), vec3f(0.94, 0.99, 1.0),
    smoothstep(-1.0, waterline - 0.1, q.y));
  col = col + vec3f(1.0, 0.97, 0.85) * falloff(length(q - sun), 3.4)
    * (0.34 + 0.20 * P(5)) * (0.9 + 0.12 * nrg);

  // Clouds: broad and low in the sky, lit from the sun's side.
  let cover = 0.25 + 0.55 * P(3);
  let cl = clouds(vec2f(q.x * 1.3, q.y * 2.1), t);
  let cloud = smoothstep(0.62 - cover * 0.22, 0.88 - cover * 0.12, cl)
    * smoothstep(0.30, -0.75, q.y);
  col = mix(col, vec3f(0.98, 1.0, 1.0), cloud * (0.45 + 0.4 * P(3)));
  col = col + vec3f(1.0, 0.98, 0.9) * cloud * falloff(length(q - sun), 2.4) * 0.20;

  // The three hills, far to near: literal geometry, so each is one or two sines.
  let height = 0.08 + 0.26 * P(1);
  let amp_far = height * 0.55 * (1.0 + 0.10 * U.bass);
  let amp_mid = height * 0.85 * (1.0 + 0.10 * U.bass);
  let amp_near = height * 1.10 * (1.0 + 0.10 * U.bass);

  // Far ridge, washed blue: one sine (no second harmonic; the haze hides it anyway).
  let y_far = waterline - 0.30 + amp_far * sin(q.x * 1.4 + drift * 0.7 + 0.7);
  let d_far = q.y - y_far;
  let aa_far = max(px * abs(amp_far * 1.4) * 1.6, 1e-4);
  let far = smoothstep(-aa_far, aa_far, d_far);
  var col_far = mix(vec3f(0.40, 0.66, 0.68), vec3f(0.20, 0.52, 0.40),
    smoothstep(0.0, 0.30, d_far));
  col = mix(col, col_far + vec3f(0.35, 0.60, 0.70) * 0.10, far);

  // Middle ridge.
  let y_mid = waterline - 0.12 + ridge(q.x, 0.0, amp_mid, 1.9, drift + 2.1, 0.36);
  let d_mid = q.y - y_mid;
  let aa_mid = max(px * (amp_mid * 2.2 + 1.0) * 1.4, 1e-4);
  let mid = smoothstep(-aa_mid, aa_mid, d_mid);
  let grass_mid = vec3f(0.16, 0.58, 0.28) + tint * 0.06;
  var col_mid = mix(grass_mid * 1.20, grass_mid * 0.55, smoothstep(0.0, 0.42, d_mid));
  col_mid = col_mid + vec3f(0.55, 0.95, 0.72) * falloff(max(d_mid, 0.0), 900.0)
    * (0.20 + 0.50 * P(4)) * (0.6 + 0.5 * U.mid);
  col = mix(col, col_mid, mid);

  // Near ridge: the darkest green, the glossiest crest, the only one with a second harmonic.
  let y_near = waterline + 0.02 + ridge(q.x, 0.0, amp_near, 2.6, -drift + 4.3, 0.42);
  let d_near = q.y - y_near;
  let aa_near = max(px * (amp_near * 3.2 + 1.0) * 1.4, 1e-4);
  let near = smoothstep(-aa_near, aa_near, d_near);
  let grass_near = vec3f(0.08, 0.40, 0.19) + tint * 0.08;
  var col_near = mix(grass_near * 1.30, grass_near * 0.50, smoothstep(0.0, 0.50, d_near));
  col_near = col_near + vec3f(0.60, 1.0, 0.75) * falloff(max(d_near, 0.0), 700.0)
    * (0.25 + 0.60 * P(4)) * (0.6 + 0.5 * U.mid + 0.3 * U.pulse);
  // Long soft bands of light down the slope: gloss, not noise.
  let sheen = falloff(abs(fract(q.x * 4.0 + y_near * 9.0) - 0.5) * 2.0 - 0.5, 40.0);
  col_near = col_near + vec3f(0.55, 1.0, 0.70) * sheen * 0.12 * P(4) * (1.0 - smoothstep(0.0, 0.35, d_near));
  col = mix(col, col_near, near);

  // Water: a strip at the bottom, aqua at the shoreline and deep teal at the viewer, with near
  // horizontal bands of light running across it and the sun's column down the middle.
  if (q.y > waterline) {
    let wy = (q.y - waterline) / (1.0 - waterline);
    var water = mix(vec3f(0.62, 0.88, 0.95) + tint * 0.05, vec3f(0.03, 0.22, 0.36), wy);
    // Two overlapping wave trains, soft and wide, so the surface reads as moving water and not as
    // contour lines: the light is a sheen, not an outline.
    let swell = sin(q.y * 22.0 + sin(q.x * 5.0 + t * 0.6) * 2.2 - t * 1.2);
    let chop = sin(q.y * 57.0 - q.x * 4.0 - t * 2.1 + 1.7);
    let shimmer = (0.5 + 0.5 * swell) * (0.65 + 0.35 * (0.5 + 0.5 * chop));
    water = water + vec3f(0.85, 1.0, 1.0) * shimmer * shimmer * shimmer * 0.20
      * (1.0 - wy * 0.8) * (0.6 + 0.4 * U.mid);
    // The sun's column, broken by the swell.
    let column = falloff(abs(q.x - sun.x), 26.0) * (0.55 + 0.45 * swell);
    water = water + vec3f(1.0, 0.95, 0.80) * column * 0.25 * (0.4 + 0.6 * P(5));
    col = mix(col, water, smoothstep(0.0, 0.02, q.y - waterline));
    // The water's own horizon: one bright hairline, and a wet sheen just under it.
    col = col + vec3f(1.0, 0.98, 0.90) * hairline(q.y - waterline, px * 2.0) * 0.35;
    col = col + vec3f(0.85, 1.0, 1.0) * falloff(abs(q.y - waterline), 220.0) * 0.18
      * (1.0 - P(5));
  }

  // Fireflies over the hills: sparse glints, low in the frame, on one hash.
  let low = smoothstep(waterline - 0.55, waterline + 0.05, q.y);
  col = col + vec3f(1.0, 0.98, 0.85) * glints(q, t, 34.0, 0.964) * (0.10 + 0.9 * P(7)) * low;
  col = col + tint2 * 0.02;
  return fin(col);
}
