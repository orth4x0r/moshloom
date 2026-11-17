// Y2K · Cyber Horizon.
//
// A wireframe horizon: a graded sky with a banded sun sitting on it, a chrome wire globe hanging
// over the grid, and the grid itself running to the vanishing point below, scrolling toward the
// viewer and mirrored into the sun's column on the ground. The sun is the album cover, the grid is
// the screensaver, the globe is the Y2K logo.
//
// p0 Speed, p1 Grid density, p2 Hue, p3 Sun size, p4 Scanlines, p5 Horizon glow, p6 Globe
// (0 hides it), p7 Sparkle.
//
// Cost: the sky is a gradient and a disc, the grid is one reciprocal per pixel with its line width
// taken from the analytic derivative of the perspective coordinate (no `fwidth`), and the globe is
// one sphere evaluation with four by six wire lines. One palette, one `atan2`, one hash.

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let t = U.T;
  let px = 2.0 / U.res.y;
  let hue = P(2);
  let density = 0.5 + 1.5 * P(1);
  let sun_r = 0.14 + 0.34 * P(3);
  let scan = P(4);
  let glow = P(5);
  let globe_amt = P(6);
  let sparkle = P(7);
  let nrg = energy();
  // The horizon lifts on bass: the world breathing, never more than a few pixels.
  let horizon = 0.04 - 0.03 * U.bass;
  let sun_c = vec2f(0.0, horizon - sun_r * 0.92);

  let hue_ang = TAU * (0.44 + hue);
  let tint = pal(cos(hue_ang), sin(hue_ang));
  let tint2 = pal_next(tint);
  let tint3 = next2(tint);
  // The sun is the warm end of the palette, the sky the cool end: no extra trigonometry.
  let hot = tint2 * 0.85 + vec3f(0.15, 0.05, 0.0);

  var col = vec3f(0.0);
  if (q.y < horizon) {
    // Sky: the hot end of the palette at the horizon, deep indigo above it.
    let up = (horizon - q.y) / (1.0 + horizon);
    let sky_top = vec3f(0.02, 0.01, 0.075) + tint * 0.05;
    col = mix(hot * 0.9, sky_top, smoothstep(0.0, 0.85, up));
    // The sun, cut into bands over its lower half, glowing harder on the beat.
    let sd = length(q - sun_c) / sun_r;
    let w = px * 1.6 / sun_r;
    let disc = smoothstep(1.0 + w, 1.0 - w, sd);
    // Below the centre the bands bite in; the control sets how deep.
    let band = fract((q.y - sun_c.y) * 34.0 + 0.5);
    let stripes = select(1.0, step(0.45, band), q.y > sun_c.y);
    let sun_face = mix(vec3f(1.0, 0.96, 0.80), hot * 1.25,
      smoothstep(-1.0, 1.0, (q.y - sun_c.y) / sun_r));
    col = mix(col, sun_face, disc * mix(1.0, stripes, scan));
    col = col + hot * falloff(sd, 2.2 - glow) * (0.32 + 0.25 * nrg);
    // A chromed ring around the sun and a hot core: the Y2K halo, both lines, never a flash.
    col = col + tint * falloff(sd - 1.18, 900.0) * (0.4 + 0.5 * U.pulse);
    col = col + vec3f(1.0, 0.98, 0.9) * falloff(sd, 40.0) * (0.25 + 0.3 * U.pulse);

    // The wire globe, hanging to the right of the sun.
    let gc = vec2f(0.55, horizon - 0.24 - 0.14 * sun_r);
    let gR = 0.06 + 0.20 * globe_amt;
    let gs = q - gc;
    let grr = dot(gs, gs) / (gR * gR);
    if (grr < 1.0) {
      let gz = sqrt(max(0.0, 1.0 - grr));
      let p3 = vec3f(gs.x / gR, gs.y / gR, gz);
      // One extra sine and cosine spin the wire, shared with nothing else in this scene.
      let spin = t * 0.4;
      let yr = p3.y * 0.9492 - p3.z * 0.3146;
      let zr = p3.y * 0.3146 + p3.z * 0.9492;
      let latv = yr * 4.0;
      let lonv = (atan2(p3.x, zr) + spin) * (6.0 / TAU);
      let wl = 4.0 * px * 0.6 / gR;
      let wn = 6.0 * px * 0.6 / (TAU * max(length(vec2f(p3.x, zr)), 0.15) * gR);
      let wire = max(hairline(fract(latv + 0.5) - 0.5, wl), hairline(fract(lonv + 0.5) - 0.5, wn));
      let g_rim = 1.0 - gz;
      col = col + tint * (wire * 0.85 + g_rim * g_rim * g_rim * g_rim * 0.7) * (0.5 + 0.6 * nrg);
    }
    col = col + tint * glints(q, t, 38.0, 0.962) * (0.15 + 0.9 * sparkle) * 0.7;
  } else {
    // Ground: the grid in perspective. `depth` is 1 / distance below the horizon, so every screen
    // row is one distance and the lines converge on their own.
    let dz = q.y - horizon + 0.0015;
    let depth = 0.32 / dz;
    let v = depth + t * 0.9 + 0.5 * U.pulse;
    let u = q.x * depth;
    // dv/dy = v^2 / 0.32 and du/dx = depth: the line widths are exact, one pixel wide everywhere.
    let lv = abs(fract(v * density) - 0.5);
    let lu = abs(fract(u * density) - 0.5);
    let wv = density * v * v / 0.32 * px * 2.2;
    let wu = density * depth * px * 2.0;
    let grid = (hairline(lv, wv) + hairline(lu, wu)) * smoothstep(0.05, 0.40, dz);
    // A bloom around the lines, so they read as light on a surface rather than as hairlines.
    let bloom = falloff(lv * 7.0, 1.0) + falloff(lu * 7.0, 1.0);
    let gc = tint2 * 0.55 + vec3f(0.02, 0.09, 0.18);
    col = mix(vec3f(0.008, 0.002, 0.020), vec3f(0.030, 0.006, 0.055), falloff(dz, 6.0));
    col = col + gc * (grid * 1.9 + bloom * 0.010 * smoothstep(0.05, 0.40, dz))
      * (0.5 + 0.55 * nrg);
    // The sun's column on the ground: a narrow bright road under it, the way a wet floor carries
    // light, plus the smear of the disc itself just past the horizon.
    col = col + hot * falloff(abs(q.x), 34.0) * falloff(dz, 45.0) * (0.25 + 0.25 * nrg);
    col = col + hot * falloff(length(vec2f(q.x * 0.8, dz + 0.02)) / 0.16, 3.0) * 0.30;
    // The horizon line itself, the brightest thing in the frame, but a line.
    col = col + tint * hairline(dz, px * 2.0 + 0.004) * (0.6 + 0.5 * glow);
    col = col + hot * falloff(dz, 60.0) * 0.35 * (0.5 + 0.5 * glow);
  }
  return fin(col);
}
