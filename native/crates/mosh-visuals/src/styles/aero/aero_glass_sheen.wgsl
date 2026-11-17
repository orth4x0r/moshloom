// Frutiger Aero · Glass Sheen.
//
// Three sheets of translucent glass over a pale sky, tilted at different angles and stacked, so
// their edges cross where the eye wants them to. Each sheet is frosted, lit from the top right,
// edged with a bright hairline, and swept by one diagonal sheen that travels with the music. Bokeh
// behind them, droplets on the front one, and a soft shaft of light across the frame. Clean, glossy,
// optimistic: the style's poster.
//
// p0 Speed, p1 Panel drift (spreads the sheets apart), p2 Hue, p3 Frost, p4 Sheen, p5 Bokeh,
// p6 Droplets, p7 Rim.
//
// Cost: three rounded boxes with one sine of frost each, five bokeh discs, four droplets, one
// palette. A sheet whose distance field is far from zero costs one subtraction.

fn round_box(p: vec2f, half: vec2f, radius: f32) -> f32 {
  let d = abs(p) - half + radius;
  return length(max(d, vec2f(0.0, 0.0))) + min(max(d.x, d.y), 0.0) - radius;
}

// One sheet, composited over `col`. `u` is the sheet's own frame (already rotated and centred),
// `tint` its colour, `sweep` where its sheen currently is, `frost` how much fog it carries.
fn sheet(
  col: vec3f,
  u: vec2f,
  half: vec2f,
  tint: vec3f,
  frost: f32,
  sweep: f32,
  sheen: f32,
  rim: f32,
  px: f32,
) -> vec3f {
  let d = round_box(u, half, 0.07);
  if (d > 0.25) {
    return col;
  }
  let inside = smoothstep(px, -px, d);
  // Frosted body: the tint, lightened towards the top, with a fine fog and the sheet's own caustic
  // where its bottom edge throws the light back up.
  let fog = 0.5 + 0.5 * sin(u.x * 6.1 + u.y * 5.3 + 1.7);
  var glass = mix(vec3f(0.74, 0.94, 1.0), tint * 0.95 + 0.10, 0.50);
  glass = glass * (1.0 - 0.28 * smoothstep(-0.28, 0.32, u.y));
  glass = glass + (fog - 0.5) * 0.18 * frost;
  glass = glass + tint * falloff(max(u.y - 0.10, 0.0), 40.0) * 0.16 * (1.0 + 0.9 * U.pulse);

  var out = mix(col, glass, inside * 0.55);
  // One diagonal sheen, travelling.
  let band = falloff(u.x * 0.62 + u.y * 0.5 - sweep, 34.0);
  out = out + vec3f(1.0) * band * inside * sheen;
  // The hairline rim: what makes it read as glass and not as paper. Brighter on the lit edge.
  let lit = 0.55 + 0.45 * smoothstep(-0.25, 0.25, -u.y);
  return out + vec3f(0.92, 0.99, 1.0) * hairline(d, px * 1.6) * rim * lit;
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let t = U.T;
  let px = 2.0 / U.res.y;
  let hue = P(2);
  let nrg = energy();
  let sun = vec2f(0.78, -0.92);
  let hue_ang = TAU * (0.46 + hue);
  let tint = pal(cos(hue_ang), sin(hue_ang));
  let tint2 = pal_next(tint);
  let tint3 = next2(tint);
  let sway = sin(t * 0.2);
  let sway_c = cos(t * 0.2);

  // Sky: one clean gradient, plus the sun's bloom.
  var col = mix(vec3f(0.94, 0.99, 1.0), vec3f(0.50, 0.79, 0.95), smoothstep(-1.0, 1.0, q.y));
  col = col + vec3f(1.0, 0.99, 0.94) * falloff(length(q - sun), 2.2) * (0.5 + 0.12 * nrg);
  col = col + tint * 0.05 * smoothstep(0.4, -0.8, q.y);

  // Bokeh behind the glass, on literal orbits, all drifting with one sway.
  let b5 = 0.5 + 0.5 * P(5);
  col = col + bokeh(q, vec2f(-0.72, 0.26) + vec2f(0.05, 0.02) * sway, 0.17, tint3, nrg) * b5;
  col = col + bokeh(q, vec2f(0.58, 0.62) + vec2f(-0.04, 0.03) * sway_c, 0.12, tint2, nrg) * b5;
  col = col + bokeh(q, vec2f(0.20, -0.72) + vec2f(0.03, -0.02) * sway, 0.21, tint, nrg) * b5;
  col = col + bokeh(q, vec2f(-0.34, -0.48) + vec2f(-0.03, 0.02) * sway_c, 0.10, tint2, nrg) * b5;
  col = col + bokeh(q, vec2f(0.90, -0.22) + vec2f(0.02, 0.03) * sway, 0.14, tint3, nrg) * b5;

  // The three sheets. Each tilt's cosine and sine are literal constants; the drift control moves
  // them, and the sheen sweep is one travelling phase per sheet.
  let drift = (P(1) - 0.5) * 0.5;
  let sheen = 0.10 + 0.34 * P(4);
  let rim = 0.18 + 0.55 * P(7);
  let frost = P(3);

  // Back sheet, tilted -0.34 rad.
  {
    let center = vec2f(0.34 + drift * 0.5, -0.30 - drift * 0.3);
    let dx = q.x - center.x;
    let dy = q.y - center.y;
    let u = vec2f(dx * 0.9428 - dy * -0.3335, dx * -0.3335 + dy * 0.9428);
    let sweep = fract(t * 0.10 + 0.13) * 2.8 - 1.4;
    col = sheet(col, u, vec2f(0.46, 0.24), tint3, frost, sweep, sheen * 0.7, rim * 0.8, px);
  }
  // Middle sheet, nearly level.
  {
    let center = vec2f(0.02 + drift * 0.2, 0.02);
    let dx = q.x - center.x;
    let dy = q.y - center.y;
    let u = vec2f(dx * 0.9998 - dy * -0.0200, dx * -0.0200 + dy * 0.9998);
    let sweep = fract(t * 0.11 + 0.37) * 2.8 - 1.4;
    col = sheet(col, u, vec2f(0.52, 0.28), tint2, frost, sweep, sheen, rim, px);
  }
  // Front sheet, tilted +0.30 rad, the glossiest.
  {
    let center = vec2f(-0.30 - drift * 0.5, 0.34 - drift * 0.3);
    let dx = q.x - center.x;
    let dy = q.y - center.y;
    let u = vec2f(dx * 0.9553 - dy * 0.2955, dx * 0.2955 + dy * 0.9553);
    let sweep = fract(t * 0.12 + 0.71) * 2.8 - 1.4;
    col = sheet(col, u, vec2f(0.42, 0.22), tint, frost, sweep, sheen * 1.2, rim * 1.15, px);
  }

  // Droplets on the front sheet, in front of everything: a lens, a rim, a hot dot.
  let amount = P(6);
  let c0 = vec2f(-0.44 + 0.02 * sway, 0.30);
  let c1 = vec2f(-0.16, 0.10 + 0.02 * sway_c);
  let c2 = vec2f(0.44 + 0.02 * sway, 0.46);
  let c3 = vec2f(0.16, -0.08);
  var lens = vec3f(0.0);
  var alpha = 0.0;
  for (var i = 0; i < 4; i = i + 1) {
    let r = select(select(select(0.050, 0.030, i == 1), 0.042, i == 2), 0.026, i == 3)
      * (0.35 + 1.1 * amount);
    let c = select(select(select(c0, c1, i == 1), c2, i == 2), c3, i == 3);
    let d = (q - c) / max(r, 1e-4);
    let r2 = dot(d, d);
    if (r2 < 1.1) {
      let z = sqrt(max(0.0, 1.0 - min(r2, 1.0)));
      let edge = 1.0 - z;
      let fres = edge * edge;
      let ndl = max(d.x * -0.35 + d.y * -0.7 + z * 0.62, 0.0);
      let s2 = ndl * ndl;
      let hot = s2 * s2 * s2 * s2 * s2;
      let a = smoothstep(1.1, 0.95, r2) * clamp(0.18 + 0.8 * fres, 0.0, 1.0);
      lens = lens + (mix(vec3f(0.55, 0.88, 0.95), vec3f(0.98, 1.0, 1.0), fres * 0.9)
        + vec3f(1.0) * hot * 1.8) * a;
      alpha = alpha + a;
    }
  }
  if (alpha > 0.001) {
    col = mix(col, lens / max(alpha, 1e-4), clamp(alpha, 0.0, 1.0));
  }

  // The shaft of light across the frame.
  col = col + vec3f(1.0, 0.99, 0.90) * falloff(q.x * 0.7 + q.y * 0.5 - 0.85, 22.0)
    * 0.06 * (0.5 + 0.5 * P(5));
  return fin(col);
}
