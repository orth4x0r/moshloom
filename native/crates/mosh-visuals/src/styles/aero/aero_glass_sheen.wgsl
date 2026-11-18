// Glass · Shatter Blades: long slivers of glass crossing a deep gradient, each one catching the light.
//
// Not a stack of panels: four blades — rounded boxes with a ten-to-one aspect — tilted at different angles
// and drifting apart, so the frame is a set of criss-crossing glass edges over a grainy plate. Each blade
// dims and cools the picture behind it, carries a bright sheen along its length, has a shadow just inside
// its rim and a hairline of light at its edge; frost scatters its body, and `Rim` decides how hard the
// edges read. Droplets on the front blade are little lenses with an inverted image and one hot dot. A
// shoulder keeps the speculars from clipping: a glass scene that paints a white patch is a bug.
//
// Audio: `U.pulse` lifts the sheen, the edges and the sun's bloom, `U.bass` tilts the blades further,
// `U.high` brightens the rims. Everything moves at the beat's rate or slower.
//
// p0 Speed, p1 Blade drift, p2 Hue, p3 Frost, p4 Sheen, p5 Bokeh, p6 Droplets, p7 Rim.

fn round_box(p: vec2f, half: vec2f, radius: f32) -> f32 {
  let d = abs(p) - half + radius;
  return length(max(d, vec2f(0.0, 0.0))) + min(max(d.x, d.y), 0.0) - radius;
}

/// One blade, composited over `col`. `u` is the blade's own frame (already rotated and centred), `half`
/// its extent, `tint` its colour, `sweep` where its sheen currently is.
fn blade(
  col: vec3f,
  u: vec2f,
  half: vec2f,
  tint: vec3f,
  frost: f32,
  sweep: f32,
  sheen: f32,
  rim: f32,
  px: f32
) -> vec3f {
  let d = round_box(u, half, 0.06);
  if (d > 0.30) {
    return col;
  }
  let inside = smoothstep(px, -px, d);
  // The body: the picture behind, dimmed and cooled. Glass is mostly colourless; its colour lives in the
  // sheen and at the edges.
  let body = col * 0.56 + vec3f(0.022, 0.032, 0.050);
  let glass = mix(body, tint * 0.26 + vec3f(0.10, 0.13, 0.18), 0.10 + 0.40 * frost);
  var out = mix(col, glass, inside);
  // The blade's thickness: a shadow just inside the rim, so the glass has a body.
  out = out * (1.0 - 0.45 * inside * (1.0 - smoothstep(0.0, 0.09, -d)));
  // The sheen: a band travelling along the blade, hard-cored, the reflection of a long light in the sheet.
  let band = glow(u.y * 1.7 + u.x * 0.35 - sweep, 2.2);
  out = out + vec3f(0.94, 0.98, 1.0) * band * inside * sheen * 1.5;
  out = out + tint * band * inside * sheen * 0.60;
  // The rim: a bright hairline on the lit side, a dim one on the shadow side.
  let lit = 0.55 + 0.45 * smoothstep(-0.20, 0.20, -u.y);
  return out + vec3f(0.90, 0.96, 1.0) * hairline(d, px * 1.7) * rim * lit * 0.85;
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let t = U.T;
  let px = 2.0 / U.res.y;
  let hue = P(2);
  let nrg = energy();
  let sun = vec2f(0.72, -0.86);
  let hue_ang = TAU * (0.32 + hue);
  let tint = pal(cos(hue_ang), sin(hue_ang));
  let tint2 = pal_next(tint);
  let tint3 = next2(tint);
  let sway = sin(t * 0.2);
  let sway_c = cos(t * 0.2);

  // --- the backdrop: a deep gradient with two aura pools, the sun's bloom, and a colour wash.
  var col = plate(q, tint * 0.8, t, 3.7);
  col = col + tint2 * 0.16 * glow(length(q - sun), 0.65) * (0.45 + 0.20 * clamp(nrg, 0.0, 1.3));
  col = col + tint3 * 0.10 * glow(length(q - vec2f(-0.62, -0.50)), 0.80);
  col = col + tint * 0.07 * smoothstep(0.4, -0.9, q.y);

  // Bokeh behind the glass, on a literal orbit, drifting with the sway.
  let b5 = 0.5 + 0.5 * P(5);
  col = col + bokeh(q, vec2f(-0.72, 0.26) + vec2f(0.05, 0.02) * sway, 0.19, tint3, nrg) * b5;
  col = col + bokeh(q, vec2f(0.20, -0.72) + vec2f(0.03, -0.02) * sway, 0.23, tint, nrg) * b5 * 0.7;

  // --- the blades. Each tilt's cosine and sine are literal constants; `Blade drift` spreads them apart
  // and `U.bass` tilts the whole family a little further.
  let drift = (P(1) - 0.5) * 0.45;
  let tilt = 0.10 * clamp(U.bass, 0.0, 1.0);
  let sheen = (0.14 + 0.42 * P(4)) * (1.0 + 0.45 * U.pulse);
  let rim = (0.22 + 0.62 * P(7)) * (1.0 + 0.40 * U.pulse);
  let frost = P(3);

  // Blade one: flat, low, across the lower half.
  {
    let c = vec2f(-0.10 + drift * 0.3, -0.34 - drift * 0.2);
    let dx = q.x - c.x;
    let dy = q.y - c.y;
    let u = vec2f(dx * 0.9801 - dy * 0.1987, dx * 0.1987 + dy * 0.9801);
    let sweep = fract(t * 0.09 + 0.13) * 2.8 - 1.4;
    col = blade(col, u, vec2f(1.05, 0.10), tint3, frost, sweep, sheen * 0.7, rim * 0.8, px);
  }
  // Blade two: steeply rising to the right.
  {
    let c = vec2f(0.26 - drift * 0.4, 0.16);
    let dx = q.x - c.x;
    let dy = q.y - c.y;
    let cs = cos(0.72 + tilt);
    let sn = sin(0.72 + tilt);
    let u = vec2f(dx * cs - dy * sn, dx * sn + dy * cs);
    let sweep = fract(t * 0.11 + 0.41) * 2.8 - 1.4;
    col = blade(col, u, vec2f(0.94, 0.11), tint2, frost, sweep, sheen, rim, px);
  }
  // Blade three: falling to the right, in front.
  {
    let c = vec2f(-0.34 + drift * 0.5, 0.40 - drift * 0.3);
    let dx = q.x - c.x;
    let dy = q.y - c.y;
    let cs = cos(-0.58 - tilt);
    let sn = sin(-0.58 - tilt);
    let u = vec2f(dx * cs - dy * sn, dx * sn + dy * cs);
    let sweep = fract(t * 0.13 + 0.67) * 2.8 - 1.4;
    col = blade(col, u, vec2f(0.88, 0.13), tint, frost, sweep, sheen * 1.2, rim * 1.15, px);
  }

  // --- droplets on the front blade, in front of everything: a lens, a rim, a hot dot.
  let amount = P(6);
  let c0 = vec2f(-0.44 + 0.02 * sway, 0.30);
  let c1 = vec2f(-0.16, 0.10 + 0.02 * sway_c);
  let c2 = vec2f(0.44 + 0.02 * sway, 0.46);
  var lens = vec3f(0.0);
  var alpha = 0.0;
  for (var i = 0; i < 2; i = i + 1) {
    let r = select(0.048, 0.030, i == 1) * (0.35 + 1.1 * amount);
    let c = select(c0, c1, i == 1);
    let d = (q - c) / max(r, 1e-4);
    let r2 = dot(d, d);
    if (r2 < 1.1) {
      let z = sqrt(max(0.0, 1.0 - min(r2, 1.0)));
      let edge = 1.0 - z;
      let fres = edge * edge;
      let ndl = max(d.x * -0.35 + d.y * -0.7 + z * 0.62, 0.0);
      let s2 = ndl * ndl;
      let hot = s2 * s2 * s2 * s2 * s2;
      let a = smoothstep(1.1, 0.95, r2) * clamp(0.16 + 0.8 * fres, 0.0, 1.0);
      lens = lens + (mix(tint2 * 0.6, tint * 0.95 + vec3f(0.10, 0.14, 0.18), fres * 0.9)
        + vec3f(1.0) * hot * (0.9 + 0.6 * U.pulse)) * a;
      alpha = alpha + a;
    }
  }
  if (alpha > 0.001) {
    col = mix(col, lens / max(alpha, 1e-4), clamp(alpha, 0.0, 1.0) * 0.85);
  }

  // The shaft of light across the frame, and the beat's bloom on it — local, never a full-frame flash.
  col = col + tint2 * glow(q.x * 0.7 + q.y * 0.5 - 0.85, 2.2) * 0.10 * (0.5 + 0.5 * P(5));
  col = col + tint * 0.14 * U.pulse * glow(length(q - sun), 0.70);
  col *= grain(pos.xy, 0.12 + 0.10 * P(3));
  return fin(shoulder(col));
}
