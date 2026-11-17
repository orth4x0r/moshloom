// Glitch CRT: a broken tube in front of a sunset grid.
//
// The picture is a wireframe floor and a striped sun, sampled through a curved tube: the barrel
// distortion moves the sample point, the aperture grille and the scanlines darken it in stripes, and
// the three colour channels are sampled at slightly different radii so the edges of the tube fringe
// red and blue. A tracking band slides up the screen at a third of a hertz, displacing and
// desaturating the rows it crosses, and the tube's own dropouts punch dark bars through it.
//
// Audio: the sun swells with `U.bass`, the grid scrolls with `U.T` and `U.mid`, the tracking band
// gets wider and more violent with `U.high` and `U.pulse`. The tracking band is the only global
// brightness change and it moves at 0.33 Hz, far inside the 400 ms flash limit.
//
// P(0) clock rate, P(1) curvature, P(2) hue, P(3) tracking, P(4) grille, P(5) aberration,
// P(6) brightness, P(7) noise.

const RED = vec3f(0.86, 0.10, 0.16);
const CYAN = vec3f(0.10, 0.72, 0.90);
const AMBER = vec3f(1.0, 0.62, 0.22);

/// The picture on the tube: a striped sun over a perspective grid, with a horizon glow.
fn screen(p: vec2f, t: f32, sun: f32) -> vec3f {
  let horizon = 0.16;
  var col = vec3f(0.020, 0.010, 0.030);
  // Sky: a band of colour above the horizon, darkest at the top.
  col += mix(vec3f(0.06, 0.02, 0.10), vec3f(0.55, 0.10, 0.22), smoothstep(1.0, horizon, p.y));
  // The sun: a disc with scanline gaps, cut by the horizon.
  let sc = vec2f(0.0, horizon + 0.30 + 0.05 * sun);
  let sr = 0.42 + 0.10 * sun;
  let d = length(p - sc);
  let disc = smoothstep(sr, sr - 0.01, d);
  let stripes = step(0.42, fract((p.y - sc.y) * 26.0 + t * 0.10));
  col = mix(col, mix(AMBER, RED, smoothstep(sc.y - sr, sc.y + sr, p.y)), disc * (0.72 + 0.28 * stripes));
  col += RED * 0.25 * exp(-abs(d - sr) * 22.0) * (0.5 + 0.5 * sun);
  // The floor: a perspective grid, scrolling toward the viewer.
  if (p.y < horizon) {
    let depth = horizon - p.y;
    let z = 0.55 / max(depth, 0.006);
    let lx = abs(fract(p.x * z * 0.55 + 0.5) - 0.5);
    let lz = abs(fract(z * 0.60 + t * 0.55) - 0.5);
    let line = min(lx * z * 0.40, lz * z * 0.45);
    let grid = smoothstep(0.055, 0.0, line) * smoothstep(0.0, 0.05, depth);
    let floor_col = mix(vec3f(0.010, 0.012, 0.030), vec3f(0.030, 0.055, 0.090), smoothstep(0.55, 0.0, depth));
    col = mix(col, floor_col, smoothstep(0.0, 0.03, depth));
    col = mix(col, mix(CYAN, vec3f(0.85, 0.95, 1.0), 0.30) * (0.55 + 0.45 * U.mid), grid);
    col += CYAN * 0.12 * exp(-depth * 7.0) * (0.4 + 0.6 * sun);
  }
  return col;
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  // `XY` is measured from the top of the target, so +y is down; flip it so the sun is up and
  // the floor is down.
  let raw = XY(pos);
  let q = vec2f(raw.x, -raw.y);
  let asp = U.res.x / U.res.y;
  let t = U.T * (0.5 + 0.6 * P(0));
  let sun = clamp(U.bass, 0.0, 1.0) * (0.5 + 0.7 * U.pulse);

  // --- the tube: barrel curvature, then the rounded mask of the glass.
  let curve = 0.055 + 0.16 * clamp(P(1), 0.0, 1.0);
  let r2 = dot(q * vec2f(0.55, 1.0), q * vec2f(0.55, 1.0));
  let warped = q * (1.0 + curve * r2) * 0.985;
  let mask = smoothstep(1.02, 0.92, abs(warped.y)) * smoothstep(asp * 1.02, asp * 0.92, abs(warped.x));

  // --- the tracking band: a sliding row that tears, desaturates and darkens.
  let tb = fract(t * 0.33);
  let wid = 0.035 + 0.13 * clamp(P(3), 0.0, 1.0) + 0.08 * clamp(U.high, 0.0, 1.0);
  let band = exp(-pow(abs((warped.y + 1.0) * 0.5 - tb) / wid, 2.0) * 2.2);
  let shift = band * (0.02 + 0.20 * U.pulse + 0.10 * U.high) * (h1(floor(warped.y * 60.0) + floor(t * 9.0)) - 0.4);

  // --- the picture, with chromatic aberration that grows toward the glass's edge.
  let ab = (0.0015 + 0.006 * P(5)) * (1.0 + 2.5 * r2);
  let base_p = warped + vec2f(shift, 0.0);
  // One extra evaluation, not three: the fringe is the red channel read a fraction outside the
  // green one, and blue keeps the green sample's own colour. Three full evaluations of the tube was
  // the most expensive thing in this pass.
  let cr = screen(base_p * (1.0 + ab), t, sun);
  let cg = screen(base_p * (1.0 - ab * 0.35), t, sun);
  var c = vec3f(cr.r, cg.g, cg.b);

  // --- the aperture grille: three colour stripes that the tube's mask puts between the pixels.
  let grille = fract(pos.x / 3.0);
  let stripes = vec3f(
    smoothstep(0.42, 0.0, abs(grille - 0.17)),
    smoothstep(0.42, 0.0, abs(grille - 0.50)),
    smoothstep(0.42, 0.0, abs(grille - 0.83))
  );
  c *= mix(vec3f(1.0), 0.55 + 0.75 * stripes, P(4));

  // --- scanlines, the tube's gamma, and a slow flicker (0.4 Hz: never a strobe).
  let scan = 0.55 + 0.45 * pow(0.5 + 0.5 * cos(pos.y * PI * 0.62), 0.7);
  c *= mix(1.0, scan, 0.35 + 0.45 * P(7));
  c = pow(max(c, vec3f(0.0)), vec3f(1.0 / (1.0 + 0.55 * P(6))));
  c *= 0.86 + 0.14 * sin(t * 2.5) * (0.3 + 0.7 * sun);
  // The tracking band eats the colour it passes.
  c = mix(c, vec3f(dot(c, vec3f(0.33))) * 1.15, band * (0.35 + 0.5 * P(3)));
  c *= 0.45 + 0.55 * band * 0.4 + 0.55 * (1.0 - band);
  // Noise: the tube's snow, denser in the band.
  let snow = h2(floor(pos.xy * 0.5) + floor(t * 24.0));
  c += vec3f(0.55, 0.62, 0.72) * step(0.995 - 0.021 * P(7), snow) * (0.30 + 0.5 * band);
  // The glass's rounded corners and a dark bezel.
  c *= mask;
  c *= smoothstep(1.30, 0.35, length(vec2f(q.x * 0.62, q.y * 0.92)));
  c += vec3f(0.004, 0.005, 0.008);
  // Hue: a channel rotation on the finished frame, so P(2) shifts the capture's colour cast.
  c *= mix(vec3f(1.0), vec3f(0.88, 0.62, 1.24), U.hue);
  return fin(c);
}
