// Y2K · Holo Globe.
//
// A holographic wire globe: orthographic sphere, meridians and parallels in true perspective, an
// iridescent tint that runs with position, a fresnel rim and a bright scan band that climbs the
// front. Two tilted orbit rings, a lens-flare streak, ghosts along the flare axis and a star field:
// the early-2000s "world wide web" screensaver, at album-cover brightness.
//
// p0 Speed, p1 Globe size, p2 Hue, p3 Grid density, p4 Rim glow, p5 Sparkle, p6 Orbit tilt,
// p7 Lens flare.
//
// Cost: no ray marching. Latitude lines are drawn on `yr`, the projected height of the sphere
// (which is exact: a parallel projects to a horizontal line), so the only transcendentals are the
// shared orbit angle, one `atan2` for the meridians and the palette. Line widths come from the
// analytic screen gradient, not from `fwidth`.

// The orbit angle is shared by the globe's meridians and both rings: one sine, one cosine.
const TILT_C = 0.9394;
const TILT_S = 0.3429;

// A thin ellipse: distance to a circle of `radius`, rotated by the shared (c, s) and squashed on
// its minor axis. `px` keeps it a pixel wide at any radius (dw/de is 1 / (radius * squash)).
fn orbit_line(p: vec2f, radius: f32, squash: f32, c: f32, s: f32, px: f32) -> f32 {
  let u = vec2f(p.x * c - p.y * s, p.x * s + p.y * c);
  let e = length(vec2f(u.x, u.y / squash)) - radius;
  return hairline(e, px * 1.4 / max(radius * squash, 0.06));
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let t = U.T;
  let px = 2.0 / U.res.y;
  let hue = P(2);
  let rim_amt = P(4);
  let sparkle = P(5);
  let flare_amt = P(7);
  let nrg = energy();
  let R = 0.40 + 0.14 * P(1) + 0.03 * U.pulse + 0.04 * U.bass;
  let c0 = vec2f(0.0, 0.02 * sin(t * 0.4));
  let sv = q - c0;
  let rr = dot(sv, sv) / (R * R);

  let hue_ang = TAU * (0.42 + hue);
  let tint = pal(cos(hue_ang), sin(hue_ang));
  let tint2 = pal_next(tint);
  let tint3 = next2(tint);
  // One orbit angle for the wire, the rings and the ghosts; the bands are smoothed, so scaling it
  // by the mid band reads as "faster when it is louder" without ever jumping.
  let ang = t * (0.18 + 0.25 * U.mid);
  let sc = cos(ang);
  let ss = sin(ang);

  // Background: near-black indigo, a halo behind the globe, stars.
  var col = mix(vec3f(0.012, 0.01, 0.04), vec3f(0.05, 0.015, 0.11), smoothstep(-1.0, 1.0, -q.y));
  col = col + tint2 * 0.10 * falloff(sqrt(max(rr, 0.0)), 2.2) * (0.5 + 0.6 * U.pulse);
  col = col + tint * glints(q, t, 62.0, 0.985) * (0.20 + 0.9 * sparkle) * 0.55;

  // The globe, drawn additively so it reads as a hologram rather than a solid ball.
  if (rr < 1.0) {
    let z = sqrt(max(0.0, 1.0 - rr));
    let p3 = vec3f(sv.x / R, sv.y / R, z);
    let yr = p3.y * TILT_C - p3.z * TILT_S;
    let zr = p3.y * TILT_S + p3.z * TILT_C;
    let count = 4.0 + 10.0 * P(3);
    // A parallel projects to `yr` = sin(latitude): lines of constant yr are the latitude circles.
    let latv = yr * count + sc * 0.0;
    // Meridians are the curves of constant atan2(x, z): the one transcendental the grid needs.
    let lonv = (atan2(p3.x, zr) + ang) * (count / TAU);
    let wl = count * px * 0.55 / R;
    let wn = count * px * 0.55 / (TAU * max(length(vec2f(p3.x, zr)), 0.12) * R);
    let grid = max(hairline(fract(latv + 0.5) - 0.5, wl), hairline(fract(lonv + 0.5) - 0.5, wn));
    // The scan band climbs the front once every three seconds.
    let scan = falloff(p3.y - mix(-1.0, 1.0, fract(t * 0.33)), 30.0) * z;
    let rim = 1.0 - z;
    let fres = rim * rim * rim * rim;
    var globe = vec3f(0.01, 0.03, 0.08) + tint * (0.06 + 1.1 * grid) * (0.6 + 0.7 * nrg);
    globe = globe + tint * fres * (0.35 + 1.2 * rim_amt) * (0.6 + 0.5 * U.pulse);
    globe = globe + tint3 * scan * 0.30 * (0.4 + 0.8 * U.mid);
    // The wire is dimmer at the silhouette, where it bunches up; the fill deepens to the centre.
    globe = globe * (0.45 + 0.55 * z);
    col = col + globe;
  }

  // Orbit rings: additive, so they pass over the wire instead of cutting a hole in it.
  let tilt = 0.10 + 0.55 * P(6);
  col = col + tint3 * orbit_line(sv, R * 1.16, tilt, sc, ss, px) * (0.55 + 0.8 * U.pulse);
  col = col + tint2 * orbit_line(sv, R * 1.34, 0.22 + 0.4 * (1.0 - P(6)), sc, -ss, px)
    * (0.35 + 0.6 * U.mid);

  // Lens flare: a streak through the light and three ghosts along the axis.
  let light = vec2f(0.62, -0.55);
  col = col + vec3f(0.45, 0.65, 1.0) * falloff(abs(q.y - light.y), 900.0)
    * falloff(abs(q.x - light.x), 1.2) * flare_amt * (0.5 + 0.5 * U.pulse);
  let g1 = c0 - (light - c0) * 0.5;
  let g2 = c0 - (light - c0) * 1.1;
  let g3 = c0 - (light - c0) * 1.7;
  col = col + tint * falloff(length(q - g1) / 0.07, 1.0) * 0.30 * flare_amt;
  col = col + tint2 * falloff(length(q - g2) / 0.10, 1.0) * 0.20 * flare_amt;
  col = col + tint3 * falloff(length(q - g3) / 0.05, 1.0) * 0.30 * flare_amt;
  return fin(col);
}
