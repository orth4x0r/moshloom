// Hyper · Poincare Tessellation: exact inversive geometry in the Poincare disk model of the
// hyperbolic plane H^2.
//
// A Mobius isometry T_a(z) = (z - a) / (1 - conj(a) * z) glides continuously through hyperbolic
// space, then an initial dihedral wedge fold followed by 5 trig-free orthogonal circle-inversion
// and reflection iterations maps every point inside the unit disk into the fundamental hyperbolic
// triangle while accumulating the exact conformal Jacobian (`scale`) and reflection parity
// (`flips`). Screen-space edge thickness is divided by `scale`, keeping geodesics crisp all the
// way toward the ideal boundary |z| = 1 without `fwidth`.
//
// Audio: `U.bass` drives the Mobius translation, `mel()` bands brighten geodesic tiers, and
// `U.pulse` sends a hyperbolic radial ring wave from the disk center.
//
// p0 Speed, p1 Symmetry, p2 Hue, p3 Curvature, p4 Drift, p5 Edge glow, p6 Contrast, p7 Grain.

fn cmul(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cdiv(a: vec2f, b: vec2f) -> vec2f {
  let d = max(dot(b, b), 1e-6);
  return vec2f(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / d;
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let t = U.T * 0.48;
  let nrg = energy();
  let hue = P(2);
  let hc = cos(hue * TAU);
  let hs = sin(hue * TAU);
  let c_a = pal(hc, hs);
  let c_b = pal_next(c_a);
  let c_c = next2(c_a);

  // Fit the Poincare disk cleanly inside the viewport with room for the horizon aura.
  let disk_scale = 0.90;
  let p = q / disk_scale;
  let r2 = dot(p, p);
  let r = sqrt(r2);

  // Subtle dark plate behind and outside the ideal boundary |z| = 1.
  var col = plate(q, c_a * 0.72, t, 1.3);
  let rim_halo = glow(r - 1.0, 6.2) * (0.32 + 0.36 * P(5)) * (0.72 + 0.28 * U.pulse);
  col = col + mix(c_a, c_b, 0.5) * rim_halo;

  if (r2 < 0.996) {
    // Rotate slowly and apply a Mobius hyperbolic isometry T_a(z) = (z - a) / (1 - conj(a) z).
    let rot_a = t * 0.16;
    let rc = cos(rot_a);
    let rs = sin(rot_a);
    var z = vec2f(rc * p.x - rs * p.y, rs * p.x + rc * p.y);

    let drift_amp = (0.10 + 0.44 * P(4)) * (0.65 + 0.35 * clamp(U.bass, 0.0, 1.0));
    let a = vec2f(cos(t * 0.37), sin(t * 0.29)) * drift_amp;
    let num = z - a;
    let den = vec2f(1.0, 0.0) - cmul(vec2f(a.x, -a.y), z);
    let den2 = max(dot(den, den), 1e-5);
    z = cdiv(num, den);
    // Exact conformal scale of T_a at z is (1 - |a|^2) / |1 - conj(a) z|^2.
    var scale = max(1.0 - dot(a, a), 0.05) / den2;

    // Hyperbolic (sym, q, 2) triangle group parameters: symmetry order 4..8 via P(1).
    let sym = floor(4.0 + P(1) * 4.99);
    let sector = PI / sym;
    let c_sec = cos(sector);
    let s_sec = sin(sector);
    let n1 = vec2f(-s_sec, c_sec);
    // Second mirror at 2*sector via double-angle identity (zero extra trig).
    let c2 = 2.0 * c_sec * c_sec - 1.0;
    let s2 = 2.0 * s_sec * c_sec;
    let n2 = vec2f(-s2, c2);

    // Initial one-shot dihedral wedge fold using a single atan2 + cos/sin pair.
    var flips = 0.0;
    if (z.y < 0.0) {
      z.y = -z.y;
      flips = flips + 1.0;
    }
    let ang0 = atan2(z.y, z.x + 1e-7);
    let k0 = floor(ang0 / (2.0 * sector) + 0.5);
    if (abs(k0) > 0.5) {
      let th = -k0 * 2.0 * sector;
      let tc = cos(th);
      let ts = sin(th);
      z = vec2f(tc * z.x - ts * z.y, ts * z.x + tc * z.y);
    }

    // Orthogonal inversion circle: center (cx, 0), radius sqrt(cx^2 - 1).
    let cx = mix(1.16, 1.56, P(3)) - 0.05 * clamp(U.bass, 0.0, 1.0);
    let inv_r2 = max(cx * cx - 1.0, 0.18);
    let inv_r = sqrt(inv_r2);

    // 5 trig-free folding + orthogonal circle-inversion iterations.
    var depth = 0.0;
    for (var i = 0; i < 5; i = i + 1) {
      if (z.y < 0.0) {
        z.y = -z.y;
        flips = flips + 1.0;
      }
      // Mirror across 2*sector, then x-axis, then sector: keeps z inside [0, PI/sym] with 0 trig.
      let p2 = dot(z, n2);
      if (p2 > 0.0) {
        z = z - 2.0 * p2 * n2;
        flips = flips + 1.0;
      }
      if (z.y < 0.0) {
        z.y = -z.y;
        flips = flips + 1.0;
      }
      let p1 = dot(z, n1);
      if (p1 > 0.0) {
        z = z - 2.0 * p1 * n1;
        flips = flips + 1.0;
      }
      if (z.y < 0.0) {
        z.y = -z.y;
        flips = flips + 1.0;
      }

      // Circle inversion across the orthogonal geodesic circle centered at (cx, 0).
      let d = z - vec2f(cx, 0.0);
      let d2 = dot(d, d);
      if (d2 < inv_r2) {
        let inv = inv_r2 / max(d2, 1e-5);
        z = vec2f(cx, 0.0) + d * inv;
        scale = scale * inv;
        flips = flips + 1.0;
        depth = depth + 1.0;
      }
    }

    // Distance to the three geodesic edges of the fundamental hyperbolic triangle, scaled back
    // into screen units using the exact conformal Jacobian `scale`.
    let px = 2.0 / (U.res.y * disk_scale);
    let inv_s = 1.0 / max(scale, 1e-4);
    let d_arc = abs(length(z - vec2f(cx, 0.0)) - inv_r) * inv_s;
    let d_x = abs(z.y) * inv_s;
    let d_wedge = abs(dot(z, n1)) * inv_s;
    let d_edge = min(d_arc, min(d_x, d_wedge));

    // Parity (checkerboard dual) + hyperbolic fold depth + iridescent turn (log-free radial metric).
    let parity = abs(fract(flips * 0.5) * 2.0);
    let hyp_r = r / max(1.08 - r, 0.08);
    let mel_band = mel(i32(clamp(depth * 4.0, 0.0, 23.0)));
    let tile_irid = irid(hue + depth * 0.13 + parity * 0.28 + 0.07 * t + 0.06 * hyp_r);

    let contrast = 0.30 + 0.65 * P(6);
    let tile_tint = mix(
      mix(c_a * 0.24, tile_irid * 0.60, 0.55),
      mix(c_b * 0.52, tile_irid * 0.94, 0.65),
      mix(0.5, parity, contrast)
    );

    // Subtle interior vault shading so each hyperbolic tile feels sculpted.
    let vault = smoothstep(0.0, px * 18.0, d_edge);
    var inside = tile_tint * (0.48 + 0.52 * vault) * (0.68 + 0.40 * mel_band);

    // Hyperbolic beat shockwave travelling outward along radial hyperbolic metric on the beat.
    let wave_pos = fract(t * 0.22) * 5.2;
    let wave = glow(hyp_r - wave_pos, 1.5) * (0.18 + 0.55 * U.pulse);
    inside = inside + c_c * wave * 0.45;

    // Crisp screen-constant geodesic ribs + soft glow.
    let edge_w = px * (1.2 + 2.4 * P(5));
    let line = 1.0 - smoothstep(edge_w * 0.5, edge_w * 1.6, d_edge);
    let edge_bloom = glow(d_edge, 1.0 / max(px * 9.0, 1e-4)) * (0.20 + 0.65 * P(5));
    let edge_col = mix(vec3f(0.95, 0.97, 1.0), mix(c_c, tile_irid, 0.5), 0.45);
    inside = inside + edge_col * (line * (0.55 + 0.45 * P(5)) + edge_bloom * (0.45 + 0.35 * nrg));

    // Smooth anti-aliased boundary fade into the Poincare horizon ring.
    let mask = smoothstep(0.998, 0.980, r);
    let horizon = glow(1.0 - r, 10.0) * (0.40 + 0.45 * U.pulse);
    col = mix(col, inside, mask) + mix(c_b, vec3f(0.96, 0.98, 1.0), 0.4) * horizon * mask;
  }

  col = col * grain(pos.xy, 0.06 + 0.14 * P(7));
  return fin(shoulder(col));
}
