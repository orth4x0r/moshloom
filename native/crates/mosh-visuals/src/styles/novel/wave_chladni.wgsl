// Wave · Chladni Cymatics: exact Helmholtz standing-wave eigenmodes of a vibrating resonant plate
// with analytic gradient-normalized nodal caustics (`|Psi| / |grad Psi|`).
//
// Four acoustic eigenmode pairs are weighted directly by live `mel()` spectral bands (sub-bass,
// low-mid, upper-mid, presence). Harmonic overtones are generated from fundamental trig evaluations
// via exact Chebyshev double/triple-angle identities, cutting transcendental cost in half while
// keeping exact 2D analytical gradients for razor-crisp pixel-uniform nodal sand contours.
//
// Audio: directly spectral — each eigenmode's amplitude comes from its matching `mel()` band
// cluster; `U.bass` flexes the plate warp, `U.pulse` rings the nodal caustics brighter.
//
// p0 Speed, p1 Mode order, p2 Hue, p3 Symmetry, p4 Crispness, p5 Phase glow, p6 Plate warp, p7 Grain.

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let t = U.T * 0.46;
  let nrg = energy();
  let hue = P(2);
  let hc = cos(hue * TAU);
  let hs = sin(hue * TAU);
  let c_a = pal(hc, hs);
  let c_b = pal_next(c_a);
  let c_c = next2(c_a);

  // Subtle anharmonic plate warp (radial lens + slow breathing) controlled by P(6).
  let r2 = dot(q, q);
  let warp = P(6) * (0.28 + 0.18 * clamp(U.bass, 0.0, 1.0));
  let p = q * (1.0 + warp * r2) + 0.06 * vec2f(sin(t * 0.27), cos(t * 0.21));

  // Fundamental wavenumber controlled by Mode order P(1); Symmetry P(3) blends anti/symmetric modes.
  let k = mix(1.4, 3.8, P(1)) * PI;
  let sym = mix(-1.0, 1.0, P(3));

  // Spectral band weights from the 24-band mel analyzer (floored so quiet bars stay lit).
  let w0 = 0.42 + 0.65 * (0.5 * (mel(1) + mel(3)));
  let w1 = 0.36 + 0.60 * (0.5 * (mel(6) + mel(9)));
  let w2 = 0.30 + 0.55 * (0.5 * (mel(12) + mel(15)));
  let w3 = 0.24 + 0.50 * (0.5 * (mel(18) + mel(21)));

  // Fundamental axial angles (only 4 trig calls produce harmonics 1x, 2x, 3x via Chebyshev!).
  let ax = k * p.x + t * 0.38;
  let ay = k * p.y - t * 0.29;
  let c1x = cos(ax);
  let s1x = sin(ax);
  let c1y = cos(ay);
  let s1y = sin(ay);

  let c2x = 2.0 * c1x * c1x - 1.0;
  let s2x = 2.0 * s1x * c1x;
  let c2y = 2.0 * c1y * c1y - 1.0;
  let s2y = 2.0 * s1y * c1y;

  let c3x = c2x * c1x - s2x * s1x;
  let s3x = s2x * c1x + c2x * s1x;
  let c3y = c2y * c1y - s2y * s1y;
  let s3y = s2y * c1y + c2y * s1y;

  // Mode 0: (1, 2) fundamental plate eigenmode
  var psi = w0 * (c1x * c2y + sym * c2x * c1y);
  var gx = w0 * k * (-s1x * c2y - sym * 2.0 * s2x * c1y);
  var gy = w0 * k * (-2.0 * c1x * s2y - sym * s2x * s1y);

  // Mode 1: (1, 3) fifth-harmonic cross eigenmode (0 extra trigs!)
  psi = psi + w1 * (c1x * c3y - sym * c3x * c1y);
  gx = gx + w1 * k * (-s1x * c3y + sym * 3.0 * s3x * c1y);
  gy = gy + w1 * k * (-3.0 * c1x * s3y + sym * c3x * s1y);

  // Mode 2: (2, 3) high-order overtone (0 extra trigs!)
  psi = psi + w3 * (c2x * c3y + sym * c3x * c2y);
  gx = gx + w3 * k * (-2.0 * s2x * c3y - sym * 3.0 * s3x * c2y);
  gy = gy + w3 * k * (-3.0 * c2x * s3y - sym * 2.0 * c3x * s2y);

  // Mode 3: 45-degree rotated diagonal eigenmode (4 trigs -> harmonics 1x and 2x for rosette stars)
  let kd = k * 1.15;
  let u = 0.7071068 * (p.x + p.y);
  let v = 0.7071068 * (p.y - p.x);
  let c1u = cos(kd * u + t * 0.45);
  let s1u = sin(kd * u + t * 0.45);
  let c1v = cos(kd * v - t * 0.31);
  let s1v = sin(kd * v - t * 0.31);
  let c2u = 2.0 * c1u * c1u - 1.0;
  let s2u = 2.0 * s1u * c1u;
  let c2v = 2.0 * c1v * c1v - 1.0;
  let s2v = 2.0 * s1v * c1v;

  psi = psi + w2 * (c1u * c2v + sym * c2u * c1v);
  let du = kd * (-s1u * c2v - sym * 2.0 * s2u * c1v);
  let dv = kd * (-2.0 * c1u * s2v - sym * c2u * s1v);
  gx = gx + w2 * 0.7071068 * (du - dv);
  gy = gy + w2 * 0.7071068 * (du + dv);

  // Exact analytic distance to the Chladni nodal contour `Psi = 0` and secondary harmonic fringe.
  let grad_len = max(length(vec2f(gx, gy)), 1e-4);
  let d_node = abs(psi) / grad_len;
  let d_sub = abs(abs(psi) - 0.55) / grad_len;

  let px = 2.0 / U.res.y;
  let crisp = mix(4.2, 1.35, P(4));
  let sand_core = 1.0 - smoothstep(px * 0.4, px * crisp * 2.2, d_node);
  let sand_halo = glow(d_node, 1.0 / max(px * 14.0, 1e-4));
  let sub_line = (1.0 - smoothstep(px * 0.5, px * crisp * 1.8, d_sub)) * 0.42;

  // Surface normal of the vibrating standing-wave membrane for iridescent sheen.
  let norm = normalize(vec3f(-gx * 0.04, -gy * 0.04, 1.0));
  let irid_col = irid(hue + psi * 0.22 + 0.35 * norm.x + 0.06 * t);

  var col = plate(q, c_a * 0.50, t, 5.3);

  // Subtle antinode phase pools so the dark plate frames the glowing Chladni sand lines.
  let phase_glow = 0.08 + 0.34 * P(5);
  let pos_lobe = smoothstep(0.15, 1.35, psi);
  let neg_lobe = smoothstep(0.15, 1.35, -psi);
  col = col + (c_a * pos_lobe + c_b * neg_lobe) * phase_glow * (0.38 + 0.32 * nrg);
  col = col + irid_col * (0.06 + 0.16 * P(5)) * (0.35 + 0.65 * norm.y) * smoothstep(0.1, 1.2, abs(psi));

  // Luminous crystalline Chladni sand concentrated on the nodal curves.
  let sand_tint = mix(vec3f(0.96, 0.98, 1.0), mix(c_c, irid_col, 0.55), 0.42);
  let beat_ring = 0.72 + 0.42 * U.pulse + 0.22 * clamp(U.high, 0.0, 1.0);
  col = col + sand_tint * (sand_core * 0.92 + sub_line * 0.44) * beat_ring;
  col = col + mix(c_b, irid_col, 0.65) * sand_halo * (0.16 + 0.32 * P(5)) * beat_ring;

  col = col * (0.90 + 0.16 * smoothstep(1.8, 0.0, length(q)));
  col = col * grain(pos.xy, 0.06 + 0.14 * P(7));
  return fin(shoulder(col));
}
