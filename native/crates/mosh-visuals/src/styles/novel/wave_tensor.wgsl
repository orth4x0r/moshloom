// Wave · Tensor Streamlines: divergence-free curl vector field advection with analytical Line
// Integral Convolution (LIC) filament ridges.
//
// A divergence-free 2D flow field v(p) = (dPsi/dy, -dPsi/dx) is generated analytically from orbiting
// logarithmic dipole vortices plus a divergence-free harmonic shear wave. Each fragment integrates
// 3 symmetric forward and backward steps along the unit tangent of v(p), convolving a coherent
// high-frequency carrier into silky anisotropic magnetic/fluid streamlines curving around saddles
// and vortex cores. Flow colour uses the continuous unit tangent (zero atan2 branch cuts!).
//
// Audio: `U.bass` orbits and strengthens the vortex dipoles, `mel()` bands illuminate velocity
// strata, and `U.pulse` sends bright luminous packets flowing along the streamlines.
//
// p0 Speed, p1 Filament density, p2 Hue, p3 Turbulence, p4 Vortex pull, p5 Streak length, p6 Sheen, p7 Grain.

// Divergence-free curl velocity v(p) = (dPsi/dy, -dPsi/dx): 2 rational dipoles + 2-trig curl wave.
fn flow_vel(p: vec2f, c0: vec2f, c1: vec2f, vort: f32, turb: f32, t: f32) -> vec2f {
  let d0 = p - c0;
  let v0 = vec2f(-d0.y, d0.x) * (vort / (dot(d0, d0) + 0.14));

  let d1 = p - c1;
  let v1 = vec2f(d1.y, -d1.x) * (vort / (dot(d1, d1) + 0.14));

  // d/dx(sin(k*y)) + d/dy(cos(k*x)) = 0 -> strictly divergence-free with only 2 trig calls.
  let k = 1.85 + 1.55 * turb;
  let vh = vec2f(sin(p.y * k + t * 0.37), cos(p.x * k - t * 0.29)) * (0.45 + 0.90 * turb);
  return v0 + v1 + vh;
}

// Fast 2-trig anisotropic filament carrier sampled along the streamline.
fn lic_carrier(p: vec2f, freq: f32, t: f32) -> f32 {
  let a = sin(p.x * freq + 0.72 * cos(p.y * freq * 0.83 + t * 0.22));
  let b = sin((p.y * 1.12 - p.x * 0.64) * freq - t * 0.18);
  return 0.5 + 0.5 * (a * 0.62 + b * 0.38);
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let t = U.T * 0.50;
  let nrg = energy();
  let hue = P(2);
  let hc = cos(hue * TAU);
  let hs = sin(hue * TAU);
  let c_a = pal(hc, hs);
  let c_b = pal_next(c_a);
  let c_c = next2(c_a);

  let freq = mix(9.0, 27.0, P(1));
  let turb = P(3);
  let vort = mix(0.28, 1.40, P(4)) * (0.75 + 0.35 * clamp(U.bass, 0.0, 1.0));
  let step_h = mix(0.016, 0.058, P(5));
  let sheen = 0.24 + 0.76 * P(6);

  // Orbiting vortex dipole centers.
  let orb = 0.44 + 0.20 * sin(t * 0.29);
  let c0 = vec2f(cos(t * 0.31), sin(t * 0.23)) * orb;
  let c1 = -c0 + vec2f(0.16 * sin(t * 0.41), -0.14 * cos(t * 0.37));

  // Local velocity for smooth branch-cut-free orientation and speed magnitude.
  let v_here = flow_vel(q, c0, c1, vort, turb, t);
  let speed = length(v_here);
  let v_dir = v_here / max(speed, 0.04);

  // Symmetric 3-step forward + 3-step backward analytical LIC advection along unit tangent.
  var pf = q;
  var pb = q;
  var acc = lic_carrier(q, freq, t) * 0.26;
  var diff_acc = 0.0;

  for (var i = 0; i < 3; i = i + 1) {
    let w = 0.22 - f32(i) * 0.05;

    let vf = flow_vel(pf, c0, c1, vort, turb, t);
    pf = pf + (vf / max(length(vf), 0.08)) * step_h;
    let cf = lic_carrier(pf, freq, t);

    let vb = flow_vel(pb, c0, c1, vort, turb, t);
    pb = pb - (vb / max(length(vb), 0.08)) * step_h;
    let cb = lic_carrier(pb, freq, t);

    acc = acc + (cf + cb) * w;
    diff_acc = diff_acc + abs(cf - cb) * w;
  }

  // Sculpted high-contrast anisotropic streamline hairlines over deep obsidian valleys.
  let ridge = smoothstep(0.40, 0.80, acc);
  let silk = ridge * ridge * ridge;
  let fine_hair = glow(acc - 0.65, 9.0);
  let speed_norm = clamp(speed * 0.40, 0.0, 1.0);
  let mel_v = mel(i32(clamp(speed_norm * 22.0, 0.0, 23.0)));

  // Completely C^infinity smooth iridescent flow colouring from unit tangent components (no atan2 seam!).
  let dir_phase = 0.24 * v_dir.x + 0.19 * v_dir.y + 0.14 * (v_dir.x * v_dir.y);
  let flow_irid = irid(hue + dir_phase + speed_norm * 0.34 + 0.05 * t);
  let stream_col = mix(mix(c_a, c_b, speed_norm), flow_irid, 0.64);

  var col = plate(q, c_a * 0.55, t, 7.9);
  col = col + stream_col * (0.06 + 0.86 * silk) * (0.52 + 0.48 * mel_v + 0.22 * nrg);

  // Specular metallic sheen along high-coherence filament crests.
  let crest = mix(vec3f(0.95, 0.98, 1.0), mix(c_c, flow_irid, 0.4), 0.38);
  col = col + crest * fine_hair * sheen * (0.48 + 0.48 * speed_norm + 0.38 * U.pulse);

  // Glowing beat packets racing through high-shear streamlines.
  let packet_wave = 0.5 + 0.5 * sin(speed * 4.2 - t * 2.8 + dot(q, v_dir) * 5.0);
  col = col + mix(c_c, flow_irid, 0.5) * diff_acc * packet_wave * (0.60 + 1.35 * U.pulse) * (0.55 + 0.45 * sheen);

  // Luminous halos around the two orbiting dipole vortex cores.
  let core_glow = (glow(length(q - c0), 2.5) + glow(length(q - c1), 2.5))
    * (0.16 + 0.26 * P(4)) * (0.70 + 0.35 * U.pulse);
  col = col + mix(c_b, c_c, 0.5) * core_glow;

  col = col * (0.90 + 0.16 * smoothstep(1.8, 0.0, length(q)));
  col = col * grain(pos.xy, 0.06 + 0.14 * P(7));
  return fin(shoulder(col));
}
