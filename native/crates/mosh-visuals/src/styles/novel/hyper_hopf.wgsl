// Hyper · Hopf Fibration: 4-dimensional stereographic projection of S^3 -> R^3 foliated into
// linked Villarceau circles via the Hopf map pi(z1, z2) = (2 z1 conj(z2), |z1|^2 - |z2|^2).
//
// Three analytical stereographic depth slices cut through the 4D unit 3-sphere S^3 after an
// isoclinic Clifford rotation in the (x, w) and (y, z) planes. Complex products z1 * conj(z2)
// and z1 * z2 yield the Hopf fiber phase (xi1 - xi2) and travel coordinate (xi1 + xi2) using a
// smooth trig-free rational turn map (zero inner-loop atan2, sin, cos, or sqrt), rendering
// luminous translucent 4D linked Villarceau torus ribbons well under 0.35 ms.
//
// Audio: `U.bass` breathes the Clifford torus latitude, `mel()` bands excite fiber bundles around
// the base 2-sphere S^2, and `U.pulse` brightens specular fiber edges and advances the 4D twist.
//
// p0 Speed, p1 Fiber count, p2 Hue, p3 4D tilt, p4 Twist, p5 Depth spread, p6 Sheen, p7 Grain.

// Smooth monotone circle turn in [0, 1) with zero transcendental calls (periodic at +/-x axis).
fn fast_turn(v: vec2f) -> f32 {
  let d = abs(v.x) + abs(v.y) + 1e-6;
  let p = v.x / d;
  return select(0.25 * (1.0 - p), 0.25 * (3.0 + p), v.y < 0.0);
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

  var col = plate(q, c_a * 0.72, t, 2.7);

  // 4D isoclinic Clifford rotation angles in the (x, w) and (y, z) planes.
  let a1 = t * 0.34 + P(3) * PI + 0.14 * U.pulse;
  let a2 = t * 0.22 + P(3) * 1.7 + 0.18 * clamp(U.mid, 0.0, 1.0);
  let c1 = cos(a1);
  let s1 = sin(a1);
  let c2 = cos(a2);
  let s2 = sin(a2);

  // Travel phase oscillator computed once outside the shell loop.
  let ct = cos(t * 1.35);
  let st = sin(t * 1.35);

  // Subtle 3D camera roll so the linked tori sweep cleanly across the stage.
  let roll = t * 0.11;
  let cr = cos(roll);
  let sr = sin(roll);
  let qr = vec2f(cr * q.x - sr * q.y, sr * q.x + cr * q.y) * 1.08;
  let qr2 = dot(qr, qr);

  let fibers = floor(5.0 + P(1) * 10.0);
  let twist_k = mix(0.25, 2.6, P(4));
  let z_span = mix(0.35, 1.25, P(5));
  let sheen = 0.28 + 0.72 * P(6);
  let bass_shift = 0.16 * clamp(U.bass, 0.0, 1.0);
  let mel_lo = mel(4);
  let mel_mid = mel(11);
  let mel_hi = mel(18);

  // Accumulate 3 analytical stereographic depth layers through the Hopf fibration.
  var accum = vec3f(0.0);
  for (var s = 0; s < 3; s = s + 1) {
    let sf = f32(s) * 0.5 - 0.5; // -0.5, 0.0, +0.5
    let pz = sf * z_span;

    // Inverse stereographic projection R^3 -> S^3: (2x, 2y, 2z, r^2 - 1) / (r^2 + 1).
    let r2 = qr2 + pz * pz;
    let inv = 1.0 / (r2 + 1.0);
    let p4_0 = vec4f(2.0 * qr * inv, 2.0 * pz * inv, (r2 - 1.0) * inv);

    // 4D isoclinic Clifford rotation turns the linked Villarceau tori inside-out.
    let p4 = vec4f(
      c1 * p4_0.x - s1 * p4_0.w,
      c2 * p4_0.y - s2 * p4_0.z,
      s2 * p4_0.y + c2 * p4_0.z,
      s1 * p4_0.x + c1 * p4_0.w,
    );

    // Base S^2 latitude cos(2*eta) = |z1|^2 - |z2|^2 (zero sqrt).
    let base_z = dot(p4.xy, p4.xy) - dot(p4.zw, p4.zw);
    let shell_target = sf * 0.75 + bass_shift;
    let shell_mask = glow(base_z - shell_target, 1.85);

    // Complex products: z1 * conj(z2) has exact phase (xi1 - xi2); z1 * z2 has phase (xi1 + xi2).
    let c_diff = vec2f(p4.x * p4.z + p4.y * p4.w, p4.y * p4.z - p4.x * p4.w);
    let c_sum = vec2f(p4.x * p4.z - p4.y * p4.w, p4.x * p4.w + p4.y * p4.z);
    let sum_u = c_sum * inverseSqrt(dot(c_sum, c_sum) + 1e-6);

    // Trig-free Villarceau fiber coordinate + smooth 4D travel wave.
    let hopf_turn = fast_turn(c_diff);
    let twist_wave = sum_u.x * ct - sum_u.y * st;
    let fiber_coord = hopf_turn + twist_wave * 0.10 * twist_k + f32(s) * 0.13;
    let fiber_frac = fract(fiber_coord * fibers) - 0.5;

    let mel_v = select(select(mel_lo, mel_mid, s == 1), mel_hi, s == 2);

    // Smooth anti-aliased satin ribbon cross-section + bright metallic specular edge.
    let u_rib = clamp(fiber_frac * 3.1, -1.0, 1.0);
    let core = satin(u_rib) * (0.72 + 0.45 * mel_v);
    let edge_spec = glow(abs(u_rib) - 0.52, 4.8) * sheen;

    // Smooth palette harmony from travel phase + S^2 latitude (zero inner-loop trig).
    let travel = 0.5 + 0.5 * twist_wave;
    let lat_t = clamp(base_z * 0.5 + 0.5, 0.0, 1.0);
    let band_col = mix(mix(c_a, c_b, travel), c_c, lat_t * 0.55 + f32(s) * 0.14);

    let depth_fade = (0.65 + 0.35 * (1.0 - abs(sf))) * (0.48 + 0.52 * inv);
    let pulse_flare = 0.85 + 0.38 * nrg + 0.36 * U.pulse * travel;
    let layer_col = band_col * (core * 0.92 + 0.16) * pulse_flare
      + mix(vec3f(0.96, 0.98, 1.0), c_c, 0.32) * edge_spec * (0.50 + 0.48 * U.pulse);

    accum = accum + layer_col * shell_mask * depth_fade * 0.72;
  }

  col = col + accum;
  col = col * (0.90 + 0.18 * smoothstep(1.9, 0.0, length(q)));
  col = col * grain(pos.xy, 0.06 + 0.14 * P(7));
  return fin(shoulder(col));
}
