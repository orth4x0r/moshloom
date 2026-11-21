// Wave · Quasicrystal Lattice: de Bruijn multigrid N-fold non-periodic plane-wave interference
// (5-fold Penrose / 7-fold heptagonal / 9-fold enneagonal) with analytic 3D facet relief and
// thin-film iridescence.
//
// Wavevectors k_j and golden-angle phason oscillators advance via exact 2D rotation recurrences
// (zero extra trig inside the loop), summing cos(k_j . p + phi_j) alongside its exact analytical
// 2D gradient. Live `mel()` bands drive per-axis phason shifts so tiles smoothly re-tessellate in
// long-range aperiodic order without ever repeating periodically across the stage.
//
// Audio: `mel()` bands drive per-axis phason flips, `U.high` sharpens metallic facet ridges, and
// `U.pulse` shifts the thin-film interference angle across the crystal facets.
//
// p0 Speed, p1 Lattice scale, p2 Hue, p3 Fold order, p4 Phason, p5 Facet relief, p6 Iridescence, p7 Grain.

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

  // Slow rotation + scale controlled by P(1).
  let rot = t * 0.06;
  let rc = cos(rot);
  let rs = sin(rot);
  let scale = mix(7.5, 22.0, P(1));
  let p = vec2f(rc * q.x - rs * q.y, rs * q.x + rc * q.y) * scale;

  // Choose odd rotational symmetry 5 (Penrose), 7 (heptagonal), or 9 (enneagonal) via P(3).
  let fold_n = select(select(5, 7, P(3) >= 0.33), 9, P(3) >= 0.67);
  let nf = f32(fold_n);
  let d_ang = PI / nf;
  let cd = cos(d_ang);
  let sd = sin(d_ang);

  // Golden-angle oscillator recurrence for per-axis phason drift (0 inner-loop trig!).
  let phason_amp = 0.25 + 1.85 * P(4);
  var ph_v = vec2f(cos(t * 0.23), sin(t * 0.23));
  let cg = -0.04718; // cos(1.618 rad)
  let sg = 0.99889;  // sin(1.618 rad)

  var kv = vec2f(1.0, 0.0);
  var w_sum = 0.0;
  var gx = 0.0;
  var gy = 0.0;

  for (var j = 0; j < 9; j = j + 1) {
    if (j >= fold_n) {
      break;
    }
    let jf = f32(j);
    let mel_j = mel(j * 2 + 1);
    let sign_j = select(-1.0, 1.0, (j & 1) == 0);
    let phase = t * (0.35 + 0.09 * jf) * sign_j
      + phason_amp * (0.55 * ph_v.y + 0.85 * mel_j);

    let arg = dot(kv, p) + phase;
    let cw = cos(arg);
    let sw = sin(arg);
    w_sum = w_sum + cw;
    gx = gx - sw * kv.x;
    gy = gy - sw * kv.y;

    // Advance wavevector and golden-angle phason oscillator via 2D rotation.
    kv = vec2f(kv.x * cd - kv.y * sd, kv.x * sd + kv.y * cd);
    ph_v = vec2f(ph_v.x * cg - ph_v.y * sg, ph_v.x * sg + ph_v.y * cg);
  }

  let inv_n = 1.0 / nf;
  let w = w_sum * inv_n;
  let grad = vec2f(gx, gy) * inv_n;
  let grad_len = length(grad);

  // Embossed 3D crystalline normal from the exact analytic gradient.
  let relief = mix(0.65, 3.8, P(5));
  let normal = normalize(vec3f(-grad.x * relief, -grad.y * relief, 1.0));

  // Dual-grid facet edges occur where the interference field crosses zero.
  let d_facet = abs(w) / max(grad_len, 0.12);
  let ridge = glow(d_facet, 2.8);
  let crisp_edge = 1.0 - smoothstep(0.06, 0.24, d_facet);

  // Directional key + rim light over the embossed facets.
  let light_dir = normalize(vec3f(-0.45, 0.68, 0.58));
  let ndl = clamp(dot(normal, light_dir), 0.0, 1.0);
  let spec = ndl * ndl * ndl * ndl;

  // Thin-film iridescence indexed by view tilt + interference height + beat shift.
  let irid_mix = 0.22 + 0.74 * P(6);
  let film = irid(hue + 0.55 * normal.x + 0.42 * normal.y + 0.30 * w + 0.14 * U.pulse + 0.05 * t);
  let metal_base = mix(c_a * 0.35 + c_b * 0.25, film, irid_mix);

  var col = plate(q, c_a * 0.5, t, 6.2) * 0.5;
  col = col + metal_base * (0.28 + 0.82 * ndl) * (0.72 + 0.36 * nrg);

  // Brushed silver specular + facet-edge glint.
  let silver = mix(vec3f(0.86, 0.90, 0.96), film, 0.35);
  col = col + silver * spec * (0.35 + 0.55 * P(5));
  let edge_boost = (0.25 + 0.55 * P(5)) * (0.70 + 0.40 * clamp(U.high, 0.0, 1.0) + 0.30 * U.pulse);
  col = col + mix(c_c, vec3f(0.98, 0.98, 1.0), 0.5) * (crisp_edge * 0.52 + ridge * 0.32) * edge_boost;

  // Deep shadow inside negative interference troughs for sculptural contrast.
  col = col * (0.62 + 0.42 * smoothstep(-0.55, 0.55, w));
  col = col * (0.90 + 0.16 * smoothstep(1.8, 0.0, length(q)));
  col = col * grain(pos.xy, 0.06 + 0.14 * P(7));
  return fin(shoulder(col));
}
