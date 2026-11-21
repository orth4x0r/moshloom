// Hyper · Kleinian Apollyon: Apollonian / Schottky Kleinian limit-set folding.
//
// Six iterations of seamless periodic box-folding and circle-inversion with periodic Maskit
// parabolic shear deform space into self-similar fractal pearl necklaces, vaulted cathedral arches,
// and Apollonian gaskets. Conformal expansion (`lyap`) and continuous geodesic orbit traps shade
// the limit set as sculpted iridescent chrome filigree over a deep velvet plate.
//
// Audio: `U.bass` opens/closes the Schottky kissing radius, `U.high` brightens high-order filigree
// arcs, and `U.pulse` illuminates the orbit-trap filaments.
//
// p0 Speed, p1 Scale, p2 Hue, p3 Cusp, p4 Shear, p5 Filigree, p6 Glow, p7 Grain.

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let t = U.T * 0.45;
  let nrg = energy();
  let hue = P(2);
  let hc = cos(hue * TAU);
  let hs = sin(hue * TAU);
  let c_a = pal(hc, hs);
  let c_b = pal_next(c_a);
  let c_c = next2(c_a);

  // Gentle camera drift + zoom controlled by P(1).
  let zoom = mix(1.38, 0.58, P(1));
  let rot = t * 0.07;
  let rc = cos(rot);
  let rs = sin(rot);
  var p = vec2f(rc * q.x - rs * q.y, rs * q.x + rc * q.y) * zoom;
  p = p + vec2f(0.16 * sin(t * 0.23), 0.12 * cos(t * 0.19));

  // Schottky inversion radius k: near 1.0 gives kissing Apollonian circles; larger k creates
  // dramatic hyperbolic lacing.
  let k_inv = mix(0.88, 1.36, P(3)) + 0.08 * clamp(U.bass, 0.0, 1.0);
  // Periodic Maskit parabolic shear smoothly warps the lattice without boundary seams.
  let shear = (P(4) - 0.5) * 0.58 + 0.08 * sin(t * 0.31);
  let box_w = vec2f(1.0, 0.94 + 0.10 * sin(t * 0.17));
  let inv_2box = 0.5 / box_w;

  var scale = 1.0;
  var trap_arc = 1e4;
  var trap_axis = 1e4;
  var fold_sum = 0.0;

  for (var i = 0; i < 6; i = i + 1) {
    // Seamless C^0 periodic lattice wrap first, then periodic triangle-wave shear in y
    // (equals 0 at +/-box_w.y so adjacent lattice cells join seamlessly with zero seam).
    p = p - 2.0 * box_w * round(p * inv_2box);
    let ny = p.y / box_w.y;
    p.x = p.x + shear * ny * (1.0 - abs(ny));
    p = p - 2.0 * box_w * round(p * inv_2box);

    // Circle inversion centered at the cell origin.
    let r2 = dot(p, p);
    let safe_r2 = max(r2, 0.05);
    let factor = k_inv / safe_r2;
    p = p * factor;
    scale = scale * factor;

    // Continuous orbit traps: distance to the kissing pearl circle and to symmetry axes.
    let r = sqrt(safe_r2);
    trap_arc = min(trap_arc, abs(r - 0.72));
    trap_axis = min(trap_axis, min(abs(p.x), abs(p.y)) / max(scale, 1.0));
    fold_sum = fold_sum + clamp(r, 0.0, 1.4);
  }

  // Smooth bounded expansion measure (no log() needed).
  let lyap = 1.5 * scale / (scale + 18.0);
  let filigree_d = abs(p.y) / max(scale, 1.0);
  let px = 2.0 * zoom / U.res.y;

  // Fine limit-set ribs + sculpted pearl arches.
  let detail_w = px * mix(1.0, 3.0, P(5));
  let rib = 1.0 - smoothstep(detail_w * 0.4, detail_w * 2.2, filigree_d);
  let axis_line = 1.0 - smoothstep(px * 0.5, px * 2.6, trap_axis);
  let pearl_glow = glow(trap_arc, mix(2.2, 6.8, P(5)));

  // Spectral weighting from mel bands keyed to fold depth.
  let mel_v = mel(i32(clamp(lyap * 15.0, 0.0, 23.0)));
  let foil = irid(hue + lyap * 0.52 + fold_sum * 0.055 + 0.06 * t);

  var col = plate(q, c_a * 0.75, t, 4.1);

  // Body of the vaulted Apollonian cells: dark chrome + iridescent depth gradient.
  let body = mix(c_a * 0.24, foil * 0.76, clamp(lyap * 0.75, 0.0, 1.0));
  col = col + body * (0.24 + 0.66 * pearl_glow) * (0.70 + 0.35 * mel_v);

  // Crisp metallic filigree along the limit set.
  let rim_col = mix(vec3f(0.95, 0.97, 1.0), mix(c_b, foil, 0.55), 0.45);
  let fine_boost = (0.35 + 0.65 * P(5)) * (0.75 + 0.35 * clamp(U.high, 0.0, 1.0));
  col = col + rim_col * (rib * 0.68 + axis_line * 0.45) * fine_boost;

  // Soft cathedral aura glow driven by P(6) and beat pulse.
  let aura = glow(trap_arc, 1.9) * (0.14 + 0.52 * P(6)) * (0.65 + 0.45 * U.pulse + 0.25 * nrg);
  col = col + mix(c_c, foil, 0.5) * aura;

  col = col * (0.90 + 0.16 * smoothstep(1.8, 0.0, length(q)));
  col = col * grain(pos.xy, 0.06 + 0.14 * P(7));
  return fin(shoulder(col));
}
