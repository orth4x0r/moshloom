// Y2K · Bubble Bloom.
//
// The early-2000s screensaver: a field of glossy soap bubbles drifting up the screen, each one a
// translucent film with an iridescent skin, a big soft highlight and a hot specular dot, plus a
// lens-flare cross on the large ones. Behind them, the same picture out of focus: five huge bokeh
// discs, half a bubble's brightness. Glints and dust over the top.
//
// p0 Speed, p1 Bubble size, p2 Hue, p3 Spread, p4 Gloss, p5 Iridescence, p6 Squash, p7 Sparkle.
//
// Cost: the bubbles are written out one call each instead of a dynamic loop, so every bubble's
// lane, phase, speed and size are literal constants and no table is indexed at runtime. One sine
// and one cosine drive every wobble and every bokeh drift; each bubble is one `sqrt` (no
// `normalize`) and two integer powers.

// One drifting bubble's centre: it climbs, wraps, and sways with the shared sway.
fn rise(lane: f32, phase: f32, speed: f32, sway: f32, t: f32) -> vec2f {
  return vec2f(lane + 0.06 * sway, 1.25 - fract(t * speed + phase) * 2.55);
}

// One bubble's contribution. `gloss` tightens the highlight, `irid` lifts the film's rim, `flare`
// adds the lens cross the big ones have. The normal is (dn, z) with dn = d / r and z = sqrt(1 - r2),
// which is already unit length, so no `normalize` is needed.
fn bubble(
  col: vec3f,
  q: vec2f,
  c: vec2f,
  r: f32,
  tint: vec3f,
  gloss: f32,
  irid: f32,
  squash: f32,
  sway: f32,
  flare: f32,
) -> vec3f {
  let d = vec2f(q.x - c.x, (q.y - c.y) / (1.0 + squash * 0.22 * sway));
  let r2 = dot(d, d) / (r * r);
  if (r2 > 1.06) {
    return col;
  }
  let dn = d / r;
  let z = sqrt(max(0.0, 1.0 - r2));
  let fres = (1.0 - z) * (1.0 - z);
  // A soap film: almost invisible in the middle, where the background shows through, then an
  // iridescent skin, then the rim. The alpha follows the same curve, so the middle is glass and
  // only the edges are body.
  var b = tint * (0.10 + 0.80 * fres) + vec3f(0.02, 0.04, 0.06) * (1.0 - fres);
  let key = vec3f(-0.5, -0.65, 0.62);
  let ndl = max(dn.x * key.x + dn.y * key.y + z * key.z, 0.0);
  let s2 = ndl * ndl;
  let s4 = s2 * s2;
  let s8 = s4 * s4;
  let soft = mix(s4, s8, gloss);
  let hot = s8 * s8 * s8;
  b = b + vec3f(0.90, 0.96, 1.0) * soft * 1.1 + vec3f(1.0) * hot * 1.8;
  // The mirrored crescent on the lower rim: the second light every bubble has.
  b = b + tint * max(-dn.y, 0.0) * fres * 0.8;
  // A dark refraction band just inside the top edge gives the film its thickness.
  b = b * (1.0 - 0.35 * smoothstep(0.25, 0.75, dn.y) * smoothstep(0.55, 0.95, r2));
  let a = smoothstep(1.05, 0.97, r2) * clamp(0.10 + 1.0 * fres, 0.0, 1.0);
  var out = mix(col, b, a);
  if (flare > 0.0) {
    let cross = falloff(abs(d.y), 3800.0) * smoothstep(2.2, 0.0, abs(d.x))
      + falloff(abs(d.x), 5200.0) * smoothstep(1.2, 0.0, abs(d.y)) * 0.7;
    out = out + vec3f(0.75, 0.9, 1.0) * cross * flare;
  }
  return out;
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let t = U.T;
  let hue = P(2);
  let spread = P(3);
  let gloss = P(4);
  let irid = P(5);
  let squash = P(6);
  let sparkle = P(7);
  let nrg = energy();
  let grow = (0.65 + 0.9 * P(1)) * (1.0 + 0.05 * U.pulse + 0.06 * nrg);
  // Every wobble in the scene is this one sway (and its cosine for the bokeh).
  let sway = sin(t * 0.55);
  let sway_c = cos(t * 0.55);
  let hue_ang = TAU * (0.46 + hue);
  let tint = pal(cos(hue_ang), sin(hue_ang));
  let tint2 = pal_next(tint);
  let tint3 = next2(tint);

  // Background: a deep teal-to-indigo wash, brighter at the bottom where the bubbles come from,
  // with two soft room lights behind the field.
  var col = mix(vec3f(0.02, 0.06, 0.11), vec3f(0.012, 0.014, 0.05), smoothstep(-1.0, 1.0, q.y));
  col = col + tint * 0.05 * smoothstep(1.0, -0.2, q.y);
  col = col + tint2 * 0.12 * falloff(length(q - vec2f(0.30, -0.55)), 1.1) * (0.6 + 0.4 * nrg);
  col = col + tint * 0.09 * falloff(length(q - vec2f(-0.62, 0.62)), 1.4);

  // Out-of-focus bokeh: five big, soft rings, drifting with the same sway, each on a literal orbit.
  col = col + bokeh(q, vec2f(-0.75, 0.35) + vec2f(0.10, 0.0) * sway, 0.30, tint3, nrg);
  col = col + bokeh(q, vec2f(0.55, -0.45) + vec2f(-0.08, 0.05) * sway_c, 0.42, tint2, nrg);
  col = col + bokeh(q, vec2f(0.15, 0.75) + vec2f(0.06, -0.04) * sway, 0.24, tint, nrg);
  col = col + bokeh(q, vec2f(-0.35, -0.70) + vec2f(0.05, 0.0) * sway_c, 0.34, tint2, nrg);
  col = col + bokeh(q, vec2f(0.85, 0.60) + vec2f(-0.05, 0.03) * sway, 0.20, tint3, nrg);

  // The bubbles, back to front: the later ones are larger and closer.
  let lane = 0.55 + 0.75 * spread;
  col = bubble(col, q, rise(-0.82 * lane, 0.13, 0.055, sway, t), 0.09 * grow, tint, gloss, irid, squash, sway, 0.0);
  col = bubble(col, q, rise(0.63 * lane, 0.61, 0.075, sway, t), 0.11 * grow, tint2, gloss, irid, squash, sway, 0.0);
  col = bubble(col, q, rise(-0.31 * lane, 0.34, 0.045, sway, t), 0.14 * grow, tint3, gloss, irid, squash, sway, 0.35);
  col = bubble(col, q, rise(0.88 * lane, 0.82, 0.065, sway, t), 0.10 * grow, tint, gloss, irid, squash, sway, 0.0);
  col = bubble(col, q, rise(-0.62 * lane, 0.07, 0.085, sway, t), 0.17 * grow, tint2, gloss, irid, squash, sway, 0.45);
  col = bubble(col, q, rise(0.24 * lane, 0.46, 0.035, sway, t), 0.21 * grow, tint3, gloss, irid, squash, sway, 0.55);
  col = bubble(col, q, rise(-0.10 * lane, 0.71, 0.058, sway, t), 0.13 * grow, tint, gloss, irid, squash, sway, 0.25);
  col = col + vec3f(0.85, 0.92, 1.0) * glints(q, t, 30.0, 0.968) * (0.2 + 1.0 * sparkle)
    * (0.4 + 0.4 * nrg);
  return fin(col);
}
