// Sigil Halo: concentric rings of barbed chrome around a red core, on a grainy olive-black plate.
//
// The wire is implicit, not a list of capsules: each ring is a wavy circle, and the barbs are one family
// of folded angles — a rectangle in arc/radius space — evaluated in the ring loop. That is what makes five
// barbed rings affordable on an integrated
// GPU: a fold and a rectangle per ring instead of three square roots. Each stroke is then lit as a
// rod the full width of its line (`chrome_ridge` in `styles/paint.wgsl`), so the halo is polished
// metal with a bevel and a shadowed edge rather than neon: cold silver where the studio dome is,
// blood red on the rim, iridescent where the beat swings it.
//
// Audio: the outer rings' thorns grow with a mel band, `U.bass` opens the spacing, `U.pulse` sends a
// red shock ring outward once per beat and flares the core. Nothing flashes faster than the beat.
//
// P(0) clock rate, P(1) rings, P(2) hue, P(3) twist, P(4) thorn length, P(5) core glow,
// P(6) ring spacing, P(7) grain.

const STEEL = vec3f(0.88, 0.91, 0.97);
const RED = vec3f(0.84, 0.045, 0.070);

/// The halo: hub, rings, thorns and the crown, as distances to their centrelines.
fn halo(p: vec2f, rings: f32, grow: f32, twist: f32, spacing: f32, band: f32) -> f32 {
  let r = length(p);
  let a = atan2(p.y, p.x);
  var d = abs(r - 0.082);

  // The barb family: folded angles, so every barb is one rectangle in arc/radius space.
  let n2 = 6.0 + floor(U.dens * 5.0);
  let sg2 = TAU / n2;
  let off2 = (fract(a / sg2 - U.T * 0.05 + twist * 1.6) - 0.5) * sg2;

  for (var i = 0; i < 5; i = i + 1) {
    let fi = f32(i);
    if (fi < rings) {
      // One shared wobble for every ring (its phase per ring is a multiple of the shared angle): five
      // separate sines here would cost more than the rest of the halo together.
      let R = 0.20 + fi * 0.132 * spacing + 0.028 * sin(a * 2.0 + U.T * 0.35 + fi * 0.9);
      d = min(d, abs(r - R));
      // The thorns: every ring's answer to the spectrum, longer on the outside.
      let len = (0.024 + 0.085 * clamp(band, 0.0, 1.2)) * grow * (0.55 + 0.25 * fi);
      let thorn = max(abs(off2 * R), abs(r - R - len * 0.5) - len * 0.5);
      d = min(d, thorn);
    }
  }

  // The crown: a spiked circle just past the last ring, its points sharp in the same way the crest's
  // are. `1 - |sin|` rather than `1 - sqrt(|sin|)`: no square root, and the cusps still read sharp.
  let Rout = 0.20 + (rings - 1.0) * 0.132 * spacing;
  let crown = Rout + (0.06 + 0.30 * clamp(band, 0.0, 1.2)) * grow
    * (1.0 - abs(sin(a * 4.0 + U.T * 0.10)));
  d = min(d, abs(r - crown));
  return d;
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let px = 2.0 / U.res.y;
  let aa = px * 1.10;
  let dens = clamp(U.dens, 0.0, 1.0);
  let r = length(q);

  let rings = 3.0 + floor(dens * 2.0);
  let grow = 0.85 + 0.45 * dens + 0.40 * P(4) + 0.25 * U.pulse;
  let twist = (0.05 + 0.25 * P(3)) * (1.0 + 0.5 * U.vari);
  let spacing = 0.85 + 0.30 * P(6) + 0.10 * clamp(U.bass, 0.0, 1.0);
  let band = mel(i32(3.0 + dens * 9.0)) * 1.4;
  let d = halo(q, rings, grow, twist, spacing, band);
  let g = vec2f(dpdx(d), dpdy(d));
  let w = 0.0170 + 0.0070 * P(4);

  // --- the plate: a grainy gradient with a cold pool above and a red heart in the middle.
  let hue_ang = TAU * P(2);
  let hue_tint = pal(cos(hue_ang), sin(hue_ang));
  var c = plate(q, hue_tint * 0.30 + vec3f(0.08, 0.07, 0.09), U.T, 4.1);
  c += RED * (0.045 + 0.095 * P(5)) * falloff(r, 2.9) * (0.35 + 1.05 * U.pulse * U.pulse);
  c += vec3f(0.10, 0.16, 0.24) * 0.14 * glow(q.y - 0.90, 1.5) * (0.4 + 0.6 * U.high);

  // The core: a dark disc with a red heart that swells on the beat, ringed by a bright meniscus.
  let core = 1.0 - smoothstep(-aa, aa, r - 0.052);
  c = mix(c, vec3f(0.016, 0.004, 0.008) + RED * (0.30 + 1.10 * U.pulse * U.pulse), core);
  c += vec3f(0.80, 0.88, 1.0) * 0.22 * glow(r - 0.052, 150.0);

  // The beat's shock ring: it expands once per beat and fades, never strobes.
  let shock = abs(r - (0.26 + 1.05 * (1.0 - U.pulse)));
  c += RED * 0.42 * glow(shock, 30.0) * U.pulse;

  // --- the wire. Studio light is only sampled near a stroke.
  let cov = 1.0 - smoothstep(w - aa, w + aa, d);
  if (d < w + px * 2.5) {
    let accent = mix(RED, hue_tint, 0.38) * (0.75 + 0.55 * P(4));
    let metal = chrome_ridge(d, g, w, STEEL, accent, 0.45 + 0.45 * clamp(U.high, 0.0, 1.0));
    c = mix(c, metal, cov);
  }
  c *= 1.0 - 0.26 * glow(d - w, 30.0);
  c *= grain(pos.xy, 0.10 + 0.20 * P(7));
  c *= mix(vec3f(1.0), vec3f(0.96, 0.80, 1.10), P(2) * 0.55);
  return fin(shoulder(c));
}
