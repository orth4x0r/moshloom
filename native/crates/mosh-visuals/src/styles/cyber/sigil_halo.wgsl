// Sigil Halo: concentric rings of barbed wire in cold chrome around a red core.
//
// The wire is implicit, not a list of capsules: each ring is a wavy circle, and the barbs are two
// folded angles — one family of short ticks, one of long thorns — evaluated in the ring loop as a
// rectangle in arc/radius space. That is what makes eight barbed rings affordable on an integrated
// GPU: a fold and a rectangle per ring instead of three square roots.
//
// Lighting: the field's screen gradient is the normal and the reflection vector samples a fake cold
// studio (bright ceiling, dark floor, one hard strip light), so the chrome has a bevel and a black
// core. `U.pulse` sends a red shock ring outward at the beat, once per beat, never faster.
//
// P(0) clock rate, P(1) rings, P(2) hue, P(3) twist, P(4) thorn length, P(5) core glow,
// P(6) ring spacing, P(7) grain.

const RED = vec3f(0.72, 0.030, 0.055);
const ICE = vec3f(0.62, 0.82, 1.0);
const STEEL = vec3f(0.56, 0.62, 0.72);

/// The halo: hub, rings, barbs and thorns. `rings` is a uniform, so the ring loop is not divergent.
fn halo(p: vec2f, rings: f32, grow: f32, twist: f32, spacing: f32, band: f32) -> f32 {
  let r = length(p);
  let a = atan2(p.y, p.x);
  var d = 1e6;

  // The hub: a heavy ring with eight radial ticks, the halo's centre.
  d = min(d, abs(r - 0.085) - 0.010);

  // The two barb families: folded angles, so every barb is one rectangle in arc/radius space.
  let n1 = 12.0 + floor(U.dens * 10.0);
  let sg1 = TAU / n1;
  let off1 = (fract(a / sg1 + U.T * 0.07 + twist * 0.4) - 0.5) * sg1;
  let n2 = 5.0 + floor(U.dens * 4.0);
  let sg2 = TAU / n2;
  let off2 = (fract(a / sg2 - U.T * 0.05 + twist * 1.6) - 0.5) * sg2;

  for (var i = 0; i < 7; i = i + 1) {
    let fi = f32(i);
    if (fi < rings) {
      let R = 0.19 + fi * 0.115 * spacing + 0.022 * sin(a * (2.0 + fi) + U.T * 0.35 + fi * 1.7);
      d = min(d, abs(r - R) - (0.0042 + 0.0016 * fi));
      // The short ticks every ring wears.
      let tick = max(abs(off1 * R) - (0.0032 + 0.0008 * fi), abs(r - R) - (0.007 + 0.013 * grow));
      d = min(d, tick);
      // The long thorn: the outer rings' answer to the spectrum.
      if (fi > rings - 3.5) {
        let len = (0.030 + 0.095 * clamp(band, 0.0, 1.2)) * grow;
        let thorn = max(abs(off2 * R) - 0.0034, abs(r - R - len * 0.5) - len * 0.5);
        d = min(d, thorn);
      }
    }
  }

  // The outer crown: a spiked circle just past the last ring, its points sharp in the same way the
  // crest's are. This is the halo's silhouette.
  let Rout = 0.19 + (rings - 1.0) * 0.115 * spacing;
  let crown = Rout + (0.06 + 0.30 * clamp(band, 0.0, 1.2)) * (1.0 - sqrt(abs(sin(a * 4.0 + U.T * 0.10))));
  d = min(d, abs(r - crown) - 0.0032);
  return d;
}

/// A soft cold studio for a chrome stroke: bright ceiling, dark floor, one hard strip light.
fn chrome(n: vec3f, refl: vec3f) -> vec3f {
  let up = smoothstep(0.05, 0.92, refl.y);
  let down = smoothstep(0.0, -0.85, refl.y);
  var env = mix(vec3f(0.17, 0.20, 0.27), vec3f(0.94, 0.98, 1.0), up);
  env = mix(env, vec3f(0.03, 0.035, 0.055), down * 0.85);
  let strip = smoothstep(0.45, 0.05, abs(n.x)) * smoothstep(-0.35, 0.55, n.y);
  env += vec3f(0.95, 0.98, 1.0) * strip * 0.9;
  return env;
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let px = 2.0 / U.res.y;
  let aa = px * 1.15;
  let e = px * 2.2;
  let dens = clamp(U.dens, 0.0, 1.0);
  let r = length(q);

  let rings = 3.0 + floor(dens * 3.0);
  let grow = 0.70 + 0.60 * dens + 0.35 * P(4) + 0.30 * U.pulse;
  let twist = (0.05 + 0.25 * P(3)) * (1.0 + 0.5 * U.vari);
  let spacing = 0.82 + 0.36 * P(6);
  let band = mel(i32(3.0 + dens * 9.0)) * 1.4;
  let d = halo(q, rings, grow, twist, spacing, band);
  let g = vec2f(dpdx(d), dpdy(d)) * (e / px);

  // --- background: a black plate, a red core glow, a cold rim, grain.
  var c = vec3f(0.010, 0.011, 0.016) + vec3f(0.026, 0.030, 0.042) * exp(-r * 1.4);
  c += RED * (0.05 + 0.10 * P(5)) * exp(-r * 3.4) * (0.30 + 1.00 * U.pulse * U.pulse);
  c += ICE * 0.014 * exp(-r * 2.2) * (0.30 + 0.70 * U.high);
  c *= 0.92 + 0.16 * h2(pos.xy * 0.75 + U.frame * 0.7) * (0.3 + 0.7 * P(7));

  // The core: a dark disc with a red heart that swells on the beat.
  let core = 1.0 - smoothstep(0.0, aa * 1.6, r - 0.055);
  c = mix(c, vec3f(0.010, 0.002, 0.004) + RED * (0.25 + 1.10 * U.pulse * U.pulse), core);
  c += ICE * 0.25 * exp(-abs(r - 0.055) * 140.0);

  // The beat's shock ring: it expands once per beat and fades, never strobes.
  let shock = abs(r - (0.30 + 1.05 * (1.0 - U.pulse))) - 0.0035;
  c += RED * 0.55 * exp(-max(shock, 0.0) * 55.0) * U.pulse;

  // --- the wire. The studio environment is only sampled near a stroke.
  if (d < px * 11.0) {
    let cov = 1.0 - smoothstep(-aa, aa, d);
    let nrm = normalize(vec3f(-g, e * 0.60));
    let refl = reflect(vec3f(0.0, 0.0, -1.0), nrm);
    let env = chrome(nrm, refl);
    let rim = pow(1.0 - clamp(nrm.z, 0.0, 1.0), 2.2);
    let spec = pow(max(dot(refl, normalize(vec3f(-0.40, 0.78, 0.45))), 0.0), 26.0);
    var metal = STEEL * (0.16 + env * 1.40) + vec3f(1.0, 0.99, 0.97) * spec * 1.2;
    metal = mix(metal, metal * 1.35 + ICE * 0.26, rim);
    metal += RED * rim * (0.15 + 0.65 * U.pulse);
    metal *= 0.68 + 0.32 * smoothstep(0.0, -0.007, d);
    c = mix(c, metal, cov);
    c += STEEL * 0.10 * exp(-max(d, 0.0) * 70.0) * (1.0 - cov);
  }
  // Hue: a channel rotation on the finished frame, so P(2) shifts the whole plate from blood red
  // toward violet without touching the chrome.
  c *= mix(vec3f(1.0), vec3f(0.88, 0.58, 1.28), U.hue);
  return fin(c);
}
