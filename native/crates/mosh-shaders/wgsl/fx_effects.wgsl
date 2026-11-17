// fx_effects: the effects lane in ONE fragment pass (owner: V-fx, Task 2.3).
//
// Hand written, not extracted: gpu.js (read-only reference) spreads the same eight effects over
// three passes — WG_PRE (mirror, fold, crush, streak), WG_FB (zoom, spin, echo, as feedback
// transforms of its bufT) and WG_OUT (negative) — because the web app owns a feedback buffer per
// layer. The native chain draws a layer as scene -> mosh -> effects -> transition, one target per
// stage, so the whole lane is one pass over the previous stage's picture.
//
// Bind group 0, mosh-render's fixed fullscreen layout:
//   0 uniform EU (80 B) | 1 inT | 2 sampler (immutable) | 3 histT | 4 sampler (immutable)
//
// `histT` is the effects target itself: the previous frame's effects output, which is what makes
// Echo a trail rather than a second copy of the current picture.
//
// Uniform field order matches `EffectUniforms` in mosh-visuals/src/effects.rs (80 bytes, asserted
// by mosh_shaders::uniform_bytes("fx_effects") in that crate's tests):
//   fxa = fold, mirror, zoom, spin          fpa = fold segments, mirror y, zoom out, spin back
//   fxb = streak, crush, echo, negative     fpb = streak bands, crush size, echo decay, invert tint
// Every amount is 0..1 and 0 is the exact identity: each branch multiplies its own contribution
// by the amount, so an inactive slot cannot move a pixel.

struct EU {
  res: vec2f,
  echo_decay: f32,
  pad0: f32,
  fxa: vec4f,
  fxb: vec4f,
  fpa: vec4f,
  fpb: vec4f,
};

@group(0) @binding(0) var<uniform> E: EU;
@group(0) @binding(1) var inT: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
@group(0) @binding(3) var histT: texture_2d<f32>;
@group(0) @binding(4) var smp2: sampler;

fn tx(uv: vec2f) -> vec3f { return textureSampleLevel(inT, smp, uv, 0.).rgb; }
fn lum(c: vec3f) -> f32 { return dot(c, vec3f(0.3, 0.59, 0.11)); }

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let asp = vec2f(E.res.x / max(E.res.y, 1.), 1.);
  let p0 = (pos.xy / E.res - 0.5) * asp;

  // Zoom and spin: a geometric transform about the centre, blended by its own amount (in the web
  // app these are feedback transforms; see the header).
  let spin = 0.45 * E.fxa.w * select(1., -1., E.fpa.w > 0.5);
  let zoom = 1. - 0.4 * E.fxa.z * select(1., -1., E.fpa.z > 0.5);
  var p = vec2f(p0.x * cos(spin) - p0.y * sin(spin), p0.x * sin(spin) + p0.y * cos(spin)) * zoom;
  var uv = p / asp + 0.5;
  var c = tx(uv);

  // Mirror: fold the picture onto its left (and top) half.
  if (E.fxa.y > 0.) {
    var pm = p0;
    pm.x = -abs(pm.x);
    if (E.fpa.y > 0.5) { pm.y = -abs(pm.y); }
    c = mix(c, tx(pm / asp + 0.5), E.fxa.y);
  }

  // Fold: a kaleidoscope with `round(2 + 10 * param)` wedges, as in WG_PRE.
  if (E.fxa.x > 0.) {
    let n = round(2. + E.fpa.x * 10.);
    let sg = 6.2831853 / n;
    let r = length(p);
    var a = atan2(p.y, p.x) + 1.5707963;
    a = abs((a - sg * floor(a / sg)) - sg * 0.5);
    let uf = r * vec2f(sin(a), -cos(a)) / asp + 0.5;
    c = mix(c, tx(uf), E.fxa.x);
    uv = mix(uv, uf, select(0., 1., E.fxa.x >= 0.5));
  }

  // Crush: block quantise, then posterise to fewer levels as the amount rises.
  if (E.fxb.y > 0.) {
    let bs = max(mix(4., 40., clamp(E.fpb.y, 0., 1.)) * E.res.y / 360., 1.);
    c = mix(c, tx((floor(uv * E.res / bs) + 0.5) * bs / E.res), E.fxb.y);
    let levels = mix(24., 2.5, clamp(E.fxb.y, 0., 1.));
    c = floor(c * levels + 0.5) / levels;
  }

  // Streak: pull a bright row above the current band down over this one.
  if (E.fxb.x > 0.) {
    let bands = mix(3., 14., clamp(E.fpb.x, 0., 1.));
    let s = tx(vec2f(uv.x, (floor(uv.y * bands) + 0.02) / bands));
    if (lum(s) > 1. - E.fxb.x * 0.85) { c = s; }
  }

  // Echo: keep the brightest of this frame and the previous effect output, offset to the left so
  // a moving picture leaves a trail within one pass.
  if (E.fxb.z > 0.) {
    let h = textureSampleLevel(histT, smp2, uv - vec2f(0.012, 0.), 0.).rgb;
    c = mix(c, max(c, h * E.echo_decay), E.fxb.z);
  }

  // Negative: invert, tinted red like the web output pass when the slot's parameter is > 0.5.
  if (E.fxb.w > 0.) {
    let inv = select(vec3f(1.) - c, vec3f(1.6, 0.24, 0.16) * c, E.fpb.w > 0.5);
    c = mix(c, inv, E.fxb.w);
  }

  return vec4f(clamp(c, vec3f(0.), vec3f(1.)), 1.);
}
