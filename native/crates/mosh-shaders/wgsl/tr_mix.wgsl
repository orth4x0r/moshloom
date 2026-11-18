// tr_mix: a per-layer transition in one fragment pass: Crossfade, Melt, and the full-frame flash
// that Strobe adds on top (owner: V-fx, Task 2.3).
//
// Hand written, not extracted. Reference: gpu.js's WG_MIX (`xf`, a straight pointwise mix) plus
// `decide()` in engine.c (read-only) for the melt freeze and the strobe invert/tint pair.
//
// Bind group 0, mosh-render's fixed fullscreen layout:
//   0 uniform TU (32 B) | 1 fromT | 2 sampler (immutable) | 3 srcT | 4 sampler (immutable)
//
// `fromT` is TransitionPass's own copy of last frame's picture, taken in `begin` and kept for the
// whole transition; `srcT` is this frame's picture after the mosh and effects stages.
//
// Uniform field order matches `TransitionUniforms` in mosh-visuals/src/transitions.rs:
//   res, progress (0..1, smoothstepped here like engine.c's `xf`), mode (0 crossfade, 1 unused,
//   2 melt), flash (0 none, 1 invert, 2 tint), param (reserved), two pads.
//
// Cut never reaches this shader: it is a copy of `srcT`, so a cut is exact and instant.
//
// Datamosh never reaches it either. The web app's datamosh is the mosh feedback buffer carried
// across the cut (`MoshPass`, V-mosh), so `TransitionPass::apply` returns without drawing and this
// pass keeps no mode for it.

struct TU {
  res: vec2f,
  progress: f32,
  mode: f32,
  flash: f32,
  param: f32,
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var<uniform> T: TU;
@group(0) @binding(1) var fromT: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
@group(0) @binding(3) var srcT: texture_2d<f32>;
@group(0) @binding(4) var smp2: sampler;

fn smoothstep01(x: f32) -> f32 { let t = clamp(x, 0., 1.); return t * t * (3. - 2. * t); }
fn hh(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy / T.res;
  let p = smoothstep01(T.progress);
  let b = textureSampleLevel(srcT, smp2, uv, 0.).rgb;
  var c = textureSampleLevel(fromT, smp, uv, 0.).rgb;

  if (T.mode == 2.) {
    // Melt: the old picture sags by a per-column drip while the new one fades in over it.
    let cols = 24.;
    let drip = ((hh(vec2f(floor(uv.x * cols), 7.)) - 0.5) * 0.12 + 0.06) * p;
    c = textureSampleLevel(fromT, smp, uv + vec2f(0., -drip), 0.).rgb;
  }

  c = mix(c, b, p);
  if (T.flash == 1.) {
    c = vec3f(1.) - c;
  } else if (T.flash == 2.) {
    c = c * vec3f(1.6, 0.24, 0.16);
  }
  return vec4f(clamp(c, vec3f(0.), vec3f(1.)), 1.);
}
