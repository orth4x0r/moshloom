// extracted by scripts/extract_wgsl.mjs from gpu.js: WG_OUT
// Do not edit: change gpu.js and re-run the script.

struct Op { rect: vec4f, src: vec4f };
struct FU { res: vec2f, iframe: f32, skipres: f32, leak: f32, mvScale: f32, coarse: f32, sh: f32, inv: f32, fade: f32, xf: f32, frame: f32, nops: f32, gain: f32, p0: f32, p1: f32, fxa: vec4f, fxb: vec4f, fxc: vec4f, fxd: vec4f, fpa: vec4f, fpb: vec4f, fpc: vec4f, fpd: vec4f, ops: array<Op,48> };
@group(0) @binding(0) var<uniform> F: FU;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return vec4f(f32((i << 1u) & 2u) * 2. - 1., f32(i & 2u) * 2. - 1., 0., 1.); }
@group(0) @binding(1) var ns: sampler; @group(0) @binding(2) var bufT: texture_2d<f32>;
fn hh(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy / F.res; let o = vec2f(F.sh, 0.);
  var c = clamp(vec3f(textureSampleLevel(bufT, ns, uv - o, 0.).r, textureSampleLevel(bufT, ns, uv, 0.).g, textureSampleLevel(bufT, ns, uv + o, 0.).b), vec3f(0.), vec3f(1.));
  c = mix(c, 1. - c, F.fxb.w);
  if (F.inv == 1.) { c = 1. - c; } else if (F.inv == 2.) { c = c * vec3f(1.6, 0.24, 0.16); }
  let v = (uv - 0.5) * 2.; let vig = 1. - 0.45 * min(dot(v, v), 1.6);
  let n = (hh(floor(pos.xy * 360. / F.res.y) + F.frame * 0.618) - 0.5) * 0.11;
  return vec4f((c * vig + n) * F.fade, 1.); }
