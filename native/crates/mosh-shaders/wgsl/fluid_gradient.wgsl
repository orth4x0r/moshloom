// extracted by scripts/extract_wgsl.mjs from gpu.js: WG_FGRD
// Do not edit: change gpu.js and re-run the script.
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return vec4f(f32((i << 1u) & 2u) * 2. - 1., f32(i & 2u) * 2. - 1., 0., 1.); }

struct FUB { texel: vec2f, dt: f32, visc: f32, imp: vec4f, beat: f32, bass: f32, pulse: f32, gain: f32, vort: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var<uniform> Q: FUB;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var tA: texture_2d<f32>;
@group(0) @binding(3) var tB: texture_2d<f32>;
@group(0) @binding(4) var tC: texture_2d<f32>;
@group(0) @binding(5) var tD: texture_2d<f32>;
fn UV(pos: vec4f) -> vec2f { return pos.xy * Q.texel; }
fn VEL(p: vec2f) -> vec2f { return textureSampleLevel(tA, smp, p, 0.).xy; }

@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = UV(pos); let dx = vec2f(Q.texel.x, 0.); let dy = vec2f(0., Q.texel.y);
  var v = textureSampleLevel(tA, smp, p, 0.).xy;
  let l = textureSampleLevel(tB, smp, p - dx, 0.).x; let r = textureSampleLevel(tB, smp, p + dx, 0.).x;
  let b = textureSampleLevel(tB, smp, p - dy, 0.).x; let t = textureSampleLevel(tB, smp, p + dy, 0.).x;
  v -= vec2f(r - l, t - b) * 0.5;
  return vec4f(v, 0., 1.); }
