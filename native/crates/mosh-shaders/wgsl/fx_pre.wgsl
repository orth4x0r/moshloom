// extracted by scripts/extract_wgsl.mjs from gpu.js: WG_PRE
// Do not edit: change gpu.js and re-run the script.

struct Op { rect: vec4f, src: vec4f };
struct FU { res: vec2f, iframe: f32, skipres: f32, leak: f32, mvScale: f32, coarse: f32, sh: f32, inv: f32, fade: f32, xf: f32, frame: f32, nops: f32, gain: f32, p0: f32, p1: f32, fxa: vec4f, fxb: vec4f, fxc: vec4f, fxd: vec4f, fpa: vec4f, fpb: vec4f, fpc: vec4f, fpd: vec4f, ops: array<Op,48> };
@group(0) @binding(0) var<uniform> F: FU;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return vec4f(f32((i << 1u) & 2u) * 2. - 1., f32(i & 2u) * 2. - 1., 0., 1.); }
@group(0) @binding(1) var ms: sampler; @group(0) @binding(2) var inT: texture_2d<f32>;
fn tx(uv: vec2f) -> vec3f { return textureSampleLevel(inT, ms, uv, 0.).rgb; }
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let asp = vec2f(F.res.x / F.res.y, 1.); var p = (pos.xy / F.res - 0.5) * asp;
  if (F.fxa.y > 0.) { p.x = -abs(p.x); if (F.fpa.y > 0.5) { p.y = -abs(p.y); } }
  var uv = p / asp + 0.5; var c = tx(uv);
  if (F.fxa.x > 0.) { let n = round(2. + F.fpa.x * 10.); let sg = 6.2831853 / n; let r = length(p); var a = atan2(p.y, p.x) + 1.5707963;
    a = abs((a - sg * floor(a / sg)) - sg * 0.5); let uf = r * vec2f(sin(a), -cos(a)) / asp + 0.5; c = mix(c, tx(uf), F.fxa.x); uv = mix(uv, uf, step(0.5, F.fxa.x)); }
  if (F.fxb.y > 0.) { let bs = mix(4., 40., F.fpb.y) * F.res.y / 360.; c = mix(c, tx((floor(uv * F.res / bs) + 0.5) * bs / F.res), F.fxb.y);
    let L = mix(24., 2.5, F.fxb.y); c = floor(c * L + 0.5) / L; }
  if (F.fxb.x > 0.) { let K = mix(3., 14., F.fpb.x); let s = tx(vec2f(uv.x, (floor(uv.y * K) + 0.02) / K)); if (dot(s, vec3f(0.3, 0.59, 0.11)) > 1. - F.fxb.x * 0.85) { c = s; } }
  return vec4f(c, 1.); }
