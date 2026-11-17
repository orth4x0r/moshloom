// extracted by scripts/extract_wgsl.mjs from gpu.js: WG_MIX
// Do not edit: change gpu.js and re-run the script.

struct Op { rect: vec4f, src: vec4f };
struct FU { res: vec2f, iframe: f32, skipres: f32, leak: f32, mvScale: f32, coarse: f32, sh: f32, inv: f32, fade: f32, xf: f32, frame: f32, nops: f32, gain: f32, p0: f32, p1: f32, fxa: vec4f, fxb: vec4f, fxc: vec4f, fxd: vec4f, fpa: vec4f, fpb: vec4f, fpc: vec4f, fpd: vec4f, ops: array<Op,48> };
@group(0) @binding(0) var<uniform> F: FU;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return vec4f(f32((i << 1u) & 2u) * 2. - 1., f32(i & 2u) * 2. - 1., 0., 1.); }
@group(0) @binding(1) var aT: texture_2d<f32>; @group(0) @binding(2) var bT: texture_2d<f32>;
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f { let p = vec2i(pos.xy); return mix(textureLoad(aT, p, 0), textureLoad(bT, p, 0), F.xf); }
