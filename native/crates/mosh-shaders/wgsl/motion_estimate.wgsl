// extracted by scripts/extract_wgsl.mjs from gpu.js: WG_ME
// Do not edit: change gpu.js and re-run the script.

struct Op { rect: vec4f, src: vec4f };
struct FU { res: vec2f, iframe: f32, skipres: f32, leak: f32, mvScale: f32, coarse: f32, sh: f32, inv: f32, fade: f32, xf: f32, frame: f32, nops: f32, gain: f32, p0: f32, p1: f32, fxa: vec4f, fxb: vec4f, fxc: vec4f, fxd: vec4f, fpa: vec4f, fpb: vec4f, fpc: vec4f, fpd: vec4f, ops: array<Op,48> };
@group(0) @binding(0) var<uniform> F: FU;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return vec4f(f32((i << 1u) & 2u) * 2. - 1., f32(i & 2u) * 2. - 1., 0., 1.); }
@group(0) @binding(1) var ls: sampler; @group(0) @binding(2) var curT: texture_2d<f32>; @group(0) @binding(3) var prvT: texture_2d<f32>;
const GS = vec2f(320., 180.);
fn lumc(p: vec2f) -> f32 { return clamp(dot(textureSampleLevel(curT, ls, p / GS, 0.).rgb, vec3f(0.3, 0.59, 0.11)), 0., 1.); }
fn lump(p: vec2f) -> f32 { return clamp(dot(textureSampleLevel(prvT, ls, p / GS, 0.).rgb, vec3f(0.3, 0.59, 0.11)), 0., 1.); }
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let o = floor(pos.xy) * 8.; var cur: array<f32, 64>; var best = 0.;
  for (var i = 0; i < 64; i++) { let p = o + vec2f(f32(i % 8), f32(i / 8)) + 0.5; cur[i] = lumc(p); best += abs(cur[i] - lump(p)) * 255.; }
  var bd = vec2f(0.);
  if (best > 0.5) {
    for (var dy = -6; dy <= 6; dy++) { for (var dx = -6; dx <= 6; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      let d = vec2f(f32(dx), f32(dy)); let lo = o + d;
      if (lo.x < 0. || lo.y < 0. || lo.x + 8. > 320. || lo.y + 8. > 184.) { continue; }
      var cost = 3. * (abs(d.x) + abs(d.y));
      for (var i = 0; i < 64; i++) { cost += abs(cur[i] - lump(lo + vec2f(f32(i % 8), f32(i / 8)) + 0.5)) * 255.; if (cost >= best) { break; } }
      if (cost < best) { best = cost; bd = d; }
    } }
  }
  let mv = clamp(bd / GS * F.gain, vec2f(-56. / 640., -56. / 360.), vec2f(56. / 640., 56. / 360.));
  return vec4f(mv, 0., 1.); }
