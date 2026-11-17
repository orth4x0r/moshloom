// extracted by scripts/extract_wgsl.mjs from gpu.js: WG_FB
// Do not edit: change gpu.js and re-run the script.

struct Op { rect: vec4f, src: vec4f };
struct FU { res: vec2f, iframe: f32, skipres: f32, leak: f32, mvScale: f32, coarse: f32, sh: f32, inv: f32, fade: f32, xf: f32, frame: f32, nops: f32, gain: f32, p0: f32, p1: f32, fxa: vec4f, fxb: vec4f, fxc: vec4f, fxd: vec4f, fpa: vec4f, fpb: vec4f, fpc: vec4f, fpd: vec4f, ops: array<Op,48> };
@group(0) @binding(0) var<uniform> F: FU;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return vec4f(f32((i << 1u) & 2u) * 2. - 1., f32(i & 2u) * 2. - 1., 0., 1.); }
@group(0) @binding(1) var ns: sampler; @group(0) @binding(2) var srcT: texture_2d<f32>; @group(0) @binding(3) var prvT: texture_2d<f32>;
@group(0) @binding(4) var bufT: texture_2d<f32>; @group(0) @binding(5) var mvT: texture_2d<f32>;
@group(0) @binding(6) var velT: texture_2d<f32>;
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  var uv = pos.xy / F.res; var sw = false;
  for (var i = i32(F.nops) - 1; i >= 0; i--) { let o = F.ops[i];
    if (uv.x >= o.rect.x && uv.x < o.rect.x + o.rect.z && uv.y >= o.rect.y && uv.y < o.rect.y + o.rect.w) {
      if (o.src.z == 2.) { uv = o.rect.xy + o.rect.zw * 0.5; } else { uv = vec2f(fract(uv.x + o.src.x), uv.y + o.src.y); }
      sw = o.src.z == 4.; break; } }
  let s = textureSampleLevel(srcT, ns, uv, 0.).rgb; var c = s;
  if (F.iframe < 0.5) {
    var bc = clamp(vec2i(floor(uv * vec2f(40., 22.5))), vec2i(0), vec2i(39, 22)); if (F.coarse > 0.5) { bc = bc & vec2i(-2); }
    var d = round(textureLoad(mvT, bc, 0).xy * F.mvScale * F.res) / F.res;
    if (F.fxc.x > 0.) { d += textureSampleLevel(velT, ns, uv, 0.).xy * F.fxc.x * 0.03; }   // Fluid: the solver's velocity pushes the pixels
    let asp = vec2f(F.res.x / F.res.y, 1.); var q = (uv + d - 0.5) * asp;     // feedback zoom / spin
    let an = 0.03 * F.fxa.w * select(1., -1., F.fpa.w > 0.5); let z = 1. - 0.035 * F.fxa.z * select(1., -1., F.fpa.z > 0.5);
    q = vec2f(q.x * cos(an) - q.y * sin(an), q.x * sin(an) + q.y * cos(an)) * z;
    c = textureSampleLevel(bufT, ns, q / asp + 0.5, 0.).rgb;
    if (F.skipres < 0.5) { c = c + s - textureSampleLevel(prvT, ns, uv, 0.).rgb; c = mix(c, s, F.leak); }
    if (F.fxb.z > 0.) { c = mix(c, max(c, textureSampleLevel(bufT, ns, uv, 0.).rgb * (0.72 + 0.22 * F.fpb.z)), F.fxb.z); }
    c = clamp(c, vec3f(-0.15), vec3f(1.15));
  }
  if (sw) { c = c.bgr; }
  return vec4f(c, 1.); }
