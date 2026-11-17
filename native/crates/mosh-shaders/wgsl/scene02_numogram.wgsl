// extracted by scripts/extract_wgsl.mjs from gpu.js: WG_SCENES[2]
// Do not edit: change gpu.js and re-run the script.

struct SU { res: vec2f, T: f32, hue: f32, bass: f32, mid: f32, high: f32, pulse: f32, vari: f32, dens: f32, beat: f32, frame: f32, pal: f32, aux0: f32, aux1: f32, aux2: f32, l0: vec4f, l1: vec4f, l2: vec4f, mel: array<vec4f,6> };
@group(0) @binding(0) var<uniform> U: SU;
const PI = 3.14159265; const TAU = 6.2831853;
fn h1(n: f32) -> f32 { return fract(sin(n * 127.1) * 43758.5453); }
fn h2(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }

@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var palT: texture_2d<f32>;
@group(0) @binding(3) var caT: texture_2d<f32>;
@group(0) @binding(4) var melT: texture_2d<f32>;
@group(0) @binding(5) var rdT: texture_2d<f32>;
@group(0) @binding(6) var mediaT: texture_2d<f32>;
fn lut(v: f32) -> vec3f { return textureSampleLevel(palT, lin, vec2f(v, (U.pal + 0.5) / 8.), 0.).rgb; }
fn mel(i: i32) -> f32 { let k = ((i % 24) + 24) % 24; return U.mel[k / 4][k % 4]; }
fn melh(band: f32, age: f32) -> f32 { return textureSampleLevel(melT, lin, vec2f((clamp(band, 0., 23.) + 0.5) / 24., (clamp(age, 0., 127.) + 0.5) / 128.), 0.).r * 1.3; }
fn XY(p: vec4f) -> vec2f { return (p.xy - U.res * 0.5) / (U.res.y * 0.5); }
fn ca(x: i32, y: i32, r: i32) -> f32 { return textureLoad(caT, vec2i(x & 511, y & 511), 0)[r]; }
fn rule() -> i32 { return ((i32(U.vari) % 3) + 3) % 3; }
fn fin(c: vec3f) -> vec4f { return vec4f(clamp(c, vec3f(0.), vec3f(1.)), 1.); }
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return vec4f(f32((i << 1u) & 2u) * 2. - 1., f32(i & 2u) * 2. - 1., 0., 1.); }
const ZP = array<vec2f, 10>(vec2f(-.25,.75), vec2f(-.55,0.), vec2f(-.3,.35), vec2f(-.25,-.75), vec2f(-.3,-.35), vec2f(.3,-.35), vec2f(.25,-.75), vec2f(.3,.35), vec2f(.55,0.), vec2f(.25,.75));
const LP = array<i32, 6>(1, 2, 7, 8, 5, 4);
const SZ = array<vec2i, 5>(vec2i(0,9), vec2i(1,8), vec2i(2,7), vec2i(3,6), vec2i(4,5));
const GT = array<vec2i, 6>(vec2i(2,3), vec2i(4,1), vec2i(5,6), vec2i(7,1), vec2i(8,9), vec2i(6,3));
fn seg(p: vec2f, a: vec2f, b: vec2f) -> f32 { let pa = p - a; let ba = b - a; let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-9), 0., 1.); return length(pa - ba * h); }
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos); let r = length(q) + 1e-4; let px = 2. / U.res.y; let ss = U.res.x / 640.;
  let tw = f32(i32(U.vari) % 4);
  let rg = step(0.96, sin(log(r) * 14. - U.T * 1.5 + atan2(q.y, q.x) * tw));
  var c = rg * (0.10 * U.pulse * U.l1.rgb + 0.18 * U.l0.rgb);
  var zp = ZP; var lp = LP; var sz = SZ; var gt = GT; let ph = fract(U.T * 0.13);
  for (var k = 0; k < 5; k++) {
    let p = (ph + f32(k)) / 5.; let s = 0.12 * exp2(p * 5.5) * (0.5 + U.dens); let b = pow(sin(PI * p), 1.5);
    let ang = U.T * 0.11 * select(-1., 1., (k & 1) == 1) + f32(k); let cs = cos(ang); let sn = sin(ang);
    var pt: array<vec2f, 10>;
    for (var z = 0; z < 10; z++) { pt[z] = vec2f(zp[z].x * cs - zp[z].y * sn, zp[z].x * sn + zp[z].y * cs) * s; }
    let th = select(2.2, 1.2, s < 1.2) * ss * px;
    var d0 = 1e9; var d1 = 1e9; var d2 = 1e9;
    for (var i = 0; i < 6; i++) { d0 = min(d0, seg(q, pt[lp[i]], pt[lp[(i + 1) % 6]])); d2 = min(d2, seg(q, pt[gt[i].x], pt[gt[i].y])); }
    for (var i = 0; i < 5; i++) { d1 = min(d1, seg(q, pt[sz[i].x], pt[sz[i].y])); }
    c = max(c, U.l0.rgb * b * clamp((th * 0.5 - d0) / px + 0.5, 0., 1.));
    c = max(c, U.l2.rgb * b * 0.8 * clamp((ss * px * 0.5 - d2) / px + 0.5, 0., 1.));
    c = max(c, U.l1.rgb * b * clamp((th * 0.5 - d1) / px + 0.5, 0., 1.));
    for (var z = 0; z < 10; z++) {
      let e = mel((z * 2 + 3) % 24); let rad = s * (0.05 + 0.07 * e); let dd = length(q - pt[z]);
      let ain = clamp((rad - dd) / px + 0.5, 0., 1.); c = c * (1. - ain);
      if (e > 0.75) { c = mix(c, vec3f(b), ain); } else { c = max(c, U.l0.rgb * b * clamp((th * 0.5 - abs(dd - rad)) / px + 0.5, 0., 1.)); }
    }
  }
  return fin(c); }
