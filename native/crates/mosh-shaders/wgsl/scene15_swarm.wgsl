// extracted by scripts/extract_wgsl.mjs from gpu.js: WG_SCENES[15]
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
struct VO { @builtin(position) p: vec4f, @location(0) uv: vec2f, @location(1) col: vec3f };
fn ppos(i: f32, t: f32) -> vec2f {
  let a = h1(i) * TAU; let b = h1(i + 31.) * TAU; let f1 = 0.3 + h1(i + 7.) * 0.9; let f2 = 0.2 + h1(i + 13.) * 0.8;
  var p = vec2f(sin(t * f1 + a) + 0.6 * sin(t * f2 * 2.3 + b), cos(t * f2 + b) + 0.6 * cos(t * f1 * 1.7 + a)) * vec2f(1.1, 0.65);
  let c = vec2f(0.8 * sin(t * 0.37), 0.5 * cos(t * 0.29));
  p = mix(p, c, 0.35 + 0.45 * sin(t * 0.5 + h1(i + 3.) * 2.));
  return c + (p - c) * (1. + 1.2 * U.pulse * U.bass); }
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  var cx = array<vec2f, 6>(vec2f(0., -1.), vec2f(1., -1.), vec2f(0., 1.), vec2f(0., 1.), vec2f(1., -1.), vec2f(1., 1.));
  let i = f32(ii); let asp = U.res.x / U.res.y; let p0 = ppos(i, U.T); let p1 = ppos(i, U.T - 0.09);
  var dir = p0 - p1; let len = length(dir); if (len > 1e-5) { dir = dir / len; } else { dir = vec2f(1., 0.); }
  let w = 0.006 + 0.012 * h1(i + 5.) + 0.01 * U.bass; let c = cx[vi];
  let p = mix(p1 - dir * w, p0 + dir * w, c.x) + vec2f(-dir.y, dir.x) * w * c.y;
  var o: VO; o.p = vec4f(p.x / asp, -p.y, 0., 1.); o.uv = c;
  o.col = (lut(h1(i + 9.) * 0.5 + U.hue) + 0.15) * (0.5 + mel(i32(ii % 24u))); return o; }
@fragment fn fs(v: VO) -> @location(0) vec4f { let a = (1. - v.uv.y * v.uv.y) * (0.25 + 0.75 * v.uv.x);
  return vec4f(v.col * a * a, 1.); }
