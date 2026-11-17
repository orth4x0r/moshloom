// extracted by scripts/extract_wgsl.mjs from gpu.js: WG_SCENES[12]
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
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos); let asp = U.res.x / U.res.y; let h = -0.08 + 0.04 * sin(U.T * 0.3); let dy = q.y - h; let n = (2. + U.dens * 8.) * 0.25;
  let z = 0.6 / (abs(dy) + 1e-3); let wx = q.x * z; let wz = z + U.T * 1.6;
  let gx = abs(fract(wx * n) - 0.5); let gz = abs(fract(wz * n) - 0.5);
  let ax = 1. - smoothstep(0., min(0.5, 0.004 * z * n * 4. + 0.012), gx); let az = 1. - smoothstep(0., min(0.5, 0.004 * z * z * n * 4. + 0.012), gz);
  var c = select(U.l1.rgb, U.l0.rgb, dy > 0.) * max(ax, az) * exp(-z * 0.12) * (0.7 + 0.6 * U.pulse * U.bass);
  c += U.l2.rgb * exp(-abs(dy) * 14.) * (0.4 + U.pulse * 0.8);
  let band = (q.x / asp * 0.5 + 0.5) * 24.; let sk = mel(i32(band)) * 0.24;
  if (dy < 0. && -dy < sk && fract(band) > 0.12) { c += lut(U.hue + floor(band) * 0.03) * (0.35 + 0.65 * step(0.5, fract(dy * 40.))); }
  return fin(c); }
