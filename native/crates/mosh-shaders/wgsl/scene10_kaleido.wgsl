// extracted by scripts/extract_wgsl.mjs from gpu.js: WG_SCENES[10]
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
  let q = XY(pos); let r = length(q); let n = round(3. + U.dens * 10.); let sg = TAU / n;
  var a = atan2(q.y, q.x) + U.T * 0.1; a = abs((a - sg * floor(a / sg)) - sg * 0.5);
  var p = r * vec2f(cos(a), sin(a)); p += 0.3 * sin(p.yx * 3. + U.T * vec2f(0.7, 0.9)); p *= 2. + sin(U.T * 0.2) + U.bass * 0.5;
  let v = sin(p.x * 4. + U.T) + sin(p.y * 5. - U.T * 1.3) + sin((p.x + p.y) * 3. + U.T * 0.5) + sin(length(p) * 6. - U.T * 2.);
  return fin(lut(v * 0.18 + U.hue + r * 0.2) * (0.55 + 0.45 * step(0., sin(v * 6.)))); }
