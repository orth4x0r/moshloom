// extracted by scripts/extract_wgsl.mjs from gpu.js: WG_RDSIM
// Do not edit: change gpu.js and re-run the script.

struct SU { res: vec2f, T: f32, hue: f32, bass: f32, mid: f32, high: f32, pulse: f32, vari: f32, dens: f32, beat: f32, frame: f32, pal: f32, aux0: f32, aux1: f32, aux2: f32, l0: vec4f, l1: vec4f, l2: vec4f, mel: array<vec4f,6> };
@group(0) @binding(0) var<uniform> U: SU;
const PI = 3.14159265; const TAU = 6.2831853;
fn h1(n: f32) -> f32 { return fract(sin(n * 127.1) * 43758.5453); }
fn h2(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return vec4f(f32((i << 1u) & 2u) * 2. - 1., f32(i & 2u) * 2. - 1., 0., 1.); }
@group(0) @binding(1) var rdT: texture_2d<f32>;
fn at(p: vec2i) -> vec2f { let sz = vec2i(320, 180); return textureLoad(rdT, (p + sz) % sz, 0).rg; }
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = vec2i(pos.xy); let c = at(p); let uv = pos.xy / vec2f(320., 180.);
  let lap = 0.2 * (at(p + vec2i(1, 0)) + at(p + vec2i(-1, 0)) + at(p + vec2i(0, 1)) + at(p + vec2i(0, -1)))
          + 0.05 * (at(p + vec2i(1, 1)) + at(p + vec2i(-1, 1)) + at(p + vec2i(1, -1)) + at(p + vec2i(-1, -1))) - c;
  let F = 0.022 + 0.034 * U.dens + 0.008 * sin(uv.x * 6. + U.T * 0.1) * sin(uv.y * 5.); let K = 0.060 + 0.004 * cos(uv.y * 4. - U.T * 0.07);
  let abb = c.r * c.g * c.g;
  var a = c.r + lap.r - abb + F * (1. - c.r); var b = c.g + 0.5 * lap.g + abb - (K + F) * c.g;
  if (U.aux2 > 0.5) { a = 1.; b = select(0., 1., h2(floor(pos.xy / 6.)) > 0.93); }
  if (U.aux0 >= 0. && length((uv - vec2f(U.aux0, U.aux1)) * vec2f(16. / 9., 1.)) < 0.035) { b = 1.; }
  return vec4f(clamp(a, 0., 1.), clamp(b, 0., 1.), 0., 1.); }
