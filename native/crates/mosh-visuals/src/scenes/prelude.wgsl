// The native scene prelude: the web app's `WG_HEAD + WG_VS` (gpu.js), bound to the descriptor
// layout `mosh-render` actually builds, plus the one helper that could not survive unchanged.
//
// A scene's body is the extracted web body, verbatim, appended after this text: only the prelude
// differs. What differs, and why:
//
//   * Bindings. `mosh-render`'s set layout is uniform at 0, then one image and one immutable
//     sampler per input: (1,2), (3,4), (5,6), (7,8), (9,10). The web app puts the sampler at 1 and
//     the five textures at 2..6. Inputs, in order: palT, caT, melT, rdT, mediaT.
//   * `lut(v)` wraps with `fract(v)`. The web sampler is `repeat` in u; `mosh-render`'s one
//     sampler is `clamp-to-edge`, and `lut()` is called with values far outside 0..1 (tunnel:
//     `u * 0.22 + ...` grows without bound), so the wrap has to be explicit or those scenes
//     flatten to one colour.
//   * `ca(x, y, r)` is generated instead of read from engine.c's 512x512 table: `mosh-render` has
//     no texture upload, so the table cannot reach the GPU. The same three rules (110, 30, 90),
//     a 32-cell periodic row seeded from the row index and evolved 24 steps. Structured like a
//     CA, not the engine's table; only Loom and Punch cards use it, as a mask.
//   * There is no vertex stage here: `mosh-render` supplies its own full-screen triangle and names
//     the stages `vs_main`/`fs_main`, so only the fragment entry point is written (`fs_main`).
//
// Everything else — `U`, `PI`, `TAU`, `h1`, `h2`, `mel`, `melh`, `XY`, `rule`, `fin`, the uniform
// field order (208 bytes, the web `SU`) — is the web prelude, unchanged.

struct SU { res: vec2f, T: f32, hue: f32, bass: f32, mid: f32, high: f32, pulse: f32, vari: f32, dens: f32, beat: f32, frame: f32, pal: f32, aux0: f32, aux1: f32, aux2: f32, l0: vec4f, l1: vec4f, l2: vec4f, mel: array<vec4f,6> };
@group(0) @binding(0) var<uniform> U: SU;
const PI = 3.14159265; const TAU = 6.2831853;
fn h1(n: f32) -> f32 { return fract(sin(n * 127.1) * 43758.5453); }
fn h2(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453); }

@group(0) @binding(1) var palT: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var caT: texture_2d<f32>;
@group(0) @binding(5) var melT: texture_2d<f32>;
@group(0) @binding(7) var rdT: texture_2d<f32>;
@group(0) @binding(9) var mediaT: texture_2d<f32>;
fn lut(v: f32) -> vec3f { return textureSampleLevel(palT, lin, vec2f(fract(v), (U.pal + 0.5) / 8.), 0.).rgb; }
fn mel(i: i32) -> f32 { let k = ((i % 24) + 24) % 24; return U.mel[k / 4][k % 4]; }
fn melh(band: f32, age: f32) -> f32 { return textureSampleLevel(melT, lin, vec2f((clamp(band, 0., 23.) + 0.5) / 24., (clamp(age, 0., 127.) + 0.5) / 128.), 0.).r * 1.3; }
fn XY(p: vec4f) -> vec2f { return (p.xy - U.res * 0.5) / (U.res.y * 0.5); }
fn ca(x: i32, y: i32, r: i32) -> f32 {
  let rule = ((r % 3) + 3) % 3;
  var row: u32 = u32(h1(f32(y & 511) + f32(rule) * 311.7) * 4294967040.0);
  for (var step = 0; step < 24; step = step + 1) {
    let left = (row << 1u) | (row >> 31u);
    let right = (row >> 1u) | (row << 31u);
    var next: u32;
    if (rule == 0) { next = ((~left) & (row | right)) | (right & (~row)); }
    else if (rule == 1) { next = left ^ (row | right); }
    else { next = left ^ right; }
    row = next;
  }
  return f32((row >> u32(x & 31)) & 1u);
}
fn rule() -> i32 { return ((i32(U.vari) % 3) + 3) % 3; }
fn fin(c: vec3f) -> vec4f { return vec4f(clamp(c, vec3f(0.), vec3f(1.)), 1.); }
