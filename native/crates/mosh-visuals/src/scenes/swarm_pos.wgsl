// The swarm's particle table — first generation, kept for the perf test in `scenes.rs`.
//
// `scenes/swarm_pos_fast.wgsl` replaces it (three rows per particle: the streak's box, its ends,
// and its finished width and colour). Nothing draws this file any more.
//
// The original note:
// One texel per particle, `(head.xy, tail.xy)` in the scene's own
// units. The web app gets this for free from instancing (`p.draw(6, 150 + 1300*dens)`); with one
// 3-vertex draw per pass the swarm fragment shader would otherwise evaluate `ppos()` twice for
// every particle in every pixel, which measured 12.7 ms at 960x540 on the integrated GPU. One
// 160-texel pass per frame replaces that with one texture load per particle.
struct SU { res: vec2f, T: f32, hue: f32, bass: f32, mid: f32, high: f32, pulse: f32, vari: f32, dens: f32, beat: f32, frame: f32, pal: f32, aux0: f32, aux1: f32, aux2: f32, l0: vec4f, l1: vec4f, l2: vec4f, mel: array<vec4f,6> };
@group(0) @binding(0) var<uniform> U: SU;
const PI = 3.14159265; const TAU = 6.2831853;
fn h1(n: f32) -> f32 { return fract(sin(n * 127.1) * 43758.5453); }

fn ppos(i: f32, t: f32) -> vec2f {
  let a = h1(i) * TAU; let b = h1(i + 31.) * TAU; let f1 = 0.3 + h1(i + 7.) * 0.9; let f2 = 0.2 + h1(i + 13.) * 0.8;
  var p = vec2f(sin(t * f1 + a) + 0.6 * sin(t * f2 * 2.3 + b), cos(t * f2 + b) + 0.6 * cos(t * f1 * 1.7 + a)) * vec2f(1.1, 0.65);
  let c = vec2f(0.8 * sin(t * 0.37), 0.5 * cos(t * 0.29));
  p = mix(p, c, 0.35 + 0.45 * sin(t * 0.5 + h1(i + 3.) * 2.));
  return c + (p - c) * (1. + 1.2 * U.pulse * U.bass);
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let i = floor(pos.x);
  return vec4f(ppos(i, U.T), ppos(i, U.T - 0.09));
}
