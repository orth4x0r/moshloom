// Scene 15's particle table, second generation. One texel per particle per row, three rows:
//
//   row 0: the streak's screen-space bounding box (min xy, max xy), already padded by its half
//          width and already in the scene's own uv units. It is the cheap per-pixel reject: four
//          compares against one texel, and only a streak whose box holds the pixel is looked at
//          again.
//   row 1: the streak's head (xy) and tail (zw) in the same units — the one texel the first
//          generation stored.
//   row 2: `w` (the half width, clamped to the 3-pixel minimum) and the colour (rgb), which the
//          fragment shader used to recompute for every pixel of every streak: a `sin` for the
//          width, a palette sample for the colour and a mel lookup for its level.
//
// The fragment shader still owns the per-pixel part (whether the pixel is inside the streak, and
// how much of it is covered). Moving the per-particle part here costs 176 x 3 texels per frame — a
// pass under the timestamp resolution — and takes 176 sines, 176 palette samples and 176 bounding
// boxes out of every pixel of the layer. See `swarm_fast.wgsl`.

struct SU { res: vec2f, T: f32, hue: f32, bass: f32, mid: f32, high: f32, pulse: f32, vari: f32, dens: f32, beat: f32, frame: f32, pal: f32, aux0: f32, aux1: f32, aux2: f32, l0: vec4f, l1: vec4f, l2: vec4f, mel: array<vec4f,6> };
@group(0) @binding(0) var<uniform> U: SU;
@group(0) @binding(1) var palT: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;

const PI = 3.14159265;
const TAU = 6.2831853;

fn h1(n: f32) -> f32 { return fract(sin(n * 127.1) * 43758.5453); }

fn lut(v: f32) -> vec3f { return textureSampleLevel(palT, lin, vec2f(fract(v), (U.pal + 0.5) / 8.), 0.).rgb; }
fn mel(i: i32) -> f32 { let k = ((i % 24) + 24) % 24; return U.mel[k / 4][k % 4]; }

// The web scene's `ppos()`: a particle's position at time `t`.
fn ppos(i: f32, t: f32) -> vec2f {
  let a = h1(i) * TAU; let b = h1(i + 31.) * TAU; let f1 = 0.3 + h1(i + 7.) * 0.9; let f2 = 0.2 + h1(i + 13.) * 0.8;
  var p = vec2f(sin(t * f1 + a) + 0.6 * sin(t * f2 * 2.3 + b), cos(t * f2 + b) + 0.6 * cos(t * f1 * 1.7 + a)) * vec2f(1.1, 0.65);
  let c = vec2f(0.8 * sin(t * 0.37), 0.5 * cos(t * 0.29));
  p = mix(p, c, 0.35 + 0.45 * sin(t * 0.5 + h1(i + 3.) * 2.));
  return c + (p - c) * (1. + 1.2 * U.pulse * U.bass);
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let i = floor(pos.x);
  let asp = U.res.x / U.res.y;
  let min_width = 3. / (U.res.y * 0.5);
  let w = max(0.006 + 0.012 * h1(i + 5.) + 0.01 * U.bass, min_width);
  // The web vertex stage negates y, so a screen-space point is (p.x / asp, -p.y).
  let a = vec2f(ppos(i, U.T).x / asp, -ppos(i, U.T).y);
  let b = vec2f(ppos(i, U.T - 0.09).x / asp, -ppos(i, U.T - 0.09).y);
  let row = floor(pos.y);
  if (row < 0.5) {
    return vec4f(min(a.x, b.x) - w, min(a.y, b.y) - w, max(a.x, b.x) + w, max(a.y, b.y) + w);
  }
  if (row < 1.5) {
    return vec4f(a, b);
  }
  let col = (lut(h1(i + 9.) * 0.5 + U.hue) + 0.15) * (0.5 + mel(i32(i) % 24));
  // `w |a - b|`: the capsule test in units of the streak's own length, so the fragment shader
  // needs neither a square root nor a multiply to compare against it.
  return vec4f(w * length(a - b), col);
}
