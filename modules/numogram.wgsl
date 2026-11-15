// numogram.wgsl - the decimal numogram tables, copied verbatim from the "2 numogram" scene
// in gpu.js (ZP, LP, SZ, GT: same values, same order, same types).
// Include after WG_HEAD. Declares no bindings and no entry points.
//
// Exports:
//   const ZP: array<vec2f, 10>   - zone positions on the decimal labyrinth, in -0.75..0.75
//   const LP: array<i32, 6>      - the six currents, as zone indices (1,2,7,8,5,4)
//   const SZ: array<vec2i, 5>    - the five syzygies, as zone pairs
//   const GT: array<vec2i, 6>    - the six gates, as zone pairs
//   const NUM_ZONES: i32 = 10
//   const NUM_CURRENTS: i32 = 6
//   const NUM_SYZYGY: i32 = 5
//   const NUM_GATES: i32 = 6
//   fn num_zone(i: i32) -> vec2f        - ZP[i], wrapped into 0..9
//   fn num_current(i: i32) -> i32       - current i (0..5) as a zone index
//   fn num_syzygy(i: i32) -> vec2f      - the two zone positions of syzygy i, as (zone a, zone b)
//   fn num_gate(i: i32) -> vec2f        - the two zone positions of gate i
//   fn num_seg(p: vec2f, a: vec2f, b: vec2f) -> f32 - distance to a segment (the scene's own `seg`)
//   fn num_rot(p: vec2f, a: f32) -> vec2f           - rotate a point by a radians
//   fn num_labyrinth(p: vec2f, n: i32) -> vec2f     - the zone ring scaled by n and rotated, n<0 mirrors

const ZP = array<vec2f, 10>(vec2f(-.25,.75), vec2f(-.55,0.), vec2f(-.3,.35), vec2f(-.25,-.75), vec2f(-.3,-.35), vec2f(.3,-.35), vec2f(.25,-.75), vec2f(.3,.35), vec2f(.55,0.), vec2f(.25,.75));
const LP = array<i32, 6>(1, 2, 7, 8, 5, 4);
const SZ = array<vec2i, 5>(vec2i(0,9), vec2i(1,8), vec2i(2,7), vec2i(3,6), vec2i(4,5));
const GT = array<vec2i, 6>(vec2i(2,3), vec2i(4,1), vec2i(5,6), vec2i(7,1), vec2i(8,9), vec2i(6,3));

const NUM_ZONES: i32 = 10;
const NUM_CURRENTS: i32 = 6;
const NUM_SYZYGY: i32 = 5;
const NUM_GATES: i32 = 6;

fn num_zone(i: i32) -> vec2f {
  let k = ((i % NUM_ZONES) + NUM_ZONES) % NUM_ZONES;
  var zp = ZP;
  return zp[k];
}

fn num_current(i: i32) -> i32 {
  var lp = LP;
  return lp[((i % NUM_CURRENTS) + NUM_CURRENTS) % NUM_CURRENTS];
}

fn num_syzygy(i: i32) -> vec2f {
  var sz = SZ;
  let p = sz[((i % NUM_SYZYGY) + NUM_SYZYGY) % NUM_SYZYGY];
  return vec2f(f32(p.x), f32(p.y));
}

fn num_gate(i: i32) -> vec2f {
  var gt = GT;
  let p = gt[((i % NUM_GATES) + NUM_GATES) % NUM_GATES];
  return vec2f(f32(p.x), f32(p.y));
}

fn num_seg(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-9), 0., 1.);
  return length(pa - ba * h);
}

fn num_rot(p: vec2f, a: f32) -> vec2f {
  let cs = cos(a);
  let sn = sin(a);
  return vec2f(p.x * cs - p.y * sn, p.x * sn + p.y * cs);
}

fn num_labyrinth(p: vec2f, n: i32) -> vec2f {
  let s = pow(2., f32(abs(n))) * 0.5;
  let a = f32(n) * 0.11;
  var q = num_rot(p, -a) / s;
  return q;
}
