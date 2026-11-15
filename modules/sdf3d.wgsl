// ============================================================================
// modules/sdf3d.wgsl - 3D signed distance primitives and domain operators.
//
// Every name here is prefixed `sd3_` (a distance field) or `op3_` (an operator),
// so this chunk can be concatenated after WG_HEAD without colliding with the
// scenes in gpu.js or with the 2D module. No bindings, no uniforms, no
// textures: it is pure WGSL and has no dependency on the host header.
//
// Conventions
//   * primitives return a signed distance in world units: negative inside,
//     positive outside, roughly Euclidean (gyroid excepted: it is only a
//     conservative estimate, see sd3_gyroid).
//   * distance combinators take two distances and return one distance.
//   * domain ops take a point and return a *transformed* point. They do not
//     evaluate anything, so a scene chains them around a primitive:
//         let d = sd3_sphere(op3_twist(p, 1.5), 1.0);
//     Repeat/fold ops return the nearest cell's local point, so the primitive
//     call after them is the whole field.
//   * all ops are safe for p in the far plane (no division by a caller value
//     that can be zero; the few divisions are guarded).
// ============================================================================

const SD3_TAU: f32 = 6.283185307179586;
const SD3_PI: f32 = 3.141592653589793;

// ---------------------------------------------------------------- primitives

// Sphere centred at the origin.
fn sd3_sphere(p: vec3f, r: f32) -> f32 {
  return length(p) - r;
}

// Axis-aligned box with half-extents b.
fn sd3_box(p: vec3f, b: vec3f) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Box with half-extents b and corners rounded by r (r may be negative).
fn sd3_round_box(p: vec3f, b: vec3f, r: f32) -> f32 {
  let q = abs(p) - b + vec3f(r);
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// Torus lying in the XZ plane: ring radius R, tube radius r.
fn sd3_torus(p: vec3f, R: f32, r: f32) -> f32 {
  let q = vec2f(length(p.xz) - R, p.y);
  return length(q) - r;
}

// Capsule: the segment a..b inflated by r. Also the round cone when a == b.
fn sd3_capsule(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-9), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

// Capped cylinder along Y: radius r, half-height h.
fn sd3_cylinder(p: vec3f, r: f32, h: f32) -> f32 {
  let d = vec2f(length(p.xz) - r, abs(p.y) - h);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}

// Infinite plane through the origin with unit normal n, shifted h along n.
// A scene that wants a ground plane at y = -1 calls
//   sd3_plane(p, vec3f(0.0, 1.0, 0.0), 1.0)
// Points on the n side of the plane are outside (positive), so this is a solid
// half-space below the surface. n must be normalised.
fn sd3_plane(p: vec3f, n: vec3f, h: f32) -> f32 {
  return dot(p, n) + h;
}

// Gyroid: an infinite, periodic minimal surface. `scale` is cells per unit and
// `thickness` is the wall half-width in cell units. The field is not an exact
// distance; it is a conservative estimate and a raymarcher should take slightly
// smaller steps near it (multiply the returned distance by ~0.8 in the scene).
fn sd3_gyroid(p: vec3f, scale: f32, thickness: f32) -> f32 {
  let q = p * scale;
  let g = dot(sin(q), cos(q.yzx));
  return (abs(g) - thickness) / max(scale, 1e-4);
}

// Regular octahedron: the L1 ball of radius s.
fn sd3_octahedron(p: vec3f, s: f32) -> f32 {
  let q = abs(p);
  return (q.x + q.y + q.z - s) * 0.5773502691896258;
}

// ------------------------------------------------------ distance combinators

// a or b.
fn op3_union(a: f32, b: f32) -> f32 {
  return min(a, b);
}

// a minus b (b carves a into a).
fn op3_sub(a: f32, b: f32) -> f32 {
  return max(a, -b);
}

// a and b.
fn op3_inter(a: f32, b: f32) -> f32 {
  return max(a, b);
}

// Union with a rounded fillet of width k (0 degenerates to op3_union).
fn op3_smooth_union(a: f32, b: f32, k: f32) -> f32 {
  let h = clamp(0.5 + 0.5 * (b - a) / max(k, 1e-6), 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

// ------------------------------------------------------ domain operators

// Infinite repetition: fold p into the cell centred on the origin. c is the
// cell size and must be non-zero on every axis.
fn op3_repeat_inf(p: vec3f, c: vec3f) -> vec3f {
  return p - c * round(p / c);
}

// Repetition limited to l cells either side of the origin (so 2*l+1 copies per
// axis); outside that box the field is left alone and usually becomes empty.
fn op3_repeat_lim(p: vec3f, c: vec3f, l: vec3f) -> vec3f {
  let id = clamp(round(p / c), -l, l);
  return p - c * id;
}

// Mirror across the plane through the origin with normal n: keeps the half
// space where dot(p, n) > 0 and folds the other half onto it. n need not be
// normalised, but the result is only a mirror if it is.
fn op3_mirror(p: vec3f, n: vec3f) -> vec3f {
  return p - 2.0 * min(dot(p, n), 0.0) * n;
}

// Twist around Y: the XZ plane rotates by k * p.y radians.
fn op3_twist(p: vec3f, k: f32) -> vec3f {
  let a = k * p.y;
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
}

// Bend around Y: the XY plane rotates by k * p.x radians. This is the usual
// raymarching "bend" (a rotation proportional to an axis, not a true arc).
fn op3_bend(p: vec3f, k: f32) -> vec3f {
  let a = k * p.x;
  let c = cos(a);
  let s = sin(a);
  return vec3f(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
}

// Polar repetition around Y: n copies of the field around the vertical axis.
// n <= 1 leaves p essentially unchanged.
fn op3_polar(p: vec3f, n: f32) -> vec3f {
  let s = SD3_TAU / max(n, 1.0);
  var a = atan2(p.z, p.x);
  a = a - s * round(a / s);
  let r = length(p.xz);
  return vec3f(r * cos(a), p.y, r * sin(a));
}

// Kaleidoscopic fold: mirror into the positive octant, then sort (x, y, z)
// n times. Each pass folds the field across a diagonal plane, so n passes give
// an n-fold diagonal symmetry. n is clamped to 4.
fn op3_fold(p: vec3f, n: f32) -> vec3f {
  var q = abs(p);
  for (var i = 0; i < 4; i++) {
    if (f32(i) >= n) { break; }
    if (q.x < q.y) { q = vec3f(q.y, q.x, q.z); }
    if (q.y < q.z) { q = vec3f(q.x, q.z, q.y); }
    if (q.z < q.x) { q = vec3f(q.z, q.y, q.x); }
  }
  return q;
}
