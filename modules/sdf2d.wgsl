// sdf2d.wgsl - signed distance fields in 2D: circle, box, segment, polygon, smooth union.
// Include after WG_HEAD. Declares no bindings and no entry points.
// Distances are in the same units as the input point (negative = inside).
//
// Exports:
//   const SD2_BIG: f32                     - 1e9, a safe "infinitely far" distance
//   fn sd2_circle(p: vec2f, r: f32) -> f32
//   fn sd2_ring(p: vec2f, r: f32, w: f32) -> f32     - annulus of half-width w
//   fn sd2_box(p: vec2f, b: vec2f) -> f32            - axis-aligned, b = half extents
//   fn sd2_box_round(p: vec2f, b: vec2f, r: f32) -> f32
//   fn sd2_box_frame(p: vec2f, b: vec2f, e: f32) -> f32 - hollow box, e = border width
//   fn sd2_segment(p: vec2f, a: vec2f, b: vec2f) -> f32 - distance to the a-b segment
//   fn sd2_segment_w(p: vec2f, a: vec2f, b: vec2f, w: f32) -> f32 - capsule
//   fn sd2_line(p: vec2f, a: vec2f, b: vec2f) -> f32  - signed distance to the infinite line
//   fn sd2_tri(p: vec2f, a: vec2f, b: vec2f, c: vec2f) -> f32
//   fn sd2_ngon(p: vec2f, n: f32, r: f32) -> f32     - regular n-gon, circumradius r, a vertex on +x
//   fn sd2_ngon_rot(p: vec2f, n: f32, r: f32, rot: f32) -> f32
//   fn sd2_star(p: vec2f, n: f32, r: f32, rf: f32) -> f32 - n-pointed star, rf = inner/outer radius
//   fn sd2_poly_n(p: vec2f, v: array<vec2f, 8>, n: i32) -> f32 - polygon through v[0..n-1], n in 3..8
//   fn sd2_poly_pts(p: vec2f, v: array<vec2f, 8>) -> f32 - polygon through all 8 of v[0..7]
//   fn sd2_smin(a: f32, b: f32, k: f32) -> f32       - polynomial smooth union
//   fn sd2_smin_exp(a: f32, b: f32, k: f32) -> f32   - exponential smooth union
//   fn sd2_smax(a: f32, b: f32, k: f32) -> f32       - smooth intersection
//   fn sd2_smin_c(a: f32, b: f32, k: f32) -> vec2f   - smooth union result plus its blend factor
//   fn sd2_fill(d: f32, px: f32) -> f32              - 1 px anti-aliased coverage from a distance
//   fn sd2_stroke(d: f32, w: f32, px: f32) -> f32    - 1 px anti-aliased band of half-width w

const SD2_BIG: f32 = 1e9;

fn sd2_circle(p: vec2f, r: f32) -> f32 {
  return length(p) - r;
}

fn sd2_ring(p: vec2f, r: f32, w: f32) -> f32 {
  return abs(length(p) - r) - w;
}

fn sd2_box(p: vec2f, b: vec2f) -> f32 {
  let d = abs(p) - b;
  return length(max(d, vec2f(0.))) + min(max(d.x, d.y), 0.);
}

fn sd2_box_round(p: vec2f, b: vec2f, r: f32) -> f32 {
  return sd2_box(p, b - vec2f(r)) - r;
}

fn sd2_box_frame(p: vec2f, b: vec2f, e: f32) -> f32 {
  let q = abs(p) - b;
  let t = abs(q + vec2f(e)) - vec2f(e);
  return min(max(q.x, q.y), 0.) + length(max(t, vec2f(0.)));
}

fn sd2_segment(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-9), 0., 1.);
  return length(pa - ba * h);
}

fn sd2_segment_w(p: vec2f, a: vec2f, b: vec2f, w: f32) -> f32 {
  return sd2_segment(p, a, b) - w;
}

fn sd2_line(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let n = normalize(b - a);
  return dot(p - a, vec2f(-n.y, n.x));
}

fn sd2_tri(p: vec2f, a: vec2f, b: vec2f, c: vec2f) -> f32 {
  let e0 = b - a;
  let e1 = c - b;
  let e2 = a - c;
  let v0 = p - a;
  let v1 = p - b;
  let v2 = p - c;
  let q0 = v0 - e0 * clamp(dot(v0, e0) / max(dot(e0, e0), 1e-9), 0., 1.);
  let q1 = v1 - e1 * clamp(dot(v1, e1) / max(dot(e1, e1), 1e-9), 0., 1.);
  let q2 = v2 - e2 * clamp(dot(v2, e2) / max(dot(e2, e2), 1e-9), 0., 1.);
  let s = sign(e0.x * e2.y - e0.y * e2.x);
  let d = min(min(vec2f(dot(q0, q0), s * (v0.x * e0.y - v0.y * e0.x)),
                  vec2f(dot(q1, q1), s * (v1.x * e1.y - v1.y * e1.x))),
                  vec2f(dot(q2, q2), s * (v2.x * e2.y - v2.y * e2.x)));
  return -sqrt(d.x) * sign(d.y);
}

fn sd2_ngon(p: vec2f, n: f32, r: f32) -> f32 {
  // exact regular n-gon via one wedge: fold to 0..2PI/n around +x, mirror, then
  // distance to the single edge v0(r,0) -> v1(r at 2PI/n), signed by the origin side.
  let nn = max(n, 3.);
  let sg = TAU / nn;
  let a = atan2(p.y, p.x);
  let b = a - floor(a / sg + 0.5) * sg;
  let q = vec2f(cos(b), abs(sin(b))) * length(p);
  let v0 = vec2f(r, 0.);
  let v1 = r * vec2f(cos(sg), sin(sg));
  let e = v1 - v0;
  let dv = q - v0;
  let h = clamp(dot(dv, e) / max(dot(e, e), 1e-9), 0., 1.);
  let d = length(dv - e * h);
  return select(d, -d, e.x * dv.y - e.y * dv.x > 0.);
}

fn sd2_ngon_rot(p: vec2f, n: f32, r: f32, rot: f32) -> f32 {
  let cs = cos(rot);
  let sn = sin(rot);
  return sd2_ngon(vec2f(cs * p.x - sn * p.y, sn * p.x + cs * p.y), n, r);
}

fn sd2_star(p: vec2f, n: f32, r: f32, rf: f32) -> f32 {
  // exact n-pointed star: same wedge fold, distance to the outer->inner edge.
  let nn = max(n, 3.);
  let sg = PI / nn;
  let a = atan2(p.y, p.x);
  let b = a - floor(a / (2. * sg) + 0.5) * (2. * sg);
  let q = vec2f(cos(b), abs(sin(b))) * length(p);
  let ri = r * clamp(rf, 0.01, 0.99);
  let v0 = vec2f(r, 0.);
  let v1 = ri * vec2f(cos(sg), sin(sg));
  let e = v1 - v0;
  let dv = q - v0;
  let h = clamp(dot(dv, e) / max(dot(e, e), 1e-9), 0., 1.);
  let d = length(dv - e * h);
  return select(d, -d, e.x * dv.y - e.y * dv.x > 0.);
}

fn sd2_poly_n(p: vec2f, v: array<vec2f, 8>, n: i32) -> f32 {
  // general polygon: v[0..n-1] in order, n in 3..8. Exact even for concave shapes.
  let cnt = clamp(n, 3, 8);
  var d = dot(p - v[0], p - v[0]);
  var s = 1.;
  for (var i = 0; i < cnt; i++) {
    let j = (i + cnt - 1) % cnt;
    let e = v[j] - v[i];
    let w = p - v[i];
    let b = w - e * clamp(dot(w, e) / max(dot(e, e), 1e-9), 0., 1.);
    d = min(d, dot(b, b));
    let c0 = p.y >= v[i].y;
    let c1 = p.y < v[j].y;
    let c2 = e.x * w.y > e.y * w.x;
    if ((c0 && c1 && c2) || (!c0 && !c1 && !c2)) { s = -s; }
  }
  return s * sqrt(d);
}

fn sd2_poly_pts(p: vec2f, v: array<vec2f, 8>) -> f32 {
  return sd2_poly_n(p, v, 8);
}

fn sd2_smin(a: f32, b: f32, k: f32) -> f32 {
  let h = clamp(0.5 + 0.5 * (b - a) / max(k, 1e-9), 0., 1.);
  return mix(b, a, h) - k * h * (1. - h);
}

fn sd2_smin_exp(a: f32, b: f32, k: f32) -> f32 {
  let kk = max(k, 1e-6);
  let h = exp(-a / kk) + exp(-b / kk);
  return -log(max(h, 1e-30)) * kk;
}

fn sd2_smax(a: f32, b: f32, k: f32) -> f32 {
  return -sd2_smin(-a, -b, k);
}

fn sd2_smin_c(a: f32, b: f32, k: f32) -> vec2f {
  // (distance, blend factor) so a caller can mix fields along with the distance
  let h = clamp(0.5 + 0.5 * (b - a) / max(k, 1e-9), 0., 1.);
  return vec2f(mix(b, a, h) - k * h * (1. - h), h);
}

fn sd2_fill(d: f32, px: f32) -> f32 {
  return clamp(0.5 - d / max(px, 1e-9), 0., 1.);
}

fn sd2_stroke(d: f32, w: f32, px: f32) -> f32 {
  return clamp(0.5 - (abs(d) - w) / max(px, 1e-9), 0., 1.);
}
