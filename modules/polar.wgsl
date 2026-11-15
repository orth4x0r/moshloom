// polar.wgsl - to/from polar, angular repeat, kaleidoscope fold.
// Include after WG_HEAD. Declares no bindings and no entry points.
// Uses only the WG_HEAD constants PI / TAU.
// Angles are radians, 0 on +x, increasing counter-clockwise (atan2 convention).
//
// Exports:
//   fn pol_to(p: vec2f) -> vec2f                 - (r, theta)
//   fn pol_from(r: f32, th: f32) -> vec2f        - inverse of pol_to
//   fn pol_to01(p: vec2f) -> vec2f               - (r, theta/TAU + 0.5)
//   fn pol_r(p: vec2f) -> f32                    - radius only
//   fn pol_th(p: vec2f) -> f32                   - angle only, -PI..PI
//   fn pol_th01(p: vec2f) -> f32                 - angle mapped to 0..1
//   fn pol_rot(p: vec2f, a: f32) -> vec2f        - rotate by a radians
//   fn pol_repeat_th(th: f32, n: f32) -> f32     - wrap an angle into one of n sectors
//   fn pol_repeat(p: vec2f, n: f32) -> vec2f     - same, on a point (keeps r)
//   fn pol_mirror(th: f32) -> f32                - fold any angle into 0..PI
//   fn pol_kaleido(p: vec2f, n: f32) -> vec2f    - n-fold kaleidoscope fold, wedge centred on +x
//   fn pol_kaleido_rot(p: vec2f, n: f32, rot: f32) -> vec2f
//   fn pol_swirl(p: vec2f, amt: f32, falloff: f32) -> vec2f
//   fn pol_log(p: vec2f, k: f32) -> vec2f        - log-polar (log r, theta*k)
//   fn pol_exp(q: vec2f, k: f32) -> vec2f        - inverse of pol_log
//   fn pol_tile(p: vec2f, n: f32, rings: f32) -> vec2f - radial/angular tiling cell coordinates

fn pol_to(p: vec2f) -> vec2f {
  return vec2f(length(p) + 1e-9, atan2(p.y, p.x));
}

fn pol_from(r: f32, th: f32) -> vec2f {
  return r * vec2f(cos(th), sin(th));
}

fn pol_to01(p: vec2f) -> vec2f {
  return vec2f(length(p) + 1e-9, atan2(p.y, p.x) / TAU + 0.5);
}

fn pol_r(p: vec2f) -> f32 {
  return length(p);
}

fn pol_th(p: vec2f) -> f32 {
  return atan2(p.y, p.x);
}

fn pol_th01(p: vec2f) -> f32 {
  return atan2(p.y, p.x) / TAU + 0.5;
}

fn pol_rot(p: vec2f, a: f32) -> vec2f {
  let cs = cos(a);
  let sn = sin(a);
  return vec2f(p.x * cs - p.y * sn, p.x * sn + p.y * cs);
}

fn pol_repeat_th(th: f32, n: f32) -> f32 {
  let sg = TAU / max(n, 1.);
  let a = th - sg * floor(th / sg);
  return a - sg * 0.5;
}

fn pol_repeat(p: vec2f, n: f32) -> vec2f {
  return pol_from(length(p), pol_repeat_th(atan2(p.y, p.x), n));
}

fn pol_mirror(th: f32) -> f32 {
  let a = abs(th);
  return abs(a - PI * floor(a / PI) - PI * 0.5) * 2.; // folds every PI into 0..PI
}

fn pol_kaleido(p: vec2f, n: f32) -> vec2f {
  let sg = PI / max(n, 1.);
  var a = atan2(p.y, p.x);
  a = a - sg * floor(a / sg);        // one wedge
  a = abs(a - sg * 0.5);             // mirror inside the wedge
  return length(p) * vec2f(cos(a), sin(a));
}

fn pol_kaleido_rot(p: vec2f, n: f32, rot: f32) -> vec2f {
  return pol_kaleido(pol_rot(p, -rot), n);
}

fn pol_swirl(p: vec2f, amt: f32, falloff: f32) -> vec2f {
  let r = length(p);
  let a = amt / (pow(r, max(falloff, 0.001)) + 1e-3);
  return pol_rot(p, a);
}

fn pol_log(p: vec2f, k: f32) -> vec2f {
  let q = pol_to(p);
  return vec2f(log(max(q.x, 1e-6)), q.y * k);
}

fn pol_exp(q: vec2f, k: f32) -> vec2f {
  return pol_from(exp(q.x), q.y / max(k, 1e-6));
}

fn pol_tile(p: vec2f, n: f32, rings: f32) -> vec2f {
  let q = pol_to(p);
  let rr = max(rings, 1.);
  return vec2f(q.y / TAU * max(n, 1.), log(max(q.x, 1e-6)) / max(log(1. + 1. / rr), 1e-6));
}
