// noise.wgsl - hashes, value noise, simplex noise, fbm.
// Include after WG_HEAD. Declares no bindings and no entry points.
// Uses only the WG_HEAD constants PI / TAU.
//
// Exports:
//   const NZ_OCT_MAX: i32                  - octave clamp used by the fbm helpers
//   fn nz_hash11(n: f32) -> f32            - 1D hash in [0,1)
//   fn nz_hash12(p: vec2f) -> f32          - 2D -> 1 value hash in [0,1)
//   fn nz_hash22(p: vec2f) -> vec2f        - 2D -> 2 value hash in [0,1)
//   fn nz_hash13(p: vec3f) -> f32          - 3D -> 1 value hash in [0,1)
//   fn nz_hash33(p: vec3f) -> vec3f        - 3D -> 3 value hash in [0,1)
//   fn nz_fade(t: f32) -> f32              - quintic interpolant 6t^5-15t^4+10t^3
//   fn nz_vnoise2(p: vec2f) -> f32         - value noise, ~[0,1]
//   fn nz_vnoise3(p: vec3f) -> f32         - value noise, ~[0,1]
//   fn nz_snoise2(p: vec2f) -> f32         - 2D simplex noise, ~[-1,1]
//   fn nz_snoise3(p: vec3f) -> f32         - 3D simplex noise, ~[-1,1]
//   fn nz_fbm2(p: vec2f, oct: i32) -> f32  - fractal sum of value noise, ~[0,1]
//   fn nz_fbm3(p: vec3f, oct: i32) -> f32  - fractal sum of value noise, ~[0,1]
//   fn nz_sfbm2(p: vec2f, oct: i32) -> f32 - fractal sum of simplex noise, ~[-1,1]
//   fn nz_ridged2(p: vec2f, oct: i32) -> f32 - ridged fractal (creases), ~[0,1]
//   fn nz_warp2(p: vec2f, amt: f32, oct: i32) -> vec2f - domain warp of p
//
// Internal helpers (also module-level fns, all nz_ prefixed, not meant as API):
//   nz_mod289, nz_mod289v3, nz_mod289v4, nz_perm3, nz_perm4, nz_taylor4

const NZ_OCT_MAX: i32 = 12;

fn nz_hash11(n: f32) -> f32 {
  return fract(sin(n * 127.1) * 43758.5453123);
}

fn nz_hash12(p: vec2f) -> f32 {
  return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453123);
}

fn nz_hash22(p: vec2f) -> vec2f {
  let k = vec2f(dot(p, vec2f(127.1, 311.7)), dot(p, vec2f(269.5, 183.3)));
  return fract(sin(k) * 43758.5453123);
}

fn nz_hash13(p: vec3f) -> f32 {
  return fract(sin(dot(p, vec3f(127.1, 311.7, 74.7))) * 43758.5453123);
}

fn nz_hash33(p: vec3f) -> vec3f {
  let k = vec3f(dot(p, vec3f(127.1, 311.7, 74.7)), dot(p, vec3f(269.5, 183.3, 246.1)), dot(p, vec3f(113.5, 271.9, 124.6)));
  return fract(sin(k) * 43758.5453123);
}

fn nz_fade(t: f32) -> f32 {
  return t * t * t * (t * (t * 6. - 15.) + 10.);
}

fn nz_vnoise2(p: vec2f) -> f32 {
  let i = floor(p);
  let f = p - i;
  let u = vec2f(nz_fade(f.x), nz_fade(f.y));
  let a = nz_hash12(i);
  let b = nz_hash12(i + vec2f(1., 0.));
  let c = nz_hash12(i + vec2f(0., 1.));
  let d = nz_hash12(i + vec2f(1., 1.));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn nz_vnoise3(p: vec3f) -> f32 {
  let i = floor(p);
  let f = p - i;
  let u = vec3f(nz_fade(f.x), nz_fade(f.y), nz_fade(f.z));
  let x00 = mix(nz_hash13(i + vec3f(0., 0., 0.)), nz_hash13(i + vec3f(1., 0., 0.)), u.x);
  let x10 = mix(nz_hash13(i + vec3f(0., 1., 0.)), nz_hash13(i + vec3f(1., 1., 0.)), u.x);
  let x01 = mix(nz_hash13(i + vec3f(0., 0., 1.)), nz_hash13(i + vec3f(1., 0., 1.)), u.x);
  let x11 = mix(nz_hash13(i + vec3f(0., 1., 1.)), nz_hash13(i + vec3f(1., 1., 1.)), u.x);
  return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

// GLSL-style mod (always positive), needed by the simplex permutations.
fn nz_mod289(x: f32) -> f32 { return x - 289. * floor(x / 289.); }
fn nz_mod289v3(x: vec3f) -> vec3f { return x - 289. * floor(x / 289.); }
fn nz_perm3(x: vec3f) -> vec3f { return nz_mod289v3(((x * 34.) + 1.) * x); }
fn nz_mod289v4(x: vec4f) -> vec4f { return x - 289. * floor(x / 289.); }
fn nz_perm4(x: vec4f) -> vec4f { return nz_mod289v4(((x * 34.) + 1.) * x); }
fn nz_taylor4(r: vec4f) -> vec4f { return 1.79284291400159 - 0.85373472095314 * r; }

fn nz_snoise2(v: vec2f) -> f32 {
  // Ashima / Stefan Gustavson simplex noise, 2D.
  let C = vec4f(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
  var i = floor(v + vec2f(dot(v, C.yy)));
  let x0 = v - i + vec2f(dot(i, C.xx));
  let i1 = select(vec2f(0., 1.), vec2f(1., 0.), x0.x > x0.y);
  let x12 = vec4f(x0.x + C.x - i1.x, x0.y + C.x - i1.y, x0.x + C.z, x0.y + C.z);
  i = vec2f(nz_mod289(i.x), nz_mod289(i.y));
  var p = nz_perm3(vec3f(i.y, i.y, i.y) + vec3f(0., i1.y, 1.));
  p = nz_perm3(p + vec3f(i.x, i.x, i.x) + vec3f(0., i1.x, 1.));
  let d0 = dot(x0, x0);
  let d1 = dot(x12.xy, x12.xy);
  let d2 = dot(x12.zw, x12.zw);
  var m = max(vec3f(0.5) - vec3f(d0, d1, d2), vec3f(0.));
  m = m * m;
  m = m * m;
  let x = 2. * fract(p * C.www) - 1.;
  let h = abs(x) - 0.5;
  let ox = floor(x + 0.5);
  let a0 = x - ox;
  m = m * (1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h));
  let g = vec3f(a0.x * x0.x + h.x * x0.y, a0.y * x12.x + h.y * x12.y, a0.z * x12.z + h.z * x12.w);
  return 130. * dot(m, g);
}

fn nz_snoise3(v: vec3f) -> f32 {
  // Ashima / Stefan Gustavson simplex noise, 3D.
  let C = vec2f(1. / 6., 1. / 3.);
  let D = vec4f(0., 0.5, 1., 2.);
  var i = floor(v + vec3f(dot(v, C.yyy)));
  let x0 = v - i + vec3f(dot(i, C.xxx));
  let g = step(x0.yzx, x0.xyz);
  let l = 1. - g;
  let i1 = min(g, l.zxy);
  let i2 = max(g, l.zxy);
  let x1 = x0 - i1 + vec3f(C.x);
  let x2 = x0 - i2 + vec3f(C.y);
  let x3 = x0 - vec3f(D.y);
  i = vec3f(nz_mod289(i.x), nz_mod289(i.y), nz_mod289(i.z));
  var p = nz_perm4(vec4f(i.z) + vec4f(0., i1.z, i2.z, 1.));
  p = nz_perm4(p + vec4f(i.y) + vec4f(0., i1.y, i2.y, 1.));
  p = nz_perm4(p + vec4f(i.x) + vec4f(0., i1.x, i2.x, 1.));
  let n_ = 0.142857142857;
  let ns = n_ * D.wyz - D.xzx;
  let j = p - 49. * floor(p * ns.z * ns.z);
  let x_ = floor(j * ns.z);
  let y_ = floor(j - 7. * x_);
  let x = x_ * ns.x + vec4f(ns.y);
  let y = y_ * ns.x + vec4f(ns.y);
  let h = 1. - abs(x) - abs(y);
  let b0 = vec4f(x.x, x.y, y.x, y.y);
  let b1 = vec4f(x.z, x.w, y.z, y.w);
  let s0 = floor(b0) * 2. + 1.;
  let s1 = floor(b1) * 2. + 1.;
  let sh = -step(h, vec4f(0.));
  let a0 = vec4f(b0.x, b0.z, b0.y, b0.w) + vec4f(s0.x, s0.z, s0.y, s0.w) * vec4f(sh.x, sh.x, sh.y, sh.y);
  let a1 = vec4f(b1.x, b1.z, b1.y, b1.w) + vec4f(s1.x, s1.z, s1.y, s1.w) * vec4f(sh.z, sh.z, sh.w, sh.w);
  var p0 = vec3f(a0.x, a0.y, h.x);
  var p1 = vec3f(a0.z, a0.w, h.y);
  var p2 = vec3f(a1.x, a1.y, h.z);
  var p3 = vec3f(a1.z, a1.w, h.w);
  let norm = nz_taylor4(vec4f(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 = p0 * norm.x;
  p1 = p1 * norm.y;
  p2 = p2 * norm.z;
  p3 = p3 * norm.w;
  var m = max(vec4f(0.6) - vec4f(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4f(0.));
  m = m * m;
  let gg = vec4f(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3));
  return 42. * dot(m * m, gg);
}

fn nz_fbm2(p: vec2f, oct: i32) -> f32 {
  var q = p;
  var amp = 0.5;
  var sum = 0.;
  let n = clamp(oct, 1, NZ_OCT_MAX);
  for (var k = 0; k < n; k++) {
    sum += amp * nz_vnoise2(q);
    q = q * 2.02 + vec2f(11.3, 7.7);
    amp *= 0.5;
  }
  return sum;
}

fn nz_fbm3(p: vec3f, oct: i32) -> f32 {
  var q = p;
  var amp = 0.5;
  var sum = 0.;
  let n = clamp(oct, 1, NZ_OCT_MAX);
  for (var k = 0; k < n; k++) {
    sum += amp * nz_vnoise3(q);
    q = q * 2.03 + vec3f(5.1, 11.3, 3.7);
    amp *= 0.5;
  }
  return sum;
}

fn nz_sfbm2(p: vec2f, oct: i32) -> f32 {
  var q = p;
  var amp = 0.5;
  var sum = 0.;
  let n = clamp(oct, 1, NZ_OCT_MAX);
  for (var k = 0; k < n; k++) {
    sum += amp * nz_snoise2(q);
    q = q * 2.01 + vec2f(3.7, 9.1);
    amp *= 0.5;
  }
  return sum;
}

fn nz_ridged2(p: vec2f, oct: i32) -> f32 {
  var q = p;
  var amp = 0.5;
  var sum = 0.;
  var w = 1.;
  let n = clamp(oct, 1, NZ_OCT_MAX);
  for (var k = 0; k < n; k++) {
    let v = 1. - abs(nz_snoise2(q) * 2. - 1.);
    sum += amp * v * v * w;
    w = clamp(v * 1.7, 0., 1.);
    q = q * 2.07 + vec2f(1.3, 4.9);
    amp *= 0.5;
  }
  return sum;
}

fn nz_warp2(p: vec2f, amt: f32, oct: i32) -> vec2f {
  let w = vec2f(nz_fbm2(p + vec2f(1.7, 9.2), oct), nz_fbm2(p + vec2f(8.3, 2.8), oct));
  return p + (w * 2. - 1.) * amt;
}
