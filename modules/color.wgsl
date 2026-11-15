// color.wgsl - hsv <-> rgb, cosine palettes, ordered (bayer) dither.
// Include after WG_HEAD. Declares no bindings and no entry points.
// Uses only the WG_HEAD constant TAU. Nothing here reads U or a texture.
//
// Exports:
//   fn col_hsv2rgb(c: vec3f) -> vec3f        - c = (hue 0..1 turns, sat, val)
//   fn col_rgb2hsv(c: vec3f) -> vec3f        - inverse of col_hsv2rgb
//   fn col_hue(c: vec3f) -> f32             - hue of a colour, 0..1
//   fn col_luma(c: vec3f) -> f32            - Rec.709 luminance
//   fn col_sat(c: vec3f, s: f32) -> vec3f   - saturate/desaturate around luma
//   fn col_cos(t: f32, a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f - Inigo Quilez cosine palette
//   fn col_pal(t: f32, phase: f32) -> vec3f - built-in 6-palette set, phase selects/rotates
//   fn col_warm(t: f32) -> vec3f
//   fn col_cool(t: f32) -> vec3f
//   fn col_fire(t: f32) -> vec3f
//   fn col_gray(t: f32) -> vec3f
//   fn col_bayer4(x: vec2i) -> f32          - ordered dither threshold in [0,1), 4x4
//   fn col_bayer8(x: vec2i) -> f32          - ordered dither threshold in [0,1), 8x8
//   fn col_dither(c: vec3f, t: f32, amt: f32) -> vec3f - offset a colour by a threshold
//   fn col_quant(c: vec3f, levels: f32) -> vec3f       - posterise to `levels` steps
//   fn col_quant_d(c: vec3f, levels: f32, t: f32, amt: f32) -> vec3f - posterise with dither

const COL_BAYER4 = array<f32, 16>(
  0., 8., 2., 10.,
  12., 4., 14., 6.,
  3., 11., 1., 9.,
  15., 7., 13., 5.);

const COL_BAYER8 = array<f32, 64>(
  0., 32., 8., 40., 2., 34., 10., 42.,
  48., 16., 56., 24., 50., 18., 58., 26.,
  12., 44., 4., 36., 14., 46., 6., 38.,
  60., 28., 52., 20., 62., 30., 54., 22.,
  3., 35., 11., 43., 1., 33., 9., 41.,
  51., 19., 59., 27., 49., 17., 57., 25.,
  15., 47., 7., 39., 13., 45., 5., 37.,
  63., 31., 55., 23., 61., 29., 53., 21.);

fn col_hsv2rgb(c: vec3f) -> vec3f {
  let K = vec4f(1., 2. / 3., 1. / 3., 3.);
  let p = abs(fract(vec3f(c.x) + K.xyz) * 6. - vec3f(K.w));
  return c.z * mix(vec3f(K.x), clamp(p - vec3f(K.x), vec3f(0.), vec3f(1.)), c.y);
}

fn col_rgb2hsv(c: vec3f) -> vec3f {
  let K = vec4f(0., -1. / 3., 2. / 3., -1.);
  let p = mix(vec4f(c.b, c.g, K.w, K.z), vec4f(c.g, c.b, K.x, K.y), vec4f(step(c.b, c.g)));
  let q = mix(vec4f(p.x, p.y, p.w, c.r), vec4f(c.r, p.y, p.z, p.x), vec4f(step(p.x, c.r)));
  let d = q.x - min(q.w, q.y);
  let e = 1e-10;
  return vec3f(abs(q.z + (q.w - q.y) / (6. * d + e)), d / (q.x + e), q.x);
}

fn col_hue(c: vec3f) -> f32 {
  return col_rgb2hsv(c).x;
}

fn col_luma(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

fn col_sat(c: vec3f, s: f32) -> vec3f {
  return mix(vec3f(col_luma(c)), c, s);
}

fn col_cos(t: f32, a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f {
  return a + b * cos(TAU * (c * t + d));
}

fn col_warm(t: f32) -> vec3f {
  return col_cos(t, vec3f(0.5), vec3f(0.5), vec3f(1., 0.7, 0.4), vec3f(0., 0.15, 0.2));
}

fn col_cool(t: f32) -> vec3f {
  return col_cos(t, vec3f(0.5), vec3f(0.5), vec3f(1., 1., 1.), vec3f(0.5, 0.2, 0.));
}

fn col_fire(t: f32) -> vec3f {
  return col_cos(t, vec3f(0.5, 0.15, 0.), vec3f(0.55, 0.45, 0.35), vec3f(1., 0.9, 0.7), vec3f(0., 0.05, 0.1));
}

fn col_gray(t: f32) -> vec3f {
  return vec3f(clamp(t, 0., 1.));
}

fn col_pal(t: f32, phase: f32) -> vec3f {
  // a small palette bank; phase rotates the hue ramp, its integer part picks the bank
  let bank = i32(floor(phase)) % 4;
  let ph = fract(phase);
  var c: vec3f;
  if (bank == 0) { c = col_cos(t, vec3f(0.5), vec3f(0.5), vec3f(1.), vec3f(0., 0.33, 0.67) + vec3f(ph * 0.2)); }
  else if (bank == 1) { c = col_warm(t); }
  else if (bank == 2) { c = col_cool(t); }
  else { c = col_fire(t); }
  return clamp(c, vec3f(0.), vec3f(1.));
}

fn col_bayer4(x: vec2i) -> f32 {
  let B = COL_BAYER4;
  let i = (x.y & 3) * 4 + (x.x & 3);
  return (B[i] + 0.5) / 16.;
}

fn col_bayer8(x: vec2i) -> f32 {
  let B = COL_BAYER8;
  let i = (x.y & 7) * 8 + (x.x & 7);
  return (B[i] + 0.5) / 64.;
}

fn col_dither(c: vec3f, t: f32, amt: f32) -> vec3f {
  return c + (clamp(t, 0., 1.) - 0.5) * amt;
}

fn col_quant(c: vec3f, levels: f32) -> vec3f {
  let n = max(levels, 1.);
  return floor(clamp(c, vec3f(0.), vec3f(1.)) * n + 0.5) / n;
}

fn col_quant_d(c: vec3f, levels: f32, t: f32, amt: f32) -> vec3f {
  let n = max(levels, 1.);
  let k = (clamp(t, 0., 1.) - 0.5) * clamp(amt, 0., 1.) / n;
  return floor((clamp(c, vec3f(0.), vec3f(1.)) + vec3f(k)) * n + 0.5) / n;
}
