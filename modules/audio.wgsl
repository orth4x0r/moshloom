// audio.wgsl - helpers over the two spectrum accessors that WG_HEAD already declares:
//   mel(i: i32) -> f32              the 24-band current frame level, wraps i modulo 24
//   melh(band: f32, age: f32) -> f32 the 24 x 128 spectral history, band in [0,23], age in [0,127]
// This module CALLS those two, it never redeclares them. It reads no binding of its own.
// Include after WG_HEAD. Declares no bindings and no entry points.
//
// Exports:
//   const AUD_BANDS: i32 = 24
//   const AUD_AGES: f32 = 128.
//   fn aud_band(i: i32) -> f32                     - mel(i), wrapped
//   fn aud_band01(x: f32) -> f32                   - linear interpolation between bands, x in [0,24]
//   fn aud_norm(i: i32) -> f32                     - mel(i) with a mild boost, clamped to 1
//   fn aud_peak() -> f32                           - loudest of the 24 bands
//   fn aud_avg() -> f32                            - mean of the 24 bands
//   fn aud_sum(b0: i32, b1: i32) -> f32            - sum of bands b0..b1-1
//   fn aud_centroid() -> f32                       - spectral centroid, 0..1
//   fn aud_hist(band: f32, age: f32) -> f32        - melh with band/age clamped
//   fn aud_hist_max(band: f32) -> f32              - max of melh over 16 age steps
//   fn aud_hist_avg(band: f32) -> f32              - mean of melh over 16 age steps
//   fn aud_onset(band: f32, age: f32, span: f32) -> f32 - current level minus level `span` frames ago
//   fn aud_wave(x: f32, age: f32) -> f32           - history sampled by continuous band x, age fixed
//   fn aud_gate(v: f32, thr: f32, soft: f32) -> f32 - soft threshold, 0 below thr
//   fn aud_pulse_at(age: f32) -> f32               - mean of all bands `age` frames ago

const AUD_BANDS: i32 = 24;
const AUD_AGES: f32 = 128.;
const AUD_SPAN: f32 = 16.;

fn aud_band(i: i32) -> f32 {
  return mel(i);
}

fn aud_band01(x: f32) -> f32 {
  let u = clamp(x, 0., 24.);
  let i = i32(floor(u));
  let f = u - floor(u);
  return mix(mel(i), mel(i + 1), f);
}

fn aud_norm(i: i32) -> f32 {
  return clamp(mel(i) * 1.25, 0., 1.);
}

fn aud_peak() -> f32 {
  var m = 0.;
  for (var i = 0; i < AUD_BANDS; i++) { m = max(m, mel(i)); }
  return m;
}

fn aud_avg() -> f32 {
  var s = 0.;
  for (var i = 0; i < AUD_BANDS; i++) { s += mel(i); }
  return s / f32(AUD_BANDS);
}

fn aud_sum(b0: i32, b1: i32) -> f32 {
  let a = clamp(min(b0, b1), 0, AUD_BANDS);
  let b = clamp(max(b0, b1), 0, AUD_BANDS);
  var s = 0.;
  for (var i = a; i < b; i++) { s += mel(i); }
  return s;
}

fn aud_centroid() -> f32 {
  var num = 0.;
  var den = 0.;
  for (var i = 0; i < AUD_BANDS; i++) {
    let e = mel(i);
    num += e * f32(i);
    den += e;
  }
  return clamp(num / max(den, 1e-5) / f32(AUD_BANDS - 1), 0., 1.);
}

fn aud_hist(band: f32, age: f32) -> f32 {
  return melh(clamp(band, 0., f32(AUD_BANDS - 1)), clamp(age, 0., AUD_AGES - 1.));
}

fn aud_hist_max(band: f32) -> f32 {
  var m = 0.;
  for (var k = 0; k < i32(AUD_SPAN); k++) {
    m = max(m, melh(clamp(band, 0., f32(AUD_BANDS - 1)), f32(k) * (AUD_AGES / AUD_SPAN)));
  }
  return m;
}

fn aud_hist_avg(band: f32) -> f32 {
  var s = 0.;
  for (var k = 0; k < i32(AUD_SPAN); k++) {
    s += melh(clamp(band, 0., f32(AUD_BANDS - 1)), f32(k) * (AUD_AGES / AUD_SPAN));
  }
  return s / AUD_SPAN;
}

fn aud_onset(band: f32, age: f32, span: f32) -> f32 {
  let b = clamp(band, 0., f32(AUD_BANDS - 1));
  let a0 = clamp(age, 0., AUD_AGES - 1.);
  let a1 = clamp(age + max(span, 0.), 0., AUD_AGES - 1.);
  return max(melh(b, a0) - melh(b, a1), 0.);
}

fn aud_wave(x: f32, age: f32) -> f32 {
  let u = clamp(x, 0., f32(AUD_BANDS - 1));
  let i = floor(u);
  let f = u - i;
  let a = melh(i, clamp(age, 0., AUD_AGES - 1.));
  let b = melh(min(i + 1., f32(AUD_BANDS - 1)), clamp(age, 0., AUD_AGES - 1.));
  return mix(a, b, f);
}

fn aud_gate(v: f32, thr: f32, soft: f32) -> f32 {
  return smoothstep(thr, thr + max(soft, 1e-4), v);
}

fn aud_pulse_at(age: f32) -> f32 {
  var s = 0.;
  for (var i = 0; i < AUD_BANDS; i++) { s += melh(f32(i), clamp(age, 0., AUD_AGES - 1.)); }
  return s / f32(AUD_BANDS);
}
