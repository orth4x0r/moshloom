// The health probe of the particle tests: flags a NaN or an infinity in a target and reports the
// largest magnitude it saw, so a 5 000-frame run can prove the state and the trail stayed finite
// and bounded. Test-only: it is compiled by `particles/tests.rs` and never in the show.
//
// R: 0 finite, 1 NaN, 2 infinity. G: the largest |channel|, divided by 32.

@group(0) @binding(1) var src: texture_2d<f32>;

@fragment fn fs_main(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
  let v = textureLoad(src, vec2i(pixel.xy), 0);
  // `isNan`/`isInf` are not in this naga's WGSL: `x != x` is the NaN test, and anything past
  // 1e30 can only be an infinity (a real value here is below 8).
  var bad = 0.0;
  if (v.x != v.x || v.y != v.y || v.z != v.z || v.w != v.w) {
    bad = 1.0;
  }
  let m = max(max(abs(v.x), abs(v.y)), max(abs(v.z), abs(v.w)));
  if (m > 1e30) {
    bad = 2.0;
  }
  return vec4f(bad, clamp(m / 32.0, 0.0, 1.0), 0.0, 1.0);
}
