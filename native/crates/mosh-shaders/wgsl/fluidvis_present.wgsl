// fluid visualizer, the pass that turns the dye field into the layer's picture (V-fluid, 2.4).
//
// Bind group 0, as mosh-render builds it: 0 uniform, 1 image, 2 sampler.
//
// Uniform `PU`, 48 bytes, all vec4f:
//   tint:  rgb = colour gain, a = unused
//   grade: x = shoulder knee, y = vignette, z = gain, w = unused
//   texel: xy = 1 / output size, zw = unused

struct PU {
  tint: vec4f,
  grade: vec4f,
  texel: vec4f,
};

@group(0) @binding(0) var<uniform> U: PU;
@group(0) @binding(1) var dyeT: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy * U.texel.xy;
  let c = max(textureSampleLevel(dyeT, smp, uv, 0.).rgb, vec3f(0.));
  // Soft shoulder: a dense core rolls off instead of clipping to a flat disc.
  let toned = c / (1. + c * U.grade.x);
  let d = (uv - 0.5) * 2.;
  let vig = 1. - U.grade.y * min(dot(d, d), 1.5);
  // Opaque: fluid is a full-frame visualizer, not something to blend over what is behind it.
  return vec4f(clamp(toned * U.tint.rgb * U.grade.z * vig, vec3f(0.), vec3f(1.)), 1.);
}
