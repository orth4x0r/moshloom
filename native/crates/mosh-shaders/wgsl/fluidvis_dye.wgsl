// fluid visualizer, dye transport and injection (V-fluid, Task 2.4).
// Not extracted from gpu.js: the web app has no dye field, it only pushes pixels with the
// velocity. Here the dye IS the picture, so it gets its own pass.
//
// Bind group 0, as mosh-render builds it: 0 uniform, 1 image, 2 sampler, 3 image.
// (mosh-shaders' extracted fluid_* passes use the web app's layout instead: 0 uniform,
// 1 sampler, 2..5 images. fluid.rs adapts those in SPIR-V, not here.)
//
// Uniform `DU`, 128 bytes, every field a vec4f so there is no padding to get wrong:
//   col0/col1/col2: rgb = injector colour, a = dye injected this step
//   kick:           xy = kick centre in grid uv, z = dye injected by the kick, w = unused
//   pos01:          xy = injector 0 position, zw = injector 1 position (grid uv)
//   pos2:           xy = injector 2 position, z = fade per step, w = dt (seconds)
//   timing:         x = swirl, y = colour cycle phase, z = level 0..1, w = t
//   texel:          xy = 1 / grid size, zw = unused

struct DU {
  col0: vec4f,
  col1: vec4f,
  col2: vec4f,
  kick: vec4f,
  pos01: vec4f,
  pos2: vec4f,
  timing: vec4f,
  texel: vec4f,
};

@group(0) @binding(0) var<uniform> U: DU;
@group(0) @binding(1) var velT: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
@group(0) @binding(3) var dyeT: texture_2d<f32>;

fn blob(uv: vec2f, centre: vec2f, sharpness: f32) -> f32 {
  let d = uv - centre;
  return exp(-dot(d, d) * sharpness);
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy * U.texel.xy;
  let v = textureSampleLevel(velT, smp, uv, 0.).xy;
  // Semi-Lagrangian transport: the dye here comes from where the flow was a step ago, and
  // `pos2.z` is the fraction of it that survives this step.
  var c = textureSampleLevel(dyeT, smp, uv - v * U.pos2.w, 0.).rgb * U.pos2.z;
  // Three injectors, one per band, at three fixed positions that drift slowly with time.
  c += U.col0.rgb * U.col0.a * blob(uv, U.pos01.xy, 60.);
  c += U.col1.rgb * U.col1.a * blob(uv, U.pos01.zw, 60.);
  c += U.col2.rgb * U.col2.a * blob(uv, U.pos2.xy, 60.);
  // A kick splashes dye at the centre its impulse came from.
  if (U.kick.z > 0.) {
    c += U.col1.rgb * U.kick.z * blob(uv, U.kick.xy, 90.);
  }
  // Hard bound, so a pathological input cannot turn the dye into a NaN or an infinity that the
  // rest of the chain then propagates (tests/fluid.rs asserts this from SPIR-V down).
  return vec4f(clamp(c, vec3f(0.), vec3f(4.)), 1.);
}
