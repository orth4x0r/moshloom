// fluid visualizer, the velocity field a mosh layer stirs with (V-fluid, Task 2.4).
//
// The simulation keeps its velocity at 256x144 in grid uv per second (what the extracted
// fluid_* passes read and write). Mosh works in pixels, so this pass converts once per frame
// into a target at the layer's internal size and leaves RG = velocity in **pixels per draw**,
// because one `FluidVisualizer::draw` is one simulation step (dt = 1/30 s) and therefore one
// frame of the field.
//
// Bind group 0, as mosh-render builds it: 0 uniform, 1 image, 2 sampler.
//
// Uniform `VU`, 48 bytes, all vec4f:
//   scale: xy = dt * internal size in pixels (uv/s -> pixels per draw), zw = unused
//   texel: xy = 1 / internal size, zw = unused
//   limit: x = clamp on each component in pixels, yzw = unused

struct VU {
  scale: vec4f,
  texel: vec4f,
  limit: vec4f,
};

@group(0) @binding(0) var<uniform> U: VU;
@group(0) @binding(1) var velT: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy * U.texel.xy;
  let v = textureSampleLevel(velT, smp, uv, 0.).xy;
  let px = v * U.scale.xy;
  let bound = max(U.limit.x, 1.);
  return vec4f(clamp(px, vec2f(-bound), vec2f(bound)), 0., 1.);
}
