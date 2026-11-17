// fluid visualizer, the health probe the tests read back (V-fluid, Task 2.4).
//
// A `Gpu::read_back` clamps half floats to 0..1 and converts them to 8 bits, which hides a NaN
// (it lands on 0, the same as a zero) and hides how far a value ran away. This pass samples the
// dye and the velocity field at every grid texel and reports, per texel:
//
//   r = 1 when any sampled channel is NaN or infinite, else 0
//   g = the largest magnitude of any sampled channel, scaled by 1/32 (so 8.0 reads as 0.25)
//   b = the summed dye at that texel (0..1), a cheap "is there dye here" map
//
// Offscreen tests only: `fluid.rs` builds this pipeline behind `#[cfg(test)]`, so it costs the
// show nothing.
//
// Bind group 0, as mosh-render builds it: 0 uniform, 1 image, 2 sampler, 3 image.
//
// Uniform `CU`, 16 bytes: xy = 1 / grid size, zw = unused.

struct CU {
  texel: vec4f,
};

@group(0) @binding(0) var<uniform> U: CU;
@group(0) @binding(1) var dyeT: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
@group(0) @binding(3) var velT: texture_2d<f32>;

// A NaN and an infinity both fall outside this window: an ordered comparison against a NaN is
// false, and so is any comparison with an infinity that big. naga's WGSL front end has no
// `isNan`/`isInf` builtin, so the window is how the probe asks the question.
fn bad(x: f32) -> f32 {
  if (x >= -3.0e38 && x <= 3.0e38) { return 0.; }
  return 1.;
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy * U.texel.xy;
  let d = textureSampleLevel(dyeT, smp, uv, 0.);
  let v = textureSampleLevel(velT, smp, uv, 0.);
  let flag = max(max(max(bad(d.r), bad(d.g)), max(bad(d.b), bad(d.a))), max(bad(v.r), bad(v.g)));
  let mag = max(
    max(abs(d.r), abs(d.g)),
    max(max(abs(d.b), abs(d.a)), max(abs(v.r), abs(v.g))),
  );
  return vec4f(flag, clamp(mag / 32., 0., 1.), clamp(d.r + d.g + d.b, 0., 1.), 1.);
}
