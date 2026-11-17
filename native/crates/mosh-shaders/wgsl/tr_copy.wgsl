// tr_copy: an exact copy of one picture, for TransitionPass::begin's private "from" copy
// (owner: V-fx, Task 2.3).
//
// Hand written, not extracted: gpu.js's WG_COPY renders rgba8unorm for thumbnails and clamps to
// 0..1, which would throw away the half-float range a transition is blending.
//
// Bind group 0, mosh-render's fixed fullscreen layout: 0 uniform (unused, 16 B) | 1 srcT.
// textureLoad at integer pixel coordinates: no sampler, no filtering, no rounding.

@group(0) @binding(1) var srcT: texture_2d<f32>;

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return textureLoad(srcT, vec2i(pos.xy), 0);
}
