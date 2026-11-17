// The picture, from the trail: up-scale the half-size trail to the layer's own size, grade it and
// stop there. The layer is opaque (alpha 1) like the fluid layer's present pass: particles are a
// full picture with a black background, not a transparent sprite layer.

@group(0) @binding(1) var trail: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;

@fragment fn fs_main(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
  let uv = pixel.xy * U.texel.xy;
  var c = textureSampleLevel(trail, lin, uv, 0.0).rgb;
  // Exposure, then a soft shoulder: the trail is HDR (up to 6) and a hard clamp turns a dense
  // kick into a flat white disc.
  c = c * U.grade.x;
  c = c / (1.0 + c * 0.32);
  let edge = length((uv - vec2f(0.5)) * 2.0);
  c = c * (1.0 - 0.22 * smoothstep(0.75, 1.5, edge));
  return vec4f(clamp(c, vec3f(0.0), vec3f(1.0)), 1.0);
}
