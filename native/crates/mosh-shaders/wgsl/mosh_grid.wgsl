// The analysis-grid copy of one layer's picture, for datamosh block matching.
//
// The motion estimate works on the web app's fixed 320x180 luminance grid whatever the layer's
// internal size is, so it used to sample a 960x540 (or 1280x800) texture at grid coordinates:
// one bilinear read per grid pixel, with the cache-miss cost of a 4-8 MB image. Each pass also
// kept a full-size copy of the previous frame just to have a second history texture.
//
// This pass writes the grid once per layer per frame (320x180 = 57,600 texels, ~0.02 ms) and the
// motion pass then reads a 0.9 MB pair instead. One bilinear sample of the full picture at the
// grid texel centre is exactly what the motion pass used to read (`textureSampleLevel(p / grid)`
// with `p` at a texel centre), so the vectors are the same to the last half float.

struct GU {
    grid: vec2<f32>,
    pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> g: GU;
@group(0) @binding(1) var srcT: texture_2d<f32>;
@group(0) @binding(2) var srcS: sampler;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = pos.xy / g.grid;
    // The layers are 16F and the motion pass clamps its luminance anyway; clamping here keeps a
    // deliberately over-bright layer (Negative, Add chains) from poisoning the block match.
    return vec4<f32>(clamp(textureSampleLevel(srcT, srcS, uv, 0.0).rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
