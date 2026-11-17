// One layer of the compositor: sample the layer's current image, apply opacity, luma-key it
// away when asked, and let the fixed-function blend unit do `Normal`/`Add`/`Screen`/`Multiply`.
//
// Uniform layout (`COMPOSITOR_UNIFORM_SIZE` = 16 bytes, std140):
//   0: opacity (f32)
//   4: luma_threshold (f32)
//   8: flags (u32)  bit 0 = luma key enabled, bit 1 = flip V
//  12: pad

struct Params {
    opacity: f32,
    luma_threshold: f32,
    flags: u32,
    pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var source_sampler: sampler;

@fragment
fn fs_main(@location(0) uv_in: vec2<f32>) -> @location(0) vec4<f32> {
    var uv = uv_in;
    if ((params.flags & 2u) != 0u) {
        uv = vec2<f32>(uv.x, 1.0 - uv.y);
    }
    let texel = textureSample(source, source_sampler, uv);
    if ((params.flags & 1u) != 0u) {
        let luma = dot(texel.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
        if (luma < params.luma_threshold) {
            discard;
        }
    }
    return vec4<f32>(texel.rgb * params.opacity, texel.a * params.opacity);
}
