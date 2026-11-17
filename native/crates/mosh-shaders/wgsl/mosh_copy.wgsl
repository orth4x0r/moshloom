// The mosh I-frame: an exact copy of the layer's current frame. Owner: V-mosh (Task 2.2). The
// feedback pass has no I-frame branch (unlike the web's `if (F.iframe < 0.5)`) because the chain
// already has a copy pass for the previous-frame buffer, so an I-frame is two `mosh_copy` draws:
// one into the mosh output and one into the previous frame.
//
// Bindings follow mosh-render's descriptor layout: the uniform at 0, then one sampled image and
// one sampler per input.

struct CU {
    res: vec2<f32>,
    pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> c: CU;
@group(0) @binding(1) var srcT: texture_2d<f32>;
@group(0) @binding(2) var srcS: sampler;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return textureSampleLevel(srcT, srcS, pos.xy / c.res, 0.0);
}
