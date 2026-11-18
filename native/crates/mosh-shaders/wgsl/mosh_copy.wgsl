// The mosh I-frame and the previous-source copy: an exact copy of a layer's current frame.
// Owner: V-mosh (Task 2.2).
//
// Two destinations, both a copy:
//
// * the previous-source target (`prv`): drawn **once every frame**, because the P-frame's residual
//   is `src - previous src` and needs last frame's picture at the layer's internal size. The web
//   app keeps the same thing (`prv` in `gpuFrame`, its `G.src` ping-pong) without a pass, because
//   it keeps two source targets; one copy pass costs less than a second full-size target.
// * the mosh output (`dst`), **only on an I-frame**: the feedback pass has no I-frame branch
//   (unlike the web's `if (F.iframe < 0.5)`) because the chain already has this pass, so an
//   I-frame is one more draw of it. With it, an I-frame records two copies (output and `prv`) and
//   a P-frame records one (`prv`).
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
