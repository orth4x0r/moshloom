// The mosh feedback pass: carry the previous mosh buffer along the motion field and mix it with
// the new frame. Owner: V-mosh (Task 2.2). Ported from the P-frame branch of `WG_FB` in gpu.js,
// with two documented differences:
//
// * `amount` is the P-frame strength - the mix weight of the displaced buffer, so the new frame
//   gets `1 - amount` - instead of the web's `leak = (1 - amount)^5`, which is the same knob with
//   a much longer time constant (Task 2.2's brief redefines amount this way).
// * the buffer is warped by the estimated motion as-is, without the web's
//   `gain = 1 + 2.2 * amount` overshoot: with both applied the picture is carried *ahead* of the
//   motion, and the overshoot would be a second copy of the same knob.
//
// I-frames do not come through this pass; `mosh_copy` refreshes the buffer instead.
//
// Bindings follow mosh-render's descriptor layout: the uniform at 0, then one sampled image and
// one sampler per input.

struct FU {
    res: vec2<f32>,
    amount: f32,
    pad0: f32,
    grid: vec2<f32>,
    pad1: vec2<f32>,
};

@group(0) @binding(0) var<uniform> f: FU;
@group(0) @binding(1) var srcT: texture_2d<f32>;
@group(0) @binding(2) var srcS: sampler;
@group(0) @binding(3) var bufT: texture_2d<f32>;
@group(0) @binding(4) var bufS: sampler;
@group(0) @binding(5) var mvT: texture_2d<f32>;
@group(0) @binding(6) var mvS: sampler;
@group(0) @binding(7) var velT: texture_2d<f32>;
@group(0) @binding(8) var velS: sampler;

// The web looks its vector texture up per block (`textureLoad(mvT, bc, 0)`, one 8x8 grid-pixel
// block per vector). We sample the same 40x23 texture at the fragment's own uv instead: the
// linear sampler interpolates between block centres, so the displacement field does not step by a
// whole block at a block edge, which would leave a trail made of detached 24-pixel ghosts at
// 960x540. The field, its units and its sign are the web's.

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = pos.xy / f.res;
    let motion = textureSampleLevel(mvT, mvS, uv, 0.0).xy;
    let flow = textureSampleLevel(velT, velS, uv, 0.0).xy;
    // Where this pixel's content was last frame: the motion is in analysis-grid pixels per frame,
    // the fluid's velocity in internal pixels per frame, so each is divided by its own scale.
    let shift = motion / f.grid + flow / f.res;
    let fresh = textureSampleLevel(srcT, srcS, uv, 0.0).rgb;
    let carried = textureSampleLevel(bufT, bufS, uv - shift, 0.0).rgb;
    let mixed = mix(fresh, carried, clamp(f.amount, 0.0, 1.0));
    return vec4<f32>(clamp(mixed, vec3<f32>(-0.15), vec3<f32>(1.15)), 1.0);
}
