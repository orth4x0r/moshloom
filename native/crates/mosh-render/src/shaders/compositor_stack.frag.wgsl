// The whole layer stack in one fragment pass.
//
// The per-layer version of the compositor (`compositor.frag.wgsl`) draws every layer as its own
// full-screen pass and lets the fixed-function blend unit combine it with what is already in the
// output image, which means one full-output-size read-modify-write *per layer*. Four layers is
// four passes and three redundant reads and writes of a 1920x1080 RGBA16F image; the frame budget
// measured 2.6 ms for exactly that at 1080p.
//
// This shader does the same arithmetic, in the same order, in registers: start from transparent
// black, sample each layer once and apply its blend mode and opacity, then write the result with
// blending disabled. The maths is the fixed-function one, premultiplied, written out:
//
//   normal    rgb = src + acc * (1 - src.a)      a = src.a + acc.a * (1 - src.a)
//   add       rgb = src + acc                    a = src.a + acc.a
//   screen    rgb = src + acc * (1 - src)        a = src.a + acc.a * (1 - src.a)
//   multiply  rgb = acc * src + acc * (1 - src.a) a = src.a + acc.a * (1 - src.a)
//   lumakey   normal, but a texel dimmer than the threshold contributes nothing (the old pass
//             discarded it, which is the same thing when nothing is behind it in the pass)
//
// `src` is the layer's texel times its opacity, exactly what the old fragment shader returned.
//
// Uniform layout (`COMPOSITOR_STACK_UNIFORM_SIZE` = 144 bytes):
//   0:  count (u32), 4: pad, 8: pad, 12: pad
//   16: eight `vec4<f32>`: (opacity, mode, luma threshold, flip V flag)
//       mode: 0 normal, 1 add, 2 screen, 3 multiply, 4 luma key
//
// Bindings follow `mosh-render`'s fixed fullscreen layout: uniform at 0, then (image, sampler)
// pairs. Eight inputs is the show's ceiling (three visualizer layers plus an overlay); the
// renderer falls back to the per-layer passes above that.

struct Stack {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    layers: array<vec4<f32>, 8>,
};

@group(0) @binding(0) var<uniform> stack: Stack;
@group(0) @binding(1) var layer0: texture_2d<f32>;
@group(0) @binding(2) var sampler0: sampler;
@group(0) @binding(3) var layer1: texture_2d<f32>;
@group(0) @binding(4) var sampler1: sampler;
@group(0) @binding(5) var layer2: texture_2d<f32>;
@group(0) @binding(6) var sampler2: sampler;
@group(0) @binding(7) var layer3: texture_2d<f32>;
@group(0) @binding(8) var sampler3: sampler;
@group(0) @binding(9) var layer4: texture_2d<f32>;
@group(0) @binding(10) var sampler4: sampler;
@group(0) @binding(11) var layer5: texture_2d<f32>;
@group(0) @binding(12) var sampler5: sampler;
@group(0) @binding(13) var layer6: texture_2d<f32>;
@group(0) @binding(14) var sampler6: sampler;
@group(0) @binding(15) var layer7: texture_2d<f32>;
@group(0) @binding(16) var sampler7: sampler;

const NORMAL: f32 = 0.0;
const ADD: f32 = 1.0;
const SCREEN: f32 = 2.0;
const MULTIPLY: f32 = 3.0;
const LUMAKEY: f32 = 4.0;

fn take(acc: vec4<f32>, texel: vec4<f32>, params: vec4<f32>) -> vec4<f32> {
    let opacity = params.x;
    let mode = params.y;
    if (mode == LUMAKEY) {
        let luma = dot(texel.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
        if (luma < params.z) {
            return acc;
        }
    }
    let src = vec4<f32>(texel.rgb * opacity, texel.a * opacity);
    if (mode == ADD) {
        return vec4<f32>(src.rgb + acc.rgb, src.a + acc.a);
    }
    if (mode == SCREEN) {
        return vec4<f32>(src.rgb + acc.rgb * (vec3<f32>(1.0) - src.rgb),
                         src.a + acc.a * (1.0 - src.a));
    }
    if (mode == MULTIPLY) {
        return vec4<f32>(acc.rgb * src.rgb + acc.rgb * (1.0 - src.a),
                         src.a + acc.a * (1.0 - src.a));
    }
    return vec4<f32>(src.rgb + acc.rgb * (1.0 - src.a), src.a + acc.a * (1.0 - src.a));
}

fn flip(uv: vec2<f32>, params: vec4<f32>) -> vec2<f32> {
    if (params.w > 0.5) {
        return vec2<f32>(uv.x, 1.0 - uv.y);
    }
    return uv;
}

@fragment
fn fs_main(@location(0) uv_in: vec2<f32>) -> @location(0) vec4<f32> {
    var acc = vec4<f32>(0.0);
    if (stack.count > 0u) {
        acc = take(acc, textureSampleLevel(layer0, sampler0, flip(uv_in, stack.layers[0]), 0.0), stack.layers[0]);
    }
    if (stack.count > 1u) {
        acc = take(acc, textureSampleLevel(layer1, sampler1, flip(uv_in, stack.layers[1]), 0.0), stack.layers[1]);
    }
    if (stack.count > 2u) {
        acc = take(acc, textureSampleLevel(layer2, sampler2, flip(uv_in, stack.layers[2]), 0.0), stack.layers[2]);
    }
    if (stack.count > 3u) {
        acc = take(acc, textureSampleLevel(layer3, sampler3, flip(uv_in, stack.layers[3]), 0.0), stack.layers[3]);
    }
    if (stack.count > 4u) {
        acc = take(acc, textureSampleLevel(layer4, sampler4, flip(uv_in, stack.layers[4]), 0.0), stack.layers[4]);
    }
    if (stack.count > 5u) {
        acc = take(acc, textureSampleLevel(layer5, sampler5, flip(uv_in, stack.layers[5]), 0.0), stack.layers[5]);
    }
    if (stack.count > 6u) {
        acc = take(acc, textureSampleLevel(layer6, sampler6, flip(uv_in, stack.layers[6]), 0.0), stack.layers[6]);
    }
    if (stack.count > 7u) {
        acc = take(acc, textureSampleLevel(layer7, sampler7, flip(uv_in, stack.layers[7]), 0.0), stack.layers[7]);
    }
    return acc;
}
