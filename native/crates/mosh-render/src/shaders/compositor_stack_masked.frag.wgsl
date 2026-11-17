// The whole layer stack in one fragment pass, with masks and a base image.
//
// `compositor_stack.frag.wgsl` is the plain pass: no masks, no base, and the path a frame with
// no Adjustment and no Mask layer takes (unchanged, so its pixels are what they always were).
// This shader is the same arithmetic plus the two things those layers need:
//
// * **a mask per layer**: a `Mask` layer's picture, sampled at the same place as the layer above
//   it, and its luminance multiplied into that layer's opacity (`Mask` layers draw nothing
//   themselves). Black hides the layer, white leaves it alone.
// * **a base image**: the composite of the layers below an `Adjustment` layer. The accumulator
//   starts from it instead of transparent black, so the layers above the adjustment are drawn
//   over the adjusted group, exactly as they are drawn over the layers below today.
//
// `take()` is the fixed-function blend written out, premultiplied, the same as the plain pass:
//
//   normal    rgb = src + acc * (1 - src.a)      a = src.a + acc.a * (1 - src.a)
//   add       rgb = src + acc                    a = src.a + acc.a
//   screen    rgb = src + acc * (1 - src)        a = src.a + acc.a * (1 - src.a)
//   multiply  rgb = acc * src + acc * (1 - src.a) a = src.a + acc.a * (1 - src.a)
//   lumakey   normal, but a texel dimmer than the threshold contributes nothing
//
// with `src = vec4(texel.rgb, texel.a) * opacity * mask_luma`.
//
// Uniform layout (`MASKED_UNIFORM_SIZE` = 272 bytes in compositor.rs):
//   0:  count (u32), 4: flags (u32, see below), 8: pad, 12: pad
//   16: eight `vec4<f32>` per layer: (opacity, mode, luma threshold, flip V flag)
//       mode: 0 normal, 1 add, 2 screen, 3 multiply, 4 luma key
//  144: eight `vec4<f32>` per layer: (mask input slot, -1 when the layer has no mask)
//
// `flags` bit 0 = a base image is bound; bit 1 = sample at `1 - uv.y`, which a pass that writes an
// intermediate sets: the crate's `fullscreen.vert` uv is V-mirrored relative to
// `@builtin(position)`, so without this an adjustment's segment would be mirrored once per
// composite pass while the plain pass mirrors once, and the layers above would land the wrong way
// up. With it, an intermediate holds the layers in their own orientation (what the position-based
// effects and mosh passes read) and the last pass into the output keeps the plain pass's
// convention, so a stack with an Adjustment looks like the same stack without one.
//
// Bindings follow mosh-render's fixed fullscreen layout: uniform at 0, then (image, sampler)
// pairs for the base, eight layer inputs, then eight masks. Seventeen sampled images is one more
// than Vulkan guarantees, so `Compositor` only builds this pipeline when the adapter's
// `maxPerStageDescriptorSampledImages` reports room for it, and reports a missing feature instead
// of drawing a stack with an Adjustment or Mask layer.

struct Stack {
    count: u32,
    flags: u32,
    pad0: u32,
    pad1: u32,
    layers: array<vec4<f32>, 8>,
    masks: array<vec4<f32>, 8>,
};

@group(0) @binding(0) var<uniform> stack: Stack;
@group(0) @binding(1) var base_t: texture_2d<f32>;
@group(0) @binding(2) var base_s: sampler;
@group(0) @binding(3) var layer0: texture_2d<f32>;
@group(0) @binding(4) var sampler0: sampler;
@group(0) @binding(5) var layer1: texture_2d<f32>;
@group(0) @binding(6) var sampler1: sampler;
@group(0) @binding(7) var layer2: texture_2d<f32>;
@group(0) @binding(8) var sampler2: sampler;
@group(0) @binding(9) var layer3: texture_2d<f32>;
@group(0) @binding(10) var sampler3: sampler;
@group(0) @binding(11) var layer4: texture_2d<f32>;
@group(0) @binding(12) var sampler4: sampler;
@group(0) @binding(13) var layer5: texture_2d<f32>;
@group(0) @binding(14) var sampler5: sampler;
@group(0) @binding(15) var layer6: texture_2d<f32>;
@group(0) @binding(16) var sampler6: sampler;
@group(0) @binding(17) var layer7: texture_2d<f32>;
@group(0) @binding(18) var sampler7: sampler;
@group(0) @binding(19) var mask0: texture_2d<f32>;
@group(0) @binding(20) var mask_sampler0: sampler;
@group(0) @binding(21) var mask1: texture_2d<f32>;
@group(0) @binding(22) var mask_sampler1: sampler;
@group(0) @binding(23) var mask2: texture_2d<f32>;
@group(0) @binding(24) var mask_sampler2: sampler;
@group(0) @binding(25) var mask3: texture_2d<f32>;
@group(0) @binding(26) var mask_sampler3: sampler;
@group(0) @binding(27) var mask4: texture_2d<f32>;
@group(0) @binding(28) var mask_sampler4: sampler;
@group(0) @binding(29) var mask5: texture_2d<f32>;
@group(0) @binding(30) var mask_sampler5: sampler;
@group(0) @binding(31) var mask6: texture_2d<f32>;
@group(0) @binding(32) var mask_sampler6: sampler;
@group(0) @binding(33) var mask7: texture_2d<f32>;
@group(0) @binding(34) var mask_sampler7: sampler;

const NORMAL: f32 = 0.0;
const ADD: f32 = 1.0;
const SCREEN: f32 = 2.0;
const MULTIPLY: f32 = 3.0;
const LUMAKEY: f32 = 4.0;

fn take(acc: vec4<f32>, texel: vec4<f32>, params: vec4<f32>, mask: f32) -> vec4<f32> {
    let opacity = params.x * mask;
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

fn luma(texel: vec4<f32>) -> f32 {
    return dot(texel.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@fragment
fn fs_main(@location(0) uv_in: vec2<f32>) -> @location(0) vec4<f32> {
    var uv = uv_in;
    if ((stack.flags & 2u) != 0u) {
        uv = vec2<f32>(uv_in.x, 1.0 - uv_in.y);
    }
    var acc = vec4<f32>(0.0);
    if ((stack.flags & 1u) != 0u) {
        acc = textureSampleLevel(base_t, base_s, uv, 0.0);
    }
    if (stack.count > 0u) {
        var mask = 1.0;
        if (stack.masks[0].x >= 0.0) {
            mask = luma(textureSampleLevel(mask0, mask_sampler0, uv, 0.0));
        }
        acc = take(acc, textureSampleLevel(layer0, sampler0, flip(uv, stack.layers[0]), 0.0),
                   stack.layers[0], mask);
    }
    if (stack.count > 1u) {
        var mask = 1.0;
        if (stack.masks[1].x >= 0.0) {
            mask = luma(textureSampleLevel(mask1, mask_sampler1, uv, 0.0));
        }
        acc = take(acc, textureSampleLevel(layer1, sampler1, flip(uv, stack.layers[1]), 0.0),
                   stack.layers[1], mask);
    }
    if (stack.count > 2u) {
        var mask = 1.0;
        if (stack.masks[2].x >= 0.0) {
            mask = luma(textureSampleLevel(mask2, mask_sampler2, uv, 0.0));
        }
        acc = take(acc, textureSampleLevel(layer2, sampler2, flip(uv, stack.layers[2]), 0.0),
                   stack.layers[2], mask);
    }
    if (stack.count > 3u) {
        var mask = 1.0;
        if (stack.masks[3].x >= 0.0) {
            mask = luma(textureSampleLevel(mask3, mask_sampler3, uv, 0.0));
        }
        acc = take(acc, textureSampleLevel(layer3, sampler3, flip(uv, stack.layers[3]), 0.0),
                   stack.layers[3], mask);
    }
    if (stack.count > 4u) {
        var mask = 1.0;
        if (stack.masks[4].x >= 0.0) {
            mask = luma(textureSampleLevel(mask4, mask_sampler4, uv, 0.0));
        }
        acc = take(acc, textureSampleLevel(layer4, sampler4, flip(uv, stack.layers[4]), 0.0),
                   stack.layers[4], mask);
    }
    if (stack.count > 5u) {
        var mask = 1.0;
        if (stack.masks[5].x >= 0.0) {
            mask = luma(textureSampleLevel(mask5, mask_sampler5, uv, 0.0));
        }
        acc = take(acc, textureSampleLevel(layer5, sampler5, flip(uv, stack.layers[5]), 0.0),
                   stack.layers[5], mask);
    }
    if (stack.count > 6u) {
        var mask = 1.0;
        if (stack.masks[6].x >= 0.0) {
            mask = luma(textureSampleLevel(mask6, mask_sampler6, uv, 0.0));
        }
        acc = take(acc, textureSampleLevel(layer6, sampler6, flip(uv, stack.layers[6]), 0.0),
                   stack.layers[6], mask);
    }
    if (stack.count > 7u) {
        var mask = 1.0;
        if (stack.masks[7].x >= 0.0) {
            mask = luma(textureSampleLevel(mask7, mask_sampler7, uv, 0.0));
        }
        acc = take(acc, textureSampleLevel(layer7, sampler7, flip(uv, stack.layers[7]), 0.0),
                   stack.layers[7], mask);
    }
    return acc;
}
