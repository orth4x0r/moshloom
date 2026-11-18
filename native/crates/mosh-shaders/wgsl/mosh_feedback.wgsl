// The mosh P-frame: carry the previous mosh buffer along the block-motion field, add the motion
// residual of the new frame, and leak the new frame in slowly. Owner: V-mosh (Task 2.2). This is
// the P-frame branch of `WG_FB` in gpu.js, ported as written:
//
//   bc = clamp(floor(uv * (40, 22.5)), 0, (39, 22))          one vector per 8x8 analysis block
//   if (coarse) bc &= -2                                      amount > 0.85: every other block
//   d  = round(motion * gain * mvScale * res / grid)          whole output pixels
//   c  = buf(pixel - d + flow)                                nearest, mirror-repeat
//   if (!skipres) { c = c + s - prv; c = mix(c, s, leak) }
//
// `motion` is the forward displacement in analysis-grid pixels (`mosh_motion`), `prv` is last
// frame's source picture, `leak = (1 - amount)^5` floored at 0.004 and `gain = 1 + 2.2 * amount`.
// The residual and the leak are what make this a datamosh instead of a crossfade; the block
// lookup and the whole-pixel rounding are what make it blocky instead of a smooth warp.
//
// `s` and `prv` are read at this pixel's own centre, and the buffer through the web's sampler:
// `G.nm` is nearest, mirror-repeat (`gpu.js:639`), and every displacement here is a whole number
// of pixels, so the pass reads exactly one texel of each input and a smear that runs off the edge
// mirrors back in rather than clamping. `textureLoad` at the mirrored texel index is that sample
// exactly, which is why this shader declares its samplers and only uses `velS`.
//
// `skipres` is the web app's `P.skipres`: on the frame a clip change enters on a Datamosh
// transition, the buffer is carried by the new clip's motion but nothing of the new clip is
// mixed in, which is the classic datamosh cut. The residual is skipped whole - `leak` too,
// exactly as in `WG_FB`.
//
// I-frames do not come through this pass; `mosh_copy` refreshes the buffer instead.
//
// Not ported from `WG_FB`, none of it the mosh's picture content: the feedback zoom and spin
// (`fxa.z`/`fxa.w`), the `F.ops` rect remaps, and `fxb` (streak, crush, echo, negative) - those are
// effect slots of their own in this port, or V-fx's. Nor the vector clamp in `WG_ME`, which cannot
// bind at this search range (see the module docs of `MoshPass`).
//
// Bindings follow mosh-render's descriptor layout: the uniform at 0, then one sampled image and
// one sampler per input.

struct FU {
    res: vec2<f32>,
    leak: f32,
    mv_scale: f32,
    gain: f32,
    skipres: f32,
    coarse: f32,
    pad0: f32,
    grid: vec2<f32>,
};

@group(0) @binding(0) var<uniform> f: FU;
@group(0) @binding(1) var srcT: texture_2d<f32>;
@group(0) @binding(2) var srcS: sampler;
@group(0) @binding(3) var bufT: texture_2d<f32>;
@group(0) @binding(4) var bufS: sampler;
@group(0) @binding(5) var prvT: texture_2d<f32>;
@group(0) @binding(6) var prvS: sampler;
@group(0) @binding(7) var mvT: texture_2d<f32>;
@group(0) @binding(8) var mvS: sampler;
@group(0) @binding(9) var velT: texture_2d<f32>;
@group(0) @binding(10) var velS: sampler;

/// The web app's vector grid: 40 blocks across, 180 / 8 = 22.5 down, clamped to 39 x 22 - the
/// last row of the 40x23 field is only half covered, and the web clamps to the covered part.
const BLOCKS: vec2<f32> = vec2<f32>(40.0, 22.5);

/// `G.nm`'s mirror-repeat wrap of one texel index (`gpu.js:639`): `0, 1, ... n-1, n-1, ... 0`.
fn mirror(index: i32, n: i32) -> i32 {
    let period = 2 * n;
    let wrapped = ((index % period) + period) % period;
    if (wrapped < n) {
        return wrapped;
    }
    return period - 1 - wrapped;
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(i32(pos.x), i32(pos.y));
    let res = vec2<i32>(i32(f.res.x), i32(f.res.y));
    // `s` and `prv` are read at this pixel's centre, which is the texel itself through any filter.
    let fresh = textureLoad(srcT, pixel, 0).rgb;

    // One vector per block of the analysis grid, read straight out of the 40x23 field: no
    // filtering, so the displacement steps by a whole block at a block edge (`textureLoad(mvT,
    // bc, 0)` in `WG_FB`).
    var bc = vec2<i32>(floor((pos.xy / f.res) * BLOCKS));
    bc = clamp(bc, vec2<i32>(0, 0), vec2<i32>(39, 22));
    if (f.coarse > 0.5) {
        bc = bc & vec2<i32>(-2);
    }
    let motion = textureLoad(mvT, bc, 0).xy;
    // The fluid visualizer's field is a displacement, in internal pixels per frame: the web adds
    // its velocity to the sample offset (`d += vel * ...; sample(buf, uv + d)` in `WG_FB`), so the
    // carried picture moves *against* the field. Same here: this is a sample offset, and `flow` is
    // added to it rather than subtracted. The web scales by `0.03` because its velocity is in uv
    // per second; this one is already pixels per frame, so the factor is 1.
    let flow = textureSampleLevel(velT, velS, pos.xy / f.res, 0.0).xy;
    // The web rounds the displacement to whole output pixels after the gain, which is what makes
    // the smear hold still for a frame instead of sliding sub-pixel. `motion` is the forward
    // displacement in grid pixels, so the sample offset it comes from is `-shift`.
    let shift = round(motion * (f.gain * f.mv_scale) * f.res / f.grid);

    // The web samples the buffer with `G.nm` - nearest, mirror-repeat (`gpu.js:639`), so a
    // displacement of whole pixels is exactly a texel, and a smear that leaves the frame mirrors
    // back in instead of clamping. `textureLoad` at the mirrored index is that sample exactly;
    // the fluid term is rounded like a displacement, not floored like a coordinate.
    let offset = vec2<i32>(i32(-shift.x + round(flow.x)), i32(-shift.y + round(flow.y)));
    let at = vec2<i32>(mirror(pixel.x + offset.x, res.x), mirror(pixel.y + offset.y, res.y));

    var c = textureLoad(bufT, at, 0).rgb;
    if (f.skipres < 0.5) {
        let previous = textureLoad(prvT, pixel, 0).rgb;
        c = c + fresh - previous;
        c = mix(c, fresh, clamp(f.leak, 0.0, 1.0));
    }
    return vec4<f32>(clamp(c, vec3<f32>(-0.15), vec3<f32>(1.15)), 1.0);
}
