// Overlay video for HAP: a block-compressed (BCn) texture drawn into a layer target with cover
// scaling, straight from the GPU's own block decode. Owner: O-hap (Task 2.8's alpha follow-up).
// Fragment stage only: mosh-render pairs this module with its fullscreen vertex shader.
//
// HAP's variants differ in what the blocks carry, so the shader has four paths selected by
// `cover.mode` bits:
//
//   bit 0 (1): scaled YCoCg (HAP Q). The DXT5 block's alpha channel is luma, red is Co, green is
//              Cg and blue holds the chroma scale: exactly `ycocg2rgba(scaled)` in FFmpeg's
//              `texturedsp.c`, the reference decoder.
//   bit 1 (2): a second texture (BC4) holds alpha. Used by HapM / HAP Q Alpha.
//   bit 2 (4): the colour texture's own alpha channel is the alpha. Used by Hap5 (DXT5).
//   bit 3 (8): alpha only (HapA): the sole texture is a BC4 mask, drawn as a white silhouette.
//
// Output is premultiplied, because mosh-render's `BlendMode::Normal` is `src + dst * (1 - a)`:
// HAP stores straight alpha, so colour is multiplied here, once.
//
// Uniform block (32 bytes, std140): destination size, display size, padded texture size, mode.
//
// Row order: like `overlay_video.wgsl`, a layer target is V-mirrored relative to the stage (the
// compositor samples a layer with uv.y = 0 at the top of the frame the audience sees, and a layer
// target's row 0 is that frame's bottom), so v follows uv: the HAP frame's first block row (its
// top) lands at the top of the stage. The texture is padded to whole 4x4 blocks, so the display
// region is the top-left corner of it and `src / tex` crops the padding rows and columns away.

struct Cover {
    dst: vec2<f32>,
    src: vec2<f32>,
    tex: vec2<f32>,
    mode: f32,
    pad: f32,
};

@group(0) @binding(0) var<uniform> cover: Cover;
@group(0) @binding(1) var color_texture: texture_2d<f32>;
@group(0) @binding(2) var color_sampler: sampler;
@group(0) @binding(3) var alpha_texture: texture_2d<f32>;
@group(0) @binding(4) var alpha_sampler: sampler;

/// Scaled YCoCg to RGB, matching FFmpeg's `ycocg2rgba(scaled = 1)`:
/// `s = (b >> 3) + 1; y = a; co = (r - 128) / s; cg = (g - 128) / s`, then
/// `R = y + co - cg, G = y + cg, B = y - co - cg`, all in 0..255 and clipped.
fn ycocg_to_rgb(color: vec4<f32>) -> vec3<f32> {
    let red = color.r * 255.0;
    let green = color.g * 255.0;
    let blue = color.b * 255.0;
    let luma = color.a;
    let scale = floor(blue / 8.0) + 1.0;
    let co = (red - 128.0) / scale / 255.0;
    let cg = (green - 128.0) / scale / 255.0;
    return vec3<f32>(luma + co - cg, luma + cg, luma - co - cg);
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // Cover: the larger of the two axis scales fills the target and crops the other axis. The
    // crop is centred, so the visible fraction of the source is dst / (src * scale).
    let scale = max(cover.dst.x / cover.src.x, cover.dst.y / cover.src.y);
    let visible = cover.dst / (cover.src * scale);
    let source_uv = vec2<f32>(
        0.5 + (uv.x - 0.5) * visible.x,
        0.5 + (uv.y - 0.5) * visible.y,
    );
    // The texture is the display size rounded up to whole 4x4 blocks, so the display picture is
    // its top-left corner.
    let texel_uv = source_uv * (cover.src / cover.tex);
    let color = textureSample(color_texture, color_sampler, texel_uv);
    // Sampled unconditionally: WGSL wants uniform control flow around implicit derivatives. When
    // the clip has no separate alpha plane the caller binds the same texture here.
    let second = textureSample(alpha_texture, alpha_sampler, texel_uv).r;

    let mode = u32(cover.mode);
    var rgb = color.rgb;
    var alpha = 1.0;

    if ((mode & 8u) != 0u) {
        // HapA: the sole plane is a BC4 alpha mask, shown as a white silhouette.
        rgb = vec3<f32>(1.0);
        alpha = color.r;
    } else {
        if ((mode & 1u) != 0u) {
            rgb = ycocg_to_rgb(color);
        }
        if ((mode & 4u) != 0u) {
            alpha = color.a;
        }
        if ((mode & 2u) != 0u) {
            alpha = second;
        }
    }

    // Premultiplied: the layer's blend is "over" with premultiplied source.
    return vec4<f32>(rgb * alpha, alpha);
}
