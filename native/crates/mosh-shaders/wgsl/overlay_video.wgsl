// Overlay video: draw a decoded BGRA frame into a layer target with cover scaling (aspect kept,
// centred). Owner: V-overlay (Task 2.8). Fragment stage only: mosh-render pairs this module with
// its own fullscreen vertex shader, so there is no vertex entry point here.
//
// Uniform block (16 bytes, std140): the destination size and the video size, in pixels.
//
// Row order: mosh-render's compositor and the stage's present pass each flip the picture once, so
// a layer target's row 0 is the top of what the audience sees, and v is flipped here so the
// video's first row (its top) lands there. A decoded frame is "top row first", like any image.

struct Cover {
    dst: vec2<f32>,
    src: vec2<f32>,
};

@group(0) @binding(0) var<uniform> cover: Cover;
@group(0) @binding(1) var video: texture_2d<f32>;
@group(0) @binding(2) var video_sampler: sampler;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // Cover: the larger of the two axis scales fills the target and crops the other axis. The
    // crop is centred, so the visible fraction of the source is dst / (src * scale).
    let scale = max(cover.dst.x / cover.src.x, cover.dst.y / cover.src.y);
    let visible = cover.dst / (cover.src * scale);
    let source_uv = vec2<f32>(
        0.5 + (uv.x - 0.5) * visible.x,
        0.5 - (uv.y - 0.5) * visible.y,
    );
    // No alpha in v1: the decoder gives opaque RGB32, and the compositor applies layer opacity.
    return vec4<f32>(textureSample(video, video_sampler, source_uv).rgb, 1.0);
}
