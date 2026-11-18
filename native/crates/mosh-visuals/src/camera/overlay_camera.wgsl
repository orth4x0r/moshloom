// Camera layer: one captured BGRA frame drawn into a layer target, cover-scaled, with the clip's
// zoom/pan/mirror and the grade (exposure, contrast, tint, posterize). Owner: L-camera.
// Hand-written for the native camera layer, not extracted from gpu.js.
//
// Fragment stage only: mosh-render pairs this module with its own fullscreen vertex shader.
//
// Uniform block (48 bytes, std140: two vec2s then two vec4s), kept in sync with `mosh-visuals/src/camera/params.rs`:
//   dst   destination size in pixels        (`vec2<f32>`)
//   src   captured frame size in pixels     (`vec2<f32>`)
//   view  (u centre, v centre, u fraction, v fraction) of the source the target shows; the
//         fraction is what the target's -0.5..0.5 offset is multiplied by, a negative u fraction
//         is the horizontal mirror, and the cover crop, the zoom and the pan are already folded
//         together on the CPU
//   grade (exposure multiplier, contrast multiplier, tint -1..1, posterize levels: 0 = off)
//
// Row order: a layer target is V-mirrored relative to the stage. This pass's uv.y is 0 at the top
// of the frame the audience sees and 1 at its bottom, while a layer target's row 0 is that frame's
// bottom, and the present pass adds no second flip (see `text/text.wgsl`). A captured frame is
// "top row first", so texture v = 0 is its top and v follows uv: the picture's top row lands at
// the top of the stage.

struct Camera {
    dst: vec2<f32>,
    src: vec2<f32>,
    view: vec4<f32>,
    grade: vec4<f32>,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var frame_tex: texture_2d<f32>;
@group(0) @binding(2) var frame_sampler: sampler;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let source_uv = vec2<f32>(
        cam.view.x + (uv.x - 0.5) * cam.view.z,
        cam.view.y + (uv.y - 0.5) * cam.view.w,
    );
    var c = textureSample(frame_tex, frame_sampler, source_uv).rgb;
    // Exposure first: a multiplier on the light, the way a camera's exposure works.
    c = c * cam.grade.x;
    // Contrast about mid grey.
    c = (c - 0.5) * cam.grade.y + 0.5;
    // Tint on the green <-> magenta axis: positive is magenta (red and blue up, green down).
    c = c * vec3<f32>(1.0 + 0.5 * cam.grade.z, 1.0 - 0.5 * cam.grade.z, 1.0 + 0.5 * cam.grade.z);
    let levels = cam.grade.w;
    if (levels > 1.5) {
        let top = levels - 1.0;
        c = floor(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)) * top + 0.5) / top;
    }
    // The camera picture is opaque; layer opacity and blending are the compositor's job.
    return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
