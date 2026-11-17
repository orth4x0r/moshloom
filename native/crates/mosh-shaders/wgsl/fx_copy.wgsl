// extracted by scripts/extract_wgsl.mjs from gpu.js: WG_COPY
// Do not edit: change gpu.js and re-run the script.
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return vec4f(f32((i << 1u) & 2u) * 2. - 1., f32(i & 2u) * 2. - 1., 0., 1.); }
@group(0) @binding(0) var aT: texture_2d<f32>;
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f { return vec4f(clamp(textureLoad(aT, vec2i(pos.xy), 0).rgb, vec3f(0.), vec3f(1.)), 1.); }
