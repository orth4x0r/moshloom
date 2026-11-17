// The web app's melT: 24x128 r8unorm, a spectrogram history the scenes read with `melh()`.
// gpu.js rewrites the whole texture every frame (`gpuMel`); the native renderer has no upload, so
// the history lives in a 24x128 layer target and shifts by one column per frame: row 0 is the
// newest analysis frame, row a the one a frames old. That is the same layout gpuMel writes
// (mb[a*24+k] = mel of frame f-a).
struct SU { res: vec2f, T: f32, hue: f32, bass: f32, mid: f32, high: f32, pulse: f32, vari: f32, dens: f32, beat: f32, frame: f32, pal: f32, aux0: f32, aux1: f32, aux2: f32, l0: vec4f, l1: vec4f, l2: vec4f, mel: array<vec4f,6> };
@group(0) @binding(0) var<uniform> U: SU;
@group(0) @binding(1) var melT: texture_2d<f32>;

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = vec2i(pos.xy);
  if (p.y == 0) {
    // gpuMel stores min(255, mel * 196) in an r8unorm texel.
    let k = clamp(p.x, 0, 23);
    return vec4f(min(1., U.mel[k / 4][k % 4] * (196. / 255.)), 0., 0., 1.);
  }
  return textureLoad(melT, vec2i(p.x, p.y - 1), 0);
}
