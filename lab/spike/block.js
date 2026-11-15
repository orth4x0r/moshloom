// S1 spike block, injected into gpu.js by lab/spike/apply.mjs. Throwaway: prices raymarching.
// ---- S1 SPIKE (throwaway): raymarched corridor at half resolution + nearest upscale ----
const WG_UPSCALE=WG_VS+`@group(0) @binding(0) var aT: texture_2d<f32>;
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return textureLoad(aT, vec2i(floor(pos.xy * 0.5)), 0); }`;
const WG_RM_STEPS=64;
const WG_RM_SPIKE=(steps)=>WG_HEAD+WG_VS+`
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = XY(pos);
  let t = U.T * 1.3;
  let ro = vec3f(0.0, 0.0, t);
  let rd = normalize(vec3f(uv.x * 0.85, uv.y * 0.85, 1.0));
  var dist = 0.0; var hit = 0.0; var steps_used = 0.0;
  for (var i = 0; i < ${steps}; i++) {
    let p = ro + rd * dist;
    let r = length(p.xy);
    let wall = abs(r - 0.9) - 0.010;
    let zz = fract(p.z * 0.25) * 4.0 - 2.0;
    let ring = length(vec2f(r - 0.9, zz)) - 0.045;
    let d = min(wall, ring);
    steps_used += 1.0;
    if (d < 0.002) { hit = 1.0; break; }
    dist += max(d, 0.010);
    if (dist > 30.0) { break; }
  }
  let fogc = 1.0 - clamp(dist / 24.0, 0.0, 1.0);
  var band = 1.0;
  if (hit > 0.5) { let p = ro + rd * dist; band = 0.35 + 0.65 * smoothstep(0.0, 0.30, abs(fract(p.z * 0.25) - 0.5)); }
  var c = lut(0.10 + 0.55 * fogc) * (0.25 + 1.4 * fogc) * band;
  c += lut(0.78) * (1.0 - fogc) * (0.5 + 0.5 * U.bass);
  return fin(c); }`;
