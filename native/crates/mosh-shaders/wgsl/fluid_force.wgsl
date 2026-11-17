// extracted by scripts/extract_wgsl.mjs from gpu.js: WG_FFOR
// Do not edit: change gpu.js and re-run the script.
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return vec4f(f32((i << 1u) & 2u) * 2. - 1., f32(i & 2u) * 2. - 1., 0., 1.); }

struct FUB { texel: vec2f, dt: f32, visc: f32, imp: vec4f, beat: f32, bass: f32, pulse: f32, gain: f32, vort: f32, pad0: f32, pad1: f32, pad2: f32 };
@group(0) @binding(0) var<uniform> Q: FUB;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var tA: texture_2d<f32>;
@group(0) @binding(3) var tB: texture_2d<f32>;
@group(0) @binding(4) var tC: texture_2d<f32>;
@group(0) @binding(5) var tD: texture_2d<f32>;
fn UV(pos: vec4f) -> vec2f { return pos.xy * Q.texel; }
fn VEL(p: vec2f) -> vec2f { return textureSampleLevel(tA, smp, p, 0.).xy; }

@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = UV(pos); var v = VEL(p);
  let mv = textureSampleLevel(tB, smp, p, 0.).xy;
  v += (mv / max(Q.dt, 0.0001) - v) * clamp(Q.gain * Q.dt, 0., 1.);        // block motion as a body force
  if (Q.imp.z > 0.) { let d = p - Q.imp.xy; let r = length(d) + 0.001;      // kick impulse, hashed centre
    v += (d / r) * Q.imp.z * exp(-r * r * 60.); }
  v += vec2f(p.y - 0.5, 0.5 - p.x) * Q.bass * 0.35 * Q.dt;                  // slow swirl
  v *= 1. / (1. + Q.visc * 9. * Q.dt);
  return vec4f(clamp(v, vec2f(-6.), vec2f(6.)), 0., 1.); }
