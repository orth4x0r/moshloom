// ============================================================================
// modules/raymarch.wgsl - sphere-tracing marcher, camera rig and fragment entry.
//
// A scene supplies exactly two functions, declared anywhere in the module
// (WGSL module scope is order independent):
//
//   fn map(p: vec3f) -> vec2f
//       x = signed distance to the scene, y = material id (free for shade)
//
//   fn shade(p: vec3f, n: vec3f, rd: vec3f, mat: f32, steps: f32) -> vec3f
//       p = hit point, n = surface normal, rd = ray direction,
//       mat = map(p).y, steps = march steps spent (for cost-based effects)
//       returns albedo / base colour. Use lut(v) for palette colour or plain
//       albedo values; the marcher supplies the lighting.
//
// Everything here is prefixed rm_ (constants RM_, structs Rm*), and the entry
// point is `fs`. Budget: 64 steps, hit when d < 0.001 * t, far plane 40.
// Output: colour on @location(0), coverage in alpha (1 hit / 0 miss).
// All texture reads go through lut(), i.e. textureSampleLevel - no derivatives
// are taken anywhere, so non-uniform control flow is safe.
// ============================================================================

// ---------------------------------------------------------------- budget
const RM_MAX_STEPS: u32 = 64u;
const RM_FAR: f32 = 40.0;
const RM_HIT_REL: f32 = 0.001;     // hit threshold: 0.1% of travelled distance
const RM_HIT_ABS: f32 = 0.0004;    // floor, so the first steps cannot self-hit
const RM_SHADOW_STEPS: u32 = 24u;
const RM_SHADOW_FAR: f32 = 18.0;
const RM_NORMAL_EPS: f32 = 0.0012;
const RM_AO_STEPS: u32 = 4u;

// ---------------------------------------------------------------- types
struct RmRay { ro: vec3f, rd: vec3f };
struct RmHit { d: f32, steps: f32 };          // d < 0 means miss
struct RmCam { ro: vec3f, ta: vec3f, fov: f32 };
struct RmOut { col: vec3f, dist: f32, cov: f32 };  // cov: 1 hit, 0 miss

// Fragment output. Task 4 extends this struct; see the HOOK at the bottom.
struct RmFrag {
  @location(0) col: vec4f,
};

// ---------------------------------------------------------------- marcher

// Sphere-trace from ro along rd (rd normalised). maxSteps is clamped to
// RM_MAX_STEPS, and the walk stops at far. Returns the hit distance in .d and
// the number of steps taken in .steps; .d is negative on a miss. The loop is
// bounded by the constant RM_MAX_STEPS and exits early on the step budget, on
// the far plane, or as soon as the field is inside 0.001 * t.
fn rm_march(ro: vec3f, rd: vec3f, maxSteps: f32, far: f32) -> RmHit {
  var t = 0.0;
  var steps = 0.0;
  var hit = 0.0;
  for (var i = 0u; i < RM_MAX_STEPS; i++) {
    if (f32(i) >= maxSteps || t >= far) { break; }
    let d = map(ro + rd * t).x;
    if (d < RM_HIT_ABS + RM_HIT_REL * t) {
      hit = 1.0;
      steps = f32(i) + 1.0;
      break;
    }
    t += d;
    steps = f32(i) + 1.0;
  }
  return RmHit(select(-1.0, t, hit > 0.5), steps);
}

// Central-difference-free normal: four tetrahedron taps around p.
fn rm_normal(p: vec3f) -> vec3f {
  let a = RM_NORMAL_EPS;
  let b = -RM_NORMAL_EPS;
  let s1 = map(p + vec3f(a, b, b)).x;
  let s2 = map(p + vec3f(b, b, a)).x;
  let s3 = map(p + vec3f(b, a, b)).x;
  let s4 = map(p + vec3f(a, a, a)).x;
  let n = vec3f(a, b, b) * s1 + vec3f(b, b, a) * s2
        + vec3f(b, a, b) * s3 + vec3f(a, a, a) * s4;
  let l = length(n);
  return select(vec3f(0.0, 1.0, 0.0), n / max(l, 1e-9), l > 1e-9);
}

// Ambient occlusion from four samples along the normal (iq-style).
fn rm_ao(p: vec3f, n: vec3f) -> f32 {
  var occ = 0.0;
  var w = 1.0;
  for (var i = 1u; i <= RM_AO_STEPS; i++) {
    let h = 0.015 + 0.115 * f32(i);
    let d = map(p + n * h).x;
    occ += (h - d) * w;
    w *= 0.72;
  }
  return clamp(1.0 - 2.6 * occ, 0.0, 1.0);
}

// Soft shadow: march the light ray with a penumbra estimate. The loop is
// bounded by RM_SHADOW_STEPS; returns 0 (fully shadowed) .. 1 (lit).
fn rm_shadow(ro: vec3f, rd: vec3f) -> f32 {
  var res = 1.0;
  var t = 0.04;
  for (var i = 0u; i < RM_SHADOW_STEPS; i++) {
    let h = map(ro + rd * t).x;
    res = min(res, 7.0 * h / max(t, 1e-4));
    t += clamp(h, 0.03, 0.8);
    if (res < 0.02 || t > RM_SHADOW_FAR) { break; }
  }
  return clamp(res, 0.0, 1.0);
}

// ---------------------------------------------------------------- camera

// Key light direction (not a colour; colours only come from lut/albedo).
fn rm_light() -> vec3f {
  return normalize(vec3f(0.45, 0.82, 0.35));
}

// Look-at: uv is [-aspect, aspect] x [-1, 1] with y already pointing up and fov
// is the vertical field of view in radians. Returns a normalised ray direction.
fn rm_look(ro: vec3f, ta: vec3f, uv: vec2f, fov: f32) -> vec3f {
  let fw = normalize(ta - ro);
  let up0 = select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(fw.y) > 0.999);
  let rt = normalize(cross(up0, fw));
  let up = cross(fw, rt);
  let th = tan(clamp(fov, 0.05, 3.0) * 0.5);
  return normalize(rt * uv.x * th + up * uv.y * th + fw);
}

// Camera paths, driven by time and the beat. `id` selects the move:
//   0 orbit       slow revolution; radius and fov breathe with U.beat/bass
//   1 dolly       push in / pull out along Z; fov opens on the beat
//   2 flythrough  forward travel through the world, drifting laterally
//   3 spiral      high orbit looking straight down, widening on the beat
// kick is U.beat (clamped), and U.bass/U.mid add the slower audio breathing.
fn rm_cam_path(id: i32, t: f32) -> RmCam {
  let kick = clamp(U.beat, 0.0, 1.0);
  let pid = ((id % 4) + 4) % 4;
  var ro = vec3f(0.0, 1.1, 6.0);
  var ta = vec3f(0.0);
  var fov = 1.15;
  if (pid == 0) {
    let a = t * 0.23;
    let r = 5.0 + 0.7 * sin(t * 0.17) + 1.2 * kick * (0.5 + 0.5 * clamp(U.bass, 0.0, 1.0));
    ro = vec3f(cos(a) * r, 1.1 + 0.8 * sin(t * 0.31), sin(a) * r);
    ta = vec3f(0.0, 0.1 * sin(t * 0.5), 0.0);
    fov = 1.05 + 0.25 * kick;
  } else if (pid == 1) {
    let z = 1.8 + 6.2 * (0.5 + 0.5 * cos(t * 0.21));
    ro = vec3f(0.45 * sin(t * 0.4), 0.55 + 0.25 * sin(t * 0.27), z);
    ta = vec3f(0.0, 0.1, -2.0);
    fov = 1.1 + 0.3 * kick + 0.15 * clamp(U.mid, 0.0, 1.0);
  } else if (pid == 2) {
    ro = vec3f(0.6 * sin(t * 0.13), 0.9 + 0.3 * sin(t * 0.21), -t * 0.85);
    ta = ro + vec3f(0.35 * sin(t * 0.29), -0.1 * cos(t * 0.17), 1.0);
    fov = 1.2 + 0.2 * kick;
  } else {
    let a = t * 0.4;
    let r = 3.6 + 0.6 * sin(t * 0.19);
    ro = vec3f(cos(a) * r, 6.2 + 1.1 * sin(t * 0.23), sin(a) * r);
    ta = vec3f(0.0);
    fov = 1.0 + 0.3 * kick;
  }
  return RmCam(ro, ta, fov);
}

// Camera for one pixel. uv comes from XY(pos); this flips it because screen Y
// grows downwards and the camera frame grows upwards. `fov` scales the path's
// own field of view (1.0 = as authored, 0.7 tele, 1.4 wide).
fn rm_camera(uv: vec2f, fov: f32) -> RmRay {
  return rm_camera_at(uv, fov, i32(U.vari), U.T);
}

// Explicit-time/ID variant: for tests, offline renders and Task 4's depth pass.
fn rm_camera_at(uv: vec2f, fov: f32, id: i32, t: f32) -> RmRay {
  let cam = rm_cam_path(id, t);
  let rd = rm_look(cam.ro, cam.ta, vec2f(uv.x, -uv.y), cam.fov * max(fov, 0.05));
  return RmRay(cam.ro, rd);
}

// ---------------------------------------------------------------- shading

// Miss colour: a palette sky, low contrast so geometry reads against it.
fn rm_sky(rd: vec3f) -> vec3f {
  let h = clamp(0.5 + 0.5 * rd.y, 0.0, 1.0);
  return lut(0.58 + 0.22 * h + 0.04 * rd.x) * (0.10 + 0.22 * h);
}

// One ray: march, shade, fog. Keeps the hit distance even on a miss (far), so
// the depth target can be attached without recomputing anything.
fn rm_shade_ray(ro: vec3f, rd: vec3f) -> RmOut {
  let hit = rm_march(ro, rd, f32(RM_MAX_STEPS), RM_FAR);
  if (hit.d < 0.0) {
    return RmOut(rm_sky(rd), RM_FAR, 0.0);
  }
  let p = ro + rd * hit.d;
  let n = rm_normal(p);
  let mat = map(p).y;
  let ao = rm_ao(p, n);
  let sh = rm_shadow(p + n * 0.015, rm_light());
  let alb = shade(p, n, rd, mat, hit.steps);
  let diff = clamp(dot(n, rm_light()), 0.0, 1.0);
  let spec = pow(clamp(dot(reflect(rd, n), rm_light()), 0.0, 1.0), 28.0);
  var col = alb * (0.16 + 0.84 * diff * sh) * (0.30 + 0.70 * ao);
  col += alb * spec * sh * 0.35;
  let bg = rm_sky(rd);
  let fog = 1.0 - exp(-max(hit.d - 2.0, 0.0) * 0.028);
  col = mix(col, bg, clamp(fog, 0.0, 0.90));
  return RmOut(col, hit.d, 1.0);
}

// ---------------------------------------------------------------- entry
@fragment fn fs(@builtin(position) pos: vec4f) -> RmFrag {
  let ray = rm_camera(XY(pos), 1.0);
  let res = rm_shade_ray(ray.ro, ray.rd);
  let col = clamp(res.col, vec3f(0.0), vec3f(1.0));
  return RmFrag(vec4f(col, res.cov));
}

// ============================================================================
// HOOK (Task 4): second target for depth.
//
// The marcher already produces everything a depth attachment needs: `res.dist`
// is the hit distance (or RM_FAR on a miss) and `res.cov` is 0/1, both kept out
// of the write above. To attach a depth target, make exactly these changes:
//
//   1. add the member to RmFrag:
//          @location(1) depth: f32,
//   2. return it from fs (res.dist is the hit distance, already RM_FAR on a miss):
//          return RmFrag(vec4f(col, res.cov), res.dist);
//   3. in gpu.js, add the depth texture view to the render pass colour
//      attachments as target 1 and write linear view distance (world units,
//      not normalised depth) so a later pass can re-project it.
//
// Nothing else in this module has to change: no extra march, no second normal.
// ============================================================================
