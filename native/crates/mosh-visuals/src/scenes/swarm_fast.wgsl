// Scene 15 (swarm), second generation, on top of `prelude.wgsl`.
//
// The first generation (`swarm.wgsl`, kept and still compiled by this file's perf test) is the web
// scene's streaks accumulated in the fragment shader over a 176-particle table, and it measures
// 1.6-4.5 ms at 960x540 on the integrated GPU: most of a layer's share of the frame budget. It is
// the same streaks, the same widths, the same colours and the same falloff; what changed is where
// the per-particle work happens, and how many particles a pixel has to look at.
//
//   * `swarm_pos_fast.wgsl` stores three rows per particle: its screen-space bounding box, its head
//     and tail, and its finished width and colour. The old shader computed the width with a `sin`
//     (`h1`) and the colour with a palette sample plus a mel lookup *per pixel per particle* — 176
//     sines and 176 samples in each of 518k pixels.
//   * `swarm_buckets.wgsl` narrows the loop: each of 16 x 9 tiles carries the list of particles
//     whose box can reach it, so a pixel looks at the handful that could cover it instead of all
//     176. A tile with more candidates than a list holds is marked saturated and scanned in full,
//     so the result never loses a streak.
//   * A pixel's place inside a streak is measured in units of the streak's own length, so
//     "outside" is a comparison of squares and no square root is taken at all: the width the table
//     stores is already `w |u|`.
//
// The accumulation itself is unchanged: additive, premultiplied by the tapered falloff
// `(1 - d^2) * (0.25 + 0.75 u)` squared, with the 0.55..1 smoothstep cover.

const SWARM_MIN: i32 = 64;
const SWARM_MAX: i32 = 176;
const TILES_X: i32 = 16;
const TILES_Y: i32 = 9;
const CAPACITY: i32 = 24;

@group(0) @binding(11) var posT: texture_2d<f32>;
@group(0) @binding(13) var bucT: texture_2d<f32>;

/// One streak's contribution at `q`, from the particle table's rows 1 and 2.
fn streak(q: vec2f, k: i32) -> vec3f {
  let ends = textureLoad(posT, vec2i(k, 1), 0);
  let part = textureLoad(posT, vec2i(k, 2), 0);
  let u = ends.xy - ends.zw;
  let wl = part.x;
  let rel = q - ends.zw;
  // |across| |u| against (w |u|)^2: the band test without a root. A degenerate streak has
  // wl = 0 and is rejected here.
  let area = rel.x * u.y - rel.y * u.x;
  if (area * area >= wl * wl) {
    return vec3f(0.);
  }
  let len2 = dot(u, u);
  let along = dot(rel, u);
  if (along < -wl || along > len2 + wl) {
    return vec3f(0.);
  }
  let d = abs(area) / wl;
  let cover = 1. - smoothstep(0.55, 1., d);
  let u_end = clamp(along / len2, 0., 1.);
  let falloff = max(0., (1. - d * d) * (0.25 + 0.75 * u_end));
  return part.yzw * falloff * falloff * cover;
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let count = SWARM_MIN + i32(f32(SWARM_MAX - SWARM_MIN) * clamp(U.dens, 0., 1.));
  let tile = clamp(
    vec2i(i32(pos.x * f32(TILES_X) / U.res.x), i32(pos.y * f32(TILES_Y) / U.res.y)),
    vec2i(0, 0),
    vec2i(TILES_X - 1, TILES_Y - 1),
  );
  let column = tile.x + tile.y * TILES_X;
  let candidates = i32(textureLoad(bucT, vec2i(column, 0), 0).x);
  var c = vec3f(0.);
  if (candidates > CAPACITY) {
    // A saturated tile: scan the whole table rather than lose a streak.
    for (var k = 0; k < count; k = k + 1) {
      c += streak(q, k);
    }
  } else {
    for (var s = 0; s < candidates; s = s + 1) {
      let k = i32(textureLoad(bucT, vec2i(column, 1 + s), 0).x);
      c += streak(q, k);
    }
  }
  return fin(c);
}
