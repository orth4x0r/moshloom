// Scene 15's tile buckets: which streaks can reach which part of the screen.
//
// `swarm_fast.wgsl` accumulates the swarm in the fragment shader, and the loop over every particle
// is what costs: 176 texel reads plus a dozen operations in each of 518k pixels at 960x540. Almost
// none of those particles can touch the pixel. This pass answers that question once per tile
// instead of once per pixel: 16 x 9 tiles over the layer, and for each tile the list of particles
// whose bounding box overlaps it.
//
// The target is `BUCKET_TILES` wide and `1 + BUCKET_CAPACITY` tall:
//
//   row 0            the tile's candidate count, or CAPACITY + 1 for "too many, scan them all"
//   rows 1..CAPACITY  the candidate particle indices, in particle order (0 when unused)
//
// It is built by scanning the particle table per bucket texel, which sounds backwards and is not:
// 144 x 25 texels x 176 particles is 634k reads per frame, against 91M in the per-pixel loop.
//
// A tile holds every particle whose (width-padded) box overlaps it, plus one tile of slack on each
// side, so a tile can only ever have too many candidates, never miss one: the count says which,
// and the fragment shader falls back to scanning the whole table for a saturated tile. Tile
// indices are derived from the same uv `XY()` uses (`pos / res`), and the slack absorbs the
// floating-point difference between the two directions.

const TILES_X: i32 = 16;
const TILES_Y: i32 = 9;
const CAPACITY: i32 = 24;
const SWARM_MIN: i32 = 64;
const SWARM_MAX: i32 = 176;

struct SU { res: vec2f, T: f32, hue: f32, bass: f32, mid: f32, high: f32, pulse: f32, vari: f32, dens: f32, beat: f32, frame: f32, pal: f32, aux0: f32, aux1: f32, aux2: f32, l0: vec4f, l1: vec4f, l2: vec4f, mel: array<vec4f,6> };
@group(0) @binding(0) var<uniform> U: SU;
@group(0) @binding(1) var posT: texture_2d<f32>;

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let tile = i32(floor(pos.x));
  let row = i32(floor(pos.y));
  let tx = f32(tile % TILES_X);
  let ty = f32(tile / TILES_X);
  let asp = U.res.x / U.res.y;
  let count = SWARM_MIN + i32(f32(SWARM_MAX - SWARM_MIN) * clamp(U.dens, 0., 1.));
  var matches: i32 = 0;
  var picked: f32 = 0.;
  for (var k = 0; k < count; k = k + 1) {
    // The particle's screen-space box, in tiles, one tile of slack on each side.
    let box = textureLoad(posT, vec2i(k, 0), 0);
    let lo = (vec2f(box.x / asp, box.y) * 0.5 + 0.5) * vec2f(f32(TILES_X), f32(TILES_Y));
    let hi = (vec2f(box.z / asp, box.w) * 0.5 + 0.5) * vec2f(f32(TILES_X), f32(TILES_Y));
    let overlaps = tx >= floor(lo.x) - 1. && tx <= ceil(hi.x) + 1.
      && ty >= floor(lo.y) - 1. && ty <= ceil(hi.y) + 1.;
    if (overlaps) {
      if (matches == row - 1) {
        picked = f32(k);
      }
      matches = matches + 1;
    }
  }
  if (row == 0) {
    // A saturated tile is marked with CAPACITY + 1: the fragment shader then scans the whole
    // table for it, so no streak is ever dropped.
    return vec4f(f32(min(matches, CAPACITY + 1)), 0., 0., 0.);
  }
  return vec4f(picked, 0., 0., 0.);
}
