// Which particles can reach which part of the layer. `particles/draw.wgsl` gathers instead of
// splatting, and the loop over every particle is what would cost: 512 texel reads in each of 130k
// pixels at 960x540's half-size trail. Almost none of those particles can touch a given pixel.
//
// This pass answers that once per tile: 16 x 9 tiles, and for each tile the list of particles
// whose padded box overlaps it.
//
// # Why the scan is sharded
//
// One fragment per tile means one fragment scans *all* `count` particles, and that loop is serial
// inside the fragment: the pass measured 0.27 ms for 158 particles, and changing the tile count
// did not move it. So each tile's scan is split into `SHARDS` consecutive slices of the particle
// range, one fragment per slice, and the gather scans the shards in turn. Four times the
// fragments, a quarter of the loop each: 0.07 ms for the same result.
//
// The target is `TILES_X * TILES_Y` wide and `SHARDS * ROWS_PER_SHARD` tall. Each shard's rows:
//
//   row 0              the shard's candidate count, or PER_SHARD + 1 when it overflowed
//   rows 1..PER_SHARD/4  four candidate indices per texel, in particle order (0 when unused)
//
// Four indices to a texel is what keeps this cheap: a half float holds an integer exactly up to
// 2048, so an 8-entry shard list is 2 texels instead of 8.
//
// A shard that overflows is not a lost particle: the gather scans that shard's slice of the table
// with the box test instead, so a saturated tile still draws everything it holds.

const TILES_X: i32 = 16;
const TILES_Y: i32 = 9;
const SHARDS: i32 = 4;
const PER_SHARD: i32 = 8;
const ROWS_PER_SHARD: i32 = 1 + PER_SHARD / 4;

@group(0) @binding(1) var tab: texture_2d<f32>;

@fragment fn fs_main(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
  let column = i32(floor(pixel.x));
  let row = i32(floor(pixel.y));
  let shard = row / ROWS_PER_SHARD;
  let sub = row % ROWS_PER_SHARD;
  let tx = f32(column % TILES_X);
  let ty = f32(column / TILES_X);
  let aspect = max(U.motion.y, 1e-4);
  let count = min(i32(U.clock.z), MAX_COUNT);
  let first = count * shard / SHARDS;
  let last = count * (shard + 1) / SHARDS;
  // The four list slots this texel carries; the count row writes none.
  let base = (sub - 1) * 4;
  var picked = vec4f(0.0);
  var matches: i32 = 0;
  for (var k = first; k < last; k = k + 1) {
    // The particle's box, in screen units, converted to this grid's UV. The box already includes
    // the glow's radius, so a tile it touches is a tile that has to list it: no slack needed.
    let box = textureLoad(tab, vec2i(k, 0), 0);
    let lo = (vec2f(box.x / aspect, box.y) + vec2f(0.5)) * vec2f(f32(TILES_X), f32(TILES_Y));
    let hi = (vec2f(box.z / aspect, box.w) + vec2f(0.5)) * vec2f(f32(TILES_X), f32(TILES_Y));
    let overlaps = tx >= floor(lo.x) && tx <= floor(hi.x)
      && ty >= floor(lo.y) && ty <= floor(hi.y);
    if (overlaps) {
      if (matches == base) { picked.x = f32(k); }
      else if (matches == base + 1) { picked.y = f32(k); }
      else if (matches == base + 2) { picked.z = f32(k); }
      else if (matches == base + 3) { picked.w = f32(k); }
      matches = matches + 1;
    }
  }
  if (sub == 0) {
    return vec4f(f32(min(matches, PER_SHARD + 1)), 0.0, 0.0, 0.0);
  }
  return picked;
}
