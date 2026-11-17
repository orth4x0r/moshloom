// The picture: every pixel of the trail gathers the particles its tile can hold, adds their glow
// on top of last frame's trail times the decay, and writes the sum. This is the layer's trail
// buffer as well as its picture before the present pass, so the pass reads and writes the same
// ping-pong target (mosh-render hands a pass the previous frame's image and clears the other).
//
// The gather is what `mosh-render` needs instead of an instanced splat: one 3-vertex draw, no
// blending, no scatter. See `particles/buckets.wgsl` for the tile lists and their shards.
//
// This pass is the layer's most expensive one and it is per *pixel*, not per particle, so it stays
// deliberately cheap: two texel loads and about a dozen operations per candidate — position,
// radius, energy and colour all come from the table — plus one filtered read of the trail, and no
// square root and no `exp` anywhere.

const TILES_X: i32 = 16;
const TILES_Y: i32 = 9;
const SHARDS: i32 = 4;
const PER_SHARD: i32 = 8;
const ROWS_PER_SHARD: i32 = 1 + PER_SHARD / 4;

@group(0) @binding(1) var tab: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var buc: texture_2d<f32>;
@group(0) @binding(5) var previous: texture_2d<f32>;

/// Two tones around the hue wheel without a branch: a cosine palette (the cheap kind demo shaders
/// use), offset by the particle's own second-hue flag. `hue` is the clip's hue, `second` 0 or 1.
fn tint(hue: f32, second: f32) -> vec3f {
  let h = hue + second * 0.38;
  let ramp = cos((h + vec3f(0.0, 0.33, 0.67)) * TAU);
  return clamp(0.5 + 0.5 * ramp, vec3f(0.0), vec3f(1.0));
}

/// One particle's light at `q` (screen units): a bright core inside a soft round halo, tapered
/// along its capsule so a moving spark has a tail. Both reach exactly zero at the radius the box
/// in the table was padded by.
fn spark(q: vec2f, k: i32) -> vec3f {
  let ends = textureLoad(tab, vec2i(k, 1), 0);
  let part = textureLoad(tab, vec2i(k, 2), 0);
  let radius = part.x;
  if (radius <= 0.0) {
    return vec3f(0.0);
  }
  let along_vec = ends.xy - ends.zw;
  let len2 = max(dot(along_vec, along_vec), 1e-9);
  let rel = q - ends.zw;
  let along = clamp(dot(rel, along_vec) / len2, 0.0, 1.0);
  let offset = rel - along_vec * along;
  let d2 = dot(offset, offset);
  let ir2 = 1.0 / (radius * radius);
  let disc = max(0.0, 1.0 - d2 * ir2);
  let core = max(0.0, 1.0 - d2 * ir2 * 7.0);
  // Brighter at the head of the capsule, so a spark reads as moving.
  let glow = disc * disc * (0.6 + 0.4 * along) + core * core * 0.9;
  return tint(part.z, part.w) * (part.y * glow);
}

/// Every particle this pixel's tile lists, shard by shard. A shard whose list overflowed is
/// scanned from the table over its own slice, so a saturated tile loses nothing.
fn gather(column: i32, q: vec2f, count: i32) -> vec3f {
  var light = vec3f(0.0);
  for (var shard = 0; shard < SHARDS; shard = shard + 1) {
    let row = shard * ROWS_PER_SHARD;
    let candidates = i32(textureLoad(buc, vec2i(column, row), 0).x);
    if (candidates > PER_SHARD) {
      let first = count * shard / SHARDS;
      let last = count * (shard + 1) / SHARDS;
      for (var k = first; k < last; k = k + 1) {
        let box = textureLoad(tab, vec2i(k, 0), 0);
        if (q.x < box.x || q.y < box.y || q.x > box.z || q.y > box.w) {
          continue;
        }
        light = light + spark(q, k);
      }
    } else {
      // Four indices to a texel (see `particles/buckets.wgsl`). Every index a list holds overlaps
      // this tile by construction, so there is no box test here.
      let slots = (candidates + 3) / 4;
      for (var slot = 0; slot < slots; slot = slot + 1) {
        let packed = textureLoad(buc, vec2i(column, row + 1 + slot), 0);
        let base = slot * 4;
        light = light + spark(q, i32(packed.x));
        if (base + 1 < candidates) {
          light = light + spark(q, i32(packed.y));
        }
        if (base + 2 < candidates) {
          light = light + spark(q, i32(packed.z));
        }
        if (base + 3 < candidates) {
          light = light + spark(q, i32(packed.w));
        }
      }
    }
  }
  return light;
}

@fragment fn fs_main(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
  let res = U.res.zw;
  let uv = pixel.xy / res;
  let q = (uv - vec2f(0.5)) * vec2f(U.motion.y, 1.0);
  let tile = clamp(
    vec2i(i32(pixel.x * U.tiles.x / res.x), i32(pixel.y * U.tiles.y / res.y)),
    vec2i(0, 0),
    vec2i(TILES_X - 1, TILES_Y - 1),
  );
  let column = tile.x + tile.y * TILES_X;
  let count = min(i32(U.clock.z), MAX_COUNT);
  let light = gather(column, q, count);

  let before = textureSampleLevel(previous, lin, uv, 0.0).rgb;
  // A longer trail accumulates more frames of light, so this frame's light is scaled by
  // `1 - decay` (normalised so the shortest trail is unchanged): `p5` then lengthens the smears
  // instead of saturating the picture.
  let accumulate = (1.0 - U.look.x) * 5.0;
  var out = before * U.look.x + light * accumulate;
  out = min(out, vec3f(6.0));
  // A soft vignette keeps the corners from collecting a haze.
  let edge = length((uv - vec2f(0.5)) * 2.0);
  out = out * (1.0 - U.grade.w * smoothstep(0.65, 1.45, edge));
  return vec4f(max(out, vec3f(0.0)), 1.0);
}
