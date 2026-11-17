// Block-matching motion estimation for one layer's datamosh, at half the sample count of
// `mosh_motion.wgsl` (owner: V-budget).
//
// `mosh_motion.wgsl` (V-mosh) compares all 64 pixels of a block against all 64 of a candidate. On
// the reference machine that pass measured 1.6 ms per layer per frame at 1080p - the whole frame
// budget's biggest single pass - because each of the 920 fragments walks up to 169 candidate
// offsets through a 64-step serial SAD chain, and 920 fragments do not fill the GPU.
//
// This pass keeps the web app's search (one vector per 8x8 block of the fixed 320x180 grid, a
// +-6 block search, the same 3-per-block motion penalty, the same early exit) and compares a 4x4
// subsample of the block instead: 16 of 64 pixels, every other row and column. A 2x2 subsample
// still spans the whole block, so the cost peak still sits on the true offset; what it loses is
// discrimination between offsets that differ only in a block's odd rows or columns, which moves a
// vector by a block (8 grid pixels) only on content that alternates every second pixel.
//
// The unit check in `tests/mosh.rs` (`motion_subsample_matches_the_full_search`) drives both
// shaders over the same synthetic frames and reports how often the vectors differ. (A vector is
// stored as `clamp(-offset)`, so one readback cannot resolve one negative offset from another:
// the check counts differing vectors rather than measuring a signed distance.)

struct MU {
    grid: vec2<f32>,
    pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> m: MU;
@group(0) @binding(1) var curT: texture_2d<f32>;
@group(0) @binding(2) var curS: sampler;
@group(0) @binding(3) var prvT: texture_2d<f32>;
@group(0) @binding(4) var prvS: sampler;

const BLOCK: f32 = 8.0;
const SEARCH: i32 = 6;
const LUMA: vec3<f32> = vec3<f32>(0.3, 0.59, 0.11);
/// Samples per block: a 4x4 grid of block pixels, every other row and column.
const SAMPLES: u32 = 16;
const STEP: f32 = 2.0;

fn lum_cur(p: vec2<f32>) -> f32 {
    return clamp(dot(textureSampleLevel(curT, curS, p / m.grid, 0.0).rgb, LUMA), 0.0, 1.0);
}

fn lum_prv(p: vec2<f32>) -> f32 {
    return clamp(dot(textureSampleLevel(prvT, prvS, p / m.grid, 0.0).rgb, LUMA), 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let origin = floor(pos.xy) * BLOCK;
    var current: array<f32, 16>;
    var best = 0.0;
    for (var i = 0u; i < SAMPLES; i = i + 1u) {
        let p = origin + vec2<f32>(f32(i % 4u), f32(i / 4u)) * STEP + vec2<f32>(0.5);
        current[i] = lum_cur(p);
        best = best + abs(current[i] - lum_prv(p)) * 255.0;
    }
    var best_offset = vec2<f32>(0.0, 0.0);
    if (best > 0.5) {
        for (var dy = -SEARCH; dy <= SEARCH; dy = dy + 1) {
            for (var dx = -SEARCH; dx <= SEARCH; dx = dx + 1) {
                if (dx == 0 && dy == 0) {
                    continue;
                }
                let offset = vec2<f32>(f32(dx), f32(dy));
                let block = origin + offset;
                if (block.x < 0.0 || block.y < 0.0 || block.x + BLOCK > m.grid.x || block.y + BLOCK > m.grid.y) {
                    continue;
                }
                var cost = 3.0 * (abs(offset.x) + abs(offset.y));
                for (var i = 0u; i < SAMPLES; i = i + 1u) {
                    let p = vec2<f32>(f32(i % 4u), f32(i / 4u)) * STEP + vec2<f32>(0.5);
                    cost = cost + abs(current[i] - lum_prv(block + p)) * 255.0;
                    if (cost >= best) {
                        break;
                    }
                }
                if (cost < best) {
                    best = cost;
                    best_offset = offset;
                }
            }
        }
    }
    return vec4<f32>(-best_offset, 0.0, 1.0);
}
