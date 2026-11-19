// Block-matching motion estimation for one layer's datamosh, at a quarter of the sample count of
// `mosh_motion.wgsl` (owner: V-budget).
//
// `mosh_motion.wgsl` (V-mosh) compares all 64 pixels of a block against all 64 of a candidate. On
// the reference machine that pass measured 1.6 ms per layer per frame at 1080p - the whole frame
// budget's biggest single pass - because each of the 920 fragments walks up to 169 candidate
// offsets through a 64-step serial SAD chain, and 920 fragments do not fill the GPU.
//
// This pass keeps the web app's search - one vector per 8x8 block of the fixed 320x180 grid, a
// +-6 block search, the same 3-per-grid-pixel motion penalty, the same bounds and the same early
// exit - and compares a 4x4 subsample of each block instead of all 64 pixels: 16 samples, every
// other row and column. A 4x4 subsample still spans the whole block, so the cost peak still sits
// on the true offset; what it loses is discrimination between offsets that differ only in a
// block's odd rows or columns, which moves a vector by a block (8 grid pixels) only on content
// that alternates every second pixel.
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
/// `gpu.js:307` bounds a candidate block against 184, not against the grid's 180: 180 / 8 = 22.5,
/// so the vector field has 23 rows and the web lets the last, half-covered row search as a whole
/// block. Ported as written.
const BLOCK_LIMIT_Y: f32 = 184.0;
/// Web motion penalty (3.0 per grid pixel in 0..255 luma) and zero-motion early-exit threshold
/// (0.5 in 0..255 luma), scaled into 0..1 so the inner loop avoids multiplying every sample by 255.
const PENALTY: f32 = 3.0 / 255.0;
const EARLY_EXIT: f32 = 0.5 / 255.0;

fn lum_cur(uv: vec2<f32>) -> f32 {
    return clamp(dot(textureSampleLevel(curT, curS, uv, 0.0).rgb, LUMA), 0.0, 1.0);
}

fn lum_prv(uv: vec2<f32>) -> f32 {
    return clamp(dot(textureSampleLevel(prvT, prvS, uv, 0.0).rgb, LUMA), 0.0, 1.0);
}

fn sad4x4(
    origin: vec2<f32>,
    offset: vec2<f32>,
    inv_grid: vec2<f32>,
    c0: vec4<f32>,
    c1: vec4<f32>,
    c2: vec4<f32>,
    c3: vec4<f32>,
    best: f32,
) -> f32 {
    let block = origin + offset;
    if (block.x < 0.0 || block.y < 0.0 || block.x + BLOCK > m.grid.x || block.y + BLOCK > BLOCK_LIMIT_Y) {
        return best;
    }
    var cost = PENALTY * (abs(offset.x) + abs(offset.y));
    if (cost >= best) {
        return best;
    }
    let uv0 = (block + vec2<f32>(0.5)) * inv_grid;
    let sx = 2.0 * inv_grid.x;
    let sy = 2.0 * inv_grid.y;

    let y0 = uv0.y;
    cost += abs(c0.x - lum_prv(vec2<f32>(uv0.x, y0)))
          + abs(c0.y - lum_prv(vec2<f32>(uv0.x + sx, y0)))
          + abs(c0.z - lum_prv(vec2<f32>(uv0.x + 2.0 * sx, y0)))
          + abs(c0.w - lum_prv(vec2<f32>(uv0.x + 3.0 * sx, y0)));
    if (cost >= best) {
        return best;
    }
    let y1 = y0 + sy;
    cost += abs(c1.x - lum_prv(vec2<f32>(uv0.x, y1)))
          + abs(c1.y - lum_prv(vec2<f32>(uv0.x + sx, y1)))
          + abs(c1.z - lum_prv(vec2<f32>(uv0.x + 2.0 * sx, y1)))
          + abs(c1.w - lum_prv(vec2<f32>(uv0.x + 3.0 * sx, y1)));
    if (cost >= best) {
        return best;
    }
    let y2 = y0 + 2.0 * sy;
    cost += abs(c2.x - lum_prv(vec2<f32>(uv0.x, y2)))
          + abs(c2.y - lum_prv(vec2<f32>(uv0.x + sx, y2)))
          + abs(c2.z - lum_prv(vec2<f32>(uv0.x + 2.0 * sx, y2)))
          + abs(c2.w - lum_prv(vec2<f32>(uv0.x + 3.0 * sx, y2)));
    if (cost >= best) {
        return best;
    }
    let y3 = y0 + 3.0 * sy;
    cost += abs(c3.x - lum_prv(vec2<f32>(uv0.x, y3)))
          + abs(c3.y - lum_prv(vec2<f32>(uv0.x + sx, y3)))
          + abs(c3.z - lum_prv(vec2<f32>(uv0.x + 2.0 * sx, y3)))
          + abs(c3.w - lum_prv(vec2<f32>(uv0.x + 3.0 * sx, y3)));
    return cost;
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let origin = floor(pos.xy) * BLOCK;
    let inv_grid = vec2<f32>(1.0) / m.grid;
    let uv0 = (origin + vec2<f32>(0.5)) * inv_grid;
    let sx = 2.0 * inv_grid.x;
    let sy = 2.0 * inv_grid.y;

    let y0 = uv0.y;
    let y1 = y0 + sy;
    let y2 = y0 + 2.0 * sy;
    let y3 = y0 + 3.0 * sy;
    let x0 = uv0.x;
    let x1 = x0 + sx;
    let x2 = x0 + 2.0 * sx;
    let x3 = x0 + 3.0 * sx;

    // Hold the 4x4 current-block luma samples in 4 vec4 registers (no dynamic array indexing).
    let c0 = vec4<f32>(
        lum_cur(vec2<f32>(x0, y0)),
        lum_cur(vec2<f32>(x1, y0)),
        lum_cur(vec2<f32>(x2, y0)),
        lum_cur(vec2<f32>(x3, y0)),
    );
    let c1 = vec4<f32>(
        lum_cur(vec2<f32>(x0, y1)),
        lum_cur(vec2<f32>(x1, y1)),
        lum_cur(vec2<f32>(x2, y1)),
        lum_cur(vec2<f32>(x3, y1)),
    );
    let c2 = vec4<f32>(
        lum_cur(vec2<f32>(x0, y2)),
        lum_cur(vec2<f32>(x1, y2)),
        lum_cur(vec2<f32>(x2, y2)),
        lum_cur(vec2<f32>(x3, y2)),
    );
    let c3 = vec4<f32>(
        lum_cur(vec2<f32>(x0, y3)),
        lum_cur(vec2<f32>(x1, y3)),
        lum_cur(vec2<f32>(x2, y3)),
        lum_cur(vec2<f32>(x3, y3)),
    );

    let p0 = vec4<f32>(
        lum_prv(vec2<f32>(x0, y0)),
        lum_prv(vec2<f32>(x1, y0)),
        lum_prv(vec2<f32>(x2, y0)),
        lum_prv(vec2<f32>(x3, y0)),
    );
    let p1 = vec4<f32>(
        lum_prv(vec2<f32>(x0, y1)),
        lum_prv(vec2<f32>(x1, y1)),
        lum_prv(vec2<f32>(x2, y1)),
        lum_prv(vec2<f32>(x3, y1)),
    );
    let p2 = vec4<f32>(
        lum_prv(vec2<f32>(x0, y2)),
        lum_prv(vec2<f32>(x1, y2)),
        lum_prv(vec2<f32>(x2, y2)),
        lum_prv(vec2<f32>(x3, y2)),
    );
    let p3 = vec4<f32>(
        lum_prv(vec2<f32>(x0, y3)),
        lum_prv(vec2<f32>(x1, y3)),
        lum_prv(vec2<f32>(x2, y3)),
        lum_prv(vec2<f32>(x3, y3)),
    );

    let ones = vec4<f32>(1.0);
    var best = dot(abs(c0 - p0) + abs(c1 - p1) + abs(c2 - p2) + abs(c3 - p3), ones);
    var best_offset = vec2<f32>(0.0, 0.0);

    if (best > EARLY_EXIT) {
        // Search outward by Chebyshev ring r = 1..=6. Every candidate on ring r has
        // |dx| + |dy| >= r, so once PENALTY * r >= best no outer candidate can improve on best.
        for (var r = 1; r <= SEARCH; r = r + 1) {
            if (PENALTY * f32(r) >= best) {
                break;
            }
            for (var dx = -r; dx <= r; dx = dx + 1) {
                let top = vec2<f32>(f32(dx), f32(-r));
                let ct = sad4x4(origin, top, inv_grid, c0, c1, c2, c3, best);
                if (ct < best) {
                    best = ct;
                    best_offset = top;
                }
                let bot = vec2<f32>(f32(dx), f32(r));
                let cb = sad4x4(origin, bot, inv_grid, c0, c1, c2, c3, best);
                if (cb < best) {
                    best = cb;
                    best_offset = bot;
                }
            }
            for (var dy = -(r - 1); dy <= r - 1; dy = dy + 1) {
                let left = vec2<f32>(f32(-r), f32(dy));
                let cl = sad4x4(origin, left, inv_grid, c0, c1, c2, c3, best);
                if (cl < best) {
                    best = cl;
                    best_offset = left;
                }
                let right = vec2<f32>(f32(r), f32(dy));
                let cr = sad4x4(origin, right, inv_grid, c0, c1, c2, c3, best);
                if (cr < best) {
                    best = cr;
                    best_offset = right;
                }
            }
        }
    }
    return vec4<f32>(-best_offset, 0.0, 1.0);
}
