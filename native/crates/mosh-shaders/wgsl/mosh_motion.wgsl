// Block-matching motion estimation for one layer's mosh pass.
// Owner: V-mosh (Task 2.2). Ported from `WG_ME` in gpu.js: one vector per 8x8 block of a fixed
// 320x180 luminance grid, searched +-6 grid pixels against the previous frame, with the web's SAD
// cost and its 3-per-pixel motion penalty. The block grid is the web's, not the layer's internal
// size, so this pass costs the same at every resolution (as it does in the browser).
//
// The web stores the *backward* sample offset (`bd`: where this block's pixels came from). This
// pass stores the *forward* displacement instead - analysis-grid pixels per frame, the same unit
// and sign as `FluidVisualizer::velocity` - and the feedback pass subtracts it.
//
// Bindings follow mosh-render's descriptor layout: the uniform at 0, then one sampled image and
// one sampler per input.

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

fn lum_cur(p: vec2<f32>) -> f32 {
    return clamp(dot(textureSampleLevel(curT, curS, p / m.grid, 0.0).rgb, LUMA), 0.0, 1.0);
}

fn lum_prv(p: vec2<f32>) -> f32 {
    return clamp(dot(textureSampleLevel(prvT, prvS, p / m.grid, 0.0).rgb, LUMA), 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let origin = floor(pos.xy) * BLOCK;
    var current: array<f32, 64>;
    var best = 0.0;
    for (var i = 0; i < 64; i = i + 1) {
        let p = origin + vec2<f32>(f32(i % 8), f32(i / 8)) + vec2<f32>(0.5);
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
                for (var i = 0; i < 64; i = i + 1) {
                    let p = vec2<f32>(f32(i % 8), f32(i / 8)) + vec2<f32>(0.5);
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
