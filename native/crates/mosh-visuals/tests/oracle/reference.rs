//! The CPU reference: the web app's datamosh, ported from `gpu.js` and `engine.c`.
//!
//! Line references are to the committed `gpu.js` and `engine.c` at the root of the repository.
//! The three things being ported are:
//!
//! * [`motion`] — `WG_ME` (gpu.js:295-314), the block matcher that fills `G.mvT`;
//! * [`feedback`] — the `F.iframe < 0.5` branch of `WG_FB` (gpu.js:325-336), the feedback pass;
//! * [`decide`] — the mosh half of `engine.c`'s `decide()` (engine.c:330-360), which turns a
//!   clip's mosh amount and a transition into `iframe`/`skipres`/`leak`/`gain`/`coarse`.
//!
//! [`Web`] glues them together the way `gpuFrame()` does (gpu.js:878-939), so a test can feed a
//! scripted sequence of frames and read the picture a web canvas would show.

use super::{Frame, HEIGHT, WIDTH, sample_linear_clamp, sample_nearest_mirror};

/// The analysis grid, `GS` in `WG_ME` (gpu.js:296). Block matching always runs on this grid,
/// whatever size the picture is.
pub const GRID: [f32; 2] = [320.0, 180.0];

/// The luma weights `lumc`/`lump` use (gpu.js:297-298).
pub const LUMA: [f32; 3] = [0.3, 0.59, 0.11];

/// Grid pixels per block, `let o = floor(pos.xy) * 8.` (gpu.js:300).
pub const BLOCK: usize = 8;

/// The +-6 grid-pixel search (gpu.js:304).
pub const SEARCH: i32 = 6;

/// The vector texture: `G.mvT = tex(40, 23, 'rgba16float', RT)` (gpu.js:634).
pub const MOTION_WIDTH: usize = 40;
pub const MOTION_HEIGHT: usize = 23;

/// `WG_ME` clamps the stored vector to `vec2f(56. / 640., 56. / 360.)` (gpu.js:313), in uv units.
pub const MV_CLAMP: [f32; 2] = [56.0 / 640.0, 56.0 / 360.0];

/// The feedback pass clamps its result to `vec3f(-0.15)..vec3f(1.15)` (gpu.js:335).
pub const BUFFER_RANGE: [f32; 2] = [-0.15, 1.15];

/// `decide()`'s leak floor: `P.leak<0.004f?0.004f` (engine.c:349).
pub const LEAK_FLOOR: f32 = 0.004;

const GRID_WIDTH: usize = GRID[0] as usize;
const GRID_HEIGHT: usize = GRID[1] as usize;

/// `lumc`/`lump`: `clamp(dot(sample(curT, ls, p / GS), LUMA), 0, 1)` (gpu.js:297-298).
///
/// The reference's grid *is* the analysed picture, at [`GRID`]'s size, so a grid point is one
/// sample of it. `ls` is `G.linC`, a linear sampler with clamp-to-edge. Every point the search
/// visits is a texel centre (an integer offset from `origin + 0.5`), where the bilinear weights
/// are `(1, 0)`, so `at_clamped` is the same value; the assert is there to catch a search point
/// that would need interpolation.
pub fn luma_grid(img: &Frame) -> Vec<f32> {
    let mut grid = vec![0.0f32; GRID_WIDTH * GRID_HEIGHT];
    for y in 0..GRID_HEIGHT {
        for x in 0..GRID_WIDTH {
            let p = [x as f32 + 0.5, y as f32 + 0.5];
            let color = sample_linear_clamp(img, p[0] / GRID[0], p[1] / GRID[1]);
            let luma = color[0] * LUMA[0] + color[1] * LUMA[1] + color[2] * LUMA[2];
            grid[y * GRID_WIDTH + x] = luma.clamp(0.0, 1.0);
        }
    }
    grid
}

/// One luma sample of the grid at a grid-space point, clamp-to-edge.
fn luma_at(grid: &[f32], px: f32, py: f32) -> f32 {
    let x = px.floor();
    let y = py.floor();
    debug_assert_eq!(
        px - x,
        0.5,
        "a search point off a texel centre needs interpolation"
    );
    debug_assert_eq!(
        py - y,
        0.5,
        "a search point off a texel centre needs interpolation"
    );
    let x = (x as i64).clamp(0, GRID_WIDTH as i64 - 1) as usize;
    let y = (y as i64).clamp(0, GRID_HEIGHT as i64 - 1) as usize;
    grid[y * GRID_WIDTH + x]
}

/// `WG_ME` (gpu.js:299-314): one displacement per 8x8 block of the analysis grid.
///
/// Returns the field as the shader stores it: `mvT`'s texel `(bx, by)` holds
/// `clamp(bd / GS * gain, +-56/640, +-56/360)` in uv units, where `bd` is the *backward* offset
/// (the grid-space position this block's pixels came from last frame). `gain` is `F.gain`,
/// `1 + 2.2 * m` (engine.c:349).
pub fn motion(cur: &[f32], prv: &[f32], gain: f32) -> Vec<[f32; 2]> {
    let mut field = vec![[0.0f32; 2]; MOTION_WIDTH * MOTION_HEIGHT];
    for by in 0..MOTION_HEIGHT {
        for bx in 0..MOTION_WIDTH {
            let origin = [(bx * BLOCK) as f32, (by * BLOCK) as f32];
            // `var cur: array<f32, 64>` and the SAD at offset zero (gpu.js:300-301).
            let mut current = [0.0f32; BLOCK * BLOCK];
            let mut best = 0.0f32;
            for i in 0..BLOCK * BLOCK {
                let p = [
                    origin[0] + (i % BLOCK) as f32 + 0.5,
                    origin[1] + (i / BLOCK) as f32 + 0.5,
                ];
                current[i] = luma_at(cur, p[0], p[1]);
                best += (current[i] - luma_at(prv, p[0], p[1])).abs() * 255.0;
            }
            let mut bd = [0.0f32, 0.0f32];
            // `if (best > 0.5)`: a block that did not change is left with a zero vector.
            if best > 0.5 {
                for dy in -SEARCH..=SEARCH {
                    for dx in -SEARCH..=SEARCH {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let lo = [origin[0] + dx as f32, origin[1] + dy as f32];
                        // gpu.js:307 bounds the block against 184, not 180 (23 * 8), so the last
                        // block row may only search upwards. Ported as written.
                        if lo[0] < 0.0
                            || lo[1] < 0.0
                            || lo[0] + BLOCK as f32 > GRID[0]
                            || lo[1] + BLOCK as f32 > 184.0
                        {
                            continue;
                        }
                        // 3 luma units per grid pixel of displacement (gpu.js:308).
                        let mut cost = 3.0 * (dx.abs() + dy.abs()) as f32;
                        for i in 0..BLOCK * BLOCK {
                            let p = [
                                lo[0] + (i % BLOCK) as f32 + 0.5,
                                lo[1] + (i / BLOCK) as f32 + 0.5,
                            ];
                            cost += (current[i] - luma_at(prv, p[0], p[1])).abs() * 255.0;
                            if cost >= best {
                                break;
                            }
                        }
                        if cost < best {
                            best = cost;
                            bd = [dx as f32, dy as f32];
                        }
                    }
                }
            }
            let mv = [bd[0] / GRID[0] * gain, bd[1] / GRID[1] * gain];
            field[by * MOTION_WIDTH + bx] = [
                mv[0].clamp(-MV_CLAMP[0], MV_CLAMP[0]),
                mv[1].clamp(-MV_CLAMP[1], MV_CLAMP[1]),
            ];
        }
    }
    field
}

/// A clip transition, as far as the mosh pass cares: `engine.c:318`'s
/// `T_CUT`/`T_XFADE`/`T_MOSH`/`T_MELT`/`T_STROBE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transition {
    Cut,
    Xfade,
    Datamosh,
    Melt,
    Strobe,
}

/// One frame's input to [`decide`].
#[derive(Clone, Copy, Debug)]
pub struct Cue {
    /// The clip's mosh amount, `b->mosh`, 0..1.
    pub amount: f32,
    /// `Some(transition)` when this frame enters a new clip (`entered` in `decide()`).
    pub entered: Option<Transition>,
    /// `f < lastF || f > lastF + 8` (engine.c:336): playback jumped backwards, or a hitch left a
    /// hole.
    pub jump: bool,
}

/// The mosh-relevant fields of a frame's `Plan` (`engine.c:320`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    pub iframe: bool,
    pub skipres: bool,
    pub freeze: bool,
    pub leak: f32,
    pub gain: f32,
    pub coarse: bool,
}

/// The mosh half of `decide()` (engine.c:330-360).
///
/// `first_frame` stands in for `needI`: engine.c sets it for the first frame of a project, and
/// `gpuFrame` takes an I-frame whenever `G.valid` is false as well (gpu.js:893), which is part of
/// [`Web::step`] rather than here.
///
/// The I-frame rules ported are the ones the brief lists: the first frame, a time jump, a `Cut`
/// or `Strobe` entry, and `m == 0`. `Melt`'s freeze, the strobe selection, xfade and the audio
/// modulation of `m` are not, since none of them is the datamosh.
pub fn decide(cue: &Cue, first_frame: bool) -> Plan {
    let mut iframe = first_frame;
    if cue.jump {
        iframe = true;
    }
    let mut skipres = false;
    if let Some(transition) = cue.entered {
        // engine.c:337: `if(lastK<0||tr==T_CUT) iframe=1; else if(tr==T_STROBE){iframe=1;}
        // else if(tr!=T_XFADE) P.skipres=1;`
        match transition {
            Transition::Cut | Transition::Strobe => iframe = true,
            Transition::Xfade => {}
            Transition::Datamosh | Transition::Melt => skipres = true,
        }
    }
    let m = cue.amount.clamp(0.0, 1.0);
    let mut plan = Plan {
        iframe,
        skipres,
        freeze: false,
        leak: LEAK_FLOOR,
        gain: 1.0 + 2.2 * m,
        coarse: m > 0.85,
    };
    // engine.c:348-349: `q=1-m` ... `P.leak=q*q*q*q*q; if(P.leak<0.004f) P.leak=0.004f;`
    let q = 1.0 - m;
    let leak = q * q * q * q * q;
    if leak >= LEAK_FLOOR {
        plan.leak = leak;
    }
    // engine.c:350: `if(m<=0.001f) iframe=1;`
    if m <= 0.001 {
        plan.iframe = true;
    }
    plan
}

/// The P-frame branch of `WG_FB` (gpu.js:325-336): carry `buffer` along `field`, add the source's
/// residual over the previous frame, leak, clamp.
///
/// `res` is `F.res` (the canvas size), `mv_scale` is `G.mvScale` (gpu.js:902).
pub fn feedback(
    cur: &Frame,
    prv: &Frame,
    buffer: &Frame,
    field: &[[f32; 2]],
    plan: &Plan,
    res: [f32; 2],
    mv_scale: f32,
) -> Frame {
    assert_eq!((cur.width, cur.height), (prv.width, prv.height));
    assert_eq!((cur.width, cur.height), (buffer.width, buffer.height));
    let mut out = Frame::new(cur.width, cur.height);
    for y in 0..cur.height {
        for x in 0..cur.width {
            let uv = [(x as f32 + 0.5) / res[0], (y as f32 + 0.5) / res[1]];
            // gpu.js:326: `bc = clamp(vec2i(floor(uv * vec2f(40., 22.5))), vec2i(0), vec2i(39, 22))`
            // — a texel fetch, one 8x8 grid block per vector; `bc & vec2i(-2)` when coarse.
            let mut bc = [(uv[0] * 40.0).floor() as i64, (uv[1] * 22.5).floor() as i64];
            bc[0] = bc[0].clamp(0, MOTION_WIDTH as i64 - 1);
            bc[1] = bc[1].clamp(0, MOTION_HEIGHT as i64 - 1);
            if plan.coarse {
                bc[0] &= !1;
                bc[1] &= !1;
            }
            let mv = field[bc[1] as usize * MOTION_WIDTH + bc[0] as usize];
            // gpu.js:327: `d = round(textureLoad(mvT, bc, 0).xy * F.mvScale * F.res) / F.res`.
            // WGSL's `round` breaks ties to even, which is `f32::round_ties_even`.
            let d = [
                (mv[0] * mv_scale * res[0]).round_ties_even() / res[0],
                (mv[1] * mv_scale * res[1]).round_ties_even() / res[1],
            ];
            // gpu.js:332: `c = textureSampleLevel(bufT, ns, q / asp + 0.5, 0.).rgb`, which with
            // zoom and spin at their defaults is `uv + d` through the nearest mirror-repeat
            // sampler.
            let mut c = sample_nearest_mirror(buffer, uv[0] + d[0], uv[1] + d[1]);
            // `s` and `prv` are sampled at `uv`, a pixel centre, so they are the pixel itself.
            let s = cur.at(x, y);
            if !plan.skipres {
                // gpu.js:333: `c = c + s - prv; c = mix(c, s, F.leak);`
                let p = prv.at(x, y);
                for channel in 0..3 {
                    let residual = c[channel] + s[channel] - p[channel];
                    c[channel] = residual + (s[channel] - residual) * plan.leak;
                }
            }
            // gpu.js:335: `c = clamp(c, vec3f(-0.15), vec3f(1.15));`
            for channel in 0..3 {
                c[channel] = c[channel].clamp(BUFFER_RANGE[0], BUFFER_RANGE[1]);
            }
            out.set(x, y, c);
        }
    }
    out
}

/// The per-layer mosh state of the web's frame loop, plus the frame loop itself.
///
/// * `buffer` is `G.buf[G.bi]` — what the last feedback pass wrote;
/// * `previous` is `G.src[1 - G.si]` — last frame's scene;
/// * `grid`/`previous_grid` are the two frames' analysis grids (the web builds them inside
///   `WG_ME`; the native pass has a `mosh_grid` pass for the same thing);
/// * `mv_scale` is `G.mvScale` (gpu.js:902).
pub struct Web {
    buffer: Option<Frame>,
    previous: Option<Frame>,
    grid: Vec<f32>,
    previous_grid: Vec<f32>,
    mv_scale: f32,
    /// `F.res`: the canvas size. 320x180 for these tests.
    pub res: [f32; 2],
}

impl Default for Web {
    fn default() -> Web {
        Web::new()
    }
}

impl Web {
    pub fn new() -> Web {
        Web {
            buffer: None,
            previous: None,
            grid: vec![0.0; GRID_WIDTH * GRID_HEIGHT],
            previous_grid: vec![0.0; GRID_WIDTH * GRID_HEIGHT],
            mv_scale: 1.0,
            res: [WIDTH as f32, HEIGHT as f32],
        }
    }

    /// One `gpuFrame()` (gpu.js:878-939): returns the picture the output pass would scan out,
    /// i.e. the mosh buffer clamped into the video range.
    pub fn step(&mut self, cur: &Frame, plan: &Plan) -> Frame {
        // gpu.js:893: `const iframe = pl[1]>0 || !G.valid`.
        let iframe = plan.iframe || self.buffer.is_none();
        // gpu.js:902: `if(freeze) G.mvScale*=1.02; else if(!iframe) G.mvScale=1;`
        if plan.freeze {
            self.mv_scale *= 1.02;
        } else if !iframe {
            self.mv_scale = 1.0;
        }
        let grid = luma_grid(cur);
        let buffer = if iframe {
            // `WG_FB` with `F.iframe >= 0.5`: `c = s`, then the same clamp (gpu.js:324-336).
            cur.clamped(BUFFER_RANGE[0], BUFFER_RANGE[1])
        } else {
            let field = motion(&grid, &self.previous_grid, plan.gain);
            feedback(
                cur,
                self.previous
                    .as_ref()
                    .expect("a P-frame has a previous frame"),
                self.buffer.as_ref().expect("a P-frame has a buffer"),
                &field,
                plan,
                self.res,
                self.mv_scale,
            )
        };
        self.previous_grid = std::mem::replace(&mut self.grid, grid);
        self.previous = Some(cur.clone());
        self.buffer = Some(buffer.clone());
        // `WG_OUT` clamps the buffer into the video range before it is scanned out.
        buffer.clamped(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::clips::{self, Clip};
    use crate::oracle::max_abs_diff;

    #[test]
    fn the_grid_is_the_web_apps() {
        assert_eq!((GRID_WIDTH, GRID_HEIGHT), (320, 180));
        assert_eq!(MOTION_WIDTH * BLOCK, GRID_WIDTH);
        // 180 / 8 = 22.5, so the field has one more, half-covered row (gpu.js:634).
        assert_eq!(MOTION_HEIGHT * BLOCK, 184);
    }

    #[test]
    fn decide_carries_the_web_apps_numbers() {
        let plain = Cue {
            amount: 0.8,
            entered: None,
            jump: false,
        };
        let plan = decide(&plain, false);
        assert!(!plan.iframe);
        assert!(!plan.skipres);
        // (1 - 0.8)^5 = 3.2e-4, under the 0.004 floor, so every amount above ~0.67 sits on the
        // floor: 0.2^5 = 0.00032 < 0.004.
        assert_eq!(plan.leak, LEAK_FLOOR);
        assert!((plan.gain - 2.76).abs() < 1e-6);
        assert!(!plan.coarse);
        // m = 0.5: 0.5^5 = 0.03125, above the floor.
        let half = decide(
            &Cue {
                amount: 0.5,
                ..plain
            },
            false,
        );
        assert!((half.leak - 0.03125).abs() < 1e-7);
        assert_eq!(half.gain, 2.1);
        // m > 0.85 pairs the blocks when it reads the vector field.
        assert!(
            decide(
                &Cue {
                    amount: 0.86,
                    ..plain
                },
                false
            )
            .coarse
        );
        assert!(
            !decide(
                &Cue {
                    amount: 0.85,
                    ..plain
                },
                false
            )
            .coarse
        );
        // m == 0 is not a P-frame at all.
        assert!(
            decide(
                &Cue {
                    amount: 0.0,
                    ..plain
                },
                false
            )
            .iframe
        );
        assert!(
            decide(
                &Cue {
                    amount: 0.001,
                    ..plain
                },
                false
            )
            .iframe
        );
        assert!(
            !decide(
                &Cue {
                    amount: 0.002,
                    ..plain
                },
                false
            )
            .iframe
        );
        // A time jump, a cut and a strobe refresh; a Datamosh entry only skips the residual.
        assert!(
            decide(
                &Cue {
                    jump: true,
                    ..plain
                },
                false
            )
            .iframe
        );
        assert!(
            decide(
                &Cue {
                    entered: Some(Transition::Cut),
                    ..plain
                },
                false
            )
            .iframe
        );
        assert!(
            decide(
                &Cue {
                    entered: Some(Transition::Strobe),
                    ..plain
                },
                false
            )
            .iframe
        );
        let datamosh = decide(
            &Cue {
                entered: Some(Transition::Datamosh),
                ..plain
            },
            false,
        );
        assert!(!datamosh.iframe && datamosh.skipres);
        // Melt takes the same `tr != T_XFADE` branch, so it skips the residual too.
        let melt = decide(
            &Cue {
                entered: Some(Transition::Melt),
                ..plain
            },
            false,
        );
        assert!(!melt.iframe && melt.skipres);
        let xfade = decide(
            &Cue {
                entered: Some(Transition::Xfade),
                ..plain
            },
            false,
        );
        assert!(!xfade.iframe && !xfade.skipres);
        // The first frame of a project is an I-frame.
        assert!(decide(&plain, true).iframe);
    }

    /// The search finds the translation, with the web's sign: `bd` points *back* at where the
    /// block's pixels came from, so a square moving 3 pixels to the right answers `-3`.
    #[test]
    fn block_matching_finds_the_translation_and_its_sign() {
        let previous = clips::frame(Clip::A, 0);
        let current = clips::frame(Clip::A, 1);
        let previous_grid = luma_grid(&previous);
        let current_grid = luma_grid(&current);
        // At gain 1 the stored vector is the raw answer in uv units.
        let raw = motion(&current_grid, &previous_grid, 1.0);
        // Block (6, 10) covers x 48..56, y 80..88, inside the square on both frames.
        assert_eq!(
            raw[10 * MOTION_WIDTH + 6],
            [-3.0 / GRID[0], 0.0],
            "the block matcher did not find the 3-pixel rightward translation"
        );
        // A background block is unchanged, so the search is skipped and the vector is zero.
        assert_eq!(raw[0], [0.0, 0.0], "an unchanged block moved");
        // `F.gain = 1 + 2.2 * 0.8` scales the stored vector, which is what carries the picture
        // past the motion (gpu.js:313).
        let gained = motion(&current_grid, &previous_grid, 1.0 + 2.2 * 0.8);
        assert_eq!(
            gained[10 * MOTION_WIDTH + 6],
            [-3.0 / GRID[0] * (1.0 + 2.2 * 0.8), 0.0]
        );
        // And the feedback rounds that to whole pixels: 8 a frame, not 3.
        let d = (gained[10 * MOTION_WIDTH + 6][0] * WIDTH as f32).round_ties_even();
        assert_eq!(d, -8.0, "the gained vector did not round to 8 pixels");
    }

    #[test]
    fn mirror_repeat_wraps_like_the_sampler() {
        // 4-wide row 0, 1, 2, 3: mirror-repeat repeats the edge texel.
        let img = Frame::from_fn(4, 1, |x, _| [x as f32, 0.0, 0.0]);
        let at = |u: f32| sample_nearest_mirror(&img, u, 0.0)[0];
        // Texel centres: 0.125, 0.375, 0.625, 0.875.
        assert_eq!(at(0.125), 0.0);
        assert_eq!(at(0.875), 3.0);
        // One texel past the right edge mirrors back onto the last texel, then the one before it.
        assert_eq!(at(1.125), 3.0);
        assert_eq!(at(1.625), 1.0);
        // One texel before the left edge mirrors onto the first.
        assert_eq!(at(-0.125), 0.0);
        assert_eq!(at(-0.375), 1.0);
    }

    #[test]
    fn a_static_frame_is_an_identity_at_any_amount() {
        let frame = Frame::from_fn(WIDTH, HEIGHT, |x, y| {
            [x as f32 / WIDTH as f32, y as f32 / HEIGHT as f32, 0.4]
        });
        for amount in [0.2f32, 0.8, 1.0] {
            let mut web = Web::new();
            let mut out = Frame::new(WIDTH, HEIGHT);
            for index in 0..8 {
                let plan = decide(
                    &Cue {
                        amount,
                        entered: None,
                        jump: false,
                    },
                    index == 0,
                );
                out = web.step(&frame, &plan);
            }
            assert!(
                max_abs_diff(&out, &frame) <= 1e-6,
                "amount {amount}: a static picture drifted"
            );
        }
    }
}
