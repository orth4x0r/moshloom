//! The synthetic clips the tests are fed, generated here and on the GPU from the same
//! arithmetic.
//!
//! The `MoshPass::apply` cases draw their input with [`SHADER`]; the reference uses [`frame`]. Both
//! must produce the same picture, so the pattern is written twice — once in WGSL, once in Rust —
//! and `synthetic_clips_match_on_the_gpu` in `tests/mosh_oracle.rs` proves the two agree to within
//! one 8-bit level. That is why the square's texture is an *integer* hash: `fract(sin(dot(...)))`
//! differs in the last bits between a CPU and a GPU, and a cell that flips value is a 0.5 error in
//! a 3-pixel block, which would look like a mosh difference later.
//!
//! The clip-change cases are driven through `Visuals::draw_layer`, whose only way to put arbitrary
//! pixels into a layer is the overlay path — an 8-bit BGRA upload ([`bgra`]). Those cases are fed
//! [`quantized`], the same pictures rounded to the 8 bits the upload carries, so the reference and
//! the GPU see the same bytes.

use super::{Frame, HEIGHT, WIDTH};

/// Which clip a frame belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Clip {
    /// A textured square translating 3 grid pixels per frame over a gradient. The smear and
    /// leak cases: the 3-pixel cells line up with the motion, so the block matcher has one
    /// unambiguous answer.
    A,
    /// Something else entirely: dark blue stripes moving the other way. For the clip changes.
    B,
    /// A 1-pixel checkerboard inside a rectangle that moves **1 pixel** a frame and reaches the
    /// bottom of the picture. The matcher's hard case: 1 px/frame is not a divisor of the 8-pixel
    /// block grid, a 1-pixel checker only matches at the exact offset, and the last block row
    /// needs the web's odd 184 bound to search at all.
    C,
}

/// The shader that draws one frame of a clip; its uniform is [`uniform`] (16 bytes).
pub const SHADER: &str = r#"
struct Params {
    clip: f32,
    t: f32,
    pad0: f32,
    pad1: f32,
};

@group(0) @binding(0) var<uniform> params: Params;

// Integer arithmetic only, so the Rust twin below agrees bit for bit.
fn cell(i: i32, j: i32) -> f32 {
    let a = u32(i) * 374761393u;
    let b = u32(j) * 668265263u;
    var h = (a ^ b) * 1274126177u;
    h = h ^ (h >> 16u);
    return f32(h & 255u) / 255.0;
}

// Clip A: a 64x64 square of 3-pixel random cells, 3 pixels to the right per frame, over a
// gradient. The cells are 3 pixels wide because the square moves 3 pixels a frame: at the true
// offset every cell lines up exactly, so the block matcher has one clean answer.
fn clip_a(px: f32, py: f32) -> vec3<f32> {
    var c = vec3<f32>(px / 320.0, py / 180.0, 0.25);
    let x0 = 32.0 + 3.0 * params.t;
    let y0 = 48.0;
    if (px >= x0 && px < x0 + 64.0 && py >= y0 && py < y0 + 64.0) {
        let i = i32(floor((px - x0) / 3.0));
        let j = i32(floor((py - y0) / 3.0));
        c = vec3<f32>(cell(i, j), cell(i + 101, j + 57), 0.85);
    }
    return c;
}

// Clip B: 6-pixel stripes with 12-pixel bands, 2 pixels to the left per frame, over a blue
// gradient. Nothing it draws resembles clip A, so a carried-over picture is obvious.
fn clip_b(px: f32, py: f32) -> vec3<f32> {
    var c = vec3<f32>(0.10, py / 360.0, 0.55);
    let x0 = 232.0 - 2.0 * params.t;
    let y0 = 30.0;
    if (px >= x0 && px < x0 + 48.0 && py >= y0 && py < y0 + 120.0) {
        let i = i32(floor((px - x0) / 6.0));
        let j = i32(floor((py - y0) / 12.0));
        if ((i + j) % 2 == 0) {
            c = vec3<f32>(0.95, 0.10, 0.35);
        } else {
            c = vec3<f32>(0.05, 0.20, 0.90);
        }
    }
    return c;
}

// Clip C: a 1-pixel checkerboard attached to a 96-wide rectangle, 1 pixel to the right per frame,
// running from y = 76 to the bottom edge so the last 8-pixel block row has motion in it. A flat
// background leaves the matcher's early exit to work on most of the frame.
fn clip_c(px: f32, py: f32) -> vec3<f32> {
    var c = vec3<f32>(0.45, 0.45, 0.45);
    let x0 = 48.0 + params.t;
    let y0 = 76.0;
    if (px >= x0 && px < x0 + 96.0 && py >= y0 && py < 180.0) {
        let i = i32(floor(px - x0));
        let j = i32(floor(py - y0));
        if ((i + j) % 2 == 0) {
            c = vec3<f32>(0.05, 0.05, 0.05);
        } else {
            c = vec3<f32>(0.95, 0.95, 0.95);
        }
    }
    return c;
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    var c = clip_b(pos.x, pos.y);
    if (params.clip < 0.5) {
        c = clip_a(pos.x, pos.y);
    } else if (params.clip > 1.5) {
        c = clip_c(pos.x, pos.y);
    }
    return vec4<f32>(c, 1.0);
}
"#;

/// The `Params` block for [`SHADER`]: 0 = A, 1 = B, 2 = C; the frame index; two pad floats.
pub fn uniform(clip: Clip, t: i32) -> Vec<u8> {
    let index = match clip {
        Clip::A => 0.0,
        Clip::B => 1.0,
        Clip::C => 2.0,
    };
    let values = [index, t as f32, 0.0, 0.0];
    let mut bytes = Vec::with_capacity(16);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// One frame of `clip` on the CPU: the Rust twin of [`SHADER`].
pub fn frame(clip: Clip, t: i32) -> Frame {
    Frame::from_fn(WIDTH, HEIGHT, |x, y| pixel(clip, x, y, t))
}

/// The same picture, frozen: the "still frame" the leak test switches to, so the motion field is
/// zero everywhere after the first frame and only the leak moves the buffer.
pub fn still(clip: Clip) -> Frame {
    frame(clip, 0)
}

/// One frame rounded to 8 bits per channel: exactly the picture [`bgra`] carries to the GPU.
///
/// The clip-change cases run through the overlay path, whose upload is `B8G8R8A8_UNORM`. The
/// reference has to be fed what the GPU is actually given, or a comparison would also be measuring
/// the quantisation.
pub fn quantized(clip: Clip, t: i32) -> Frame {
    Frame::from_fn(WIDTH, HEIGHT, |x, y| {
        let color = pixel(clip, x, y, t);
        [to_float(color[0]), to_float(color[1]), to_float(color[2])]
    })
}

/// One frame as the overlay path's upload: `B8G8R8A8_UNORM`, rows tightly packed, top row first.
pub fn bgra(clip: Clip, t: i32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(WIDTH * HEIGHT * 4);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let color = pixel(clip, x, y, t);
            for channel in [2, 1, 0] {
                bytes.push(to_byte(color[channel]));
            }
            bytes.push(255);
        }
    }
    bytes
}

/// One pixel of one clip, at pixel `(x, y)` on frame `t`, in 0..1 RGB. The fragment shader's
/// `pos.xy` is the pixel centre, so this is where `+ 0.5` comes from.
fn pixel(clip: Clip, x: usize, y: usize, t: i32) -> [f32; 3] {
    let px = x as f32 + 0.5;
    let py = y as f32 + 0.5;
    match clip {
        Clip::A => clip_a(px, py, t),
        Clip::B => clip_b(px, py, t),
        Clip::C => clip_c(px, py, t),
    }
}

fn to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn to_float(value: f32) -> f32 {
    f32::from(to_byte(value)) / 255.0
}

fn clip_a(px: f32, py: f32, t: i32) -> [f32; 3] {
    let mut color = [px / 320.0, py / 180.0, 0.25];
    let x0 = 32.0 + 3.0 * t as f32;
    let y0 = 48.0;
    if px >= x0 && px < x0 + 64.0 && py >= y0 && py < y0 + 64.0 {
        let i = ((px - x0) / 3.0).floor() as i32;
        let j = ((py - y0) / 3.0).floor() as i32;
        color = [cell(i, j), cell(i + 101, j + 57), 0.85];
    }
    color
}

fn clip_b(px: f32, py: f32, t: i32) -> [f32; 3] {
    let mut color = [0.10, py / 360.0, 0.55];
    let x0 = 232.0 - 2.0 * t as f32;
    let y0 = 30.0;
    if px >= x0 && px < x0 + 48.0 && py >= y0 && py < y0 + 120.0 {
        let i = ((px - x0) / 6.0).floor() as i32;
        let j = ((py - y0) / 12.0).floor() as i32;
        color = if (i + j) % 2 == 0 {
            [0.95, 0.10, 0.35]
        } else {
            [0.05, 0.20, 0.90]
        };
    }
    color
}

fn clip_c(px: f32, py: f32, t: i32) -> [f32; 3] {
    let mut color = [0.45, 0.45, 0.45];
    let x0 = 48.0 + t as f32;
    let y0 = 76.0;
    if px >= x0 && px < x0 + 96.0 && py >= y0 && py < 180.0 {
        let i = (px - x0).floor() as i32;
        let j = (py - y0).floor() as i32;
        color = if (i + j) % 2 == 0 {
            [0.05, 0.05, 0.05]
        } else {
            [0.95, 0.95, 0.95]
        };
    }
    color
}

/// The GPU's `cell()`, in Rust: wrapping 32-bit integer arithmetic and nothing else.
fn cell(i: i32, j: i32) -> f32 {
    let a = (i as u32).wrapping_mul(374_761_393);
    let b = (j as u32).wrapping_mul(668_265_263);
    let mut h = (a ^ b).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    f32::from((h & 255) as u8) / 255.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_clips_actually_move() {
        // Clip A: 3 pixels to the right per frame, so the square's left edge is at 32 + 3t.
        let a0 = frame(Clip::A, 0);
        let a1 = frame(Clip::A, 1);
        assert_ne!(a0, a1, "clip A is static");
        // Clip B: 2 pixels to the left per frame.
        assert_ne!(frame(Clip::B, 0), frame(Clip::B, 1), "clip B is static");
        // Clip C: 1 pixel to the right per frame.
        assert_ne!(frame(Clip::C, 0), frame(Clip::C, 1), "clip C is static");
        // And the clips look nothing like each other, which is what the carry-over test needs: a
        // large difference to carry.
        assert!(super::super::mean_abs_diff(&a0, &frame(Clip::B, 0)) > 0.1);
    }

    /// Clip C has to reach the last block row (y 176..180) and carry 1-pixel detail, or the case
    /// tests nothing the other two clips do not.
    #[test]
    fn clip_c_reaches_the_bottom_block_row_with_one_pixel_detail() {
        let picture = frame(Clip::C, 0);
        // Two neighbouring pixels on the last row that the matcher samples (y = 179) differ: the
        // checker is 1 pixel.
        let left = picture.at(60, 179);
        let right = picture.at(61, 179);
        assert_ne!(left, right, "clip C's checker is not 1 pixel");
        // The last block row (y = 176) is inside the rectangle, not background.
        assert!(
            super::super::mean_abs_diff(
                &Frame::from_fn(1, 1, |_, _| picture.at(60, 176)),
                &Frame::from_fn(1, 1, |_, _| [0.45, 0.45, 0.45])
            ) > 0.0,
            "clip C's rectangle does not reach the last block row"
        );
    }

    #[test]
    fn bgra_and_quantized_are_the_same_picture() {
        for clip in [Clip::A, Clip::B, Clip::C] {
            let bytes = bgra(clip, 3);
            assert_eq!(bytes.len(), WIDTH * HEIGHT * 4);
            let mut restored = Frame::new(WIDTH, HEIGHT);
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    let at = (y * WIDTH + x) * 4;
                    restored.set(
                        x,
                        y,
                        [
                            f32::from(bytes[at + 2]) / 255.0,
                            f32::from(bytes[at + 1]) / 255.0,
                            f32::from(bytes[at]) / 255.0,
                        ],
                    );
                }
            }
            assert_eq!(restored, quantized(clip, 3), "BGRA round trip");
            // And the quantisation is small: the clips are 8-bit-ish already.
            assert!(super::super::max_abs_diff(&restored, &frame(clip, 3)) <= 0.5 / 255.0 + 1e-6);
        }
    }

    #[test]
    fn the_hash_is_bounded_and_spread() {
        let mut seen = [false; 256];
        for i in 0..32 {
            for j in 0..32 {
                let value = cell(i, j);
                assert!((0.0..=1.0).contains(&value));
                seen[(value * 255.0).round() as usize] = true;
            }
        }
        assert!(
            seen.iter().filter(|hit| **hit).count() > 200,
            "the texture hash is not spread over the range"
        );
    }
}
