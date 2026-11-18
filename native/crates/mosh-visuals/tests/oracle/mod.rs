//! An independent oracle for the native datamosh.
//!
//! `reference` is a line-by-line CPU port of the web app's P-frame: `WG_ME` (block matching) and
//! the P-frame branch of `WG_FB` (the feedback pass) from `gpu.js`, and the mosh decisions of
//! `decide()` in `engine.c`. `clips` are the synthetic pictures the GPU tests are fed, generated
//! here and on the GPU by the same integer arithmetic.
//!
//! Why a port instead of a reference recording: the native port has to behave like *the web app*,
//! not like the version of the native pass that happened to be in the tree when its own tests were
//! written. Every number the oracle produces is derived from `gpu.js` and `engine.c` only.
//!
//! # What the picture size buys
//!
//! The web runs the shaders at the canvas size (`G.w`x`G.h`) over a fixed 320x180 analysis grid.
//! At a 320x180 canvas every sample the two passes take lands on a texel centre:
//!
//! * `WG_ME` samples `p / GS` at `p = origin + (i % 8, i / 8) + 0.5`, i.e. the centre of grid
//!   texel `origin + (i % 8, i / 8)`;
//! * `WG_FB` samples at `pos.xy / F.res`, a pixel centre, and at `uv + d` with
//!   `d = round(mv * res) / res`, a whole number of pixels.
//!
//! so neither pass interpolates, and the reference is exact rather than approximate. The address
//! modes still matter and are ported: the motion pass samples with `G.linC` (linear,
//! clamp-to-edge) and the feedback pass with `G.nm` (nearest, mirror-repeat, gpu.js:637-639).
//!
//! # Deliberately not modelled
//!
//! The web's frame loop has more than the mosh pass in it. Left out, because they are not the
//! datamosh (and the brief excludes them): the `ops` rect swaps (`F.ops`), the feedback zoom and
//! spin (`fxa.z`/`fxa.w`), `fxb` (streak/crush/echo/negative), the fluid's velocity term, the
//! output pass's scanline shift, vignette, noise, fade and invert, and the CPU renderer's
//! `corrupt()` glitches.
//!
//! `mvT`, `srcT`, `prvT` and `bufT` are `rgba16float` in the web (gpu.js:634, :850), so a real web
//! frame quantises the vectors to f16 (relative precision 4.9e-4) and the buffer every frame.
//! This reference keeps everything in f32: the difference is far below the 0.05 thresholds and
//! the 25% tolerances these tests use, and modelling f16 storage would hide the port behind a
//! bit-twiddling helper. It is called out here so nobody reads a mismatch as a bug.

#![allow(clippy::needless_range_loop)] // the ports read like the shaders with explicit indices

pub mod clips;
pub mod reference;

/// The test picture size. The web's analysis grid is 320x180 (`GS`, gpu.js:296): at that size
/// every sample lands on a texel centre and the port is exact (see the module docs).
pub const WIDTH: usize = 320;
pub const HEIGHT: usize = 180;

/// An RGB picture, tight and row-major, 0..1 per channel.
///
/// No alpha: `WG_ME` reads `r/g/b`, the feedback pass carries `c.rgb`, and the output pass drops
/// alpha as well, so a mosh frame has three channels.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<f32>,
}

impl Frame {
    /// A black picture.
    pub fn new(width: usize, height: usize) -> Frame {
        Frame {
            width,
            height,
            rgb: vec![0.0; width * height * 3],
        }
    }

    /// A picture whose pixel `(x, y)` is `pixel(x, y)`.
    pub fn from_fn(
        width: usize,
        height: usize,
        mut pixel: impl FnMut(usize, usize) -> [f32; 3],
    ) -> Frame {
        let mut frame = Frame::new(width, height);
        for y in 0..height {
            for x in 0..width {
                frame.set(x, y, pixel(x, y));
            }
        }
        frame
    }

    /// A picture from a GPU read-back: tightly packed RGBA8, top row first (`Gpu::read_back`).
    pub fn from_rgba8(rgba: &[u8], width: usize, height: usize) -> Frame {
        assert_eq!(
            rgba.len(),
            width * height * 4,
            "a {width}x{height} read-back"
        );
        Frame::from_fn(width, height, |x, y| {
            let at = (y * width + x) * 4;
            [
                f32::from(rgba[at]) / 255.0,
                f32::from(rgba[at + 1]) / 255.0,
                f32::from(rgba[at + 2]) / 255.0,
            ]
        })
    }

    pub fn at(&self, x: usize, y: usize) -> [f32; 3] {
        let at = (y * self.width + x) * 3;
        [self.rgb[at], self.rgb[at + 1], self.rgb[at + 2]]
    }

    pub fn set(&mut self, x: usize, y: usize, color: [f32; 3]) {
        let at = (y * self.width + x) * 3;
        self.rgb[at..at + 3].copy_from_slice(&color);
    }

    /// The texel at `(x, y)`, clamping to the edge: the sampler's `clamp-to-edge` mode.
    pub fn at_clamped(&self, x: i64, y: i64) -> [f32; 3] {
        let x = x.clamp(0, self.width as i64 - 1) as usize;
        let y = y.clamp(0, self.height as i64 - 1) as usize;
        self.at(x, y)
    }

    /// The same picture with every channel clamped into `[low, high]`.
    pub fn clamped(&self, low: f32, high: f32) -> Frame {
        let mut out = self.clone();
        for value in &mut out.rgb {
            *value = value.clamp(low, high);
        }
        out
    }
}

/// `textureSampleLevel(img, G.nm, (u, v), 0)`: nearest filtering, mirror-repeat (gpu.js:639).
///
/// Nearest picks the texel containing the sample point, which in normalized coordinates is
/// `floor(u * width)`; the wrapping then mirrors that index into range. Every coordinate the
/// reference passes here is a texel centre (`(i + 0.5) / width`), so the choice is unambiguous.
pub fn sample_nearest_mirror(img: &Frame, u: f32, v: f32) -> [f32; 3] {
    let x = mirror((u * img.width as f32).floor() as i64, img.width as i64);
    let y = mirror((v * img.height as f32).floor() as i64, img.height as i64);
    img.at(x, y)
}

/// `textureSampleLevel(img, G.linC, (u, v), 0)`: linear filtering, clamp-to-edge (gpu.js:637).
pub fn sample_linear_clamp(img: &Frame, u: f32, v: f32) -> [f32; 3] {
    let x = u * img.width as f32 - 0.5;
    let y = v * img.height as f32 - 0.5;
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (x0, y0) = (x0 as i64, y0 as i64);
    let top = img.at_clamped(x0, y0);
    let top_right = img.at_clamped(x0 + 1, y0);
    let bottom = img.at_clamped(x0, y0 + 1);
    let bottom_right = img.at_clamped(x0 + 1, y0 + 1);
    let mut out = [0.0f32; 3];
    for channel in 0..3 {
        let upper = top[channel] + (top_right[channel] - top[channel]) * fx;
        let lower = bottom[channel] + (bottom_right[channel] - bottom[channel]) * fx;
        out[channel] = upper + (lower - upper) * fy;
    }
    out
}

/// The mirror-repeat wrapping of one texel index: `0, 1, ... n-1, n-1, ... 0, 0, ...`.
fn mirror(index: i64, n: i64) -> usize {
    let period = 2 * n;
    let wrapped = index.rem_euclid(period);
    if wrapped < n {
        wrapped as usize
    } else {
        (period - 1 - wrapped) as usize
    }
}

/// Mean absolute per-channel difference of two pictures, 0..1 units.
///
/// This is the one metric both sides report: the oracle computes it on f32 pictures and the GPU
/// tests on `Frame`s built from their read-backs, so the two numbers mean the same thing.
pub fn mean_abs_diff(a: &Frame, b: &Frame) -> f32 {
    assert_eq!(
        (a.width, a.height),
        (b.width, b.height),
        "pictures of different sizes"
    );
    let total: f64 = a
        .rgb
        .iter()
        .zip(&b.rgb)
        .map(|(left, right)| f64::from(*left - *right).abs())
        .sum();
    (total / a.rgb.len() as f64) as f32
}

/// Largest per-channel difference of two pictures, 0..1 units.
pub fn max_abs_diff(a: &Frame, b: &Frame) -> f32 {
    assert_eq!(
        (a.width, a.height),
        (b.width, b.height),
        "pictures of different sizes"
    );
    a.rgb
        .iter()
        .zip(&b.rgb)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max)
}
