//! The baked text sheet: one image per face, made when the words change.
//!
//! # What a sheet holds
//!
//! The sheet is the laid-out block with a margin, in the byte order `mosh_render::VideoTexture`
//! takes (B, G, R, A), and every pixel is four things at once:
//!
//! | byte | sampled as | meaning |
//! | --- | --- | --- |
//! | 0 | `.r` | the glyph's cell centre x, 0..1 across the sheet |
//! | 1 | `.g` | the glyph's cell centre y |
//! | 2 | `.b` | the glyph's own noise value, 0..1 |
//! | 3 | `.a` | the signed distance field, 0.5 = the edge |
//!
//! The cell channels are what lets the shader work per glyph in a single full-screen pass: a pixel
//! looks up its own glyph's centre and noise, then resamples the distance field around that
//! centre for the beat pop, jitter and rotation. `set_text` builds this; `draw` never touches a
//! glyph again.
//!
//! Cells are written left to right and the first glyph to reach a pixel owns that pixel's cell
//! channels; the distance field takes the union (`max`), so a kerned pair's overlapping pixel is
//! still inked.
//!
//! # Size
//!
//! The sheet is the block's ink box plus [`MARGIN`] on every side, so a short word is a small
//! texture. Words that would be larger than [`MAX_SHEET_WIDTH`] x [`MAX_SHEET_HEIGHT`] are
//! re-baked at a smaller em size: a lyrics page gets a coarser distance field, not a texture the
//! integrated GPU cannot hold.

use super::layout::{BAKE_PX, Face, PAD};

/// Margin around the block's ink box, in sheet pixels: [`PAD`] plus two pixels of slack so a
/// bilinear sample at the block's edge never reads the texture's clamp.
pub const MARGIN: u32 = PAD as u32 + 2;

/// The largest sheet `set_text` bakes. A 1792x1024 RGBA8 image is 7.3 MB; beyond that the em size
/// shrinks instead.
pub const MAX_SHEET_WIDTH: u32 = 1792;
pub const MAX_SHEET_HEIGHT: u32 = 1024;

/// Sheet sizes round up to these steps (pixels), so a text change usually reuses the texture.
const WIDTH_STEP: u32 = 256;
const HEIGHT_STEP: u32 = 64;

/// A laid-out string, rasterised into one image.
pub struct Sheet {
    pub width: u32,
    pub height: u32,
    /// The ink block's size in sheet pixels.
    pub block_w: u32,
    pub block_h: u32,
    /// The block's offset inside the sheet, in sheet pixels.
    pub margin: u32,
    /// BGRA8, `width * height * 4` bytes.
    pub pixels: Vec<u8>,
}

impl Sheet {
    /// Lays `text` out and rasterises it. `None` when there is nothing to draw (empty or
    /// whitespace-only text), which the layer renders as transparent black.
    pub fn bake(face: &Face, text: &str) -> Option<Sheet> {
        let mut px = BAKE_PX;
        let mut layout = face.layout(text, px);
        if layout.glyphs.is_empty() {
            return None;
        }
        for _ in 0..3 {
            let (width, height) = sheet_size(&layout);
            if width <= MAX_SHEET_WIDTH && height <= MAX_SHEET_HEIGHT {
                break;
            }
            let fit = ((MAX_SHEET_WIDTH as f32 - 1.0) / width as f32)
                .min((MAX_SHEET_HEIGHT as f32 - 1.0) / height as f32);
            px *= fit.clamp(0.25, 0.9);
            layout = face.layout(text, px);
            if layout.glyphs.is_empty() {
                return None;
            }
        }

        let (width, height) = sheet_size(&layout);
        let mut pixels = vec![0u8; (width * height * 4) as usize];
        for glyph in &layout.glyphs {
            let cell_x = MARGIN as i32 + glyph.x;
            let cell_y = MARGIN as i32 + glyph.y;
            let centre_x =
                quantise((cell_x as f32 + glyph.cell_w as f32 * 0.5) / width.max(1) as f32);
            let centre_y =
                quantise((cell_y as f32 + glyph.cell_h as f32 * 0.5) / height.max(1) as f32);
            let noise = quantise(glyph.random);
            for row in 0..glyph.cell_h {
                for column in 0..glyph.cell_w {
                    let x = cell_x + column as i32;
                    let y = cell_y + row as i32;
                    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
                        continue;
                    }
                    let offset = ((y as u32 * width + x as u32) * 4) as usize;
                    if pixels[offset] == 0
                        && pixels[offset + 1] == 0
                        && pixels[offset + 2] == 0
                        && pixels[offset + 3] == 0
                    {
                        pixels[offset] = centre_x;
                        pixels[offset + 1] = centre_y;
                        pixels[offset + 2] = noise;
                    }
                    let distance = glyph.sdf[(row * glyph.cell_w + column) as usize];
                    pixels[offset + 3] = pixels[offset + 3].max(distance);
                }
            }
        }

        Some(Sheet {
            width,
            height,
            block_w: layout.block_w.max(1) as u32,
            block_h: layout.block_h.max(1) as u32,
            margin: MARGIN,
            pixels,
        })
    }

    /// The block's inner rectangle in sheet UV: `(min, max)`, the space the shader maps a block
    /// coordinate into.
    pub fn inset(&self) -> ([f32; 2], [f32; 2]) {
        let min = [
            self.margin as f32 / self.width as f32,
            self.margin as f32 / self.height as f32,
        ];
        let max = [
            (self.margin + self.block_w) as f32 / self.width as f32,
            (self.margin + self.block_h) as f32 / self.height as f32,
        ];
        (min, max)
    }
}

/// The texture's size for a laid-out block.
///
/// Widths round up to [`WIDTH_STEP`] and heights to [`HEIGHT_STEP`]: a lyric changing from
/// "ILANA" to "MOshloom" keeps the same texture instead of allocating a new image and two staging
/// buffers on the render thread. The block sits at the top-left margin of the larger sheet; the
/// rest is transparent, and `Sheet::inset()` tells the shader where the block is.
fn sheet_size(layout: &super::layout::Layout) -> (u32, u32) {
    let width = layout.block_w.max(1) as u32 + 2 * MARGIN;
    let height = layout.block_h.max(1) as u32 + 2 * MARGIN;
    (
        bucket(width, WIDTH_STEP).min(MAX_SHEET_WIDTH),
        bucket(height, HEIGHT_STEP).min(MAX_SHEET_HEIGHT),
    )
}

fn bucket(value: u32, step: u32) -> u32 {
    value.div_ceil(step) * step
}

fn quantise(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ubuntu() -> Face {
        Face::new(crate::text::FACE_DISPLAY).expect("Ubuntu Light")
    }

    #[test]
    fn an_empty_text_has_no_sheet() {
        let face = ubuntu();
        for text in ["", " ", "\n\n", " \t "] {
            assert!(
                Sheet::bake(&face, text).is_none(),
                "text {text:?} baked a sheet"
            );
        }
    }

    #[test]
    fn the_sheet_holds_the_block_and_its_margin() {
        let face = ubuntu();
        let sheet = Sheet::bake(&face, "ILANA").expect("a sheet");
        assert!(sheet.width >= sheet.block_w + 2 * MARGIN);
        assert!(sheet.height >= sheet.block_h + 2 * MARGIN);
        // Sizes round up so a text change usually reuses the texture instead of allocating one.
        assert_eq!(
            sheet.width % WIDTH_STEP,
            0,
            "widths round up to {WIDTH_STEP}"
        );
        assert_eq!(
            sheet.height % HEIGHT_STEP,
            0,
            "heights round up to {HEIGHT_STEP}"
        );
        assert!(sheet.width <= MAX_SHEET_WIDTH && sheet.height <= MAX_SHEET_HEIGHT);
        assert_eq!(
            sheet.pixels.len(),
            (sheet.width * sheet.height * 4) as usize
        );
        // The block is fully inside the sheet: a one-line word is wider than it is tall.
        assert!(sheet.block_w > sheet.block_h * 2, "ILANA is a wide word");
    }

    #[test]
    fn the_margin_is_empty() {
        let face = ubuntu();
        let sheet = Sheet::bake(&face, "ILANA").expect("a sheet");
        let at = |x: u32, y: u32| {
            let offset = ((y * sheet.width + x) * 4) as usize;
            [
                sheet.pixels[offset],
                sheet.pixels[offset + 1],
                sheet.pixels[offset + 2],
                sheet.pixels[offset + 3],
            ]
        };
        // The sheet's outer two pixels are empty: a glyph cell's PAD may reach into the rest of
        // the margin, but nothing is allowed to touch the edge a bilinear sample clamps at.
        for (x, y) in [
            (0, 0),
            (1, 1),
            (sheet.width - 1, 0),
            (sheet.width - 2, 1),
            (0, sheet.height - 1),
            (1, sheet.height - 2),
            (sheet.width - 1, sheet.height - 1),
            (sheet.width / 2, 0),
            (0, sheet.height / 2),
        ] {
            assert_eq!(
                at(x, y),
                [0, 0, 0, 0],
                "pixel ({x},{y}) is on the sheet's edge"
            );
        }
        // ... and the middle of the block carries a distance sample of some kind.
        let middle = at(
            sheet.margin + sheet.block_w / 2,
            sheet.margin + sheet.block_h / 2,
        );
        assert!(
            middle != [0, 0, 0, 0],
            "the block's centre should be a glyph cell pixel"
        );
    }

    #[test]
    fn the_field_is_half_scale_at_the_ink_edge() {
        let face = ubuntu();
        let sheet = Sheet::bake(&face, "I").expect("a sheet");
        let field: Vec<u8> = sheet.pixels.chunks(4).map(|pixel| pixel[3]).collect();
        let inside = field.iter().filter(|value| **value > 160).count();
        let outside = field.iter().filter(|value| **value == 0).count();
        assert!(inside > 0, "the glyph has an inside");
        assert!(
            outside > inside,
            "most of the sheet is void: {outside} vs {inside}"
        );
    }
}
