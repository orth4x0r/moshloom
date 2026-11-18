//! The baked text sheet: one image per face, made when the words change.
//!
//! # What a sheet holds
//!
//! The sheet is the laid-out block with a margin, in the byte order `mosh_render::VideoTexture`
//! takes (B, G, R, A). Every pixel carries the field of the **whole line**, not of one glyph's
//! cell: a stroke that a neighbouring letter's outline, glow or bevel reaches keeps a real
//! distance there instead of being cut at a cell edge:
//!
//! | byte | sampled as | meaning |
//! | --- | --- | --- |
//! | 2 | `.r` | the outward surface direction x, encoded `v * 0.5 + 0.5` |
//! | 1 | `.g` | the outward surface direction y, encoded `v * 0.5 + 0.5` |
//! | 0 | `.b` | unused, zero (the bevel's depth is the distance itself, see `text.wgsl`) |
//! | 3 | `.a` | the signed distance field, 0.5 = the edge |
//!
//! # Composing a line from glyph tiles
//!
//! Each glyph's field is cached on its [`Face`] as a tile: the ink box plus [`super::layout::PAD`].
//! A tile is the *true* distance to that glyph's edge out to the field's range, so taking the
//! nearest tile per pixel (`max` of the distance, which grows inside) gives the line's own
//! distance field. That is what makes a text change cheap once the characters are known, and it
//! is why there is no cell to clip at: the tile a pixel belongs to is whichever glyph is actually
//! nearest, decided per pixel, not by a grid.
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
const WIDTH_STEP: u32 = 64;
const HEIGHT_STEP: u32 = 32;

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
    /// Lays `text` out and composes its sheet from the face's cached glyph tiles. `None` when
    /// there is nothing to draw (empty or whitespace-only text), which the layer renders as
    /// transparent black.
    pub fn bake(face: &Face, text: &str) -> Option<Sheet> {
        let mut px = BAKE_PX;
        let mut layout = face.layout(text, px);
        if layout.glyphs.is_empty() {
            return None;
        }
        // A word too long for one texture is re-baked smaller; the field still resolves the edge,
        // it just measures fewer em pixels.
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

        let mask_w = layout.block_w.max(1) as u32 + 2 * MARGIN;
        let mask_h = layout.block_h.max(1) as u32 + 2 * MARGIN;
        let width = bucket(mask_w, WIDTH_STEP).min(MAX_SHEET_WIDTH);
        let height = bucket(mask_h, HEIGHT_STEP).min(MAX_SHEET_HEIGHT);
        let mut pixels = vec![0u8; (width * height * 4) as usize];
        let pad = PAD;

        for glyph in &layout.glyphs {
            let tile = &glyph.glyph;
            // The tile starts `PAD` pixels before the glyph's ink box.
            let origin_x = MARGIN as i32 + glyph.x - pad;
            let origin_y = MARGIN as i32 + glyph.y - pad;
            for row in 0..tile.tile_h {
                let y = origin_y + row as i32;
                if y < 0 || y >= height as i32 {
                    continue;
                }
                let source = &tile.tile[row * tile.tile_w * 4..][..tile.tile_w * 4];
                let target = (y as usize * width as usize) * 4;
                for (column, texel) in source.as_chunks::<4>().0.iter().enumerate() {
                    let x = origin_x + column as i32;
                    if x < 0 || x >= width as i32 {
                        continue;
                    }
                    let at = target + x as usize * 4;
                    // The nearest glyph wins the pixel: its distance, its surface direction.
                    if texel[3] > pixels[at + 3] {
                        pixels[at..at + 4].copy_from_slice(texel);
                    }
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

    /// The block's inner rectangle in the UV of a `width` x `height` sheet: `(min, max)`, the
    /// space the shader maps a block coordinate into. The texture may be larger than this bake
    /// (its size is reused between texts), so the block always sits at its top-left margin.
    pub fn inset_in(&self, width: u32, height: u32) -> ([f32; 2], [f32; 2]) {
        let min = [
            self.margin as f32 / width as f32,
            self.margin as f32 / height as f32,
        ];
        let max = [
            (self.margin + self.block_w) as f32 / width as f32,
            (self.margin + self.block_h) as f32 / height as f32,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn unbounded() -> Face {
        Face::new(crate::text::FACE_CHROME).expect("Unbounded")
    }

    #[test]
    fn an_empty_text_has_no_sheet() {
        let face = unbounded();
        for text in ["", " ", "\n\n", " \t "] {
            assert!(
                Sheet::bake(&face, text).is_none(),
                "text {text:?} baked a sheet"
            );
        }
    }

    #[test]
    fn the_sheet_holds_the_block_and_its_margin() {
        let face = unbounded();
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
        let face = unbounded();
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
        // The sheet's outer pixels are empty: the glyphs' own padding is inside the sheet, and
        // nothing is allowed to touch the edge a bilinear sample clamps at.
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
            "the block's centre should be a field pixel"
        );
    }

    #[test]
    fn the_field_is_half_scale_at_the_ink_edge() {
        let face = unbounded();
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

    /// The defect this task fixes: a per-glyph cell clipped the field at its own border, so the
    /// sheet had a straight step at every cell edge. A real distance field is 1-Lipschitz: no two
    /// neighbouring pixels may differ by more than one pixel of distance, which is
    /// `255 / (2 * SDF_RANGE)` levels. Clipped cells produced steps of 50-200 levels, and two
    /// glyphs' padded cells overlapping cut each other the same way.
    #[test]
    fn a_baked_line_has_no_cell_seams() {
        let face = unbounded();
        let step = (255.0 / (2.0 * super::super::layout::SDF_RANGE)).ceil() as i32 + 3;
        for text in ["ILANA", "MW", "AV", "THE Loom knows"] {
            let sheet = Sheet::bake(&face, text).expect("a sheet");
            let a =
                |x: u32, y: u32| i32::from(sheet.pixels[((y * sheet.width + x) * 4 + 3) as usize]);
            for y in 0..sheet.height {
                for x in 1..sheet.width {
                    let jump = (a(x, y) - a(x - 1, y)).abs();
                    assert!(
                        jump <= step,
                        "{text:?}: the field jumps {jump} levels between ({}, {y}) and ({x}, {y}), \
                         over the {step} a continuous distance field allows",
                        x - 1
                    );
                }
            }
            for y in 1..sheet.height {
                for x in 0..sheet.width {
                    let jump = (a(x, y) - a(x, y - 1)).abs();
                    assert!(
                        jump <= step,
                        "{text:?}: the field jumps {jump} levels between ({x}, {}) and ({x}, {y})",
                        y - 1
                    );
                }
            }
        }
    }

    /// The three looks use three faces, so the same words bake different ink. This is the CPU
    /// half of "each look uses its face"; `tests/text.rs` checks the rendered pictures.
    #[test]
    fn each_face_bakes_its_own_glyphs() {
        let faces = [
            ("chrome", unbounded()),
            (
                "sigil",
                Face::new(crate::text::FACE_SIGIL).expect("Grenze Gotisch"),
            ),
            (
                "clean",
                Face::new(crate::text::FACE_CLEAN).expect("Space Grotesk"),
            ),
        ];
        let sheets: Vec<Sheet> = faces
            .iter()
            .map(|(_, face)| Sheet::bake(face, "ILANA").expect("a sheet"))
            .collect();
        for (index, (name, _)) in faces.iter().enumerate() {
            for (other, (other_name, _)) in faces.iter().enumerate().skip(index + 1) {
                let a = &sheets[index];
                let b = &sheets[other];
                println!(
                    "{name}: {}x{} block, {other_name}: {}x{} block",
                    a.block_w, a.block_h, b.block_w, b.block_h
                );
                assert!(
                    a.block_w != b.block_w || a.block_h != b.block_h,
                    "{name} and {other_name} bake the same box: the look is not using its face"
                );
            }
        }
    }
}

