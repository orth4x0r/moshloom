//! The baked text sheet: one image per look, made when the words change.
//!
//! # What a sheet holds
//!
//! The sheet is the wrapped block with a margin, in the byte order `mosh_render::VideoTexture`
//! takes (B, G, R, A). Every pixel carries the field of the **whole block**, not of one glyph's
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
//! # Composing a block from glyph tiles
//!
//! Each glyph's field is cached on its [`Face`] as a tile: the ink box plus
//! [`super::layout::PAD`]. A tile is the *true* distance to that glyph's edge out to the field's
//! range, so taking the nearest tile per pixel (`max` of the distance, which grows inside) gives
//! the block's own distance field. That is what makes a text change cheap once the characters are
//! known, and it is why there is no cell to clip at: the tile a pixel belongs to is whichever
//! glyph is actually nearest, decided per pixel, not by a grid.
//!
//! Where two tiles are within a few levels of each other — the medial axis between letters that
//! nearly touch — their surface directions are averaged. Taking one tile's direction and then the
//! other's across that line is a crease, and on the chrome look a crease through a pair like
//! "at" reads as the two letters being merged badly.
//!
//! # Size
//!
//! [`capacity`] is the largest block the layout can produce at the em size the shader bakes
//! (wrapped to [`super::layout::WRAP_EM`] and capped at [`super::layout::MAX_LINES`]), so the layer
//! can allocate one texture per look and never grow it: growing means dropping the old image, and
//! dropping it waits for the device.

use super::layout::{BAKE_PX, Face, MAX_LINES, PAD};

/// Margin around the block's ink box, in sheet pixels: [`PAD`] plus two pixels of slack so a
/// bilinear sample at the block's edge never reads the texture's clamp.
pub const MARGIN: u32 = PAD as u32 + 2;

/// The largest sheet a block may be. A 1792x1024 RGBA8 image is 7.3 MB; a block that would need
/// more is laid out at a smaller em size instead.
pub const MAX_SHEET_WIDTH: u32 = 1792;
pub const MAX_SHEET_HEIGHT: u32 = 1152;

/// Sheet sizes round up to these steps (pixels), so a test bake is compact and reusable.
const WIDTH_STEP: u32 = 64;
const HEIGHT_STEP: u32 = 32;

/// Two tiles' surface directions are averaged when their distances are this close (levels out of
/// 255), which is the medial axis between two letters: one tile's direction would be a crease.
const NORMAL_BLEND: u8 = 10;

/// A laid-out string, rasterised into one image. The layer composes into its own buffer
/// (Baked), so the struct itself is what the tests inspect.
#[cfg_attr(not(test), allow(dead_code))]
pub struct Sheet {
    pub width: u32,
    pub height: u32,
    /// The ink block's size in sheet pixels.
    pub block_w: u32,
    pub block_h: u32,
    /// The block's offset inside the sheet, in sheet pixels.
    pub margin: u32,
    /// BGRA8, `width * height * 4` bytes.
    #[cfg_attr(not(test), allow(dead_code))]
    pub pixels: Vec<u8>,
}

impl Sheet {
    /// Lays `text` out and composes it into a sheet sized for the block itself. `None` when there
    /// is nothing to draw (empty or whitespace-only text).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn bake(face: &Face, text: &str) -> Option<Sheet> {
        let layout = face.layout(text, BAKE_PX);
        if layout.glyphs.is_empty() {
            return None;
        }
        let width = bucket(layout.block_w.max(1) as u32 + 2 * MARGIN, WIDTH_STEP);
        let height = bucket(layout.block_h.max(1) as u32 + 2 * MARGIN, HEIGHT_STEP);
        let mut pixels = Vec::new();
        let baked = compose_into(face, text, width, height, &mut pixels, height)?;
        Some(Sheet {
            width,
            height,
            block_w: baked.block_w,
            block_h: baked.block_h,
            margin: MARGIN,
            pixels,
        })
    }

    /// Composes `text` into a sheet of exactly `width` x `height` pixels — the layer's
    /// preallocated texture — shrinking the em size only if the wrapped block would not fit.
    /// `None` when there is nothing to draw or the block cannot fit at any size.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn bake_capacity(face: &Face, text: &str, width: u32, height: u32) -> Option<Sheet> {
        let mut pixels = Vec::new();
        let baked = compose_into(face, text, width, height, &mut pixels, height)?;
        Some(Sheet {
            width,
            height,
            block_w: baked.block_w,
            block_h: baked.block_h,
            margin: MARGIN,
            pixels,
        })
    }

    /// [`Self::bake_capacity`] into a buffer the layer keeps: a text change then costs no
    /// allocation, only the clear and the blit.
    pub fn bake_into(
        face: &Face,
        text: &str,
        width: u32,
        height: u32,
        pixels: &mut Vec<u8>,
        clear_rows: u32,
    ) -> Option<Baked> {
        compose_into(face, text, width, height, pixels, clear_rows)
    }

}

/// What one composed sheet holds, apart from its pixels.
pub struct Baked {
    pub block_w: u32,
    pub block_h: u32,
    pub dropped: usize,
    /// Rows of the texture this bake touched (the block plus its margin).
    pub mask_h: u32,
}

/// Composes `text` into `pixels` at `width` x `height`, clearing it first.
fn compose_into(
    face: &Face,
    text: &str,
    width: u32,
    height: u32,
    pixels: &mut Vec<u8>,
    clear_rows: u32,
) -> Option<Baked> {
    // One layout per candidate size, and the one that fits is the one composed: laying out twice
    // (once to choose, once to draw) is a third of a text change at 20 characters.
    let mut chosen = None;
    for px in [BAKE_PX, 96.0, 84.0, 72.0, 60.0] {
        let layout = face.layout(text, px);
        if layout.glyphs.is_empty() {
            return None;
        }
        if layout.block_w.max(1) as u32 + 2 * MARGIN <= width
            && layout.block_h.max(1) as u32 + 2 * MARGIN <= height
        {
            chosen = Some(layout);
            break;
        }
    }
    let layout = chosen?;
    let mask_w = layout.block_w.max(1) as u32 + 2 * MARGIN;
    let mask_h = layout.block_h.max(1) as u32 + 2 * MARGIN;
    if mask_w > width || mask_h > height {
        return None;
    }
    // Only the rows a bake can have touched need clearing: the block always sits at the
    // top-left, so anything below `max(previous mask height, this mask height)` is still zero
    // from the allocation. Clearing the whole texture was half of a text change's cost.
    let len = (width * height * 4) as usize;
    let clear_rows = clear_rows.clamp(mask_h, height);
    let row_bytes = (width * 4) as usize;
    pixels.resize(len, 0);
    pixels[..clear_rows as usize * row_bytes].fill(0);
    let pad = PAD;

        for glyph in &layout.glyphs {
            let tile = &glyph.glyph;
            // The tile starts `PAD` pixels before the glyph's ink box.
            let origin_x = MARGIN as i32 + glyph.x - pad;
            let origin_y = MARGIN as i32 + glyph.y - pad;
            let tile_texels = tile.tile.as_chunks::<4>().0;
            let dest_texels = pixels.as_chunks_mut::<4>().0;
            for row in 0..tile.tile_h {
                let y = origin_y + row as i32;
                if y < 0 || y >= height as i32 {
                    continue;
                }
                let source = &tile_texels[row * tile.tile_w..][..tile.tile_w];
                let target = y as usize * width as usize;
                for (column, texel) in source.iter().enumerate() {
                    let x = origin_x + column as i32;
                    if x < 0 || x >= width as i32 {
                        continue;
                    }
                    let at = target + x as usize;
                    let held = dest_texels[at];
                    if texel[3] > held[3] {
                        if held[3] > 0 && texel[3] - held[3] <= NORMAL_BLEND {
                            // The medial axis between two glyphs: average the surface directions
                            // so the shading does not crease where the nearest glyph changes.
                            dest_texels[at][1] =
                                ((u16::from(held[1]) + u16::from(texel[1])) / 2) as u8;
                            dest_texels[at][2] =
                                ((u16::from(held[2]) + u16::from(texel[2])) / 2) as u8;
                        } else {
                            dest_texels[at][1] = texel[1];
                            dest_texels[at][2] = texel[2];
                        }
                        dest_texels[at][3] = texel[3];
                    } else if held[3] > 0
                        && texel[3] > 0
                        && held[3] - texel[3] <= NORMAL_BLEND
                    {
                        // The held tile is the nearer one, but the other's direction is close
                        // enough to matter: average only, never darken the pixel.
                        dest_texels[at][1] =
                            ((u16::from(held[1]) + u16::from(texel[1])) / 2) as u8;
                        dest_texels[at][2] =
                            ((u16::from(held[2]) + u16::from(texel[2])) / 2) as u8;
                    }
                }
            }
        }

    Some(Baked {
        block_w: layout.block_w.max(1) as u32,
        block_h: layout.block_h.max(1) as u32,
        dropped: layout.dropped,
        mask_h,
    })
}

impl Sheet {
}

impl Baked {
    /// The block's inner rectangle in the UV of a `width` x `height` texture: `(min, max)`, the
    /// space the shader maps a block coordinate into. The texture is the capacity, larger than
    /// this bake, so the block always sits at its top-left margin.
    pub fn inset_in(&self, width: u32, height: u32) -> ([f32; 2], [f32; 2]) {
        let min = [
            MARGIN as f32 / width as f32,
            MARGIN as f32 / height as f32,
        ];
        let max = [
            (MARGIN + self.block_w) as f32 / width as f32,
            (MARGIN + self.block_h) as f32 / height as f32,
        ];
        (min, max)
    }
}

/// The largest sheet [`Sheet::bake_capacity`] can ever need for this face: the wrap width and the
/// line cap bound the block, so one allocation per look lasts for the whole show.
pub fn capacity(face: &Face) -> (u32, u32) {
    let width = 2 * MARGIN + (face.wrap_width(BAKE_PX) + BAKE_PX * 0.45) as u32;
    let height = 2 * MARGIN + (MAX_LINES as f32 * face.line_height()).ceil() as u32 + 4;
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
        Face::new(crate::text::FACE_CHROME, true).expect("Unbounded")
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
        assert!(sheet.block_w > sheet.block_h * 2, "ILANA is a wide word");
    }

    #[test]
    fn a_capacity_bake_fills_the_whole_texture_and_never_needs_more() {
        let face = unbounded();
        let (width, height) = capacity(&face);
        println!("capacity: {width}x{height}");
        // The longest text the layout keeps, wrapped at the em size the layer uses.
        let long = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor";
        for text in ["I", "ILANA", "the loom knows what", long] {
            let sheet = Sheet::bake_capacity(&face, text, width, height).expect("a sheet");
            assert_eq!((sheet.width, sheet.height), (width, height));
            assert!(sheet.block_w + 2 * MARGIN <= width);
            assert!(sheet.block_h + 2 * MARGIN <= height);
        }
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
        let middle = at(
            sheet.margin + sheet.block_w / 2,
            sheet.margin + sheet.block_h / 2,
        );
        assert!(
            middle != [0, 0, 0, 0],
            "the block's centre should be a field pixel"
        );
    }

    /// Most of a one-glyph sheet is outside the field: the field's range is what keeps a glyph's
    /// tile small relative to the block.
    #[test]
    fn most_of_the_sheet_is_void() {
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

    /// The defect this rework fixes: a per-glyph cell clipped the field at its own border, so the
    /// sheet had a straight step at every cell edge. A real distance field is 1-Lipschitz: no two
    /// neighbouring pixels may differ by more than one pixel of distance, which is
    /// `255 / (2 * SDF_RANGE)` ~ 11 levels. The bound here is 2.5 pixels because a pixel whose
    /// antialiased coverage is partial takes its distance from that coverage (the sub-pixel edge),
    /// which can move it by half a pixel against its neighbour's grid value on each side. Clipped
    /// cells produced steps of 50-200 levels, and two glyphs' padded cells cut each other the same
    /// way — nowhere near this bound.
    #[test]
    fn a_baked_line_has_no_cell_seams() {
        let face = unbounded();
        let step = (255.0 / (2.0 * super::super::layout::SDF_RANGE) * 2.5).ceil() as i32;
        for text in ["ILANA", "MW", "AV", "at", "THE Loom knows"] {
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

    /// A pair that nearly touches must not crease its shading. The surface direction is baked,
    /// and where two tiles meet (the medial axis between letters) taking one tile's direction and
    /// then the other's is a crease: on the chrome look "at" read as the two letters merged badly.
    /// A crease is a *run* of large direction steps along a row, not a single step, because the
    /// direction legitimately flips across a thin stroke.
    #[test]
    fn a_tight_pair_has_no_surface_crease() {
        let face = unbounded();
        let mut worst = 0usize;
        let mut worst_text = "";
        for text in ["at", "AV", "To", "Ty", "wa", "ILANA"] {
            let sheet = Sheet::bake(&face, text).expect("a sheet");
            let normal_at = |x: u32, y: u32, channel: usize| {
                i32::from(sheet.pixels[((y * sheet.width + x) * 4) as usize + channel])
            };
            for y in 0..sheet.height {
                let mut run = 0usize;
                for x in 1..sheet.width {
                    let lit = |x: u32| {
                        sheet.pixels[((y * sheet.width + x) * 4 + 3) as usize] >= 8
                    };
                    let at = |channel: usize| normal_at(x, y, channel);
                    let before = |channel: usize| normal_at(x - 1, y, channel);
                    let step = (at(1) - before(1)).abs().max((at(2) - before(2)).abs());
                    if lit(x) && lit(x - 1) && step > 120 {
                        run += 1;
                        if run > worst {
                            worst = run;
                            worst_text = text;
                        }
                    } else {
                        run = 0;
                    }
                }
            }
        }
        println!(
            "longest run of surface-direction steps over 120 levels: {worst} pixels ({worst_text:?})"
        );
        assert!(
            worst <= 3,
            "the baked surface direction steps more than 120 levels for {worst} pixels in a row \
             ({worst_text:?}), which reads as a crease in a pair that nearly touches"
        );
    }

    /// The three looks use three faces, so the same words bake different ink. This is the CPU
    /// half of "each look uses its face"; `tests/text.rs` checks the rendered pictures.
    #[test]
    fn each_face_bakes_its_own_glyphs() {
        let faces = [
            ("chrome", unbounded()),
            (
                "sigil",
                Face::new(crate::text::FACE_SIGIL, true).expect("Grenze Gotisch"),
            ),
            (
                "clean",
                Face::new(crate::text::FACE_CLEAN, false).expect("Space Grotesk"),
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

