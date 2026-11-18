//! Font loading, glyph rasterising, the signed distance field and line layout.
//!
//! Everything in this module runs inside [`crate::text::TextLayer::set_text`], when a text clip
//! starts or its words change. Nothing here runs per frame: `draw` only records one pass.
//!
//! # One field per glyph tile, composed into one field for the line
//!
//! Each glyph is rasterised into its own *tile*: its ink box plus [`PAD`] on every side, and the
//! distance transform runs over that tile. A tile therefore holds the true distance to that
//! glyph's edge out to [`SDF_RANGE`], which is what lets the sheet take the nearest tile per pixel
//! (`Sheet::bake`): the result is the line's own distance field, and there is no cell for an
//! outline, a glow or a bevel to be cut at. A tile is a pure function of (face, character, em
//! size), so it is cached and a new text re-rasters only the characters it introduces — the old
//! per-cell sheet had to redo every glyph on every text change, and cut each one at its cell.
//!
//! # Why an SDF and not a coverage bitmap
//!
//! The sheet is baked once, at [`BAKE_PX`] em, while `size` (`p0`) scales it every frame. A signed
//! distance field stays smooth under that scaling: the shader thresholds the distance with a
//! screen-space antialiasing width, so a glyph three times its baked size still has a clean edge,
//! and the same distance drives the outline, the glow and the bevel.
//!
//! The distance field is a 3-4 chamfer transform: two sweeps over the mask, each pixel taking the
//! cheapest of its already-visited neighbours plus 3 (across) or 4 (diagonal). That is a handful of
//! `u8` adds per pixel, not a quadratic search, so the whole bake stays inside a text change.
//!
//! # Layout
//!
//! Lines come from `\n`. Within a line the pen advances by the font's own advance widths plus its
//! kerning pairs, which is what keeps letter spacing typographically right instead of whatever a
//! cell grid happens to produce; each line is then centred inside the block by its *advance* width
//! (not its ink width, which excludes side bearings and would sit the line visibly off centre) and
//! the block keeps the union of the ink boxes so a glyph that overhangs its advance cannot be cut.
//!
//! # The glyph cache
//!
//! A tile is a pure function of (face, character, em size), so it is cached on the [`Face`] and
//! reused by the next text: a lyric changing one word re-rasters (and re-transforms) only the
//! characters that are new to it, and composing the sheet is a blit per glyph. This is what keeps
//! `set_text` under the 2 ms budget it has on the render thread (see `tests/text.rs`,
//! `set_text_and_draw_costs`).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use fontdue::{Font, FontSettings};

/// Em size the glyphs are rasterised at. The sheet is a distance field, so a layer that draws at
/// 2x this size is still crisp; this is the one number that trades `set_text` time for sharpness.
pub const BAKE_PX: f32 = 112.0;

/// How far, in baked pixels, a distance sample stays meaningful on each side of the edge. The
/// shader reads the sheet as `0.5 = edge` and converts back with the same constant. An outline at
/// its maximum is about 3.5 baked pixels wide and a glow is dead by 10, so 12 leaves headroom.
pub const SDF_RANGE: f32 = 12.0;

/// Baked pixels over which the bevel's height falls from the stroke's centre to its edge. Baked
/// into the sheet's blue channel as the "depth" the shader turns into a surface normal.
pub const BEVEL_PX: f32 = 9.0;

/// Room around a glyph's ink for the distance field's outside half, in baked pixels: the field's
/// range plus two pixels of slack so a bilinear sample at the block's edge never reads the
/// texture's clamp.
pub const PAD: i32 = SDF_RANGE as i32 + 2;

/// A loaded face. `fontdue::Font` is the parser and rasteriser; the face's licence is in
/// `native/assets/fonts/`.
pub struct Face {
    font: Font,
    /// Rasterised glyphs, keyed by character and em size. Pure data, so sharing one bake between
    /// two texts of the same face is exact, not an approximation.
    glyphs: RefCell<HashMap<CacheKey, Rc<CachedGlyph>>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct CacheKey {
    ch: char,
    /// The em size in 1/16 pixel, so a float size that differs only by rounding noise is a hit.
    px_16: u32,
}

/// One rasterised glyph: its baked field tile and the metrics needed to place it.
///
/// The tile is the glyph's ink box plus [`PAD`] on every side, already in the sheet's BGRA field
/// encoding. It is the true signed distance to *this* glyph's edge within [`SDF_RANGE`] of it,
/// which is what lets [`crate::text::sheet::Sheet::bake`] compose a line by taking the nearest
/// tile per pixel: the composed field is the line's distance field, not a grid of clipped cells.
pub struct CachedGlyph {
    /// BGRA, `tile_w * tile_h * 4` bytes: `(unused, normal.y, normal.x, distance)`.
    pub tile: Vec<u8>,
    pub tile_w: usize,
    pub tile_h: usize,
    /// The ink box inside the tile: [`PAD`] on every side.
    pub width: usize,
    pub height: usize,
    /// Bitmap ink box relative to the pen, y up (font units scaled to pixels).
    pub xmin: i32,
    pub ymin: i32,
    /// The font's advance for this character, in pixels. Spacing comes from here, not from a cell.
    pub advance: f32,
}

/// One glyph placed in the laid-out block.
pub struct Glyph {
    pub glyph: Rc<CachedGlyph>,
    /// Cell top-left in block pixels, y down. The cell is the ink box; the distance field's padding
    /// lives in the sheet, not per glyph, so these boxes are allowed to overlap (kerning does).
    pub x: i32,
    pub y: i32,
}

/// One line's extent after centring, in block pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Line {
    pub ink_left: i32,
    pub ink_right: i32,
    /// The line's advance width: what the line was centred by and what a font-measured string
    /// width is comparable to. Read by this module's tests.
    pub advance: f32,
}

/// A laid-out string: the glyph placements and the ink box the sheet has to hold.
#[derive(Default)]
pub struct Layout {
    pub glyphs: Vec<Glyph>,
    /// Ink size of the whole block, in baked pixels.
    pub block_w: i32,
    pub block_h: i32,
    /// The widest line's advance width, in baked pixels. Read by this module's tests: it is what
    /// "the string is as wide as the font says" is measured against.
    #[cfg_attr(not(test), allow(dead_code))]
    pub advance_w: f32,
    /// One entry per line, in reading order, already centred in the block.
    #[cfg_attr(not(test), allow(dead_code))]
    pub lines: Vec<Line>,
}

/// One glyph before the block's origin and the per-line centring are known.
struct Placed {
    glyph: Rc<CachedGlyph>,
    x: i32,
    y: i32,
    line: usize,
}

impl Face {
    /// Loads a face from the bytes committed under `native/assets/fonts/`. Returns `None` when the
    /// font cannot be parsed; the layer then draws black instead of failing to start.
    pub fn new(bytes: &'static [u8]) -> Option<Face> {
        Font::from_bytes(bytes, FontSettings::default())
            .ok()
            .map(|font| Face {
                font,
                glyphs: RefCell::new(HashMap::new()),
            })
    }

    /// Number of glyphs currently cached, for the cache test.
    #[cfg(test)]
    fn cached(&self) -> usize {
        self.glyphs.borrow().len()
    }

    /// One character's rasterised coverage, from the cache when this face already baked it.
    fn glyph(&self, ch: char, px: f32) -> Rc<CachedGlyph> {
        let key = CacheKey {
            ch,
            px_16: (px * 16.0).round().max(1.0) as u32,
        };
        if let Some(hit) = self.glyphs.borrow().get(&key) {
            return Rc::clone(hit);
        }
        let (metrics, coverage) = self.font.rasterize(ch, px);
        let (width, height) = (metrics.width, metrics.height);
        let pad = PAD as usize;
        let mut mask = vec![0u8; (width + 2 * pad) * (height + 2 * pad)];
        for row in 0..height {
            let source = row * width;
            let target = (row + pad) * (width + 2 * pad) + pad;
            mask[target..target + width]
                .copy_from_slice(&coverage[source..source + width]);
        }
        let tile_w = width + 2 * pad;
        let tile_h = height + 2 * pad;
        let mut tile = vec![0u8; tile_w * tile_h * 4];
        write_field(&mask, tile_w, tile_h, tile_w, &mut tile);
        let glyph = Rc::new(CachedGlyph {
            tile,
            tile_w,
            tile_h,
            width,
            height,
            xmin: metrics.xmin,
            ymin: metrics.ymin,
            advance: metrics.advance_width,
        });
        self.glyphs.borrow_mut().insert(key, Rc::clone(&glyph));
        glyph
    }

    /// Lays `text` out at `px` em. Empty (or whitespace-only) text has no glyphs and no block.
    pub fn layout(&self, text: &str, px: f32) -> Layout {
        let line_metrics = self.font.horizontal_line_metrics(px);
        let (ascent, line_height) = match line_metrics {
            Some(metrics) => (metrics.ascent, metrics.new_line_size + px * 0.10),
            None => (px * 0.8, px * 1.2),
        };

        let mut placed: Vec<Placed> = Vec::new();
        let mut advances: Vec<f32> = Vec::new();
        let mut baseline = ascent;

        for line_text in text.split('\n') {
            let line_text = line_text.trim_end_matches('\r');
            let line_index = advances.len();
            let mut pen = 0.0f32;
            let mut previous: Option<char> = None;

            for mut ch in line_text.chars() {
                if ch == '\t' {
                    ch = ' ';
                }
                if ch.is_control() {
                    continue;
                }
                if let Some(previous) = previous
                    && let Some(kern) = self.font.horizontal_kern(previous, ch, px)
                {
                    pen += kern;
                }
                previous = Some(ch);
                let glyph = self.glyph(ch, px);
                if glyph.width > 0 && glyph.height > 0 {
                    // The cell's top-left, y down: the pen plus the bitmap's own left bearing,
                    // and the baseline minus the bitmap's top.
                    let x = pen.round() as i32 + glyph.xmin;
                    let y = baseline.round() as i32 - (glyph.ymin + glyph.height as i32);
                    placed.push(Placed {
                        glyph: Rc::clone(&glyph),
                        x,
                        y,
                        line: line_index,
                    });
                }
                pen += glyph.advance;
            }
            advances.push(pen);
            baseline += line_height;
        }

        let block_advance = advances.iter().copied().fold(0.0f32, f32::max);

        // Centre every line by its advance width, not by its ink: a line's side bearings are part
        // of its width, and centring by ink is what puts a line visibly off the block's centre.
        for glyph in &mut placed {
            let line_advance = advances[glyph.line];
            glyph.x += ((block_advance - line_advance) * 0.5).round() as i32;
        }

        let ink_left = placed.iter().map(|glyph| glyph.x).min().unwrap_or(0);
        let ink_right = placed
            .iter()
            .map(|glyph| glyph.x + glyph.glyph.width as i32)
            .max()
            .unwrap_or(0);
        let ink_top = placed.iter().map(|glyph| glyph.y).min().unwrap_or(0);
        let ink_bottom = placed
            .iter()
            .map(|glyph| glyph.y + glyph.glyph.height as i32)
            .max()
            .unwrap_or(0);
        let block_w = (ink_right - ink_left).max(0);
        let block_h = (ink_bottom - ink_top).max(0);

        // Line ink boxes and advances, in block space.
        let mut lines: Vec<Line> = advances
            .iter()
            .map(|advance| Line {
                ink_left: 0,
                ink_right: 0,
                advance: *advance,
            })
            .collect();
        for (line, extent) in lines.iter_mut().enumerate() {
            let of_line = placed.iter().filter(|glyph| glyph.line == line);
            let left = of_line.clone().map(|glyph| glyph.x).min();
            let right = of_line
                .map(|glyph| glyph.x + glyph.glyph.width as i32)
                .max();
            if let (Some(left), Some(right)) = (left, right) {
                extent.ink_left = left - ink_left;
                extent.ink_right = right - ink_left;
            }
        }

        let glyphs = placed
            .into_iter()
            .map(|glyph| Glyph {
                glyph: glyph.glyph,
                x: glyph.x - ink_left,
                y: glyph.y - ink_top,
            })
            .collect();

        Layout {
            glyphs,
            block_w,
            block_h,
            advance_w: block_advance,
            lines,
        }
    }
}

/// Rasterises a laid-out block into one coverage mask, then derives the distance field and the
/// bevel data from it.
///
/// `mask_w`/`mask_h` include the block's [`PAD`] on every side, so the field's outside half is
/// real samples instead of a clamp at the block's border. The returned field is `mask_w * mask_h`
/// entries per channel.
/// Turns a laid-out block's antialiased coverage into a text sheet in place: one distance field
/// for the whole block plus the bevel's surface direction, written straight into the sheet's BGRA
/// bytes (see `text/sheet.rs` for the channel layout).
///
/// `width`/`height` include the block's [`PAD`] on every side, so the field's outside half is real
/// samples instead of a clamp at the block's border. `stride` is the sheet's width in pixels: rows
/// past `width` and the area below `height` stay zero, which is what makes the sheet's margin
/// empty. `out` must be `stride * height * 4` bytes, zeroed.
pub fn write_field(coverage: &[u8], width: usize, height: usize, stride: usize, out: &mut [u8]) {
    debug_assert_eq!(coverage.len(), width * height);
    debug_assert!(out.len() >= stride * height * 4);

    /// Chamfer units: 3 = one pixel across, 4 = one pixel diagonally. `u8` saturates at 255
    /// units (85 px), far past [`SDF_RANGE`], so the whole transform is one byte array.
    const ACROSS: u8 = 3;
    const DIAGONAL: u8 = 4;
    const FAR: u8 = u8::MAX;

    // One padding pixel of "outside" on every side, so the sweeps below need no bounds checks:
    // the border is a real row and column of the array. The sweeps walk rows through `windows`,
    // which is also what lets the compiler drop the per-pixel bounds checks this loop cannot
    // afford: `set_text` is a text change, and this is its whole cost.
    let padded = width + 2;
    let total = padded * (height + 2);
    let mut inside = vec![0u8; total];
    for y in 0..height {
        let source = y * width;
        let target = (y + 1) * padded + 1;
        for x in 0..width {
            inside[target + x] = u8::from(coverage[source + x] >= 128);
        }
    }

    // The transform measures the distance to the ink's *boundary*, not separately to each class:
    // a boundary pixel is one whose 4-neighbourhood has a different class. That is half the work
    // of two fields and gives the same curve.
    let mut distance = vec![FAR; total];
    for y in 0..height {
        let row = &mut distance[(y + 1) * padded + 1..][..width];
        let side = &inside[(y + 1) * padded..][..padded];
        let up = &inside[y * padded..];
        let down = &inside[(y + 2) * padded..];
        for (x, cell) in row.iter_mut().enumerate() {
            let here = side[x + 1];
            if here != side[x] || here != side[x + 2] || here != up[x + 1] || here != down[x + 1] {
                *cell = 0;
            }
        }
    }

    // A 3-4 chamfer transform: one sweep from the top-left, one from the bottom-right. Each row is
    // walked with `windows(3)` (the row above/below) and `iter_mut()` (this row), so no index is
    // bounds-checked inside the loop.
    for y in 1..=height {
        let (head, tail) = distance.split_at_mut(y * padded);
        let above = &head[(y - 1) * padded..];
        let row = &mut tail[1..][..width];
        let mut left = FAR;
        for (cell, up) in row.iter_mut().zip(above.windows(3)) {
            let mut best = *cell;
            best = best.min(left.saturating_add(ACROSS));
            best = best.min(up[1].saturating_add(ACROSS));
            best = best.min(up[0].saturating_add(DIAGONAL));
            best = best.min(up[2].saturating_add(DIAGONAL));
            left = best;
            *cell = best;
        }
    }
    for y in (1..=height).rev() {
        let (_, tail) = distance.split_at_mut(y * padded);
        let (row_part, rest) = tail.split_at_mut(padded);
        let row = &mut row_part[1..][..width];
        let below = &rest[..padded];
        let mut right = FAR;
        for (cell, down) in row.iter_mut().rev().zip(below.windows(3).rev()) {
            let mut best = *cell;
            best = best.min(right.saturating_add(ACROSS));
            best = best.min(down[1].saturating_add(ACROSS));
            best = best.min(down[0].saturating_add(DIAGONAL));
            best = best.min(down[2].saturating_add(DIAGONAL));
            right = best;
            *cell = best;
        }
    }

    // The distance byte for every chamfer unit, once: `0.5 = edge`, brighter inside, 0 past the
    // field's range. Inside and outside pixels are one pixel apart when they touch, but the edge
    // runs halfway between two pixel centres, so both sides are pulled in by half a pixel.
    let range = (SDF_RANGE * 3.0).ceil() as i32 + 3;
    let lut_len = (range * 2 + 1) as usize;
    let mut sdf_lut = [0u8; 96];
    for (index, value) in sdf_lut.iter_mut().enumerate().take(lut_len) {
        let units = index as i32 - range;
        let pixels = (units as f32 - (units.signum() as f32) * 1.5) / 3.0;
        let encoded = 0.5 - pixels / (2.0 * SDF_RANGE);
        *value = (encoded * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    let band_units = ((BEVEL_PX + 2.0) * 3.0) as i32;

    for y in 0..height {
        let row = (y + 1) * padded;
        let dist = &distance[row + 1..][..width];
        let side = &inside[row + 1..][..width];
        let up = &distance[row - padded..];
        let down = &distance[row + padded..];
        let up_side = &inside[row - padded..];
        let down_side = &inside[row + padded..];
        let out_row = &mut out[y * stride * 4..][..width * 4];
        for x in 0..width {
            let units = if side[x] == 1 {
                -(i32::from(dist[x]))
            } else {
                i32::from(dist[x])
            };
            let index = (units + range).clamp(0, range * 2) as usize;
            let sdf = sdf_lut[index];
            // B, G, R, A: (unused), normal.y, normal.x, distance.
            let at_out = x * 4;
            out_row[at_out + 3] = sdf;
            // Beyond the field's range (and so in the sheet's margin) every byte stays zero: the
            // sheet's outer pixels are empty, and a bilinear tap there decodes to no surface.
            if sdf == 0 {
                continue;
            }
            // The outward surface direction only matters within a bevel of the edge; deeper
            // inside the surface faces the viewer, which is the encoded (0.5, 0.5).
            out_row[at_out + 1] = 128;
            out_row[at_out + 2] = 128;
            if x > 0 && x + 1 < width && units.abs() < band_units {
                let signed = |value: u8, is_inside: u8| -> f32 {
                    if is_inside == 1 {
                        -f32::from(value)
                    } else {
                        f32::from(value)
                    }
                };
                let gx = signed(dist[x + 1], side[x + 1]) - signed(dist[x - 1], side[x - 1]);
                let gy =
                    signed(down[x + 1], down_side[x + 1]) - signed(up[x + 1], up_side[x + 1]);
                let length = (gx * gx + gy * gy).sqrt();
                if length > 0.001 {
                    out_row[at_out + 1] = ((gy / length * 0.5 + 0.5) * 255.0).round() as u8;
                    out_row[at_out + 2] = ((gx / length * 0.5 + 0.5) * 255.0).round() as u8;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unbounded() -> Face {
        Face::new(crate::text::FACE_CHROME).expect("Unbounded")
    }

    #[test]
    fn line_breaks_split_lines() {
        let face = unbounded();
        assert_eq!(face.layout("ILANA", BAKE_PX).lines.len(), 1);
        assert_eq!(face.layout("IL\nANA", BAKE_PX).lines.len(), 2);
        assert_eq!(face.layout("IL\n\nANA", BAKE_PX).lines.len(), 3);
        assert_eq!(face.layout("IL\nANA\n", BAKE_PX).lines.len(), 3);
    }

    #[test]
    fn each_line_is_centred_in_the_block() {
        let face = unbounded();
        let layout = face.layout("I\nWWWW\nILANA", BAKE_PX);
        assert_eq!(layout.lines.len(), 3);
        let widths: Vec<f32> = layout.lines.iter().map(|line| line.advance).collect();
        assert!(
            widths.iter().any(|width| *width != widths[0]) && widths.len() == 3,
            "the three lines need different advances to test centring: {widths:?}"
        );
        // The centred line's ink sits on the block's centre up to its side bearings.
        for line in &layout.lines {
            let centre = (line.ink_left + line.ink_right) as f32 * 0.5;
            let block_centre = layout.block_w as f32 * 0.5;
            assert!(
                (centre - block_centre).abs() <= BAKE_PX * 0.08,
                "line {line:?} centres at {centre}, the block at {block_centre}"
            );
        }
    }

    #[test]
    fn spacing_comes_from_the_font_advances() {
        let face = unbounded();
        let text = "ILANA";
        let layout = face.layout(text, BAKE_PX);
        // The font's own answer, computed without this module's layout: advance widths plus the
        // GPOS kerning pairs. `block_advance` must be that, to within rounding of the pen.
        let mut expected = 0.0f32;
        let mut previous = None;
        for ch in text.chars() {
            if let Some(previous) = previous
                && let Some(kern) = face.font.horizontal_kern(previous, ch, BAKE_PX)
            {
                expected += kern;
            }
            expected += face.font.metrics(ch, BAKE_PX).advance_width;
            previous = Some(ch);
        }
        let error = (layout.advance_w - expected).abs() / expected;
        assert!(
            error < 0.02,
            "the line is {:.2}px wide, the font's advances say {expected:.2}px ({:.1}% off)",
            layout.advance_w,
            error * 100.0
        );
        // The ink is inside the advance box (side bearings push it in, never out past 2%).
        assert!(
            layout.block_w as f32 <= expected * 1.02,
            "glyph ink {}px is wider than the font's advances {expected:.2}px",
            layout.block_w
        );
    }

    #[test]
    fn empty_text_has_no_glyphs() {
        let face = unbounded();
        for text in ["", "\n", " \n\t\n  "] {
            let layout = face.layout(text, BAKE_PX);
            assert!(layout.glyphs.is_empty(), "text {text:?} produced glyphs");
            assert_eq!(layout.block_w, 0);
            assert_eq!(layout.block_h, 0);
        }
    }

    #[test]
    fn a_repeated_character_is_rasterised_once() {
        let face = unbounded();
        face.layout("ILANA", BAKE_PX);
        let after_first = face.cached();
        assert_eq!(after_first, 4, "ILANA has four distinct characters");
        face.layout("BANANA", BAKE_PX);
        assert_eq!(
            face.cached(),
            after_first + 1,
            "a second text must only rasterise the characters it adds (B), got {} cached",
            face.cached()
        );
        face.layout("ILANA", BAKE_PX);
        assert_eq!(
            face.cached(),
            after_first + 1,
            "laying out a known text must not rasterise anything"
        );
    }

    #[test]
    fn the_distance_field_is_an_edge() {
        // A solid 24x24 square of ink in the middle of the field: the centre is deep inside, the
        // corner is outside, and a pixel just outside the ink is half a sample from it.
        let (width, height) = (24 + 2 * PAD as usize, 24 + 2 * PAD as usize);
        let mut coverage = vec![0u8; width * height];
        for y in 0..24 {
            for x in 0..24 {
                coverage[(y + PAD as usize) * width + x + PAD as usize] = 255;
            }
        }
        let stride = width;
        let mut pixels = vec![0u8; stride * height * 4];
        write_field(&coverage, width, height, stride, &mut pixels);
        let at = |x: usize, y: usize| pixels[(y * stride + x) * 4 + 3];
        let middle = at(PAD as usize + 12, PAD as usize + 12);
        assert!(
            middle > 200,
            "the middle of a solid glyph is inside, got {middle}"
        );
        assert_eq!(at(0, 0), 0, "the field's corner is outside the glyph");
        let outside = at(PAD as usize - 1, PAD as usize + 4);
        assert!(
            (100..160).contains(&outside),
            "a pixel just outside the edge should be near 0.5, got {outside}"
        );
        // The field is 1-Lipschitz: one pixel can only move the distance by one pixel, which is
        // 255 / (2 * SDF_RANGE) ~ 11 levels. A clipped cell shows up as a jump far larger.
        let step = (255.0 / (2.0 * SDF_RANGE)).ceil() as i32 + 2;
        for y in 0..height {
            for x in 1..width {
                let jump = i32::from(at(x, y)) - i32::from(at(x - 1, y));
                assert!(
                    jump.abs() <= step,
                    "the field jumps {jump} levels between ({}, {y}) and ({x}, {y})",
                    x - 1
                );
            }
        }
    }

    #[test]
    fn a_glyph_cell_is_stable() {
        let face = unbounded();
        let first = face.layout("IL", BAKE_PX);
        let second = face.layout("IL", BAKE_PX);
        assert_eq!(first.glyphs.len(), second.glyphs.len());
        for (left, right) in first.glyphs.iter().zip(second.glyphs.iter()) {
            assert_eq!((left.x, left.y), (right.x, right.y));
            assert!(Rc::ptr_eq(&left.glyph, &right.glyph), "the bake is cached");
        }
    }
}
