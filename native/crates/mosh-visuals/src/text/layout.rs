//! Font loading, glyph rasterising, the signed distance field and line layout.
//!
//! Everything in this module runs inside [`crate::text::TextLayer::set_text`], when a text clip
//! starts or its words change. Nothing here runs per frame: `draw` only records one pass.
//!
//! # Why an SDF and not a coverage bitmap
//!
//! The sheet is baked once, at [`BAKE_PX`] em, while `size` (`p0`) and the per-glyph beat pop
//! scale it every frame. A signed distance field stays smooth under that scaling: the shader
//! thresholds the distance with a screen-space antialiasing width, so a glyph three times its
//! baked size still has a clean edge. A coverage bitmap would blur or stair-step.
//!
//! The distance field is a 3-4 chamfer transform: two sweeps over the glyph's cell, each pixel
//! taking the cheapest of its already-visited neighbours plus a step. That is a handful of integer
//! adds per pixel, not a quadratic search, so the whole bake stays inside a text change.
//!
//! # Layout
//!
//! Lines come from `\n`. Each line is laid out at its own pen, then centred inside the block (the
//! widest line's ink width), and the block keeps the union of the ink boxes, not the font's line
//! box, so an empty last line cannot push the words off centre. Kerning comes from the font when
//! it has a pair.

use std::collections::HashMap;

use fontdue::{Font, FontSettings};

/// Em size the glyphs are rasterised at. The sheet is a distance field, so a layer that draws at
/// 2x this size is still crisp; this is the one number that trades `set_text` time for sharpness.
pub const BAKE_PX: f32 = 96.0;

/// How far, in baked pixels, a distance sample stays meaningful on each side of the edge. The
/// shader reads the sheet as `0.5 = edge` and converts back with the same constant.
pub const SDF_RANGE: f32 = 6.0;

/// Room around a glyph's ink for the distance field's outside half, in baked pixels.
pub const PAD: i32 = 7;

/// A loaded face. `fontdue::Font` is the parser and rasteriser; the face's licence is in
/// `native/assets/fonts/`.
pub struct Face {
    font: Font,
}

/// One rasterised glyph, ready to be packed into a sheet cell.
pub struct Glyph {
    /// The glyph's distance field: `cell_w * cell_h` bytes, 128 = half a byte off the edge,
    /// brighter is inside. The cell includes [`PAD`] of margin on every side.
    pub sdf: Vec<u8>,
    pub cell_w: u32,
    pub cell_h: u32,
    /// The cell's top-left corner in block pixels. May be negative: an italic overhang or a
    /// kerned pair can stick out past the block's ink box, which is why the sheet has a margin.
    pub x: i32,
    pub y: i32,
    /// The glyph's own noise value, 0..1, so the shader can jitter and rotate glyphs instead of
    /// the whole word. Derived from the glyph index and the character, so it is stable.
    pub random: f32,
}

/// One line's ink extent after centring, in block pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Line {
    pub ink_left: i32,
    pub ink_right: i32,
}

/// A laid-out string: the glyph cells and the ink box the sheet has to hold.
#[derive(Default)]
pub struct Layout {
    pub glyphs: Vec<Glyph>,
    /// Ink size of the whole block, in baked pixels.
    pub block_w: i32,
    pub block_h: i32,
    /// One entry per line, in reading order, already centred in the block. Read by this module's
    /// tests (line breaks, centring); the shader only needs the block's box.
    #[cfg_attr(not(test), allow(dead_code))]
    pub lines: Vec<Line>,
}

/// A glyph before the block's origin and the per-line centring are known.
struct Raw {
    sdf: Vec<u8>,
    cell_w: u32,
    cell_h: u32,
    /// Bitmap ink box, in line pixels: left edge and top edge, y down.
    left: i32,
    top: i32,
    random: f32,
}

/// Where one line's glyphs are in the raw list, and the ink extent they cover.
struct RawLine {
    first: usize,
    ink_left: i32,
    ink_right: i32,
}

impl Face {
    /// Loads a face from the bytes committed under `native/assets/fonts/`. Returns `None` when the
    /// font cannot be parsed; the layer then draws black instead of failing to start.
    pub fn new(bytes: &'static [u8]) -> Option<Face> {
        Font::from_bytes(bytes, FontSettings::default())
            .ok()
            .map(|font| Face { font })
    }

    /// Lays `text` out at `px` em. Empty (or whitespace-only) text has no glyphs and no block.
    pub fn layout(&self, text: &str, px: f32) -> Layout {
        let line_metrics = self.font.horizontal_line_metrics(px);
        let (ascent, line_height) = match line_metrics {
            Some(metrics) => (metrics.ascent, metrics.new_line_size + px * 0.10),
            None => (px * 0.8, px * 1.2),
        };

        let mut raw: Vec<Raw> = Vec::new();
        let mut glyph_cells: HashMap<char, (u32, u32, Vec<u8>)> = HashMap::new();
        let mut raw_lines: Vec<RawLine> = Vec::new();
        let mut baseline = ascent;
        let mut glyph_index = 0u32;

        for line_text in text.split('\n') {
            let line_text = line_text.trim_end_matches('\r');
            let first = raw.len();
            let mut pen = 0.0f32;
            let mut previous: Option<char> = None;
            let mut ink_left = i32::MAX;
            let mut ink_right = i32::MIN;

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
                let (metrics, coverage) = self.font.rasterize(ch, px);
                let advance = metrics.advance_width;
                let random = noise(glyph_index, ch);
                glyph_index += 1;
                let width = metrics.width as i32;
                let height = metrics.height as i32;
                if width > 0 && height > 0 {
                    let left = pen.round() as i32 + metrics.xmin;
                    let top = baseline.round() as i32 - (metrics.ymin + height);
                    ink_left = ink_left.min(left);
                    ink_right = ink_right.max(left + width);
                    // A word repeats letters; the distance field of a repeated glyph is the same
                    // field, so it is computed once per character per bake and cloned after.
                    let (cell_w, cell_h, sdf) = glyph_cells.entry(ch).or_insert_with(|| {
                        (
                            (metrics.width + 2 * PAD as usize) as u32,
                            (metrics.height + 2 * PAD as usize) as u32,
                            distance_field(&coverage, metrics.width, metrics.height),
                        )
                    });
                    raw.push(Raw {
                        sdf: sdf.clone(),
                        cell_w: *cell_w,
                        cell_h: *cell_h,
                        left,
                        top,
                        random,
                    });
                }
                pen += advance;
            }
            // A line of only spaces has no ink: it still counts as a line (so `\n\n` is a blank
            // line) but must not widen the block.
            if ink_left > ink_right {
                ink_left = 0;
                ink_right = 0;
            }
            raw_lines.push(RawLine {
                first,
                ink_left,
                ink_right,
            });
            baseline += line_height;
        }

        let block_left = raw_lines
            .iter()
            .map(|line| line.ink_left)
            .min()
            .unwrap_or(0);
        let block_right = raw_lines
            .iter()
            .map(|line| line.ink_right)
            .max()
            .unwrap_or(0);
        let block_top = raw.iter().map(|glyph| glyph.top).min().unwrap_or(0);
        let block_bottom = raw
            .iter()
            .map(|glyph| glyph.top + (glyph.cell_h as i32 - 2 * PAD))
            .max()
            .unwrap_or(0);
        let block_w = (block_right - block_left).max(0);
        let block_h = (block_bottom - block_top).max(0);

        // Centre each line in the block, then shift the block's ink box to (0, 0). The raw glyphs
        // are contiguous and in line order, so they drain in step with the lines.
        let mut glyphs = Vec::with_capacity(raw.len());
        let mut lines = Vec::with_capacity(raw_lines.len());
        let mut raw = raw.into_iter();
        for (line_index, line) in raw_lines.iter().enumerate() {
            let line_w = line.ink_right - line.ink_left;
            let centre_offset = (block_w - line_w) / 2 - line.ink_left + block_left;
            lines.push(Line {
                ink_left: line.ink_left + centre_offset - block_left,
                ink_right: line.ink_right + centre_offset - block_left,
            });
            let end = raw_lines
                .get(line_index + 1)
                .map(|next| next.first)
                .unwrap_or(usize::MAX);
            let count = end.saturating_sub(line.first);
            for glyph in raw.by_ref().take(count) {
                glyphs.push(Glyph {
                    sdf: glyph.sdf,
                    cell_w: glyph.cell_w,
                    cell_h: glyph.cell_h,
                    x: glyph.left + centre_offset - block_left - PAD,
                    y: glyph.top - block_top - PAD,
                    random: glyph.random,
                });
            }
        }

        Layout {
            glyphs,
            block_w,
            block_h,
            lines,
        }
    }
}

/// A stable 0..1 noise value per glyph (the shader's per-glyph jitter and rotation).
fn noise(index: u32, ch: char) -> f32 {
    let mut value = index.wrapping_mul(0x9e37_79b9) ^ (ch as u32).wrapping_mul(0x85eb_ca6b);
    value ^= value >> 15;
    value = value.wrapping_mul(0x2545_f491);
    value ^= value >> 13;
    (value & 0x00ff_ffff) as f32 / 0x0100_0000u32 as f32
}

/// Signed distance field of one glyph's coverage bitmap, in its padded cell.
///
/// A 3-4 chamfer distance transform: two sweeps over the cell, each pixel taking the cheapest of
/// its already-visited neighbours plus 3 (across) or 4 (diagonal). Integer units of a third of a
/// pixel, which is why the distances divide by 3 below. The 6% error of a chamfer against an exact
/// Euclidean distance is invisible at these edge widths and costs a fraction of the arithmetic:
/// the whole bake has to stay inside a text change, not a frame.
fn distance_field(coverage: &[u8], width: usize, height: usize) -> Vec<u8> {
    let pad = PAD as usize;
    let cell_w = width + 2 * pad;
    let cell_h = height + 2 * pad;
    let count = cell_w * cell_h;
    let mut inside = vec![false; count];
    for y in 0..height {
        for x in 0..width {
            inside[(y + pad) * cell_w + x + pad] = coverage[y * width + x] >= 128;
        }
    }

    /// Chamfer units: 3 = one pixel across, 4 = one pixel diagonally.
    const ACROSS: u32 = 3;
    const DIAGONAL: u32 = 4;
    const FAR: u32 = 1 << 24;

    let mut to_inside = vec![FAR; count];
    let mut to_outside = vec![FAR; count];
    for (index, is_inside) in inside.iter().enumerate() {
        if *is_inside {
            to_inside[index] = 0;
        } else {
            to_outside[index] = 0;
        }
    }

    // One sweep in each direction, the two fields carried together so the cell is read once per
    // pass. `backward` walks from the bottom-right corner, which is the other half of the chamfer.
    fn sweep(
        inside: &mut [u32],
        outside: &mut [u32],
        cell_w: usize,
        cell_h: usize,
        backward: bool,
    ) {
        for step_y in 0..cell_h {
            let y = if backward {
                cell_h - 1 - step_y
            } else {
                step_y
            };
            for step_x in 0..cell_w {
                let x = if backward {
                    cell_w - 1 - step_x
                } else {
                    step_x
                };
                let at = y * cell_w + x;
                let (mut nearest_inside, mut nearest_outside) = (inside[at], outside[at]);
                // The neighbour across, already visited: left going forwards, right backwards.
                let across_neighbour = if backward {
                    if x + 1 < cell_w { Some(at + 1) } else { None }
                } else if x > 0 {
                    Some(at - 1)
                } else {
                    None
                };
                if let Some(neighbour) = across_neighbour {
                    nearest_inside = nearest_inside.min(inside[neighbour] + ACROSS);
                    nearest_outside = nearest_outside.min(outside[neighbour] + ACROSS);
                }
                // The neighbour row, already visited: the row above going forwards, below backwards.
                let visited_row = if backward {
                    if y + 1 < cell_h { Some(y + 1) } else { None }
                } else if y > 0 {
                    Some(y - 1)
                } else {
                    None
                };
                if let Some(up) = visited_row {
                    let same = up * cell_w + x;
                    nearest_inside = nearest_inside.min(inside[same] + ACROSS);
                    nearest_outside = nearest_outside.min(outside[same] + ACROSS);
                    if x > 0 {
                        let neighbour = up * cell_w + x - 1;
                        nearest_inside = nearest_inside.min(inside[neighbour] + DIAGONAL);
                        nearest_outside = nearest_outside.min(outside[neighbour] + DIAGONAL);
                    }
                    if x + 1 < cell_w {
                        let neighbour = up * cell_w + x + 1;
                        nearest_inside = nearest_inside.min(inside[neighbour] + DIAGONAL);
                        nearest_outside = nearest_outside.min(outside[neighbour] + DIAGONAL);
                    }
                }
                inside[at] = nearest_inside;
                outside[at] = nearest_outside;
            }
        }
    }
    sweep(&mut to_inside, &mut to_outside, cell_w, cell_h, false);
    sweep(&mut to_inside, &mut to_outside, cell_w, cell_h, true);

    let mut sdf = vec![0u8; count];
    for (index, is_inside) in inside.iter().enumerate() {
        let signed = if *is_inside {
            -(to_outside[index] as f32) / ACROSS as f32
        } else {
            to_inside[index] as f32 / ACROSS as f32
        };
        // Inside and outside pixels are one pixel apart when they touch, but the edge runs halfway
        // between two pixel centres.
        let signed = (signed - signed.signum() * 0.5).clamp(-SDF_RANGE, SDF_RANGE);
        let value = 0.5 - signed / (2.0 * SDF_RANGE);
        sdf[index] = (value * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    sdf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ubuntu() -> Face {
        Face::new(crate::text::FACE_DISPLAY).expect("Ubuntu Light")
    }

    #[test]
    fn line_breaks_split_lines() {
        let face = ubuntu();
        assert_eq!(face.layout("ILANA", BAKE_PX).lines.len(), 1);
        assert_eq!(face.layout("IL\nANA", BAKE_PX).lines.len(), 2);
        assert_eq!(face.layout("IL\n\nANA", BAKE_PX).lines.len(), 3);
        assert_eq!(face.layout("IL\nANA\n", BAKE_PX).lines.len(), 3);
    }

    #[test]
    fn each_line_is_centred_in_the_block() {
        let face = ubuntu();
        let layout = face.layout("I\nWWWW\nILANA", BAKE_PX);
        assert_eq!(layout.lines.len(), 3);
        let widths: Vec<i32> = layout
            .lines
            .iter()
            .map(|line| line.ink_right - line.ink_left)
            .collect();
        let mut by_width = widths.clone();
        by_width.sort_unstable();
        assert!(
            by_width[0] < by_width[2],
            "the three lines must have different widths to test centring: {widths:?}"
        );
        for line in &layout.lines {
            let centre = line.ink_left + (line.ink_right - line.ink_left) / 2;
            let block_centre = layout.block_w / 2;
            assert!(
                (centre - block_centre).abs() <= 2,
                "line {line:?} centres at {centre}, the block at {block_centre}"
            );
        }
    }

    #[test]
    fn empty_text_has_no_glyphs() {
        let face = ubuntu();
        for text in ["", "\n", " \n\t\n  "] {
            let layout = face.layout(text, BAKE_PX);
            assert!(layout.glyphs.is_empty(), "text {text:?} produced glyphs");
            assert_eq!(layout.block_w, 0);
            assert_eq!(layout.block_h, 0);
        }
    }

    #[test]
    fn the_distance_field_is_an_edge() {
        // A solid 8x8 square of ink in the middle of the cell: the centre is deep inside, the
        // cell's corner is outside, and a pixel just outside the ink is half a sample from it.
        let (width, height) = (8, 8);
        let coverage = vec![255u8; width * height];
        let sdf = distance_field(&coverage, width, height);
        let cell_w = width + 2 * PAD as usize;
        let at = |x: usize, y: usize| sdf[y * cell_w + x];
        let middle = at(PAD as usize + width / 2, PAD as usize + height / 2);
        assert!(
            middle > 190,
            "the middle of a solid glyph is inside, got {middle}"
        );
        assert_eq!(at(0, 0), 0, "the cell's corner is outside the glyph");
        let outside = at(PAD as usize - 1, PAD as usize + height / 2);
        assert!(
            (100..160).contains(&outside),
            "a pixel just outside the edge should be near 0.5, got {outside}"
        );
    }

    #[test]
    fn a_glyph_cell_is_stable() {
        let face = ubuntu();
        let first = face.layout("IL", BAKE_PX);
        let second = face.layout("IL", BAKE_PX);
        assert_eq!(first.glyphs.len(), second.glyphs.len());
        for (left, right) in first.glyphs.iter().zip(second.glyphs.iter()) {
            assert_eq!(
                (left.x, left.y, left.cell_w, left.cell_h),
                (right.x, right.y, right.cell_w, right.cell_h)
            );
            assert_eq!(left.sdf, right.sdf);
            assert_eq!(left.random, right.random);
        }
    }
}
