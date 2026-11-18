//! Font loading, glyph rasterising, pair kerning, line wrapping, the signed distance field and the
//! block layout.
//!
//! Everything in this module runs inside [`crate::text::TextLayer::set_text`], when a text clip
//! starts or its words change. Nothing here runs per frame: `draw` only records one pass.
//!
//! # One field per glyph tile, composed into one field for the block
//!
//! Each glyph is rasterised into its own *tile*: its ink box plus [`PAD`] on every side, and the
//! distance transform runs over that tile. A tile therefore holds the true distance to that
//! glyph's edge out to [`SDF_RANGE`], which is what lets the sheet take the nearest tile per pixel
//! (`Sheet::compose`): the result is the block's own distance field, and there is no cell for an
//! outline, a glow or a bevel to be cut at. A tile is a pure function of (face, character, em
//! size), so it is cached and a new text re-rasterises only the characters it introduces.
//!
//! # Kerning
//!
//! Spacing is the font's own: every glyph's advance width plus the pair adjustment the font's
//! **GPOS** table gives for the pair before it. `fontdue` only reads the legacy `kern` table, and
//! none of the three display faces has one, so the GPOS lookup (PairPos formats 1 and 2) is read
//! here through `ttf-parser` — fontdue's own parser, already on the dependency graph. Pair values
//! are cached per (left, right) in font units; the pixel value is `units * px / units_per_em`.
//!
//! # Wrapping and the length cap
//!
//! A display line is wrapped to [`WRAP_EM`] em of advance width (word wrap, breaking a word that is
//! longer than a whole line), because the shader fits the block to the frame: an unwrapped 100
//! character line would be fitted down to a stripe a few percent of the frame tall. Explicit `\n`
//! breaks are kept. At most [`MAX_LINES`] visual lines are drawn and at most [`MAX_CHARS`]
//! characters are laid out; anything beyond is dropped and logged once, so a pasted essay cannot
//! produce a smear or an empty frame.
//!
//! # Sub-pixel edges
//!
//! The rasteriser's antialiased coverage says where inside a pixel the edge is, so a pixel whose
//! coverage is partial gets its distance from the coverage instead of from the pixel grid. Without
//! that, every edge is quantised to whole baked pixels, which is visible as stair-stepping once
//! the shader scales the block up (the sheet is baked at [`BAKE_PX`] em and a fitted line at 540p
//! draws it 3-4x larger).

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use fontdue::{Font, FontSettings};

/// Em size the glyphs are rasterised at. The sheet is a distance field, so a layer that draws at
/// 2x this size is still crisp; this is the one number that trades `set_text` time for sharpness.
pub const BAKE_PX: f32 = 96.0;

/// How far, in baked pixels, a distance sample stays meaningful on each side of the edge. The
/// shader reads the sheet as `0.5 = edge` and converts back with the same constant.
pub const SDF_RANGE: f32 = 12.0;

/// Baked pixels over which the bevel's height falls from the stroke's centre to its edge.
pub const BEVEL_PX: f32 = 9.0;

/// Room around a glyph's ink for the distance field's outside half, in baked pixels: the field's
/// range plus two pixels of slack so a bilinear sample at the block's edge never reads the
/// texture's clamp.
pub const PAD: i32 = SDF_RANGE as i32 + 2;

/// A visual line wraps at this many characters' worth of advance (measured on the face's `o`), so
/// every face gets a similar number of letters per line — a fixed em count gives the wide
/// blackletter seven characters and the narrow grotesque twenty.
pub const WRAP_CHARS: f32 = 9.0;

/// The most visual lines a block draws (explicit `\n` breaks included).
pub const MAX_LINES: usize = 7;

/// The most characters a block lays out; the rest is dropped and logged.
pub const MAX_CHARS: usize = 600;

/// Tiles kept per face. A few hundred glyphs at one em size is far more than a show uses; the cap
/// is there so a long lyric cannot grow the cache without bound.
pub const CACHE_TILES: usize = 192;

/// A loaded face: fontdue for rasterising, ttf-parser for the GPOS pair adjustments.
pub struct Face {
    font: Font,
    outline: ttf_parser::Face<'static>,
    units_per_em: f32,
    /// GPOS pair adjustment in font units, cached per (left, right).
    kerning: RefCell<HashMap<(char, char), i16>>,
    glyphs: RefCell<GlyphCache>,
    /// True when this face's look shades with the baked surface direction (chrome, sigil). The
    /// clean look only needs the distance, so its bake skips the gradient pass.
    normals: bool,
    /// One line of text at [`BAKE_PX`], ascent to the next baseline.
    line_height: f32,
}

/// Tiles on one face, least recently used first.
struct GlyphCache {
    map: HashMap<CacheKey, Rc<CachedGlyph>>,
    order: VecDeque<CacheKey>,
}

impl GlyphCache {
    fn get(&mut self, key: &CacheKey) -> Option<Rc<CachedGlyph>> {
        let glyph = self.map.get(key).cloned()?;
        if let Some(at) = self.order.iter().position(|held| held == key) {
            self.order.remove(at);
        }
        self.order.push_back(*key);
        Some(glyph)
    }

    fn insert(&mut self, key: CacheKey, glyph: Rc<CachedGlyph>) {
        self.map.insert(key, glyph);
        self.order.push_back(key);
        while self.order.len() > CACHE_TILES {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.map.remove(&oldest);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct CacheKey {
    ch: char,
    /// The em size in 1/256 pixel. The size is a fixed [`BAKE_PX`] for every text, so this is
    /// effectively per face and character; the field keeps a later ladder honest.
    px_256: u32,
}

/// One rasterised glyph: its baked field tile and the metrics needed to place it.
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
    /// The font's advance for this character, in pixels.
    pub advance: f32,
}

/// One glyph placed in the laid-out block.
pub struct Glyph {
    pub glyph: Rc<CachedGlyph>,
    /// Cell top-left in block pixels, y down. The cell is the ink box; the field's padding lives
    /// in the tile, not per glyph, so these boxes are allowed to overlap (kerning does).
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
    /// One entry per visual line, in reading order, already centred in the block.
    #[cfg_attr(not(test), allow(dead_code))]
    pub lines: Vec<Line>,
    /// Characters the length cap dropped from the end of the text (0 for a text that fits).
    #[cfg_attr(not(test), allow(dead_code))]
    pub dropped: usize,
}

/// One glyph before the block's origin and the per-line centring are known.
struct Placed {
    glyph: Rc<CachedGlyph>,
    x: i32,
    y: i32,
    line: usize,
}

/// One visual line: the character range it draws, and its advance width.
struct VisualLine {
    first: usize,
    end: usize,
    advance: f32,
}

impl Face {
    /// Loads a face from the bytes committed under `native/assets/fonts/`. Returns `None` when the
    /// font cannot be parsed; the layer then draws black instead of failing to start.
    ///
    /// `normals` asks for the baked surface direction in the tiles (the chrome and sigil looks);
    /// the clean look passes `false` and its bake skips that pass.
    pub fn new(bytes: &'static [u8], normals: bool) -> Option<Face> {
        let font = Font::from_bytes(bytes, FontSettings::default()).ok()?;
        let outline = ttf_parser::Face::parse(bytes, 0).ok()?;
        let units_per_em = f32::from(outline.units_per_em());
        let line_height = font
            .horizontal_line_metrics(BAKE_PX)
            .map(|metrics| metrics.new_line_size + BAKE_PX * 0.10)
            .unwrap_or(BAKE_PX * 1.2);
        Some(Face {
            font,
            outline,
            units_per_em,
            kerning: RefCell::new(HashMap::new()),
            glyphs: RefCell::new(GlyphCache {
                map: HashMap::new(),
                order: VecDeque::new(),
            }),
            normals,
            line_height,
        })
    }

    /// One line of text at [`BAKE_PX`], ascent to the next baseline. The sheet's capacity is
    /// derived from this.
    pub fn line_height(&self) -> f32 {
        self.line_height
    }

    /// The advance width a visual line wraps to, at `px`: [`WRAP_CHARS`] of this face's own
    /// letters, clamped so no face gets a silly line.
    pub fn wrap_width(&self, px: f32) -> f32 {
        let reference = self.advance('o', px).max(px * 0.35);
        (WRAP_CHARS * reference).clamp(px * 5.0, px * 16.0)
    }

    /// The GPOS pair adjustment for `left` followed by `right`, in em. Zero when the face has no
    /// such pair. This is the real kerning: `fontdue`'s own call reads only the legacy `kern`
    /// table, which none of the display faces has.
    pub fn kern_em(&self, left: char, right: char) -> f32 {
        let units = {
            let cache = self.kerning.borrow();
            cache.get(&(left, right)).copied()
        };
        let units = match units {
            Some(units) => units,
            None => {
                let units = self.gpos_pair(left, right);
                self.kerning.borrow_mut().insert((left, right), units);
                units
            }
        };
        f32::from(units) / self.units_per_em
    }

    /// The `kern` feature's PairPos adjustment for the pair, in font units.
    fn gpos_pair(&self, left: char, right: char) -> i16 {
        let Some(table) = self.outline.tables().gpos else {
            return 0;
        };
        let (Some(left), Some(right)) = (
            self.outline.glyph_index(left),
            self.outline.glyph_index(right),
        ) else {
            return 0;
        };
        // Only the pair-positioning lookups the `kern` feature names: mark attachment and the
        // other GPOS lookups move glyphs in ways a display line does not want here.
        let kern_tag = u32::from_be_bytes(*b"kern");
        let mut total = 0i16;
        for feature_index in 0..table.features.len() {
            let Some(feature) = table.features.get(feature_index) else {
                continue;
            };
            if feature.tag.0 != kern_tag {
                continue;
            }
            for slot in 0..feature.lookup_indices.len() {
                let Some(index) = feature.lookup_indices.get(slot) else {
                    continue;
                };
                let Some(lookup) = table.lookups.get(index) else {
                    continue;
                };
                if let Some(value) = pair_value(&lookup, left, right) {
                    total += value;
                    break;
                }
            }
        }
        total
    }

    /// The font's advance for `ch`, in pixels, without rasterising it.
    fn advance(&self, ch: char, px: f32) -> f32 {
        self.font.metrics(ch, px).advance_width
    }

    /// One character's baked field tile, from the cache when this face already baked it.
    fn glyph(&self, ch: char, px: f32) -> Rc<CachedGlyph> {
        let key = CacheKey {
            ch,
            px_256: (px * 256.0).round().max(1.0) as u32,
        };
        if let Some(hit) = self.glyphs.borrow_mut().get(&key) {
            return hit;
        }
        let (metrics, coverage) = self.font.rasterize(ch, px);
        let (width, height) = (metrics.width, metrics.height);
        let pad = PAD as usize;
        let tile_w = width + 2 * pad;
        let tile_h = height + 2 * pad;
        let mut mask = vec![0u8; tile_w * tile_h];
        for row in 0..height {
            let source = row * width;
            let target = (row + pad) * tile_w + pad;
            mask[target..target + width].copy_from_slice(&coverage[source..source + width]);
        }
        let mut tile = vec![0u8; tile_w * tile_h * 4];
        write_field(&mask, tile_w, tile_h, tile_w, self.normals, &mut tile);
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

    /// Lays `text` out at `px` em: wrapped, capped and centred. Empty (or whitespace-only) text
    /// has no glyphs and no block.
    pub fn layout(&self, text: &str, px: f32) -> Layout {
        let ascent = self
            .font
            .horizontal_line_metrics(px)
            .map(|metrics| metrics.ascent)
            .unwrap_or(px * 0.8);
        let line_height = self.line_height * (px / BAKE_PX);

        // The characters the cap allows, with the per-glyph step (advance + the kern it takes
        // from the character before it) so a line's width is a prefix-sum lookup.
        let chars: Vec<char> = text
            .chars()
            .map(|ch| if ch == '\t' { ' ' } else { ch })
            .filter(|ch| !ch.is_control() || *ch == '\n')
            .take(MAX_CHARS)
            .collect();
        let dropped = text.chars().count().saturating_sub(chars.len());
        let mut steps = Vec::with_capacity(chars.len());
        let mut previous = None;
        for ch in &chars {
            let advance = if *ch == '\n' {
                continue;
            } else {
                self.advance(*ch, px)
            };
            let kern = previous
                .filter(|_| *ch != '\n')
                .map(|left: char| self.kern_em(left, *ch) * px)
                .unwrap_or(0.0);
            steps.push(advance + kern);
            previous = Some(*ch);
        }

        // Greedy wrap on the advance width, preferring the last space in the line.
        let wrap = self.wrap_width(px);
        let mut lines: Vec<VisualLine> = Vec::new();
        let mut first = 0usize;
        let mut widths = vec![0.0f32; chars.len() + 1];
        let mut step_index = 0usize;
        for (index, ch) in chars.iter().enumerate() {
            widths[index + 1] = widths[index]
                + if *ch == '\n' {
                    0.0
                } else {
                    let value = steps[step_index];
                    step_index += 1;
                    value
                };
        }
        while first < chars.len() {
            if lines.len() == MAX_LINES {
                break;
            }
            if chars[first] == '\n' {
                // A blank line: two breaks in a row, or one right at the start.
                lines.push(VisualLine {
                    first,
                    end: first,
                    advance: 0.0,
                });
                first += 1;
                continue;
            }
            // The furthest character that still fits, and the last space before it.
            let mut end = first;
            while end < chars.len()
                && chars[end] != '\n'
                && widths[end + 1] - widths[first] <= wrap
            {
                end += 1;
            }
            if end == first {
                end = first + 1;
            }
            // The last space, but only when the line has to be broken at all: a line that ends
            // the text (or reaches an explicit break) is not re-broken at its own space.
            let mut break_at = end;
            let must_break = end < chars.len() && chars[end] != '\n';
            if must_break {
                for index in (first + 1..=end).rev() {
                    if chars[index - 1] == ' ' {
                        break_at = index - 1;
                        break;
                    }
                }
                if break_at == first {
                    break_at = end;
                }
            }
            lines.push(VisualLine {
                first,
                end: break_at,
                advance: widths[break_at] - widths[first],
            });
            first = break_at;
            if first < chars.len() && chars[first] == '\n' {
                // The break this line was measured up to.
                first += 1;
            } else {
                while first < chars.len() && chars[first] == ' ' {
                    first += 1;
                }
            }
        }
        // A text that ends in a break has a last, empty line, and `\n` alone is one empty line.
        if first == chars.len() && chars.last() == Some(&'\n') && lines.len() < MAX_LINES {
            lines.push(VisualLine {
                first,
                end: first,
                advance: 0.0,
            });
        }
        let dropped = dropped + chars.len().saturating_sub(first.min(chars.len()));

        // Place every glyph, line by line. Kerning is the font's, and only inside a line.
        let mut placed: Vec<Placed> = Vec::new();
        let mut baseline = ascent;
        for (line_index, line) in lines.iter().enumerate() {
            let mut pen = 0.0f32;
            let mut previous: Option<char> = None;
            for ch in &chars[line.first..line.end.max(line.first)] {
                if let Some(left) = previous {
                    pen += self.kern_em(left, *ch) * px;
                }
                previous = Some(*ch);
                let glyph = self.glyph(*ch, px);
                if glyph.width > 0 && glyph.height > 0 {
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
            baseline += line_height;
        }

        let block_advance = lines
            .iter()
            .map(|line| line.advance)
            .fold(0.0f32, f32::max);

        // Centre every line by its advance width, not by its ink: a line's side bearings are part
        // of its width, and centring by ink is what puts a line visibly off the block's centre.
        for glyph in &mut placed {
            let line_advance = lines[glyph.line].advance;
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

        let mut extents: Vec<Line> = lines
            .iter()
            .map(|line| Line {
                ink_left: 0,
                ink_right: 0,
                advance: line.advance,
            })
            .collect();
        for (line, extent) in extents.iter_mut().enumerate() {
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
            lines: extents,
            dropped,
        }
    }
}

/// Turns a laid-out block's antialiased coverage into field tiles in place: one distance field for
/// the mask plus, when `normals`, the bevel's surface direction, written as BGRA (see
/// `text/sheet.rs` for the channel layout).
///
/// `width`/`height` include the block's [`PAD`] on every side, so the field's outside half is real
/// samples instead of a clamp at the block's border. `stride` is the sheet's width in pixels: rows
/// past `width` stay zero, which is what makes the margin empty. `out` must be `stride * height *
/// 4` bytes, zeroed.
pub fn write_field(
    coverage: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    normals: bool,
    out: &mut [u8],
) {
    debug_assert_eq!(coverage.len(), width * height);
    debug_assert!(out.len() >= stride * height * 4);

    /// Chamfer units: 3 = one pixel across, 4 = one pixel diagonally. `u8` saturates at 255 units
    /// (85 px), far past [`SDF_RANGE`], so the whole transform is one byte array.
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

    // A 3-4 chamfer transform: one sweep from the top-left, one from the bottom-right.
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

    let band_units = (BEVEL_PX + 2.0) * 3.0;
    for y in 0..height {
        let out_row = &mut out[y * stride * 4..][..width * 4];
        for x in 0..width {
            let pixels = signed_at(x, y, width, padded, &distance, &inside, coverage);
            let encoded = 0.5 - pixels / (2.0 * SDF_RANGE);
            let sdf = (encoded * 255.0).round().clamp(0.0, 255.0) as u8;
            let at_out = x * 4;
            out_row[at_out + 3] = sdf;
            // Beyond the field's range (and so in the sheet's margin) every byte stays zero: the
            // sheet's outer pixels are empty, and a bilinear tap there decodes to no surface.
            if sdf == 0 || !normals {
                continue;
            }
            // The outward surface direction only matters within a bevel of the edge; deeper
            // inside the surface faces the viewer, which is the encoded (0.5, 0.5).
            out_row[at_out + 1] = 128;
            out_row[at_out + 2] = 128;
            if x == 0 || x + 1 >= width || y == 0 || y + 1 >= height {
                continue;
            }
            if pixels.abs() * 3.0 >= band_units {
                continue;
            }
            let gx = signed_at(x + 1, y, width, padded, &distance, &inside, coverage)
                - signed_at(x - 1, y, width, padded, &distance, &inside, coverage);
            let gy = signed_at(x, y + 1, width, padded, &distance, &inside, coverage)
                - signed_at(x, y - 1, width, padded, &distance, &inside, coverage);
            let length = (gx * gx + gy * gy).sqrt();
            if length > 0.001 {
                out_row[at_out + 1] = ((gy / length * 0.5 + 0.5) * 255.0).round() as u8;
                out_row[at_out + 2] = ((gx / length * 0.5 + 0.5) * 255.0).round() as u8;
            }
        }
    }
}

/// The GPOS PairPos adjustment a lookup gives the pair, in font units, if it applies.
fn pair_value(
    lookup: &ttf_parser::opentype_layout::Lookup<'_>,
    left: ttf_parser::GlyphId,
    right: ttf_parser::GlyphId,
) -> Option<i16> {
    use ttf_parser::gpos::PositioningSubtable as Subtable;
    for slot in 0..lookup.subtables.len() {
        let Some(subtable) = lookup.subtables.get::<Subtable<'_>>(slot) else {
            continue;
        };
        let Subtable::Pair(pair) = subtable else {
            continue;
        };
        let Some(index) = pair.coverage().get(left) else {
            continue;
        };
        let value = match pair {
            ttf_parser::gpos::PairAdjustment::Format1 { sets, .. } => sets
                .get(index)
                .and_then(|set| set.get(right))
                .map(|(first, _)| first.x_advance),
            ttf_parser::gpos::PairAdjustment::Format2 {
                classes, matrix, ..
            } => matrix
                .get((classes.0.get(left), classes.1.get(right)))
                .map(|(first, _)| first.x_advance),
        };
        if let Some(value) = value {
            return Some(value);
        }
    }
    None
}

/// The signed distance of one mask pixel, in pixels, positive outside.
///
/// The chamfer transform measures the distance to the nearest *boundary pixel*, which puts every
/// edge on the rasteriser's grid; scaling that up 3-4x at 540p is visible as stair-steps. The
/// antialiased coverage says where inside a pixel the edge really is, so within a couple of pixels
/// of the edge the two are blended, weighted by how close the pixel is. Both estimates agree at
/// a pixel or two out (the transform is exact there, the coverage saturates), so the blend has no
/// seam of its own — unlike using the coverage for partial pixels only, which made the neighbours
/// of a partial pixel disagree by a whole pixel and scalloped the outline.
fn signed_at(
    x: usize,
    y: usize,
    width: usize,
    padded: usize,
    distance: &[u8],
    inside: &[u8],
    coverage: &[u8],
) -> f32 {
    let index = (y + 1) * padded + x + 1;
    let value = f32::from(distance[index]);
    let signed = if inside[index] == 1 { -value } else { value };
    // Half a pixel in chamfer units is 1.5; the edge runs between pixel centres.
    let transform = (signed - signed.signum() * 1.5) / 3.0;
    let coverage = f32::from(coverage[y * width + x]) / 255.0;
    let from_coverage = (0.5 - coverage) * 2.0;
    let weight = (1.0 - transform.abs() / 2.5).clamp(0.0, 1.0);
    transform * (1.0 - weight) + from_coverage * weight
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unbounded() -> Face {
        Face::new(crate::text::FACE_CHROME, true).expect("Unbounded")
    }

    fn space_grotesk() -> Face {
        Face::new(crate::text::FACE_CLEAN, false).expect("Space Grotesk")
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
    fn gpos_kerning_is_read_from_the_font() {
        // All three faces are OpenType fonts with GPOS and no legacy `kern` table, so fontdue's
        // own `horizontal_kern` returns zero for every pair. These pairs really kern.
        let face = space_grotesk();
        let mut kerned = Vec::new();
        for (left, right) in [('A', 'V'), ('T', 'o'), ('T', 'y'), ('V', 'a'), ('A', 'T')] {
            let kern = face.kern_em(left, right);
            println!("{left}{right}: {kern:+.4} em");
            if kern != 0.0 {
                kerned.push((left, right, kern));
            }
        }
        assert!(
            kerned.len() >= 2,
            "the face's GPOS kern feature must move at least two of those pairs: {kerned:?}"
        );
        // And the layout must apply it: the line's width is the advances plus those kerns.
        let text = "AV";
        let layout = face.layout(text, BAKE_PX);
        let advances = face.advance('A', BAKE_PX) + face.advance('V', BAKE_PX);
        let kern = face.kern_em('A', 'V') * BAKE_PX;
        println!(
            "AV: line {:.2}px, advances {advances:.2}px, kern {kern:+.2}px",
            layout.advance_w
        );
        assert!(
            (layout.advance_w - (advances + kern)).abs() < 0.01,
            "the pen must be the advances plus the GPOS pair value, got {:.3} vs {:.3}",
            layout.advance_w,
            advances + kern
        );
        assert!(
            (layout.advance_w - advances).abs() > 0.5,
            "this pair must actually kern: the line is {:.2}px, the advances alone {advances:.2}px",
            layout.advance_w
        );
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
    fn a_long_line_wraps_instead_of_shrinking() {
        let face = unbounded();
        let text = "the loom knows what the flesh wants and the loom is us";
        let layout = face.layout(text, BAKE_PX);
        assert!(
            layout.lines.len() > 1,
            "a {}-character line must wrap, got {} line(s)",
            text.chars().count(),
            layout.lines.len()
        );
        let aspect = layout.block_w as f32 / layout.block_h.max(1) as f32;
        println!(
            "{} chars: {} lines, {}x{} block, aspect {aspect:.2}, no input dropped ({})",
            text.chars().count(),
            layout.lines.len(),
            layout.block_w,
            layout.block_h,
            layout.dropped
        );
        assert_eq!(layout.dropped, 0, "a text this short must not be truncated");
        assert!(
            aspect < 8.0,
            "a wrapped block's aspect must stay readable, got {aspect:.2}"
        );
    }

    #[test]
    fn a_too_long_text_is_truncated_and_logged() {
        let face = unbounded();
        let text = "lorem ipsum dolor sit amet ".repeat(80);
        let layout = face.layout(&text, BAKE_PX);
        println!(
            "{} chars in: {} dropped, {} lines, {}x{} block",
            text.chars().count(),
            layout.dropped,
            layout.lines.len(),
            layout.block_w,
            layout.block_h
        );
        assert!(
            layout.dropped > 0,
            "a {} character text must be truncated",
            text.chars().count()
        );
        assert!(!layout.glyphs.is_empty(), "the visible part must still draw");
        assert!(
            layout.lines.len() <= MAX_LINES,
            "the line cap must hold, got {}",
            layout.lines.len()
        );
        assert!(
            layout.block_w as f32 <= face.wrap_width(BAKE_PX) * 1.3,
            "the wrapped block must stay inside the wrap width: {}",
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
    fn the_glyph_cache_is_bounded() {
        let face = unbounded();
        let mut drawn = 0usize;
        for seed in 0..CACHE_TILES * 3 {
            let ch = char::from_u32(0x21 + (seed % 90) as u32).unwrap_or('x');
            let glyph = face.glyph(ch, BAKE_PX);
            drawn += usize::from(glyph.width > 0);
        }
        let cached = face.glyphs.borrow().map.len();
        println!(
            "{drawn} glyphs drawn, {cached} tiles cached (cap {CACHE_TILES})"
        );
        assert!(
            cached <= CACHE_TILES,
            "the tile cache must stay bounded, got {cached}"
        );
    }

    #[test]
    fn a_repeated_character_is_rasterised_once() {
        let face = unbounded();
        face.layout("ILANA", BAKE_PX);
        let after_first = face.glyphs.borrow().map.len();
        assert_eq!(after_first, 4, "ILANA has four distinct characters");
        face.layout("BANANA", BAKE_PX);
        assert_eq!(
            face.glyphs.borrow().map.len(),
            after_first + 1,
            "a second text must only rasterise the characters it adds (B)"
        );
        face.layout("ILANA", BAKE_PX);
        assert_eq!(
            face.glyphs.borrow().map.len(),
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
        let mut pixels = vec![0u8; width * height * 4];
        write_field(&coverage, width, height, width, true, &mut pixels);
        let at = |x: usize, y: usize| pixels[(y * width + x) * 4 + 3];
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
        // 255 / (2 * SDF_RANGE) ~ 11 levels. Clipped cells would show up as jumps far larger.
        // Two and a half pixels: the antialiased coverage moves a pixel's edge by up to half a
        // pixel against its neighbour's grid value. A clipped cell jumps 50-200 levels.
        let step = (255.0 / (2.0 * SDF_RANGE) * 2.5).ceil() as i32;
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


