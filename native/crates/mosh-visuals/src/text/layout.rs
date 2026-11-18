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
//! The `kern` feature is resolved once, at load, through the script and language system (`latn`'s
//! default LangSys, then `DFLT`'s, then the first script's), and its lookups are de-duplicated:
//! every lookup applies once. Walking the FeatureList instead — as this module first did — would
//! count a font that writes one `kern` FeatureRecord per script, all pointing at the same lookup,
//! once per script. The pair cache itself is capped at [`CACHE_KERNS`] entries and cleared when it
//! reaches the cap, so a pasted essay cannot grow it without bound.
//!
//! # Wrapping and the length cap
//!
//! A display line is wrapped to [`WRAP_CHARS`] em of advance width (word wrap, breaking a word that is
//! longer than a whole line), because the shader fits the block to the frame: an unwrapped 100
//! character line would be fitted down to a stripe a few percent of the frame tall. Explicit `\n`
//! breaks are kept. A line is measured and centred without its trailing whitespace, so `"hello "`
//! centres exactly like `"hello"`. At most [`MAX_LINES`] visual lines are drawn and at most
//! [`MAX_CHARS`] characters are laid out; anything beyond is dropped and logged once, so a pasted
//! essay cannot produce a smear or an empty frame.
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

/// Pair values kept per face. A display line reuses a few dozen; the cap only stops a pasted essay
/// from growing the map without bound. The map is cleared when it reaches this many entries.
pub const CACHE_KERNS: usize = 1024;

/// The `DFLT` script tag: the script a text engine with no better idea asks for, and the one the
/// display faces list first.
const SCRIPT_DFLT: u32 = u32::from_be_bytes(*b"DFLT");

/// The `latn` script tag: the fallback when a face has no `DFLT` system.
const SCRIPT_LATN: u32 = u32::from_be_bytes(*b"latn");

/// The `kern` feature tag.
const FEATURE_KERN: u32 = u32::from_be_bytes(*b"kern");

/// A loaded face: fontdue for rasterising, ttf-parser for the GPOS pair adjustments.
pub struct Face {
    font: Font,
    outline: ttf_parser::Face<'static>,
    units_per_em: f32,
    /// The lookups the face's `kern` feature resolves to, in application order. Resolved once at
    /// load through the script and language system, so no pair can apply a lookup twice.
    kern: Vec<u16>,
    /// GPOS pair adjustment in font units, cached per (left, right).
    kerning: RefCell<HashMap<(char, char), i16>>,
    glyphs: RefCell<GlyphCache>,
    /// True when this face's look shades with the baked surface direction (chrome, sigil). The
    /// clean look only needs the distance, so its bake skips the gradient pass.
    normals: bool,
    /// True for the sigil look: the bake grows thorns out of the block's outer contour
    /// (`text/thorn.rs`). The other two looks have a clean silhouette.
    thorns: bool,
    /// One line of text at [`BAKE_PX`], ascent to the next baseline.
    line_height: f32,
}

/// One `kern` FeatureRecord as a script's language system names it, with the lookups it points at.
/// Kept free of `ttf-parser`'s types so the selection rule can be tested from plain data.
#[derive(Clone, Debug, PartialEq, Eq)]
struct KernFeature {
    /// The script tag (`DFLT`, `latn`, ...).
    script: u32,
    /// The language tag, `None` for that script's default language system.
    language: Option<u32>,
    /// The lookups the record names.
    lookups: Vec<u16>,
}

/// The lookups a `kern` feature contributes, chosen through the script and language system: the
/// `latn` script's default language system, then `DFLT`'s, then the first script's, and only the
/// chosen system's records. Each lookup appears once.
///
/// `latn` first is the OpenType practice: the text is Latin, so the Latin script's system is the
/// one that was written for it, and a face that keeps a separate `DFLT` system means that one for
/// text with no script of its own.
///
/// The hazard this closes: a font built one `kern` FeatureRecord per script points every record at
/// the same lookup. Walking the FeatureList and adding each record's first matching pair value
/// then counted that kerning once per script, so `AV` closed two or three times as far as the font
/// says. Here the records are filtered to one script and the lookups are de-duplicated.
fn kern_lookups(features: &[KernFeature]) -> Vec<u16> {
    let mut chosen: Vec<&KernFeature> = features
        .iter()
        .filter(|feature| feature.script == SCRIPT_LATN && feature.language.is_none())
        .collect();
    if chosen.is_empty() {
        chosen = features
            .iter()
            .filter(|feature| feature.script == SCRIPT_DFLT && feature.language.is_none())
            .collect();
    }
    // The first script the table lists is the last resort.
    let first_script = features.first().map(|feature| feature.script);
    let fallback: Vec<&KernFeature> = first_script
        .map(|script| {
            features
                .iter()
                .filter(|feature| feature.script == script && feature.language.is_none())
                .collect()
        })
        .unwrap_or_default();
    if chosen.is_empty() {
        chosen = fallback;
    }
    if chosen.is_empty() {
        // No script data the lookup can be reached through: use every record rather than losing
        // the font's kerning.
        chosen = features.iter().collect();
    }
    let mut lookups: Vec<u16> = Vec::new();
    for feature in chosen {
        for lookup in &feature.lookups {
            if !lookups.contains(lookup) {
                lookups.push(*lookup);
            }
        }
    }
    lookups
}

/// Every `kern` FeatureRecord in the layout table, tagged with the script and language system that
/// reaches it.
fn kern_features(table: &ttf_parser::opentype_layout::LayoutTable<'_>) -> Vec<KernFeature> {
    let mut features = Vec::new();
    for script in table.scripts {
        if let Some(default) = script.default_language {
            push_kern_feature(&mut features, table, script.tag.0, None, &default);
        }
        for language in script.languages {
            push_kern_feature(&mut features, table, script.tag.0, Some(language.tag.0), &language);
        }
    }
    features
}

/// Adds one language system's `kern` records to `features`.
fn push_kern_feature(
    features: &mut Vec<KernFeature>,
    table: &ttf_parser::opentype_layout::LayoutTable<'_>,
    script: u32,
    language: Option<u32>,
    system: &ttf_parser::opentype_layout::LanguageSystem<'_>,
) {
    let required = system.required_feature.into_iter();
    for index in required.chain(system.feature_indices) {
        let Some(feature) = table.features.get(index) else {
            continue;
        };
        if feature.tag.0 != FEATURE_KERN {
            continue;
        }
        let mut lookups = Vec::with_capacity(feature.lookup_indices.len() as usize);
        for slot in 0..feature.lookup_indices.len() {
            if let Some(lookup) = feature.lookup_indices.get(slot) {
                lookups.push(lookup);
            }
        }
        features.push(KernFeature {
            script,
            language,
            lookups,
        });
    }
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
    /// The block's words, as inclusive x ranges in block pixels plus the line they sit on, in
    /// reading order: within a line, a gap wider than a space separates two words. Thorns use them
    /// to stay over the word they grow from (`text/thorn.rs`).
    #[cfg_attr(not(test), allow(dead_code))]
    pub words: Vec<(i32, i32, usize)>,
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

/// One visual line: the character range it draws. Its advance width is measured when the glyphs
/// are placed, because kerning depends on the glyphs and on the line's own first character.
struct VisualLine {
    first: usize,
    end: usize,
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
        let kern = match outline.tables().gpos {
            Some(table) => kern_lookups(&kern_features(&table)),
            None => Vec::new(),
        };
        let line_height = font
            .horizontal_line_metrics(BAKE_PX)
            .map(|metrics| metrics.new_line_size + BAKE_PX * 0.10)
            .unwrap_or(BAKE_PX * 1.2);
        Some(Face {
            font,
            outline,
            units_per_em,
            kern,
            kerning: RefCell::new(HashMap::new()),
            glyphs: RefCell::new(GlyphCache {
                map: HashMap::new(),
                order: VecDeque::new(),
            }),
            normals,
            thorns: false,
            line_height,
        })
    }

    /// Marks this face as the sigil look: its bake grows thorns out of the block's outer contour
    /// (`text/thorn.rs`) and uploading them is what the shader draws. The chrome and clean looks
    /// leave this off: their silhouette is the glyph's own.
    pub fn with_thorns(mut self, thorns: bool) -> Face {
        self.thorns = thorns;
        self
    }

    /// True when the bake should grow thorns for this face.
    pub fn thorns(&self) -> bool {
        self.thorns
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
                let mut cache = self.kerning.borrow_mut();
                if cache.len() >= CACHE_KERNS {
                    // A text long enough to fill the map is a text whose pairs are about to
                    // repeat; clearing costs one pass and keeps the map bounded.
                    cache.clear();
                }
                cache.insert((left, right), units);
                units
            }
        };
        f32::from(units) / self.units_per_em
    }

    /// The `kern` feature's PairPos adjustment for the pair, in font units: every lookup the
    /// face's `kern` feature resolves to, applied once, added up (that is how OpenType applies
    /// lookups — several pair-positioning lookups can each move the same pair).
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
        pair_sum(&self.kern, |index| {
            let lookup = table.lookups.get(index)?;
            pair_value(&lookup, left, right)
        })
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
            // A line is measured and centred without its trailing whitespace: "hello " centres
            // exactly like "hello", so a space typed before an explicit break cannot shift the
            // line off the block's centre.
            while break_at > first && chars[break_at - 1].is_whitespace() {
                break_at -= 1;
            }
            lines.push(VisualLine {
                first,
                end: break_at,
            });
            first = break_at;
            while first < chars.len() && chars[first] == ' ' {
                first += 1;
            }
            if first < chars.len() && chars[first] == '\n' {
                // The break this line was measured up to.
                first += 1;
            }
        }
        // A text that ends in a break has a last, empty line, and `\n` alone is one empty line.
        if first == chars.len() && chars.last() == Some(&'\n') && lines.len() < MAX_LINES {
            lines.push(VisualLine {
                first,
                end: first,
            });
        }
        let dropped = dropped + chars.len().saturating_sub(first.min(chars.len()));

        // Place every glyph, line by line, and measure each line by the pen it ends with. Kerning
        // is the font's and only inside a line: the pen starts at zero on every line, so a line's
        // first glyph never kerns from the character before the break (nor from a space the wrap
        // dropped). That is what the centring uses; measuring a line from the prefix widths
        // instead counted a kern across the break and needed a correction of its own.
        let mut placed: Vec<Placed> = Vec::new();
        let mut baseline = ascent;
        let mut advances = vec![0.0f32; lines.len()];
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
            advances[line_index] = pen;
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

        let mut extents: Vec<Line> = advances
            .iter()
            .map(|advance| Line {
                ink_left: 0,
                ink_right: 0,
                advance: *advance,
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

        // The block's words, per line: a gap wider than a space (0.15 em of ink between two
        // glyphs) starts a new word. Thorns use these so a spike cannot cross a space and spear the
        // next word.
        let mut words: Vec<(i32, i32, usize)> = Vec::new();
        for line in 0..lines.len() {
            let mut of_line: Vec<&Placed> = placed.iter().filter(|glyph| glyph.line == line).collect();
            of_line.sort_by_key(|glyph| glyph.x);
            let mut start: Option<i32> = None;
            let mut end = 0i32;
            for glyph in of_line {
                if start.is_some() && (glyph.x - end) as f32 > 0.15 * px {
                    words.push((start.unwrap_or(0) - ink_left, end - ink_left, line));
                    start = None;
                }
                start.get_or_insert(glyph.x);
                end = glyph.x + glyph.glyph.width as i32;
            }
            if let Some(start) = start {
                words.push((start - ink_left, end - ink_left, line));
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
            words,
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

/// The GPOS PairPos adjustment a lookup gives the pair, in font units, if it applies: the lookup's
/// **first** subtable that handles the pair wins. One lookup is one adjustment — a later subtable
/// of the same lookup never adds to an earlier one.
fn pair_value(
    lookup: &ttf_parser::opentype_layout::Lookup<'_>,
    left: ttf_parser::GlyphId,
    right: ttf_parser::GlyphId,
) -> Option<i16> {
    use ttf_parser::gpos::PositioningSubtable as Subtable;
    let subtables = (0..lookup.subtables.len()).map(|slot| {
        let subtable = lookup.subtables.get::<Subtable<'_>>(slot)?;
        let Subtable::Pair(pair) = subtable else {
            return None;
        };
        let index = pair.coverage().get(left)?;
        match pair {
            ttf_parser::gpos::PairAdjustment::Format1 { sets, .. } => sets
                .get(index)
                .and_then(|set| set.get(right))
                .map(|(first, _)| first.x_advance),
            ttf_parser::gpos::PairAdjustment::Format2 {
                classes, matrix, ..
            } => matrix
                .get((classes.0.get(left), classes.1.get(right)))
                .map(|(first, _)| first.x_advance),
        }
    });
    first_subtable(subtables)
}

/// The adjustment a lookup list gives a pair, in font units: **across** lookups the adjustments
/// add up (each lookup applies in turn), while inside one lookup [`first_subtable`] keeps a single
/// value. Saturating, so two lookups at the `i16` limit cannot wrap.
fn pair_sum(lookups: &[u16], mut value_of: impl FnMut(u16) -> Option<i16>) -> i16 {
    let mut total = 0i32;
    for index in lookups {
        if let Some(value) = value_of(*index) {
            total += i32::from(value);
        }
    }
    total.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

/// The first subtable that handles the pair: a lookup's subtables are alternatives, not
/// adjustments to be added.
fn first_subtable(values: impl IntoIterator<Item = Option<i16>>) -> Option<i16> {
    values.into_iter().flatten().next()
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
        // own `horizontal_kern` returns zero for every pair. These pairs really kern, and these are
        // the values a reviewer measured from the committed Space Grotesk: AV -0.0550 em,
        // To -0.1030, Ty -0.0790. (A commit or a doc that quotes AV -0.064, Ty -0.058, To -0.049 is
        // wrong: those numbers were never the font's.)
        //
        // This test reads the font, not the summation rule: every committed face has one `kern`
        // FeatureRecord, so the old per-record walk produced these same values.
        // `a_pair_sums_every_lookup_and_one_subtable_per_lookup` is what pins the rule.
        let face = space_grotesk();
        let measured = [('A', 'V', -0.0550), ('T', 'o', -0.1030), ('T', 'y', -0.0790)];
        let mut kerned = Vec::new();
        for (left, right, expected) in measured {
            let kern = face.kern_em(left, right);
            println!("{left}{right}: {kern:+.4} em (the font says {expected:+.4})");
            assert!(
                (kern - expected).abs() < 0.002,
                "{left}{right} kerns {kern:+.4} em, not the {expected:+.4} this font has"
            );
            assert!(kern != 0.0);
            kerned.push((left, right, kern));
        }
        for (left, right) in [('V', 'a'), ('A', 'T')] {
            let kern = face.kern_em(left, right);
            println!("{left}{right}: {kern:+.4} em");
            if kern != 0.0 {
                kerned.push((left, right, kern));
            }
        }
        assert!(
            kerned.len() >= 3,
            "the face's GPOS kern feature must move at least three of those pairs: {kerned:?}"
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

    /// The summation rule, from data: **across lookups** the pair adjustments add up (each lookup
    /// applies in turn), while **within one lookup** only the first subtable that handles the pair
    /// counts. The walk this replaces (a2d8e4a) stopped at the first lookup of a feature that
    /// matched, so two lookups could not both move a pair.
    #[test]
    fn a_pair_sums_every_lookup_and_one_subtable_per_lookup() {
        // Two lookups, both handling AV: they add up.
        assert_eq!(
            pair_sum(&[3, 5], |index| match index {
                3 => Some(40),
                5 => Some(-15),
                _ => None,
            }),
            25
        );
        // A lookup that does not handle the pair contributes nothing.
        assert_eq!(
            pair_sum(&[3, 9], |index| match index {
                3 => Some(40),
                _ => None,
            }),
            40
        );
        // Two lookups at the i16 limit saturate instead of wrapping.
        assert_eq!(pair_sum(&[0, 1], |_| Some(i16::MAX)), i16::MAX);
        // Within one lookup the first subtable that handles the pair wins, and a later one does
        // not add to it.
        assert_eq!(first_subtable([None, Some(7), Some(-9)]), Some(7));
        assert_eq!(first_subtable([Some(3), Some(7)]), Some(3));
        assert_eq!(first_subtable([None, None]), None);
        assert_eq!(first_subtable([]), None);
    }

    /// The record hazard: a font built one `kern` FeatureRecord per script points every record at
    /// the same lookup. Walking the FeatureList and adding each record's pair value counted that
    /// kerning once per script — `AV` closed two or three times as far as the font says. The
    /// chosen script's lookups are de-duplicated, so each lookup applies once.
    #[test]
    fn a_kern_lookup_named_by_two_scripts_is_counted_once() {
        let cyrl = u32::from_be_bytes(*b"cyrl");
        let features = [
            KernFeature {
                script: SCRIPT_DFLT,
                language: None,
                lookups: vec![3],
            },
            KernFeature {
                script: SCRIPT_LATN,
                language: None,
                lookups: vec![3],
            },
            KernFeature {
                script: cyrl,
                language: None,
                lookups: vec![3],
            },
        ];
        assert_eq!(
            kern_lookups(&features),
            vec![3],
            "lookup 3 is named by three scripts' kern records and must be applied once"
        );
        // Several lookups inside one record all apply (that is how OpenType works), each once.
        let features = [KernFeature {
            script: SCRIPT_DFLT,
            language: None,
            lookups: vec![0, 4, 0],
        }];
        assert_eq!(kern_lookups(&features), vec![0, 4]);
    }

    #[test]
    fn kern_lookups_follow_the_script_and_language_system() {
        let feature = |script: &[u8; 4], language: Option<&[u8; 4]>, lookups: Vec<u16>| KernFeature {
            script: u32::from_be_bytes(*script),
            language: language.map(|tag| u32::from_be_bytes(*tag)),
            lookups,
        };
        // `latn`'s default system wins: the words are Latin, so the Latin script's system is the
        // one written for them, ahead of the generic `DFLT` system.
        assert_eq!(
            kern_lookups(&[
                feature(b"cyrl", None, vec![2]),
                feature(b"DFLT", None, vec![0]),
                feature(b"latn", None, vec![1]),
            ]),
            vec![1]
        );
        // A face with only a `DFLT` system uses it.
        assert_eq!(
            kern_lookups(&[feature(b"cyrl", None, vec![2]), feature(b"DFLT", None, vec![0])]),
            vec![0]
        );
        // Without either, the first script in the table.
        assert_eq!(
            kern_lookups(&[feature(b"cyrl", None, vec![2]), feature(b"grek", None, vec![3])]),
            vec![2]
        );
        // A language-specific record is not mixed into the script's default system.
        assert_eq!(
            kern_lookups(&[
                feature(b"latn", None, vec![1]),
                feature(b"latn", Some(b"TRK "), vec![9]),
            ]),
            vec![1]
        );
        // A face with no default system at all keeps every record rather than losing its kerning.
        assert_eq!(
            kern_lookups(&[feature(b"ZZZZ", Some(b"ZZZ "), vec![5])]),
            vec![5]
        );
    }

    #[test]
    fn the_kern_cache_is_bounded() {
        let face = space_grotesk();
        let chars: Vec<char> = ('!'..='~').collect();
        let mut queried = 0usize;
        for left in &chars {
            for right in &chars {
                face.kern_em(*left, *right);
                queried += 1;
            }
        }
        let cached = face.kerning.borrow().len();
        println!("{queried} pairs queried, {cached} cached (cap {CACHE_KERNS})");
        assert!(
            cached <= CACHE_KERNS,
            "the pair cache must stay bounded: {cached} entries, cap {CACHE_KERNS}"
        );
        assert!(cached > 0, "the cache should hold something");
    }

    /// A line is measured and centred without its trailing whitespace: `"hello \nworld"` must
    /// centre `hello` exactly where `"hello\nworld"` does, and a line's advance is its own glyphs
    /// plus the kerns inside it — never a kern across the break. The space used to be part of the
    /// line's advance, and the correction that removed it subtracted a pair the prefix widths had
    /// already accounted for, so a line could end up a couple of pixels too wide and sit off the
    /// block's centre.
    #[test]
    fn a_trailing_space_does_not_shift_the_line() {
        let face = unbounded();
        // The width the font itself gives a string, as the line's advance must be.
        let measured = |face: &Face, text: &str| {
            let mut width = 0.0;
            let mut previous: Option<char> = None;
            for ch in text.chars() {
                if let Some(left) = previous {
                    width += face.kern_em(left, ch) * BAKE_PX;
                }
                width += face.advance(ch, BAKE_PX);
                previous = Some(ch);
            }
            width
        };
        let plain = face.layout("hello\nworld", BAKE_PX);
        let padded = face.layout("hello \nworld", BAKE_PX);
        assert_eq!(plain.lines.len(), 2, "two lines, not three");
        assert_eq!(padded.lines.len(), 2, "a space before a break is not a line");
        assert_eq!(plain.lines, padded.lines, "the same two lines, measured alike");
        assert_eq!(plain.advance_w, padded.advance_w);
        let placed = |layout: &Layout| {
            layout
                .glyphs
                .iter()
                .map(|glyph| (glyph.x, glyph.y))
                .collect::<Vec<_>>()
        };
        assert_eq!(placed(&plain), placed(&padded), "every glyph sits the same");
        // The second line is `world`: its advance is that string's own width, not the whole
        // prefix's, and not the prefix's with a kern taken out twice.
        for layout in [&plain, &padded] {
            let expected = measured(&face, "world");
            println!(
                "line 2: {:.4}px, the font says {expected:.4}px",
                layout.lines[1].advance
            );
            assert!(
                (layout.lines[1].advance - expected).abs() < 0.01,
                "the second line is {:.4}px wide, `world` is {expected:.4}px",
                layout.lines[1].advance
            );
        }
        // A trailing space at the end of the text is not a line (or a dropped character) either.
        let single = face.layout("hello ", BAKE_PX);
        assert_eq!(single.lines.len(), 1);
        assert_eq!(single.lines, face.layout("hello", BAKE_PX).lines);
        assert_eq!(single.dropped, 0, "trimming is not truncation");

        // A pair that kerns hard across the break. Space Grotesk's `To` is -0.103 em: the second
        // line is one `o`, so its width is exactly that glyph's advance.
        let clean = space_grotesk();
        assert!(
            clean.kern_em('T', 'o') < -0.05,
            "the clean face's To must really kern, it is {:.4} em",
            clean.kern_em('T', 'o')
        );
        let plain = clean.layout("T\no", BAKE_PX);
        let padded = clean.layout("T \no", BAKE_PX);
        assert_eq!(plain.lines.len(), 2);
        assert_eq!(padded.lines.len(), 2);
        assert_eq!(
            plain.lines, padded.lines,
            "a space before the break changed a line's width"
        );
        assert_eq!(
            placed(&plain),
            placed(&padded),
            "a space before the break moved a glyph"
        );
        let bare = clean.advance('o', BAKE_PX);
        for layout in [&plain, &padded] {
            println!(
                "`o` after a break: {:.4}px, the glyph is {bare:.4}px",
                layout.lines[1].advance
            );
            assert!(
                (layout.lines[1].advance - bare).abs() < 0.01,
                "a line that starts with `o` is {:.4}px wide, the glyph is {bare:.4}px",
                layout.lines[1].advance
            );
        }
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
