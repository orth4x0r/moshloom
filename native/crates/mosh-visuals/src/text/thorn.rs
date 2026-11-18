//! Thorns and sigil marks: the sigil look's linework, grown from the block at bake time.
//!
//! # What the linework is
//!
//! Cybersigilism is thin tribal linework: long, whip-like spikes that leave the silhouette where it
//! turns hardest — a stroke's end, a corner, a point — and sweep outward, plus abstract thorny
//! marks that flank the word. Both are *geometry*, found or drawn at bake time and handed to the
//! shader as tapered capsules, so they stay sharp at any size and can never be baked into a
//! letter's face.
//!
//! A [`Stroke`] is two tapered capsules `a -> b -> c`, wide at `a` and a needle point at `c`. A
//! thorn is one or two strokes (two make an S-curve, and a long thorn may carry a secondary barb);
//! a symbol is five strokes on one side of the block, which the shader draws again mirrored about
//! the block's centre, so the pair is symmetric.
//!
//! # Where a thorn may grow
//!
//! Only on an **outer** contour (a pixel the block's border can reach through empty space, so a
//! counter — the hole in `A` — is never an anchor), only so the spike itself stays in that empty
//! space, and only over its own word: a spike that would reach into another word's column box
//! would spear the next word across a space, which reads as a scratch, not a thorn. Every pixel a
//! candidate covers is checked against the block's ink and against the other words' boxes, and a
//! candidate that fails either is dropped before it is kept. `thorn::tests` pins both, plus that
//! no linework lands inside a glyph's interior.
//!
//! Sparse and asymmetric by construction: anchors are taken in order of how sharply the contour
//! turns, no two closer than a fifth of a line's height, at most [`MAX_THORNS`] strokes per block,
//! each with its own length, sweep and bend from a hash of its anchor, and none that would cross
//! another. The marks come from a hash of the text, so the same words always draw the same sigil.

use std::collections::VecDeque;

use super::layout::SDF_RANGE;

/// The most thorn strokes one block carries. A thorn is one stroke, an S-curve or a long thorn
/// with a barb is two or three, so this is roughly a dozen thorns on a ten-letter word: sparse,
/// and the uniform block stays small (twenty strokes of 32 bytes).
pub const MAX_THORNS: usize = 20;

/// The most symbol strokes one mark carries. Each is drawn twice (mirrored), so the pair is
/// symmetric without storing the reflection.
pub const MAX_SYMBOLS: usize = 8;

/// How far, in baked pixels, the far end of a stroke must reach beyond the ink's edge for the
/// anchor to be worth a thorn. Shorter than this and the mark reads as a fringe, not a spike.
pub const REACH_PX: f32 = 4.0;

/// The second capsule's fraction of the base half-width at the bend, and at the tip. A thorn ends
/// in a needle point, so the tip is zero and the whip keeps most of its thickness until the bend.
pub(crate) const BEND_WIDTH: f32 = 0.7;
pub(crate) const TIP_WIDTH: f32 = 0.0;

/// How far, in baked pixels, a thorn's body must stay from a mark's. The marks sit just outside
/// the block and so do the thorns: on the left, where the shader draws the authored mark mirrored,
/// a thorn hugging the block's edge merged into the mark and the pair stopped reading as a mark
/// beside the word. The gap is a little under the mark's own distance from the block (`0.06` of a
/// line), floored in pixels so a small block keeps a visible gap.
fn mark_clearance(line_h: f32) -> f32 {
    (0.06 * line_h).clamp(4.0, 12.0)
}

/// One stroke of the sigil look's linework, in block pixels (origin at the block's top-left, y
/// down — the same space `Layout` places glyphs in, and the space `text.wgsl` maps `block_uv`
/// into): two tapered capsules from `a` through `b` to `c`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stroke {
    pub a: [f32; 2],
    pub b: [f32; 2],
    pub c: [f32; 2],
    /// Half-width at `a`, in block pixels. The stroke tapers to a point at `c`.
    pub width: f32,
    /// True for a symbol stroke: the shader draws it a second time, mirrored about the block's
    /// vertical centre, which is what makes the pair of marks symmetric.
    pub mirrored: bool,
}

impl Stroke {
    /// The signed distance from `p` to the stroke's body, in block pixels: negative inside. The
    /// union of its two tapered capsules, which is what `text.wgsl` evaluates too (the tests use
    /// this one to rasterise the same body).
    pub fn signed(&self, p: [f32; 2]) -> f32 {
        let first = capsule(p, self.a, self.b, self.width, self.width * BEND_WIDTH);
        let second = capsule(
            p,
            self.b,
            self.c,
            self.width * BEND_WIDTH,
            self.width * TIP_WIDTH,
        );
        first.min(second)
    }

    /// True when `p` is in the stroke's body, away from the root where it joins what it grows
    /// from: the root is buried in the silhouette by design (a thorn grows out of the letter, a
    /// symbol stroke out of the mark's spine), so it is not part of what has to stay clear.
    pub fn covers_clear(&self, p: [f32; 2]) -> bool {
        let dx = p[0] - self.a[0];
        let dy = p[1] - self.a[1];
        let root = self.width + 1.5;
        dx * dx + dy * dy > root * root && self.signed(p) <= 0.0
    }

    /// The length of the stroke's centreline, in block pixels.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn length(&self) -> f32 {
        distance(self.a, self.b) + distance(self.b, self.c)
    }
}

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
}

/// The slack `text.wgsl`'s `out_of_reach` adds to a stroke's reach, in block heights. The shader
/// works in block-height units; this module works in block pixels, so the two copies scale it by
/// the block's height. `text::tests::the_thorn_geometry_constants_match_the_shader` parses the
/// shader's literal and compares it with this one.
///
/// The mirror below is test-only: the shader is what draws, and these are the copy its tests hold
/// every authored stroke to.
#[cfg(test)]
pub(crate) const REACH_SLACK: f32 = 0.01;

/// The radius, in block pixels, of the circle the shader's `out_of_reach` rejects a stroke by: the
/// length of the stroke's **path** — `|a-b| + |b-c|` — plus its base half-width twice and a little
/// slack. The chord `|c-a|` is shorter than the path whenever the stroke bends, and a bound built
/// from it skips pixels near the bend: `W`, `I I I` and `THE LOOM IS US` each lost part of a mark
/// barb that way (two or three baked pixels per stroke).
#[cfg(test)]
pub fn reach_radius(stroke: &Stroke, block_h: f32) -> f32 {
    distance(stroke.a, stroke.b)
        + distance(stroke.b, stroke.c)
        + stroke.width * 2.0
        + REACH_SLACK * block_h
}

/// True when the shader's early-out would discard `p` for `stroke`: `p` is farther from the
/// stroke's root than the bound above allows. This mirrors `out_of_reach` in `text.wgsl`.
#[cfg(test)]
pub fn out_of_reach(p: [f32; 2], stroke: &Stroke, block_h: f32) -> bool {
    distance(p, stroke.a) > reach_radius(stroke, block_h)
}

/// The signed distance to a capsule from `a` to `b` whose radius goes from `ra` to `rb`.
fn capsule(p: [f32; 2], a: [f32; 2], b: [f32; 2], ra: f32, rb: f32) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ap = [p[0] - a[0], p[1] - a[1]];
    let length2 = ab[0] * ab[0] + ab[1] * ab[1];
    let t = if length2 > 1e-6 {
        ((ap[0] * ab[0] + ap[1] * ab[1]) / length2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let d = [ap[0] - ab[0] * t, ap[1] - ab[1] * t];
    (d[0] * d[0] + d[1] * d[1]).sqrt() - (ra + (rb - ra) * t)
}

/// A hash of the words, so the marks are a function of the text and nothing else.
pub fn text_seed(text: &str) -> u32 {
    let mut hash = 2_166_136_261u32;
    for byte in text.bytes() {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(16_777_619);
    }
    hash
}

/// The block's ink, read from the sheet's baked distance field, plus the empty space the block's
/// border can reach and the columns its words occupy. Thorns grow in that empty space only.
struct Block<'a> {
    /// The sheet's BGRA field.
    field: &'a [u8],
    /// The sheet's width in pixels.
    width: u32,
    /// The block's top-left in the sheet.
    origin: (u32, u32),
    /// The block's size in pixels.
    size: (u32, u32),
    ink: Vec<bool>,
    /// Empty space reachable from the block's border: outside the glyphs, and not a counter.
    outside: Vec<bool>,
}

impl<'a> Block<'a> {
    fn new(field: &'a [u8], width: u32, origin: (u32, u32), size: (u32, u32)) -> Block<'a> {
        let count = (size.0 * size.1) as usize;
        let mut ink = vec![false; count];
        for y in 0..size.1 {
            for x in 0..size.0 {
                ink[(y * size.0 + x) as usize] = field[index(width, origin, size, x, y)] >= 128;
            }
        }
        // Flood the empty space from the block's border: a pixel the flood reaches is outside the
        // glyphs, and a hole it cannot reach (the counter of an `A`) is not somewhere a thorn may
        // go.
        let mut outside = vec![false; count];
        let mut queue: VecDeque<(u32, u32)> = VecDeque::new();
        let at = |x: u32, y: u32| (y * size.0 + x) as usize;
        for x in 0..size.0 {
            for y in [0, size.1.saturating_sub(1)] {
                if !ink[at(x, y)] && !outside[at(x, y)] {
                    outside[at(x, y)] = true;
                    queue.push_back((x, y));
                }
            }
        }
        for y in 0..size.1 {
            for x in [0, size.0.saturating_sub(1)] {
                if !ink[at(x, y)] && !outside[at(x, y)] {
                    outside[at(x, y)] = true;
                    queue.push_back((x, y));
                }
            }
        }
        while let Some((x, y)) = queue.pop_front() {
            for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || ny < 0 || nx >= size.0 as i32 || ny >= size.1 as i32 {
                    continue;
                }
                let (nx, ny) = (nx as u32, ny as u32);
                if ink[at(nx, ny)] || outside[at(nx, ny)] {
                    continue;
                }
                outside[at(nx, ny)] = true;
                queue.push_back((nx, ny));
            }
        }
        Block {
            field,
            width,
            origin,
            size,
            ink,
            outside,
        }
    }

    fn alpha(&self, x: f32, y: f32) -> Option<f32> {
        if x < 0.0 || y < 0.0 {
            return None;
        }
        let (x, y) = (x.round() as u32, y.round() as u32);
        if x >= self.size.0 || y >= self.size.1 {
            return None;
        }
        Some(f32::from(
            self.field[index(self.width, self.origin, self.size, x, y)],
        ))
    }

    /// The signed distance at a block pixel, in baked pixels: positive outside the ink.
    fn distance(&self, x: f32, y: f32) -> f32 {
        let Some(alpha) = self.alpha(x, y) else {
            return SDF_RANGE;
        };
        (0.5 - alpha / 255.0) * (2.0 * SDF_RANGE)
    }

    fn is_ink(&self, x: u32, y: u32) -> bool {
        self.ink[(y * self.size.0 + x) as usize]
    }

    fn is_outside(&self, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 || x >= self.size.0 as i32 || y >= self.size.1 as i32 {
            return false;
        }
        self.outside[(y as u32 * self.size.0 + x as u32) as usize]
    }

    /// The outward normal at a block pixel, from the field's own gradient. `None` where the
    /// gradient is flat (deep inside a stroke, or far outside the field's range).
    fn normal(&self, x: u32, y: u32) -> Option<[f32; 2]> {
        let gx = self.alpha(x as f32 + 1.0, y as f32)? - self.alpha(x as f32 - 1.0, y as f32)?;
        let gy = self.alpha(x as f32, y as f32 + 1.0)? - self.alpha(x as f32, y as f32 - 1.0)?;
        let length = (gx * gx + gy * gy).sqrt();
        if length < 2.0 {
            return None;
        }
        // The field rises toward the ink, so the outward direction is against its gradient.
        Some([-gx / length, -gy / length])
    }

    /// The word (x range and line) a spike at `x` grows from, or a box that contains nothing when
    /// the anchor is outside every word.
    fn own_word(&self, words: &[(i32, i32, usize)], x: f32) -> (u32, u32, usize) {
        words
            .iter()
            .copied()
            .find(|word| x >= word.0 as f32 && x <= word.1 as f32 + 1.0)
            .map(|word| (word.0.max(0) as u32, word.1.max(0) as u32, word.2))
            .unwrap_or((self.size.0, 0, usize::MAX))
    }

    /// True when every pixel the stroke covers sits in the block's outside space and, wherever it
    /// is over the block's *columns*, over `own` word: no ink, no counter (both are the block's
    /// outside space), and no gap column and no other word's column. The column band holds all the
    /// way out — a spike may lean above or below the line, but not sideways out of its word — which
    /// is what keeps a spike from spearing the next word across a space.
    fn is_clear(&self, stroke: &Stroke, own: (u32, u32, usize)) -> bool {
        let (min_x, max_x, min_y, max_y) = stroke_bounds(stroke, self.size);
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let p = [x as f32 + 0.5, y as f32 + 0.5];
                if !stroke.covers_clear(p) {
                    continue;
                }
                if !self.is_outside(x as i32, y as i32) {
                    return false;
                }
                if own.0 > own.1 {
                    // The anchor is in no word (a block with no ink boxes): nothing to check.
                    continue;
                }
                if x >= own.0 && x <= own.1 {
                    continue;
                }
                return false;
            }
        }
        true
    }

    /// The farthest distance a stroke's clear body reaches beyond the ink, in baked pixels: the
    /// number that says a spike is a spike and not a fringe.
    fn reach(&self, stroke: &Stroke) -> f32 {
        let (min_x, max_x, min_y, max_y) = stroke_bounds(stroke, self.size);
        let mut reach = f32::NEG_INFINITY;
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let p = [x as f32 + 0.5, y as f32 + 0.5];
                if stroke.covers_clear(p) {
                    reach = reach.max(self.distance(p[0], p[1]));
                }
            }
        }
        reach
    }
}

/// The alpha byte of the sheet's field at a block pixel.
fn index(width: u32, origin: (u32, u32), size: (u32, u32), x: u32, y: u32) -> usize {
    let sheet_x = (origin.0 + x).min(origin.0 + size.0 - 1);
    let sheet_y = (origin.1 + y).min(origin.1 + size.1 - 1);
    ((sheet_y * width + sheet_x) * 4 + 3) as usize
}

/// The box a stroke can cover, in block pixels, as inclusive integer bounds.
fn stroke_bounds(stroke: &Stroke, size: (u32, u32)) -> (u32, u32, u32, u32) {
    let points = [stroke.a, stroke.b, stroke.c];
    let reach = stroke.width + 1.5;
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for p in points {
        min_x = min_x.min(p[0]);
        max_x = max_x.max(p[0]);
        min_y = min_y.min(p[1]);
        max_y = max_y.max(p[1]);
    }
    (
        (min_x - reach).max(0.0) as u32,
        (max_x + reach).min(size.0.saturating_sub(1) as f32).max(0.0) as u32,
        (min_y - reach).max(0.0) as u32,
        (max_y + reach).min(size.1.saturating_sub(1) as f32).max(0.0) as u32,
    )
}

/// Grows the thorns for a baked block: `field` is the sheet's BGRA distance field, `width` its
/// width, `origin` the block's top-left in it, `size` the block's size in pixels, `lines` how many
/// visual lines the block has and `marks` the flanking strokes the bake has already placed. At most
/// [`MAX_THORNS`] strokes, and none at all for a block with no contour (empty or whitespace-only
/// text).
///
/// A thorn's length is a fraction of one *line's* height, not of the whole block, so a two-line
/// phrase grows the same spikes as a single word on one line. The longest reach about 0.6 of the
/// line's height out from the silhouette. A thorn that would come within [`mark_clearance`] of a
/// mark's body — the authored mark or its mirror — is dropped instead: the marks must read as
/// separate linework beside the word, not as part of the band of spikes on the block's edge.
pub fn grow(
    field: &[u8],
    width: u32,
    origin: (u32, u32),
    size: (u32, u32),
    lines: usize,
    words: &[(i32, i32, usize)],
    marks: &[Stroke],
) -> Vec<Stroke> {
    if size.0 < 3 || size.1 < 3 || field.len() < (width * (origin.1 + size.1) * 4) as usize {
        return Vec::new();
    }
    let block = Block::new(field, width, origin, size);
    let line_h = size.1 as f32 / lines.max(1) as f32;
    // The marks are already fixed, so a thorn that would merge into one is dropped here: the two
    // read as separate linework, a mark beside the word and spikes on its contour.
    let clearance = mark_clearance(line_h);

    // Every boundary pixel where the contour turns sharply enough to be a corner or a stroke's
    // end: the turn is the angle between the outward normals a few pixels either side of the
    // pixel along its own tangent. A smooth stroke's normals agree; a corner's do not.
    struct Candidate {
        turn: f32,
        x: u32,
        y: u32,
        normal: [f32; 2],
    }
    let mut candidates: Vec<Candidate> = Vec::new();
    for y in 1..size.1.saturating_sub(1) {
        for x in 1..size.0.saturating_sub(1) {
            if !block.is_ink(x, y) {
                continue;
            }
            let on_edge = !block.is_ink(x - 1, y)
                || !block.is_ink(x + 1, y)
                || !block.is_ink(x, y - 1)
                || !block.is_ink(x, y + 1);
            if !on_edge {
                continue;
            }
            let Some(normal) = block.normal(x, y) else {
                continue;
            };
            // The anchor's outside has to be the block's outside: a counter's boundary has ink
            // on the other side of its outward normal, and no thorn may grow into a counter.
            let probe = [
                x as f32 + normal[0] * 3.0,
                y as f32 + normal[1] * 3.0,
            ];
            if !block.is_outside(probe[0].round() as i32, probe[1].round() as i32) {
                continue;
            }
            let tangent = [-normal[1], normal[0]];
            let ahead = block.normal(
                (x as f32 + tangent[0] * 4.0).round().max(0.0) as u32,
                (y as f32 + tangent[1] * 4.0).round().max(0.0) as u32,
            );
            let behind = block.normal(
                (x as f32 - tangent[0] * 4.0).round().max(0.0) as u32,
                (y as f32 - tangent[1] * 4.0).round().max(0.0) as u32,
            );
            let turn = match (ahead, behind) {
                (Some(ahead), Some(behind)) => 1.0 - (ahead[0] * behind[0] + ahead[1] * behind[1]),
                _ => 0.0,
            };
            candidates.push(Candidate {
                turn,
                x,
                y,
                normal,
            });
        }
    }
    // Sharpest first; the tie-break is positional so a given block always grows the same thorns.
    candidates.sort_by(|a, b| {
        b.turn
            .total_cmp(&a.turn)
            .then_with(|| (a.y, a.x).cmp(&(b.y, b.x)))
    });

    let spacing = (0.20 * line_h).clamp(7.0, 26.0);
    let mut anchors: Vec<[f32; 2]> = Vec::new();
    let mut kept: Vec<Stroke> = Vec::new();
    for candidate in candidates {
        if kept.len() >= MAX_THORNS {
            break;
        }
        let at = [candidate.x as f32, candidate.y as f32];
        if anchors
            .iter()
            .any(|other| (other[0] - at[0]).hypot(other[1] - at[1]) < spacing)
        {
            continue;
        }
        let own = block.own_word(words, at[0]);
        let strokes = spawn(candidate.normal, at, line_h);
        if kept.len() + strokes.len() > MAX_THORNS {
            continue;
        }
        if !strokes.iter().all(|stroke| {
            block.is_clear(stroke, own)
                && block.reach(stroke) >= REACH_PX
                && clear_of_marks(stroke, marks, size.0 as f32, clearance)
        }) {
            continue;
        }
        // Two spikes crossing each other read as a scratch, so a crossing is not kept. A thorn's
        // own strokes share an end point on purpose, so they are not compared with each other.
        if kept
            .iter()
            .any(|held| strokes.iter().any(|stroke| crosses(held, stroke)))
        {
            continue;
        }
        anchors.push(at);
        kept.extend(strokes);
    }
    kept
}

/// One anchor's thorns: a whip that grows outward, sometimes S-curved through a second stroke or
/// carrying a short barb, each with its own length, sweep and bend from the anchor's hash, so no
/// two are the same and nothing is mirrored.
fn spawn(normal: [f32; 2], at: [f32; 2], line_h: f32) -> Vec<Stroke> {
    let seed = (at[0] as u32).wrapping_mul(73_856_093) ^ (at[1] as u32).wrapping_mul(19_349_663);
    let length = line_h * (0.24 + 0.36 * hash(seed, 1));
    let sweep = (hash(seed, 2) - 0.5) * 1.4;
    let bend_side = (hash(seed, 3) - 0.5) * 0.9 * length;
    let width = (0.030 * line_h).clamp(1.1, 2.4);
    let (sin, cos) = sweep.sin_cos();
    let direction = [
        normal[0] * cos - normal[1] * sin,
        normal[0] * sin + normal[1] * cos,
    ];
    let across = [-direction[1], direction[0]];
    let base = [at[0] + normal[0] * 0.6, at[1] + normal[1] * 0.6];
    let along = |t: f32, side: f32| {
        [
            base[0] + direction[0] * length * t + across[0] * side,
            base[1] + direction[1] * length * t + across[1] * side,
        ]
    };
    let mut strokes = Vec::with_capacity(3);
    if hash(seed, 4) < 0.35 && length > 0.35 * line_h {
        // An S: two strokes round a shared elbow, bending the other way, so the whip snakes.
        let flip = if hash(seed, 5) < 0.5 { -1.0 } else { 1.0 };
        let elbow = along(0.45, bend_side);
        let mut strokes_at = |points: [[f32; 2]; 3], width: f32| {
            strokes.push(Stroke {
                a: points[0],
                b: points[1],
                c: points[2],
                width,
                mirrored: false,
            });
        };
        strokes_at([base, along(0.22, bend_side * 0.5), elbow], width);
        strokes_at(
            [
                elbow,
                along(0.72, flip * bend_side * 0.9),
                along(1.0, flip * bend_side * 0.3),
            ],
            width * 0.8,
        );
    } else {
        strokes.push(Stroke {
            a: base,
            b: along(0.38, bend_side),
            c: along(1.0, bend_side * 0.4),
            width,
            mirrored: false,
        });
        // A long whip may carry one short barb, branching off its belly.
        if length > 0.45 * line_h && hash(seed, 6) < 0.4 {
            let root = along(0.55, bend_side * 0.7);
            let branch = [
                direction[0] * 0.55 + across[0] * flip_side(seed) * 0.83,
                direction[1] * 0.55 + across[1] * flip_side(seed) * 0.83,
            ];
            let reach = length * 0.34;
            strokes.push(Stroke {
                a: root,
                b: [
                    root[0] + branch[0] * reach * 0.55,
                    root[1] + branch[1] * reach * 0.55,
                ],
                c: [
                    root[0] + branch[0] * reach,
                    root[1] + branch[1] * reach,
                ],
                width: width * 0.7,
                mirrored: false,
            });
        }
    }
    strokes
}

/// -1 or +1, from the seed, for a barb that may leave either side of a thorn.
fn flip_side(seed: u32) -> f32 {
    if hash(seed, 7) < 0.5 {
        -1.0
    } else {
        1.0
    }
}

/// The sigil marks that flank the word: five strokes in a box just outside the block on the right
/// (the shader mirrors them for the left), so the pair is symmetric, abstract — a bramble: one
/// spine in two halves that bends twice, with three barbs on one side, an open line, never a closed
/// figure, no letterforms and nothing from an alphabet. The same words always draw the same marks
/// (`text_seed`), and they sit outside the ink's box, so they can never cover a glyph.
pub fn symbols(text: &str, size: (u32, u32), lines: usize) -> Vec<Stroke> {
    let line_h = size.1 as f32 / lines.max(1) as f32;
    if size.0 < 3 || size.1 < 3 || line_h < 8.0 {
        return Vec::new();
    }
    let seed = text_seed(text);
    // Small enough to stay inside the layer: the block is fitted to most of the frame's width, so
    // there are only a few tenths of a line height to the side.
    let mark_w = 0.20 * line_h;
    let mark_h = 0.72 * line_h;
    let gap = 0.06 * line_h;
    let left = size.0 as f32 + gap;
    let top = size.1 as f32 * 0.5 - mark_h * 0.5;
    // (u, v) in the mark's own box, v down, to block pixels.
    let at = |u: f32, v: f32| [left + u * mark_w, top + v * mark_h];
    let jitter = |salt: u32, amount: f32| (hash(seed.wrapping_add(17 * salt), salt) - 0.5) * amount;
    let width = (0.014 * line_h).clamp(0.7, 1.4);
    let bend = jitter(1, 0.14);

    // A bramble: a spine that bends twice, three barbs out to one side, all open (a barb never
    // closes back onto the spine, which is what would make a star or any other figure).
    let spine = [at(0.40 + bend, 0.00), at(0.56 + bend, 0.34), at(0.40 + bend, 0.60)];
    let tail = [at(0.40 + bend, 0.60), at(0.24 + bend, 0.84), at(0.40 + bend, 1.00)];
    let barb_high = [
        at(0.56 + bend, 0.34),
        at(0.84, 0.22 + jitter(2, 0.10)),
        at(1.00, 0.30 + jitter(3, 0.08)),
    ];
    let barb_mid = [
        at(0.40 + bend, 0.60),
        at(0.82, 0.56 + jitter(4, 0.10)),
        at(0.96, 0.68 + jitter(5, 0.08)),
    ];
    let barb_low = [
        at(0.24 + bend, 0.84),
        at(0.62, 0.90 + jitter(6, 0.08)),
        at(0.74, 1.00),
    ];
    let strokes = [spine, tail, barb_high, barb_mid, barb_low];
    strokes
        .iter()
        .take(MAX_SYMBOLS)
        .enumerate()
        .map(|(index, points)| Stroke {
            a: points[0],
            b: points[1],
            c: points[2],
            // The spine leads; the barbs are lighter, so the mark reads as one line with thorns.
            width: if index < 2 { width } else { width * 0.75 },
            mirrored: true,
        })
        .collect()
}

/// True when two strokes' bodies would touch or cross.
fn crosses(a: &Stroke, b: &Stroke) -> bool {
    let joins = |stroke: &Stroke| [(stroke.a, stroke.b), (stroke.b, stroke.c)];
    let mut closest = f32::INFINITY;
    for (a0, a1) in joins(a) {
        for (b0, b1) in joins(b) {
            closest = closest.min(segment_gap(a0, a1, b0, b1));
        }
    }
    closest < (a.width + b.width) * 2.0
}

/// The closest approach of two segments, sampled along the first: enough for "do these two spikes
/// cross", and a fraction of the cost of the exact form.
fn segment_gap(a0: [f32; 2], a1: [f32; 2], b0: [f32; 2], b1: [f32; 2]) -> f32 {
    let mut closest = f32::INFINITY;
    for step in 0..=8u32 {
        let t = step as f32 / 8.0;
        let p = [
            a0[0] + (a1[0] - a0[0]) * t,
            a0[1] + (a1[1] - a0[1]) * t,
        ];
        closest = closest.min(point_gap(p, b0, b1));
    }
    closest
}

/// The distance from a point to a segment.
fn point_gap(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ap = [p[0] - a[0], p[1] - a[1]];
    let length2 = ab[0] * ab[0] + ab[1] * ab[1];
    let t = if length2 > 1e-6 {
        ((ap[0] * ab[0] + ap[1] * ab[1]) / length2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (ap[0] - ab[0] * t).hypot(ap[1] - ab[1] * t)
}

/// A mark stroke as the shader draws the left copy: mirrored about the block's vertical centre,
/// which in block pixels is `block_w - x`.
pub fn mirrored(stroke: &Stroke, block_w: f32) -> Stroke {
    let about = |p: [f32; 2]| [block_w - p[0], p[1]];
    Stroke {
        a: about(stroke.a),
        b: about(stroke.b),
        c: about(stroke.c),
        width: stroke.width,
        mirrored: stroke.mirrored,
    }
}

/// True when `stroke`'s body keeps `clearance` pixels from every mark's body, on both sides: the
/// authored mark and the copy the shader mirrors out of it. Radii are the strokes' full base
/// half-widths, never the taper, so the test is a little stricter than the drawn ink.
fn clear_of_marks(stroke: &Stroke, marks: &[Stroke], block_w: f32, clearance: f32) -> bool {
    marks.iter().all(|mark| {
        [*mark, mirrored(mark, block_w)].iter().all(|copy| {
            centreline_gap(stroke, copy) >= stroke.width + copy.width + clearance
        })
    })
}

/// The closest the two strokes' centrelines come, in block pixels: the nearest pair of points on
/// their two-segment polylines. Both directions are sampled, so neither stroke's shape can slip
/// between the other's samples.
fn centreline_gap(a: &Stroke, b: &Stroke) -> f32 {
    let mut closest = f32::INFINITY;
    for (a0, a1) in [(a.a, a.b), (a.b, a.c)] {
        for (b0, b1) in [(b.a, b.b), (b.b, b.c)] {
            closest = closest.min(segment_gap(a0, a1, b0, b1));
            closest = closest.min(segment_gap(b0, b1, a0, a1));
        }
    }
    closest
}

/// A deterministic hash of a seed and a salt, in `0..1`.
fn hash(seed: u32, salt: u32) -> f32 {
    let mut value = seed ^ salt.wrapping_mul(0x9e37_79b9);
    value ^= value >> 15;
    value = value.wrapping_mul(0x2545_f491);
    value ^= value >> 13;
    (value & 0x00ff_ffff) as f32 / 16_777_216.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::layout::Face;
    use crate::text::sheet::Sheet;

    fn sigil() -> Face {
        Face::new(crate::text::FACE_SIGIL, true)
            .map(|face| face.with_thorns(true))
            .expect("Grenze Gotisch")
    }

    /// A block's linework and its surroundings, read from a sheet the same way `grow` reads it.
    struct Probe {
        sheet: Sheet,
        outside: Vec<bool>,
        size: (u32, u32),
        line_h: f32,
    }

    impl Probe {
        fn new(text: &str) -> Probe {
            let sheet = Sheet::bake(&sigil(), text).expect("a sheet");
            let size = (sheet.block_w, sheet.block_h);
            let block = Block::new(
                &sheet.pixels,
                sheet.width,
                (sheet.margin, sheet.margin),
                size,
            );
            let outside = block.outside.clone();
            let line_h = sheet.block_h as f32 / sheet.lines.max(1) as f32;
            Probe {
                sheet,
                outside,
                size,
                line_h,
            }
        }

        /// The thorns the bake grew: the same list the layer uploads.
        fn thorns(&self) -> &[Stroke] {
            &self.sheet.thorns
        }

        fn symbols(&self) -> &[Stroke] {
            &self.sheet.symbols
        }

        fn is_outside(&self, x: u32, y: u32) -> bool {
            self.outside[(y * self.size.0 + x) as usize]
        }

        /// The field's signed distance at a block pixel, in baked pixels.
        fn distance(&self, x: u32, y: u32) -> f32 {
            let at = ((self.sheet.margin + y) * self.sheet.width + self.sheet.margin + x) * 4 + 3;
            (0.5 - f32::from(self.sheet.pixels[at as usize]) / 255.0) * (2.0 * SDF_RANGE)
        }

        /// The farthest the stroke's own centreline gets from the ink, in baked pixels: the spike's
        /// reach, measured on the curve rather than on its (sub-pixel at the tip) raster.
        fn reach(&self, stroke: &Stroke) -> f32 {
            let mut reach = f32::NEG_INFINITY;
            for step in 0..=24u32 {
                let t = step as f32 / 24.0;
                let p = if t < 0.5 {
                    lerp(stroke.a, stroke.b, t * 2.0)
                } else {
                    lerp(stroke.b, stroke.c, (t - 0.5) * 2.0)
                };
                if p[0] < 0.0 || p[1] < 0.0 {
                    reach = reach.max(SDF_RANGE);
                    continue;
                }
                let (x, y) = (p[0] as u32, p[1] as u32);
                if x >= self.size.0 || y >= self.size.1 {
                    reach = reach.max(SDF_RANGE);
                    continue;
                }
                reach = reach.max(self.distance(x, y));
            }
            reach
        }

        /// The block's words, from the layout that baked it: a thorn may not leave its own word.
        fn words(&self) -> &[(i32, i32, usize)] {
            &self.sheet.words
        }
    }

    fn lerp(a: [f32; 2], b: [f32; 2], t: f32) -> [f32; 2] {
        [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
    }

    #[test]
    fn a_block_grows_long_thin_thorns_on_its_outer_contour() {
        for text in ["ILANA", "SIGIL", "THE LOOM\nIS US"] {
            let probe = Probe::new(text);
            let thorns = probe.thorns();
            let lengths: Vec<f32> = thorns.iter().map(Stroke::length).collect();
            let longest = lengths.iter().copied().fold(0.0f32, f32::max);
            let shortest = lengths.iter().copied().fold(f32::INFINITY, f32::min);
            println!(
                "{text:?}: {}x{} block, {} strokes, lengths {:.1}..{:.1}px (line {:.1}px), \
                 widths {:.2}..{:.2}",
                probe.size.0,
                probe.size.1,
                thorns.len(),
                shortest,
                longest,
                probe.line_h,
                thorns.iter().map(|s| s.width).fold(f32::INFINITY, f32::min),
                thorns.iter().map(|s| s.width).fold(0.0f32, f32::max),
            );
            assert!(
                !thorns.is_empty(),
                "{text:?} grew no thorns: the sigil look would be a plain serif"
            );
            assert!(
                thorns.len() <= MAX_THORNS,
                "{text:?} grew {} strokes, over the cap",
                thorns.len()
            );
            // Whip-like: the longest reach about half a line or more, and every stroke is many
            // times longer than it is thick.
            assert!(
                longest >= 0.45 * probe.line_h,
                "{text:?}: the longest thorn is {longest:.1}px on a {:.1}px line, not long enough",
                probe.line_h
            );
            assert!(
                shortest >= 0.15 * probe.line_h,
                "{text:?}: a thorn is {shortest:.1}px on a {:.1}px line, shorter than a fringe",
                probe.line_h
            );
            for stroke in thorns {
                assert!(
                    stroke.length() > stroke.width * 6.0,
                    "{text:?}: a stroke {:.1}px long and {:.1}px thick is a blob",
                    stroke.length(),
                    stroke.width
                );
            }
            // Sparse: no two thorns start on top of each other.
            for (index, stroke) in thorns.iter().enumerate() {
                for other in thorns.iter().skip(index + 1) {
                    let gap = distance(stroke.a, other.a);
                    if std::ptr::eq(stroke, other) {
                        continue;
                    }
                    assert!(
                        gap > 5.0 || (stroke.a == other.a),
                        "{text:?}: two thorns start {gap}px apart without sharing a root"
                    );
                }
            }
        }
    }

    /// The look's contract, on the same raster the shader draws: no linework ink inside the ink or
    /// a counter, and the thorns reach well beyond the silhouette. The lattice this replaces
    /// failed the first half: it grew a barb in the hole of every `A` and `N`.
    #[test]
    fn linework_never_lies_across_a_glyph_face() {
        for text in ["ILANA", "SIGIL", "THE LOOM\nIS US"] {
            let probe = Probe::new(text);
            let (width, height) = probe.size;
            let mut inside = 0usize;
            let mut ink = 0usize;
            for y in 0..height {
                for x in 0..width {
                    let p = [x as f32 + 0.5, y as f32 + 0.5];
                    let covered = probe
                        .thorns()
                        .iter()
                        .any(|stroke| stroke.covers_clear(p) && stroke.signed(p) <= 0.0);
                    if !covered {
                        continue;
                    }
                    if probe.is_outside(x, y) {
                        ink += 1;
                    } else {
                        inside += 1;
                    }
                }
            }
            let reach = probe
                .thorns()
                .iter()
                .map(|stroke| probe.reach(stroke))
                .fold(f32::NEG_INFINITY, f32::max);
            let nearest = probe
                .thorns()
                .iter()
                .map(|stroke| probe.reach(stroke))
                .fold(f32::INFINITY, f32::min);
            let longest = probe
                .thorns()
                .iter()
                .map(Stroke::length)
                .fold(0.0f32, f32::max);
            println!(
                "{text:?}: {ink} thorn px, reaching {nearest:.1}..{reach:.1}px beyond the ink \
                 (the field stops at {SDF_RANGE}), the longest stroke {longest:.1}px on a {:.1}px \
                 line, {inside} px not in the block's outside space",
                probe.line_h
            );
            assert!(ink > 0, "{text:?} has no thorn ink at all");
            assert_eq!(
                inside, 0,
                "{text:?}: {inside} thorn pixels lie inside the glyph's interior (a counter, or \
                 the ink itself)"
            );
            assert!(
                nearest > 1.5,
                "{text:?}: a stroke reaches only {nearest:.1}px beyond the ink, so it is a fringe \
                 against the letter, not a spike"
            );
            assert!(
                reach >= SDF_RANGE * 0.99,
                "{text:?}: no thorn leaves the field's range ({reach:.1}px), so none is long"
            );
            assert!(
                longest > 0.45 * probe.line_h,
                "{text:?}: the longest whip is {longest:.1}px on a {:.1}px line, too short to read \
                 as sigil linework",
                probe.line_h
            );
        }
    }

    /// A thorn must not spear the next word: it stays over its own word's columns while it is over
    /// the block, and (on a single line) never enters a gap column at all. The word boxes come from
    /// the layout that baked the sheet; the gap columns come from the ink itself.
    #[test]
    fn a_thorn_stays_over_its_own_word() {
        for text in ["THE LOOM", "one two three", "THE LOOM IS US"] {
            let probe = Probe::new(text);
            let words = probe.words().to_vec();
            let single_line = probe.sheet.lines == 1;
            // The spaces between the words on each line, from the layout's own boxes: the columns
            // strictly between one word's right edge and the next word's left edge.
            let mut gaps: Vec<(i32, i32)> = Vec::new();
            for (index, word) in words.iter().enumerate() {
                let next = words.iter().skip(index + 1).find(|next| next.2 == word.2);
                if let Some(next) = next.filter(|next| next.0 > word.1 + 1) {
                    gaps.push((word.1 + 1, next.0 - 1));
                }
            }
            let gap_columns = gaps.iter().map(|gap| gap.1 - gap.0 + 1).sum::<i32>();
            println!(
                "{text:?}: {} words at {words:?}, {gap_columns} gap columns in {gaps:?}, {} thorn \
                 strokes",
                words.len(),
                probe.thorns().len()
            );
            if text != "ILANA" {
                assert!(
                    !gaps.is_empty(),
                    "{text:?} needs a space between two words for this test to mean anything"
                );
            }
            // The gap test is single-line only: on a two-line block the columns of one line's space
            // sit over the other line's words, so a column-wise gap means nothing there.
            let check_gaps = single_line;
            let (width, _) = probe.size;
            let mut crossers = 0usize;
            let mut in_gap = 0usize;
            for stroke in probe.thorns() {
                let own = words.iter().copied().find(|word| {
                    stroke.a[0] >= word.0 as f32 && stroke.a[0] <= word.1 as f32 + 1.0
                });
                for y in 0..probe.size.1 {
                    for x in 0..width {
                        let p = [x as f32 + 0.5, y as f32 + 0.5];
                        if !stroke.covers_clear(p) || stroke.signed(p) > 0.0 {
                            continue;
                        }
                        let mine = own.is_some_and(|own| x as i32 >= own.0 && x as i32 <= own.1);
                        if own.is_some() && !mine {
                            crossers += 1;
                        }
                        if check_gaps
                            && gaps
                                .iter()
                                .any(|gap| x as i32 >= gap.0 && x as i32 <= gap.1)
                        {
                            in_gap += 1;
                        }
                    }
                }
            }
            assert_eq!(
                crossers, 0,
                "{text:?}: {crossers} thorn pixels lie over another word or a gap; a spike \
                 crossing a space spears the next word"
            );
            assert_eq!(
                in_gap, 0,
                "{text:?}: {in_gap} thorn pixels lie in a space between words"
            );
        }
    }

    /// The marks flank the word, mirrored by the shader, and never touch it: every authored stroke
    /// sits to the right of the block, inside a mark's width of it, and the same words always draw
    /// the same marks.
    #[test]
    fn symbols_flank_the_word_and_never_touch_it() {
        for text in ["ILANA", "THE LOOM\nIS US"] {
            let probe = Probe::new(text);
            let symbols = probe.symbols();
            println!(
                "{text:?}: {} symbol strokes, x {:.0}..{:.0} for a {:.0}px block",
                symbols.len(),
                symbols.iter().map(|s| s.a[0]).fold(f32::INFINITY, f32::min),
                symbols
                    .iter()
                    .flat_map(|s| [s.a[0], s.b[0], s.c[0]])
                    .fold(0.0f32, f32::max),
                probe.size.0 as f32
            );
            assert!(!symbols.is_empty(), "{text:?}: no sigil mark at all");
            assert!(symbols.len() <= MAX_SYMBOLS, "over the symbol cap");
            for stroke in symbols {
                assert!(stroke.mirrored, "a mark's stroke is drawn on both sides");
                for point in [stroke.a, stroke.b, stroke.c] {
                    assert!(
                        point[0] > probe.size.0 as f32,
                        "{text:?}: a mark's point {point:?} overlaps the block"
                    );
                    assert!(
                        point[0] < probe.size.0 as f32 + 0.40 * probe.line_h,
                        "{text:?}: a mark reaches {:.0}px past the block, too far to stay in the \
                         layer",
                        point[0] - probe.size.0 as f32
                    );
                    assert!(
                        point[1] >= 0.0 && point[1] <= probe.size.1 as f32,
                        "{text:?}: a mark's point {point:?} is outside the block's rows"
                    );
                }
                assert!(
                    stroke.length() > stroke.width * 4.0,
                    "a mark stroke is {:.1}px long and {:.1}px thick",
                    stroke.length(),
                    stroke.width
                );
            }
        }
    }

    /// Same words, same sigil: the thorns and the marks are a function of the block and the text,
    /// with nothing random and nothing from a hash map's order.
    #[test]
    fn the_same_text_draws_the_same_sigil() {
        for text in ["ILANA", "THE LOOM\nIS US"] {
            let first = Probe::new(text);
            let second = Probe::new(text);
            assert_eq!(first.thorns(), second.thorns(), "{text:?}: thorns differ");
            assert_eq!(first.symbols(), second.symbols(), "{text:?}: marks differ");
        }
        // Different words draw different marks.
        let one = Probe::new("THE LOOM");
        let two = Probe::new("THE FLESH");
        assert_ne!(one.symbols(), two.symbols(), "the text does not reach the marks");
    }

    /// The same predicate against the *class* of defect the previous shader had: a diagonal lattice
    /// over the whole block, one barb per cell, gated only by the distance to the nearest edge, so
    /// it grows a barb inside a counter (the hole of an `A`). This is not a transcription of that
    /// shader — its constants and hashes are gone — it is the shape of the rule, kept so the
    /// property has something that fails it.
    #[test]
    fn the_old_diagonal_lattice_would_fail_the_same_test() {
        let probe = Probe::new("ILANA");
        let (width, height) = probe.size;
        let line_h = probe.line_h;
        let bold = 0.012 * line_h;
        let barb_len = 0.075 * line_h;
        let mut inside = 0usize;
        let mut total = 0usize;
        for y in 0..height {
            for x in 0..width {
                let distance = probe.distance(x, y);
                let beyond = distance - bold;
                let along = (x as f32 * 0.94 + y as f32 * 0.34) / barb_len;
                let cell = along.floor();
                let bend = hash(cell as i32 as u32, 7);
                let grow = hash(cell as i32 as u32, 3);
                let phase = along.fract() - 0.5;
                let rise = (beyond / barb_len).clamp(0.0, 1.0);
                let curved = phase + (bend - 0.5) * 1.2 * rise;
                let bar_width = 0.30 + (0.01 - 0.30) * rise;
                let pointed = 1.0 - smoothstep(0.0, bar_width, curved.abs());
                let sparse = smoothstep(0.74, 0.88, grow);
                let present = smoothstep(-0.02, 0.08, beyond) * (1.0 - smoothstep(0.9, 1.0, rise));
                let spike = pointed * sparse * present * (1.0 - rise * rise);
                let shape = distance - spike * barb_len;
                // Only the barb's own ink: the glyph's body is ink too, and the point here is
                // what the lattice adds *outside* it. `plain` is the body's own edge; a pixel
                // counts only where the spike is what makes it ink.
                let plain = distance - bold;
                if plain >= 0.0 && shape - bold < 0.0 {
                    total += 1;
                    if !probe.is_outside(x, y) {
                        // Barb ink where the block's border cannot reach: an `A`'s counter.
                        inside += 1;
                    }
                }
            }
        }
        println!("old lattice: {total} barb px, {inside} of them across a glyph's interior");
        assert!(
            total > 0,
            "the emulated lattice drew nothing, so this comparison proves nothing"
        );
        assert!(
            inside > 0,
            "the emulated old lattice was expected to paint inside the glyphs' interiors"
        );
    }

    fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
        let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// Every integer pixel within half a pixel of `stroke`'s body, in block pixels: the body
    /// itself plus the antialiased fringe a pixel centre at the edge still carries. The bounds are
    /// the stroke's own, never the block's: a mark sits outside the block's box, and its bounds
    /// must be checked out there too.
    fn body_pixels(stroke: &Stroke) -> Vec<[f32; 2]> {
        let reach = stroke.width + 2.0;
        let points = [stroke.a, stroke.b, stroke.c];
        let min = [
            points.iter().map(|p| p[0]).fold(f32::MAX, f32::min) - reach,
            points.iter().map(|p| p[1]).fold(f32::MAX, f32::min) - reach,
        ];
        let max = [
            points.iter().map(|p| p[0]).fold(f32::MIN, f32::max) + reach,
            points.iter().map(|p| p[1]).fold(f32::MIN, f32::max) + reach,
        ];
        let mut inside = Vec::new();
        for y in (min[1].floor() as i32)..=(max[1].ceil() as i32) {
            for x in (min[0].floor() as i32)..=(max[0].ceil() as i32) {
                let p = [x as f32 + 0.5, y as f32 + 0.5];
                if stroke.signed(p) <= 0.5 {
                    inside.push(p);
                }
            }
        }
        inside
    }

    /// The shader rejects a stroke with one bounding circle before it runs the capsule maths
    /// (`out_of_reach` in `text.wgsl`), so every pixel of every authored stroke must lie inside
    /// that circle. A bound that is too small drops ink silently: the review found two or three
    /// baked pixels of a mark barb missing on `W` and `THE LOOM IS US`.
    ///
    /// One check covers both copies: the shader draws the left mark by evaluating the *authored*
    /// stroke at the mirrored position, so the authored body — including the part outside the
    /// block's box — is exactly what the bound is applied to.
    #[test]
    fn every_authored_stroke_fits_the_shader_reach_bound() {
        let mut checked = 0usize;
        for text in ["W", "I I I", "THE LOOM IS US", "ILANA", "SIGIL"] {
            let probe = Probe::new(text);
            let block_h = probe.sheet.block_h as f32;
            for (name, strokes) in [("thorn", probe.thorns()), ("mark", probe.symbols())] {
                for (index, stroke) in strokes.iter().enumerate() {
                    for p in body_pixels(stroke) {
                        let from_root = distance(p, stroke.a);
                        let bound = reach_radius(stroke, block_h);
                        assert!(
                            !out_of_reach(p, stroke, block_h),
                            "{text:?}: {name} {index} at {p:?} is {from_root:.2}px from its root \
                             and inside the stroke, but the shader's bound is {bound:.2}px: the \
                             early-out would drop that pixel"
                        );
                        checked += 1;
                    }
                }
            }
        }
        println!("{checked} authored-stroke pixels inside the shader's release bound");
        assert!(checked > 0, "no stroke has any pixels in its own body");
    }

    /// The bound the review caught, as a number: the chord `|c-a|` instead of the path
    /// `|a-b| + |b-c|`. A chord is shorter than the path for every bent stroke, so pixels near
    /// the bend fall outside the circle and the shader never draws them. This is the defect
    /// `every_authored_stroke_fits_the_shader_reach_bound` fails on with the old formula; it stays
    /// here so the chord's insufficiency has something that asserts it.
    #[test]
    fn the_chord_bound_would_skip_part_of_a_bent_stroke() {
        let mut total = 0usize;
        for text in ["W", "I I I", "THE LOOM IS US", "SIGIL"] {
            let probe = Probe::new(text);
            let slack = REACH_SLACK * probe.sheet.block_h as f32;
            for (name, strokes) in [("thorn", probe.thorns()), ("mark", probe.symbols())] {
                let skipped: usize = strokes
                    .iter()
                    .map(|stroke| {
                        let chord = distance(stroke.a, stroke.c) + stroke.width * 2.0 + slack;
                        body_pixels(stroke)
                            .into_iter()
                            .filter(|p| distance(*p, stroke.a) > chord)
                            .count()
                    })
                    .sum();
                if skipped > 0 {
                    println!("{text:?}: {skipped} {name} pixel(s) outside the chord bound");
                }
                total += skipped;
            }
        }
        assert!(
            total > 0,
            "the chord bound skipped nothing, so it is not the bound this test is about"
        );
    }

    /// The marks flank the word and the thorns grow from the same edges, so on the left — where
    /// the shader draws the authored mark mirrored — a thorn hugging the block merged into the mark
    /// and the pair stopped reading as a mark beside the word. This checks the raster: no thorn ink
    /// lands within the marks' clearance, on either side, and the gap is measured and printed.
    #[test]
    fn a_thorn_keeps_its_distance_from_the_marks() {
        for text in ["ILANA", "THE LOOM IS US", "SIGIL", "W"] {
            let probe = Probe::new(text);
            let block_w = probe.size.0 as f32;
            let clearance = mark_clearance(probe.line_h);
            let marks: Vec<Stroke> = probe
                .symbols()
                .iter()
                .flat_map(|mark| [*mark, mirrored(mark, block_w)])
                .collect();
            let flank = (2.0 * probe.line_h) as i32;
            let mut violations = 0usize;
            let mut closest = f32::INFINITY;
            for y in 0..probe.size.1 as i32 {
                for x in -flank..probe.size.0 as i32 + flank {
                    let p = [x as f32 + 0.5, y as f32 + 0.5];
                    if !probe.thorns().iter().any(|thorn| thorn.signed(p) <= 0.0) {
                        continue;
                    }
                    let near_mark = marks
                        .iter()
                        .map(|mark| mark.signed(p))
                        .fold(f32::INFINITY, f32::min);
                    closest = closest.min(near_mark);
                    if near_mark <= clearance {
                        violations += 1;
                    }
                }
            }
            println!(
                "{text:?}: {} thorns, closest thorn ink {closest:.1}px from a mark's body \
                 (clearance {clearance:.1}px)",
                probe.thorns().len()
            );
            assert!(
                !probe.thorns().is_empty(),
                "{text:?} grew no thorns at all: the clearance must not clear the contour"
            );
            assert_eq!(
                violations, 0,
                "{text:?}: {violations} thorn pixels lie within {clearance:.1}px of a mark, so the \
                 mark merges into the thorn band"
            );
            assert!(
                closest >= 1.0,
                "{text:?}: thorn ink touches a mark's body (gap {closest:.2}px)"
            );
        }
    }

    #[test]
    fn an_empty_block_grows_no_thorns() {
        assert!(Sheet::bake(&sigil(), " ").is_none(), "empty text has no sheet");
    }

    #[test]
    fn a_stroke_is_clear_where_it_touches_its_base() {
        let stroke = Stroke {
            a: [10.0, 10.0],
            b: [12.0, 4.0],
            c: [14.0, -2.0],
            width: 2.0,
            mirrored: false,
        };
        assert!(!stroke.covers_clear([10.0, 10.0]), "the base joint is not the stroke");
        assert!(stroke.covers_clear([12.0, 6.0]), "the body is");
        assert!(stroke.signed([12.0, 6.0]) <= 0.0);
        assert!(stroke.signed([40.0, 6.0]) > 0.0);
        assert!(stroke.length() > 12.0 && stroke.length() < 14.0);
        // A needle: the tip capsule has no radius, so the very end is a point.
        assert!(stroke.width * TIP_WIDTH == 0.0);
    }
}
