//! Thorns: the sigil look's spikes, grown from the block's own contour at bake time.
//!
//! # What a thorn is
//!
//! Cybersigilism is thin tribal linework, not a fringe: a few long, tapered, curved spikes that
//! leave the silhouette where it turns hardest — a stroke's end, a corner, a point — and sweep
//! outward. So a thorn here is *geometry*, found from the baked block and handed to the shader as
//! a curve ([`Thorn`], two tapered capsules through `bend`), and the shader draws it in its own
//! silver. The previous look grew its barbs as a diagonal lattice in distance space: a mark in
//! every cell, all of them parallel, which read as scratches laid across the letters and painted
//! the counters of `A`, `N` and `o`.
//!
//! # Where a thorn may grow
//!
//! Only on an **outer** contour (a pixel the block's border can reach through empty space, so a
//! counter — the hole in `A` — is never an anchor), and only so the spike itself stays in that
//! empty space: every pixel a candidate thorn covers is checked against the block's ink, and a
//! candidate that would touch ink or lean into a counter is dropped before it is kept. That is the
//! property `thorn::tests` pins: thorn ink exists beyond the ink's dilation, and no thorn ink lies
//! inside the ink or a counter.
//!
//! Sparse by construction: anchors are taken in order of how sharply the contour turns, no two
//! closer than a quarter of a line's height, at most [`MAX_THORNS`] per block, each with its own
//! length, sweep and bend from a hash of its anchor, and none that would cross another. Nothing is
//! mirror-symmetric.

use std::collections::VecDeque;

use super::layout::SDF_RANGE;

/// The most thorns one block carries. A long text already has many corners; the cap keeps the
/// look sparse and the shader's uniform block bounded.
pub const MAX_THORNS: usize = 10;

/// How far, in baked pixels, the far end of a thorn must reach beyond the ink's edge for the
/// anchor to be worth a thorn. Shorter than this and the mark reads as a fringe, not a spike.
pub const REACH_PX: f32 = 4.0;

/// The second capsule's fraction of the base half-width at the bend, and at the tip. The thorn is
/// a point by the end, so it tapers from a stroke to nothing instead of ending in a cut.
const BEND_WIDTH: f32 = 0.55;
const TIP_WIDTH: f32 = 0.01;

/// One thorn: a curved, tapered spike that grows outward from the contour, in block pixels
/// (origin at the block's top-left, y down — the same space `Layout` places glyphs in, and the
/// space `text.wgsl` maps `block_uv` into).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thorn {
    pub base: [f32; 2],
    pub bend: [f32; 2],
    pub tip: [f32; 2],
    /// Half-width where the thorn leaves the ink, in block pixels.
    pub width: f32,
}

impl Thorn {
    /// The signed distance from `p` to the thorn's body, in block pixels: negative inside the
    /// thorn. The union of its two tapered capsules, which is what `text.wgsl` evaluates too (the
    /// tests use this one to rasterise the same body).
    pub fn signed(&self, p: [f32; 2]) -> f32 {
        let first = capsule(p, self.base, self.bend, self.width, self.width * BEND_WIDTH);
        let second = capsule(
            p,
            self.bend,
            self.tip,
            self.width * BEND_WIDTH,
            self.width * TIP_WIDTH,
        );
        first.min(second)
    }

    /// True when `p` is in the thorn's body, away from the root where it joins the ink: the root
    /// is buried in the silhouette by design (a thorn grows out of the letter, it does not float),
    /// so it is not part of what has to stay clear. Everything the spike adds past the root must be
    /// outside the ink.
    pub fn covers_clear(&self, p: [f32; 2]) -> bool {
        let dx = p[0] - self.base[0];
        let dy = p[1] - self.base[1];
        let root = self.width + 1.5;
        dx * dx + dy * dy > root * root && self.signed(p) <= 0.0
    }

    /// The length of the thorn's centreline, in block pixels.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn length(&self) -> f32 {
        distance(self.base, self.bend) + distance(self.bend, self.tip)
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
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

/// The block's ink, read from the sheet's baked distance field, plus the empty space the block's
/// border can reach. Thorns grow in that empty space only.
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

    /// True when every pixel the thorn covers sits in the block's outside space.
    fn is_clear(&self, thorn: &Thorn) -> bool {
        let (min_x, max_x, min_y, max_y) = thorn_bounds(thorn, self.size);
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let p = [x as f32 + 0.5, y as f32 + 0.5];
                if !thorn.covers_clear(p) {
                    continue;
                }
                if !self.is_outside(x as i32, y as i32) {
                    return false;
                }
            }
        }
        true
    }

    /// The farthest distance a thorn's clear body reaches beyond the ink, in baked pixels: the
    /// number that says a thorn is a spike and not a fringe.
    fn reach(&self, thorn: &Thorn) -> f32 {
        let (min_x, max_x, min_y, max_y) = thorn_bounds(thorn, self.size);
        let mut reach = f32::NEG_INFINITY;
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let p = [x as f32 + 0.5, y as f32 + 0.5];
                if thorn.covers_clear(p) {
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

/// The box a thorn can cover, in block pixels, as inclusive integer bounds.
fn thorn_bounds(thorn: &Thorn, size: (u32, u32)) -> (u32, u32, u32, u32) {
    let points = [thorn.base, thorn.bend, thorn.tip];
    let reach = thorn.width + 1.5;
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
/// width, `origin` the block's top-left in it, `size` the block's size in pixels and `lines` how
/// many visual lines the block has. At most [`MAX_THORNS`], and none at all for a block with no
/// contour (empty or whitespace-only text).
///
/// A thorn's length is a fraction of one *line's* height, not of the whole block, so a two-line
/// phrase grows the same spikes as a single word on one line.
pub fn grow(
    field: &[u8],
    width: u32,
    origin: (u32, u32),
    size: (u32, u32),
    lines: usize,
) -> Vec<Thorn> {
    if size.0 < 3 || size.1 < 3 || field.len() < (width * (origin.1 + size.1) * 4) as usize {
        return Vec::new();
    }
    let block = Block::new(field, width, origin, size);
    let line_h = size.1 as f32 / lines.max(1) as f32;

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

    let spacing = (0.24 * line_h).clamp(7.0, 30.0);
    let mut chosen: Vec<[f32; 2]> = Vec::new();
    let mut thorns: Vec<Thorn> = Vec::new();
    for candidate in candidates {
        if thorns.len() == MAX_THORNS {
            break;
        }
        let at = [candidate.x as f32, candidate.y as f32];
        if chosen
            .iter()
            .any(|other| (other[0] - at[0]).hypot(other[1] - at[1]) < spacing)
        {
            continue;
        }
        let thorn = spawn(candidate.normal, at, line_h);
        // Two spikes crossing each other read as a scratch, so a crossing is not kept.
        if !block.is_clear(&thorn)
            || block.reach(&thorn) < REACH_PX
            || thorns.iter().any(|kept| crosses(kept, &thorn))
        {
            continue;
        }
        chosen.push(at);
        thorns.push(thorn);
    }
    thorns
}

/// True when two thorns' bodies would touch or cross.
fn crosses(a: &Thorn, b: &Thorn) -> bool {
    let joins = |thorn: &Thorn| [(thorn.base, thorn.bend), (thorn.bend, thorn.tip)];
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

/// One thorn from an anchor: a length, a sweep away from the normal and a bend, each from the
/// anchor's own hash, so no two are the same and nothing is mirrored.
fn spawn(normal: [f32; 2], at: [f32; 2], line_h: f32) -> Thorn {
    let seed = (at[0] as u32).wrapping_mul(73_856_093) ^ (at[1] as u32).wrapping_mul(19_349_663);
    let length = line_h * (0.18 + 0.08 * hash(seed, 1));
    let sweep = (hash(seed, 2) - 0.5) * 0.8;
    let bend_side = (hash(seed, 3) - 0.5) * 0.28 * length;
    let width = (0.042 * line_h).clamp(1.8, 4.0);
    let (sin, cos) = sweep.sin_cos();
    let direction = [
        normal[0] * cos - normal[1] * sin,
        normal[0] * sin + normal[1] * cos,
    ];
    let across = [-direction[1], direction[0]];
    let base = [at[0] + normal[0] * 0.6, at[1] + normal[1] * 0.6];
    let tip = [
        base[0] + direction[0] * length,
        base[1] + direction[1] * length,
    ];
    let bend = [
        base[0] + direction[0] * length * 0.35 + across[0] * bend_side,
        base[1] + direction[1] * length * 0.35 + across[1] * bend_side,
    ];
    Thorn {
        base,
        bend,
        tip,
        width,
    }
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

    /// A block's thorns and its reachable empty space, from a sheet the same way `grow` reads it.
    struct Probe {
        sheet: Sheet,
        outside: Vec<bool>,
        size: (u32, u32),
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
            Probe {
                sheet,
                outside,
                size,
            }
        }

        /// The thorns the bake grew for this block: the same list the layer uploads.
        fn thorns(&self) -> &[Thorn] {
            &self.sheet.thorns
        }

        fn is_outside(&self, x: u32, y: u32) -> bool {
            self.outside[(y * self.size.0 + x) as usize]
        }

        /// The field's signed distance at a block pixel, in baked pixels.
        fn distance(&self, x: u32, y: u32) -> f32 {
            let at = ((self.sheet.margin + y) * self.sheet.width + self.sheet.margin + x) * 4 + 3;
            (0.5 - f32::from(self.sheet.pixels[at as usize]) / 255.0) * (2.0 * SDF_RANGE)
        }

        /// The farthest the thorn's own centreline gets from the ink, in baked pixels: the spike's
        /// reach, measured on the curve rather than on its (sub-pixel at the tip) raster.
        fn reach(&self, thorn: &Thorn) -> f32 {
            let mut reach = f32::NEG_INFINITY;
            for step in 0..=24u32 {
                let t = 0.15 + 0.85 * (step as f32 / 24.0);
                let p = if t < 0.42 {
                    lerp(thorn.base, thorn.bend, t / 0.42)
                } else {
                    lerp(thorn.bend, thorn.tip, (t - 0.42) / 0.58)
                };
                if p[0] < 0.0 || p[1] < 0.0 {
                    // Past the block's box there is no ink anywhere: the field's whole range.
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
    }

    fn lerp(a: [f32; 2], b: [f32; 2], t: f32) -> [f32; 2] {
        [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
    }

    #[test]
    fn a_block_grows_few_long_thorns_on_its_outer_contour() {
        for text in ["ILANA", "SIGIL", "THE LOOM\nIS US"] {
            let probe = Probe::new(text);
            let thorns = probe.thorns();
            println!(
                "{text:?}: {}x{} block, {} thorns, lengths {:?}",
                probe.size.0,
                probe.size.1,
                thorns.len(),
                thorns
                    .iter()
                    .map(|thorn| (thorn.length() * 10.0).round() / 10.0)
                    .collect::<Vec<_>>()
            );
            assert!(
                !thorns.is_empty(),
                "{text:?} grew no thorns: the sigil look would be a plain serif"
            );
            assert!(
                thorns.len() <= MAX_THORNS,
                "{text:?} grew {} thorns, over the cap",
                thorns.len()
            );
            for thorn in thorns {
                // A thorn is a spike, not a blob: several times longer than it is thick.
                assert!(
                    thorn.length() > thorn.width * 3.5,
                    "{text:?}: a thorn {}px long and {}px thick is a blob",
                    thorn.length(),
                    thorn.width
                );
                // Sparse: thorns start well apart.
                for other in thorns {
                    if std::ptr::eq(thorn, other) {
                        continue;
                    }
                    let gap = distance(thorn.base, other.base);
                    assert!(
                        gap > 5.0,
                        "{text:?}: two thorns start {gap}px apart, which reads as a fringe"
                    );
                }
            }
        }
    }

    /// The look's contract, on the same raster the shader draws: a thorn's ink lies outside the
    /// ink's dilation and never inside the glyph's interior — not in the ink, and not in a
    /// counter. The old diagonal lattice failed the second half: it grew a barb in the hole of
    /// every `A` and `N`.
    #[test]
    fn thorn_ink_lies_outside_the_ink_and_never_across_a_face() {
        for text in ["ILANA", "SIGIL", "THE LOOM\nIS US"] {
            let probe = Probe::new(text);
            let thorns = probe.thorns();
            let (width, height) = probe.size;
            let mut ink = 0usize;
            let mut inside = 0usize;
            for y in 0..height {
                for x in 0..width {
                    let p = [x as f32 + 0.5, y as f32 + 0.5];
                    let covered = thorns
                        .iter()
                        .any(|thorn| thorn.covers_clear(p) && thorn.signed(p) <= 0.0);
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
            // How far each spike's own centreline gets from the ink: a thorn whose far end is
            // still on top of a letter is a scratch, not a thorn.
            let reach: Vec<f32> = thorns.iter().map(|thorn| probe.reach(thorn)).collect();
            let shallowest = reach.iter().copied().fold(f32::INFINITY, f32::min);
            let longest = reach.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            println!(
                "{text:?}: {} thorns, {ink} thorn px, each reaching {shallowest:.1}..{longest:.1}px \
                 beyond the ink, {inside} px not in the block's outside space",
                thorns.len(),
            );
            assert!(!thorns.is_empty(), "{text:?} has no thorn ink at all");
            assert_eq!(
                inside, 0,
                "{text:?}: {inside} thorn pixels lie inside the glyph's interior (a counter, or \
                 the ink itself)"
            );
            assert!(
                shallowest > REACH_PX - 1.0,
                "{text:?}: a thorn reaches only {shallowest:.1}px beyond the ink, so it is a \
                 fringe against the letter, not a spike (the others reach {longest:.1}px)"
            );
        }
    }

    /// The same predicate against the rule the previous shader used: a diagonal lattice across the
    /// whole block, one barb per cell, gated only by the distance to the nearest edge. It grows a
    /// barb inside a counter (the hole of an `A`), which is the "scratches across the face" the
    /// coordinator saw.
    #[test]
    fn the_old_diagonal_lattice_would_fail_the_same_test() {
        let probe = Probe::new("ILANA");
        let (width, height) = probe.size;
        let block_h = height as f32;
        let bold = 0.012 * block_h;
        let barb_len = 0.075 * block_h;
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

    #[test]
    fn an_empty_block_grows_no_thorns() {
        assert!(Sheet::bake(&sigil(), " ").is_none(), "empty text has no sheet");
    }

    #[test]
    fn a_thorn_is_clear_where_it_touches_its_base() {
        let thorn = Thorn {
            base: [10.0, 10.0],
            bend: [12.0, 4.0],
            tip: [14.0, -2.0],
            width: 2.0,
        };
        assert!(!thorn.covers_clear([10.0, 10.0]), "the base joint is not the thorn");
        assert!(thorn.covers_clear([12.0, 6.0]), "the body is");
        assert!(thorn.signed([12.0, 6.0]) <= 0.0);
        assert!(thorn.signed([40.0, 6.0]) > 0.0);
        assert!(thorn.length() > 12.0 && thorn.length() < 14.0);
    }
}


