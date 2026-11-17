//! Unsupervised checks on an analysis run: no hand labels, only what the run itself says.
//!
//! The tuning material has no agreed section or drop marks, so the section and drop rules are
//! judged by internal consistency and by comparison with the largest measured changes in the
//! same run:
//!
//! - **(a)** two sections whose mean loudness differs by more than 3 dB must land on different
//!   levels, in the right order;
//! - **(b)** the loudest section must reach level 4 and the quietest must read 1 or 2;
//! - **(c)** every detected drop must coincide (within one bar) with one of the largest
//!   bar-to-bar energy jumps of the track, and report how many of those jumps were caught;
//! - **(d)** every section boundary must be one of the largest novelty peaks of the track;
//! - **(e)** the level never moves more than one step per bar;
//! - **(f)** consecutive sections that get louder must get a higher level.
//!
//! Pure functions over [`BarRow`]s, so the rules can be tested without audio.

use std::fmt::Write as _;

/// One bar as the checks see it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BarRow {
    pub bar_index: i64,
    pub loudness_db: f32,
    /// Fast level of the bar (mean block RMS, dBFS): the bar-to-bar energy a drop is judged on.
    pub level_db: f32,
    pub level: u8,
    /// 0-based section number.
    pub section: u32,
    /// This bar is the first bar of its section.
    pub section_started: bool,
    /// Mel novelty at this bar.
    pub novelty: f32,
    /// A DROP event fired on this bar.
    pub drop: bool,
}

/// One section as the run produced it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SectionSummary {
    pub index: u32,
    pub first_bar: i64,
    pub last_bar: i64,
    pub mean_db: f32,
    pub mean_level: f32,
    pub min_level: u8,
    pub max_level: u8,
}

/// Two sections more than 3 dB apart, and whether their levels are ordered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SectionPair {
    pub louder: u32,
    pub quieter: u32,
    pub db: f32,
    pub ordered: bool,
}

/// What the checks found. Numbers, not adjectives: every field is something to quote.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Eval {
    pub sections: Vec<SectionSummary>,
    pub pairs: Vec<SectionPair>,
    pub ordered_pairs: usize,
    /// Mean level of the loudest section, and of the quietest.
    pub loudest_level: f32,
    pub quietest_level: f32,
    pub drops: usize,
    /// Drops within one bar of one of the largest bar-to-bar energy jumps.
    pub drops_on_top_jumps: usize,
    /// The largest bar-to-bar jumps that got a drop: recall of the same comparison.
    pub top_jumps: usize,
    pub top_jumps_with_drop: usize,
    /// The largest bar-to-bar jump the run produced, in dB.
    pub largest_jump_db: f32,
    pub boundaries: usize,
    /// Boundaries among the largest novelty peaks of the track.
    pub boundaries_on_peaks: usize,
    /// The largest level step between consecutive bars.
    pub max_level_step: u8,
    /// Consecutive sections that got louder, and how many of them got a higher level.
    pub louder_steps: usize,
    pub louder_steps_ordered: usize,
    /// Bars a DROP fired on, and bars a section started on (1-based, as the tracks are counted).
    pub drop_bars: Vec<i64>,
    pub boundary_bars: Vec<i64>,
}

impl Eval {
    /// `(a)`-`(f)` as a compact table, one line per check.
    pub fn report(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "# (a) pairs>3dB ordered {}/{}",
            self.ordered_pairs,
            self.pairs.len()
        );
        let _ = writeln!(
            out,
            "# (b) loudest level {:.2} (needs 4), quietest {:.2} (needs 1-2)",
            self.loudest_level, self.quietest_level
        );
        let _ = writeln!(
            out,
            "# (c) drops on top rises {}/{} (largest rise {:.1} dB), top rises with a drop {}/{}",
            self.drops_on_top_jumps,
            self.drops,
            self.largest_jump_db,
            self.top_jumps_with_drop,
            self.top_jumps
        );
        let _ = writeln!(
            out,
            "# (d) boundaries on novelty peaks {}/{}",
            self.boundaries_on_peaks, self.boundaries
        );
        let _ = writeln!(out, "# (e) max level step per bar {}", self.max_level_step);
        let _ = writeln!(
            out,
            "# (f) louder consecutive sections ordered {}/{}",
            self.louder_steps_ordered, self.louder_steps
        );
        let _ = writeln!(
            out,
            "# drops at bars [{}]",
            self.drop_bars
                .iter()
                .map(|b| (b + 1).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(
            out,
            "# boundaries at bars [{}]",
            self.boundary_bars
                .iter()
                .map(|b| (b + 1).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        out
    }

    /// One line per section: `sec 1 bars 8-22 db -16.9 level 3 (3-3)`.
    pub fn sections_report(&self) -> String {
        let mut out = String::new();
        for s in &self.sections {
            let _ = writeln!(
                out,
                "# sec {} bars {}-{} db {:.1} level {:.2} ({}-{})",
                s.index,
                s.first_bar + 1,
                s.last_bar + 1,
                s.mean_db,
                s.mean_level,
                s.min_level,
                s.max_level
            );
        }
        out
    }
}

/// A section is "louder by more than this" for `(a)`.
pub const PAIR_DB: f32 = 3.0;
/// A bar-to-bar level jump smaller than this is not a jump worth matching a drop against.
pub const JUMP_FLOOR_DB: f32 = 2.0;

/// Run every check over one track's rows.
pub fn evaluate(rows: &[BarRow]) -> Eval {
    let mut eval = Eval::default();
    if rows.is_empty() {
        return eval;
    }

    // Sections, in bar order.
    let mut section = rows[0].section;
    let mut start = 0usize;
    for i in 1..=rows.len() {
        let ends = i == rows.len() || rows[i].section != section;
        if ends {
            let slice = &rows[start..i];
            let n = slice.len() as f32;
            let mean_db = slice.iter().map(|r| r.loudness_db).sum::<f32>() / n;
            let mean_level = slice.iter().map(|r| f32::from(r.level)).sum::<f32>() / n;
            eval.sections.push(SectionSummary {
                index: section,
                first_bar: slice[0].bar_index,
                last_bar: slice[slice.len() - 1].bar_index,
                mean_db,
                mean_level,
                min_level: slice.iter().map(|r| r.level).min().unwrap_or(1),
                max_level: slice.iter().map(|r| r.level).max().unwrap_or(1),
            });
            if i < rows.len() {
                section = rows[i].section;
                start = i;
            }
        }
    }

    // (a) and (f).
    for i in 0..eval.sections.len() {
        for j in (i + 1)..eval.sections.len() {
            let (a, b) = (&eval.sections[i], &eval.sections[j]);
            let db = (a.mean_db - b.mean_db).abs();
            if db <= PAIR_DB {
                continue;
            }
            let (louder, quieter) = if a.mean_db > b.mean_db { (a, b) } else { (b, a) };
            let ordered = louder.mean_level > quieter.mean_level;
            if ordered {
                eval.ordered_pairs += 1;
            }
            eval.pairs.push(SectionPair {
                louder: louder.index,
                quieter: quieter.index,
                db,
                ordered,
            });
        }
    }
    for pair in eval.sections.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        if b.mean_db > a.mean_db + PAIR_DB {
            eval.louder_steps += 1;
            if b.mean_level > a.mean_level {
                eval.louder_steps_ordered += 1;
            }
        }
    }

    // (b).
    let loudest = eval
        .sections
        .iter()
        .max_by(|a, b| a.mean_db.total_cmp(&b.mean_db));
    let quietest = eval
        .sections
        .iter()
        .min_by(|a, b| a.mean_db.total_cmp(&b.mean_db));
    eval.loudest_level = loudest.map_or(0.0, |s| s.mean_level);
    eval.quietest_level = quietest.map_or(0.0, |s| s.mean_level);

    // (c): the largest bar-to-bar rises of the fast level. A jump at index i is between bar i and
    // bar i + 1, so it "is" the bar that follows it. Rises only: a breakdown is the largest
    // *change* in many tracks and it is not a drop.
    let drop_bars: Vec<i64> = rows.iter().filter(|r| r.drop).map(|r| r.bar_index).collect();
    eval.drops = drop_bars.len();
    let mut jumps: Vec<(i64, f32)> = rows
        .windows(2)
        .enumerate()
        .map(|(i, w)| (i as i64, w[1].level_db - w[0].level_db))
        .filter(|(_, rise)| *rise > 0.0)
        .collect();
    eval.largest_jump_db = jumps.iter().map(|(_, d)| *d).fold(0.0, f32::max);    let mut magnitudes: Vec<f32> = jumps.iter().map(|(_, d)| *d).collect();
    magnitudes.sort_by(|a, b| b.total_cmp(a));
    let top = magnitudes.len().min(eval.drops.max(3));
    let threshold = magnitudes
        .get(top.saturating_sub(1))
        .copied()
        .unwrap_or(0.0)
        .max(JUMP_FLOOR_DB);
    jumps.sort_by(|a, b| b.1.total_cmp(&a.1));
    for bar in &drop_bars {
        let hit = jumps
            .iter()
            .any(|(i, d)| *d >= threshold && (i - bar).abs() <= 1);
        if hit {
            eval.drops_on_top_jumps += 1;
        }
    }
    // The other direction: how many of the largest rises got a drop. Precision alone would be
    // satisfied by one lucky drop.
    jumps.truncate(top.min(jumps.len()));
    eval.top_jumps = jumps.len();
    eval.top_jumps_with_drop = jumps
        .iter()
        .filter(|(i, d)| {
            *d >= threshold && drop_bars.iter().any(|bar| (i - bar).abs() <= 1)
        })
        .count();

    // (d): boundaries among the largest novelty peaks. A boundary is snapped to a phrase line, so
    // the peak that caused it may be up to two bars away; the check looks in that window.
    let mut novelties: Vec<f32> = rows.iter().map(|r| r.novelty).collect();
    novelties.sort_by(|a, b| b.total_cmp(a));
    eval.boundaries = rows
        .iter()
        .filter(|r| r.section_started && r.section > 0)
        .count();
    let top = novelties.len().min(eval.boundaries.max(1));
    let novelty_floor = novelties.get(top.saturating_sub(1)).copied().unwrap_or(0.0);
    eval.boundaries_on_peaks = rows
        .iter()
        .filter(|r| r.section_started && r.section > 0)
        .filter(|r| {
            rows.iter()
                .filter(|other| (other.bar_index - r.bar_index).abs() <= 2)
                .any(|other| other.novelty >= novelty_floor)
        })
        .count();

    // (e).
    eval.max_level_step = rows
        .windows(2)
        .map(|w| w[0].level.abs_diff(w[1].level))
        .max()
        .unwrap_or(0);

    eval.drop_bars = rows.iter().filter(|r| r.drop).map(|r| r.bar_index).collect();
    eval.boundary_bars = rows
        .iter()
        .filter(|r| r.section_started && r.section > 0)
        .map(|r| r.bar_index)
        .collect();

    eval
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(bar: i64, db: f32, level: u8, section: u32, started: bool, novelty: f32) -> BarRow {
        BarRow {
            bar_index: bar,
            loudness_db: db,
            level_db: db,
            level,
            section,
            section_started: started,
            novelty,
            drop: false,
        }
    }

    /// Two loud sections 7 dB apart read 3 and 4: the reza case, as rows.
    #[test]
    fn a_louder_section_is_ordered_above_an_earlier_loud_one() {
        let mut rows = Vec::new();
        for i in 0..8 {
            rows.push(row(i, -27.0, 2, 0, i == 0, 0.0));
        }
        for i in 8..23 {
            rows.push(row(i, -17.0, 3, 1, i == 8, 0.4));
        }
        for i in 23..39 {
            rows.push(row(i, -10.0, 4, 2, i == 23, 0.4));
        }
        let eval = evaluate(&rows);
        assert_eq!(eval.sections.len(), 3);
        assert_eq!(eval.ordered_pairs, eval.pairs.len());
        assert_eq!(eval.pairs.len(), 3, "-27/-17/-10 are all more than 3 dB apart");
        assert!(eval.loudest_level >= 4.0, "{}", eval.loudest_level);
        assert!(eval.quietest_level <= 2.0, "{}", eval.quietest_level);
        assert_eq!(eval.max_level_step, 1);
        assert_eq!(eval.louder_steps, 2);
        assert_eq!(eval.louder_steps_ordered, 2);
    }

    /// A drop on the biggest jump of the track is found; one on a quiet bar is not.
    #[test]
    fn drops_are_checked_against_the_largest_jumps() {
        let mut rows: Vec<BarRow> = (0..16)
            .map(|i| {
                // A quiet run, one big jump at bar 8, then a level run.
                let db = if i < 8 { -24.0 } else { -10.0 };
                row(i, db, if i < 8 { 2 } else { 4 }, 0, i == 0, 0.0)
            })
            .collect();
        rows[8].drop = true;
        let eval = evaluate(&rows);
        assert_eq!(eval.drops, 1);
        assert_eq!(eval.drops_on_top_jumps, 1);
        assert!((eval.largest_jump_db - 14.0).abs() < 1e-4);

        rows[8].drop = false;
        rows[3].drop = true;
        let eval = evaluate(&rows);
        assert_eq!(
            eval.drops_on_top_jumps, 0,
            "a drop in the middle of a steady run is not a top jump"
        );
    }

    /// A boundary that is not a novelty peak is a false section.
    #[test]
    fn boundaries_are_checked_against_novelty_peaks() {
        let rows: Vec<BarRow> = (0..16)
            .map(|i| {
                let started = i == 8;
                row(
                    i,
                    -20.0,
                    2,
                    u32::from(i >= 8),
                    started,
                    if started { 0.5 } else { 0.02 },
                )
            })
            .collect();
        let eval = evaluate(&rows);
        assert_eq!(eval.boundaries, 1);
        assert_eq!(eval.boundaries_on_peaks, 1);
    }
}
