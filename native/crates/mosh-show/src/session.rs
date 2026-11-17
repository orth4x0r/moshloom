//! The session grid: cells, launch quantization and where a follow action goes next.

use mosh_core::{AnalysisFrame, ClipId, LaunchQuantize, SessionGrid};

/// Bars per phrase for `LaunchQuantize::Phrase`. Live's default phrase length, and the design's
/// "every 4/8/16 bars"; four keeps a phrase wait short enough to land inside a section.
pub const PHRASE_BARS: u32 = 4;

/// A phase closer than this to 0 is exactly on the line, not just before it.
const ON_LINE: f32 = 1e-4;

/// The clip in a cell, or `None` when the slot is empty or outside the grid.
pub fn cell(grid: &SessionGrid, layer: usize, scene: usize) -> Option<ClipId> {
    grid.scenes.get(scene)?.cells.get(layer).copied().flatten()
}

/// One entry per distinct clip in a layer's column: `(first scene it appears in, clip)`.
pub fn column(grid: &SessionGrid, layer: usize) -> Vec<(usize, ClipId)> {
    let mut out: Vec<(usize, ClipId)> = Vec::new();
    for scene in 0..grid.scenes.len() {
        let Some(id) = cell(grid, layer, scene) else { continue };
        if out.iter().any(|(_, seen)| *seen == id) {
            continue;
        }
        out.push((scene, id));
    }
    out
}

/// The scene after `from` that has a clip in `layer`'s column, wrapping around.
pub fn next_scene_in_column(grid: &SessionGrid, layer: usize, from: usize) -> Option<usize> {
    let n = grid.scenes.len();
    if n == 0 {
        return None;
    }
    let start = from % n;
    for step in 1..=n {
        let scene = (start + step) % n;
        if cell(grid, layer, scene).is_some() {
            return Some(scene);
        }
    }
    None
}

/// The first scene that has a clip in `layer`'s column.
pub fn first_scene_in_column(grid: &SessionGrid, layer: usize) -> Option<usize> {
    (0..grid.scenes.len()).find(|scene| cell(grid, layer, *scene).is_some())
}

/// The scene after `from` with a non-empty cell anywhere, wrapping around.
pub fn next_scene_with_content(grid: &SessionGrid, from: usize) -> Option<usize> {
    let n = grid.scenes.len();
    if n == 0 {
        return None;
    }
    let start = from % n;
    for step in 1..=n {
        let scene = (start + step) % n;
        if grid.scenes[scene].cells.iter().any(|c| c.is_some()) {
            return Some(scene);
        }
    }
    None
}

/// The first scene with a non-empty cell anywhere.
pub fn first_scene_with_content(grid: &SessionGrid) -> Option<usize> {
    (0..grid.scenes.len()).find(|scene| grid.scenes[*scene].cells.iter().any(|c| c.is_some()))
}

/// The ppq of the first launch boundary at or after `f`, given the bar length in beats.
///
/// A frame that sits exactly on a boundary gets that boundary, so a launch on the bar line
/// happens now rather than a bar later.
pub fn next_boundary(quantize: LaunchQuantize, f: &AnalysisFrame, bar_len: f64) -> f64 {
    let bar_len = if bar_len > 0.0 { bar_len } else { 4.0 };
    let offset = match quantize {
        LaunchQuantize::None => 0.0,
        LaunchQuantize::Beat => {
            if f.beat_phase < ON_LINE {
                0.0
            } else {
                1.0 - f.beat_phase as f64
            }
        }
        LaunchQuantize::Bar => bar_offset(f.bar_phase, bar_len),
        LaunchQuantize::Phrase => {
            let phrase = PHRASE_BARS as i64;
            let remainder = f.bar_index.rem_euclid(phrase);
            if remainder == 0 && f.bar_phase < ON_LINE {
                0.0
            } else {
                // The next bar start on a phrase line, counted from this bar's start.
                let target = (f.bar_index.div_euclid(phrase) + 1) * phrase;
                ((target - f.bar_index) as f64 - f.bar_phase as f64) * bar_len
            }
        }
    };
    f.ppq + offset.max(0.0)
}

fn bar_offset(bar_phase: f32, bar_len: f64) -> f64 {
    if bar_phase < ON_LINE { 0.0 } else { (1.0 - bar_phase as f64) * bar_len }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mosh_core::{EventFlags, TransportSource};

    fn at(ppq: f64, bar_phase: f32, bar_index: i64) -> AnalysisFrame {
        AnalysisFrame {
            ppq,
            bar_phase,
            bar_index,
            beat_phase: ppq.rem_euclid(1.0) as f32,
            source: TransportSource::Plugin,
            events: EventFlags::empty(),
            ..AnalysisFrame::default()
        }
    }

    #[test]
    fn none_quantizes_to_now() {
        let f = at(5.5, 0.375, 1);
        assert_eq!(next_boundary(LaunchQuantize::None, &f, 4.0), 5.5);
    }

    #[test]
    fn beat_lands_on_the_next_beat() {
        let f = at(5.5, 0.375, 1);
        assert!((next_boundary(LaunchQuantize::Beat, &f, 4.0) - 6.0).abs() < 1e-9);
        let on = at(6.0, 0.5, 1);
        assert!((next_boundary(LaunchQuantize::Beat, &on, 4.0) - 6.0).abs() < 1e-9);
    }

    #[test]
    fn bar_lands_on_the_next_bar_line() {
        let f = at(5.5, 0.375, 1);
        assert!((next_boundary(LaunchQuantize::Bar, &f, 4.0) - 8.0).abs() < 1e-4);
        let on = at(8.0, 0.0, 2);
        assert!((next_boundary(LaunchQuantize::Bar, &on, 4.0) - 8.0).abs() < 1e-9);
        // 3/4 bars are 3 beats long: ppq 4.5 is mid bar 1 (3..6).
        let three = at(4.5, 0.5, 1);
        assert!((next_boundary(LaunchQuantize::Bar, &three, 3.0) - 6.0).abs() < 1e-9);
    }

    #[test]
    fn phrase_lands_on_the_next_phrase_line() {
        // Bar 5, phase 0.7, 4/4: the next phrase line is bar 8. The phase is an f32 in the frame,
        // so the target carries about 1e-7 of it; 1e-4 of a beat is still far below a frame.
        let f = at(22.8, 0.7, 5);
        assert!((next_boundary(LaunchQuantize::Phrase, &f, 4.0) - 32.0).abs() < 1e-4);
        // On a phrase line it is now.
        let on = at(32.0, 0.0, 8);
        assert!((next_boundary(LaunchQuantize::Phrase, &on, 4.0) - 32.0).abs() < 1e-6);
        // Bar 7, phase 0: the next phrase line is bar 8.
        let last = at(28.0, 0.0, 7);
        assert!((next_boundary(LaunchQuantize::Phrase, &last, 4.0) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn column_dedupes_clips() {
        let grid = SessionGrid {
            layers: vec![Default::default()],
            scenes: vec![
                mosh_core::SceneRow { name: "0".into(), cells: vec![Some(7)] },
                mosh_core::SceneRow { name: "1".into(), cells: vec![None] },
                mosh_core::SceneRow { name: "2".into(), cells: vec![Some(7)] },
                mosh_core::SceneRow { name: "3".into(), cells: vec![Some(9)] },
            ],
            quantize: LaunchQuantize::Bar,
        };
        // Distinct clips, at the first scene each appears in.
        assert_eq!(column(&grid, 0), vec![(0, 7), (3, 9)]);
        // `Next` walks the column's non-empty slots, duplicates included, like Live.
        assert_eq!(next_scene_in_column(&grid, 0, 0), Some(2));
        assert_eq!(next_scene_in_column(&grid, 0, 2), Some(3));
        assert_eq!(next_scene_in_column(&grid, 0, 3), Some(0));
        assert_eq!(first_scene_in_column(&grid, 0), Some(0));
        assert_eq!(next_scene_with_content(&grid, 3), Some(0));
        assert_eq!(first_scene_with_content(&grid), Some(0));
    }
}
