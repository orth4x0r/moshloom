//! The APC mini's pad LEDs, as a mirror of the Session grid.
//!
//! The pads are the grid: column `x` is layer `x`, row `y` (from the top) is scene `y`, the same
//! mapping `mosh-midi` decodes a pad press with. What a pad shows:
//!
//! | pad | means |
//! | --- | --- |
//! | off | the slot is empty |
//! | yellow | a clip is loaded |
//! | green, blinking | launched, waiting for its bar line |
//! | green | playing |
//! | red, blinking | playing, and a stop is waiting for its bar line |
//!
//! Everything here is pure, so the mapping is tested without a controller; `AppCore` sends the
//! pads that changed.

use mosh_core::SessionGrid;
use mosh_midi::PadColor;

use crate::core::GridState;

/// The APC mini's pad grid is 8 x 8.
pub const PADS: usize = 8;

/// One colour per pad, `[y][x]` with `y` counted from the top row.
pub type PadGrid = [[Option<PadColor>; PADS]; PADS];

/// A grid in which no pad's colour is known: every pad of the next grid differs from it, which
/// is how the whole grid is sent to a controller that has just been plugged in.
pub fn unknown_grid() -> PadGrid {
    [[None; PADS]; PADS]
}

/// What each pad should show for this grid and what it is doing.
pub fn pad_grid(grid: &SessionGrid, state: &GridState) -> PadGrid {
    let mut pads = [[Some(PadColor::Off); PADS]; PADS];
    for (y, row) in pads.iter_mut().enumerate() {
        for (x, pad) in row.iter_mut().enumerate() {
            let cell = grid
                .scenes
                .get(y)
                .and_then(|scene| scene.cells.get(x).copied().flatten());
            let Some(clip) = cell else { continue };
            let scene = y as u16;
            let playing = state.playing[x] == Some(clip) && state.scene[x] == Some(scene);
            *pad = Some(if state.queued[x] == Some(scene) {
                PadColor::GreenBlink
            } else if playing && state.stopping & (1 << x) != 0 {
                PadColor::RedBlink
            } else if playing {
                PadColor::Green
            } else {
                PadColor::Yellow
            });
        }
    }
    pads
}

/// The pads whose colour differs between two grids, as `(x, y, colour)`.
pub fn pad_changes(previous: &PadGrid, next: &PadGrid) -> Vec<(u8, u8, PadColor)> {
    let mut changes = Vec::new();
    for y in 0..PADS {
        for x in 0..PADS {
            if previous[y][x] != next[y][x]
                && let Some(colour) = next[y][x]
            {
                changes.push((x as u8, y as u8, colour));
            }
        }
    }
    changes
}

#[cfg(test)]
mod tests {
    use super::*;
    use mosh_core::{ControlEvent, Layer, SceneRow};

    fn grid() -> SessionGrid {
        SessionGrid {
            layers: vec![Layer::default(), Layer::default()],
            scenes: vec![
                SceneRow {
                    name: "a".into(),
                    cells: vec![Some(1), Some(2)],
                },
                SceneRow {
                    name: "b".into(),
                    cells: vec![Some(3), None],
                },
            ],
            ..SessionGrid::default()
        }
    }

    #[test]
    fn pads_mirror_the_grid_through_a_launch_and_a_stop() {
        let grid = grid();
        let mut state = GridState::default();

        // Nothing playing: loaded slots are yellow, everything else is off.
        let idle = pad_grid(&grid, &state);
        assert_eq!(idle[0][0], Some(PadColor::Yellow));
        assert_eq!(idle[0][1], Some(PadColor::Yellow));
        assert_eq!(idle[1][0], Some(PadColor::Yellow));
        assert_eq!(idle[1][1], Some(PadColor::Off));
        assert_eq!(idle[7][7], Some(PadColor::Off));
        // A controller that was just plugged in gets all 64 pads.
        assert_eq!(pad_changes(&unknown_grid(), &idle).len(), 64);

        // A pad press: the slot blinks until the show plays it.
        state.note_requests(&grid, &[ControlEvent::LaunchClip { layer: 0, scene: 1 }]);
        state.settle(&grid, &[None, None]);
        let queued = pad_grid(&grid, &state);
        assert_eq!(queued[1][0], Some(PadColor::GreenBlink));
        assert_eq!(
            pad_changes(&idle, &queued),
            vec![(0, 1, PadColor::GreenBlink)],
            "one pad changed, one message"
        );

        // The bar line: clip 3 is on layer 0.
        state.settle(&grid, &[Some(3), None]);
        let playing = pad_grid(&grid, &state);
        assert_eq!(playing[1][0], Some(PadColor::Green));
        assert_eq!(playing[0][0], Some(PadColor::Yellow));

        // A scene launch queues every slot of the row; the empty one stays off.
        state.note_requests(&grid, &[ControlEvent::LaunchScene(0)]);
        let scene = pad_grid(&grid, &state);
        assert_eq!(scene[0][0], Some(PadColor::GreenBlink));
        assert_eq!(scene[0][1], Some(PadColor::GreenBlink));
        assert_eq!(scene[1][0], Some(PadColor::Green), "still playing until the bar");
        state.settle(&grid, &[Some(1), Some(2)]);
        let scene = pad_grid(&grid, &state);
        assert_eq!(scene[0][0], Some(PadColor::Green));
        assert_eq!(scene[0][1], Some(PadColor::Green));
        assert_eq!(scene[1][0], Some(PadColor::Yellow));

        // A stop blinks red on the playing pad, then the pad is just loaded again.
        state.note_requests(&grid, &[ControlEvent::StopLayer(1)]);
        assert_eq!(pad_grid(&grid, &state)[0][1], Some(PadColor::RedBlink));
        state.settle(&grid, &[Some(1), None]);
        assert_eq!(pad_grid(&grid, &state)[0][1], Some(PadColor::Yellow));

        // A clip the show launched by itself (Autopilot) lights the row that holds it.
        let mut auto = GridState::default();
        auto.settle(&grid, &[Some(3), None]);
        assert_eq!(pad_grid(&grid, &auto)[1][0], Some(PadColor::Green));

        // A launch of an empty slot or a layer off the grid queues nothing.
        let mut none = GridState::default();
        none.note_requests(
            &grid,
            &[
                ControlEvent::LaunchClip { layer: 1, scene: 1 },
                ControlEvent::LaunchClip { layer: 9, scene: 0 },
            ],
        );
        assert_eq!(none, GridState::default());
    }
}
