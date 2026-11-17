//! Input from controllers and the Moshloom Control plugin, already decoded.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ControlEvent {
    LaunchClip { layer: usize, scene: usize },
    LaunchScene(usize),
    StopLayer(usize),
    /// Freeze automation (follow actions, intensity-driven changes); manual launches still work.
    Lock(bool),
    BackToArrangement,
    SetParam { layer: usize, index: usize, value: f32 },
    SetOpacity { layer: usize, value: f32 },
    /// Added to the analysed intensity, -1..1.
    IntensityBias(f32),
    SelectSong(usize),
}
