//! The show model: what is shown, saved in the project file.

use crate::analysis::EventFlags;
use serde::{Deserialize, Serialize};

pub type ClipId = u32;
/// "tunnel", "fluid", ..., or "custom:<name>".
pub type VisualizerId = String;

pub const PROJECT_SCHEMA: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct ClipParams {
    /// Generic visualizer parameters; meanings per visualizer are listed by mosh-shaders.
    pub p: [f32; 8],
    pub palette: u16,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParamRange {
    pub lo: f32,
    pub hi: f32,
}

impl Default for ParamRange {
    fn default() -> Self {
        ParamRange { lo: 0.0, hi: 1.0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectKind {
    Fold,
    Mirror,
    Streak,
    Crush,
    Zoom,
    Spin,
    Echo,
    Negative,
    Mosh,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EffectSlot {
    pub kind: EffectKind,
    pub amount: f32,
    pub param: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TransitionKind {
    #[default]
    Cut,
    Crossfade,
    Datamosh,
    Melt,
    Strobe,
}

#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Transition {
    pub kind: TransitionKind,
    pub beats: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LaunchQuantize {
    None,
    Beat,
    #[default]
    Bar,
    Phrase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FollowAction {
    #[default]
    None,
    Next,
    Random,
    RandomProcedural,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Clip {
    pub id: ClipId,
    pub name: String,
    pub visualizer: VisualizerId,
    pub params: ClipParams,
    /// Parameter ranges per intensity level (index 0 = level 1).
    pub ranges: [[ParamRange; 8]; 4],
    /// How hard the beat pulse moves parameters, per intensity level.
    pub pulse_depth: [f32; 4],
    pub effects: Vec<EffectSlot>,
    pub transition_in: Transition,
    pub follow: FollowAction,
    pub follow_after_bars: u16,
    /// Fire the follow action on LEVEL_UP / DROP at the next downbeat.
    pub react_to_events: bool,
    /// Overlay clips: path to a video file.
    pub video: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum BlendMode {
    #[default]
    Normal,
    Add,
    Screen,
    Multiply,
    LumaKey,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LayerKind {
    #[default]
    Visualizer,
    Overlay,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    pub kind: LayerKind,
    pub blend: BlendMode,
    pub opacity: f32,
    /// Internal render scale relative to the output, e.g. 0.5 renders at half size and upscales.
    pub internal_scale: f32,
}

impl Default for Layer {
    fn default() -> Self {
        Layer { kind: LayerKind::Visualizer, blend: BlendMode::Normal, opacity: 1.0, internal_scale: 0.5 }
    }
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct SceneRow {
    pub name: String,
    /// One cell per layer.
    pub cells: Vec<Option<ClipId>>,
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct SessionGrid {
    pub layers: Vec<Layer>,
    pub scenes: Vec<SceneRow>,
    pub quantize: LaunchQuantize,
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct SongMarker {
    pub name: String,
    pub start_ppq: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct SigChange {
    pub ppq: f64,
    pub num: u16,
    pub den: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct ArrClip {
    pub clip: ClipId,
    /// Half-open range [start_ppq, end_ppq).
    pub start_ppq: f64,
    pub end_ppq: f64,
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Arrangement {
    pub songs: Vec<SongMarker>,
    /// One lane per layer.
    pub lanes: Vec<Vec<ArrClip>>,
    pub sig_map: Vec<SigChange>,
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Preset {
    pub name: String,
    pub visualizer: VisualizerId,
    pub params: ClipParams,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ShowMode {
    /// Follow Live's playhead through the arrangement.
    #[default]
    Arrangement,
    /// Clip grid, launched by hand.
    Session,
    /// Clip grid, launched by follow actions and intensity events, no hands.
    Autopilot,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Project {
    pub schema: u32,
    pub mode: ShowMode,
    pub clips: Vec<Clip>,
    pub presets: Vec<Preset>,
    pub session: SessionGrid,
    pub arrangement: Arrangement,
}

impl Default for Project {
    fn default() -> Self {
        Project {
            schema: PROJECT_SCHEMA,
            mode: ShowMode::default(),
            clips: Vec::new(),
            presets: Vec::new(),
            session: SessionGrid::default(),
            arrangement: Arrangement::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActiveTransition {
    pub kind: TransitionKind,
    /// 0..1
    pub progress: f32,
    pub from: Option<ClipId>,
}

/// What one layer should draw this frame.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerState {
    pub layer: usize,
    pub clip: Option<ClipId>,
    pub visualizer: Option<VisualizerId>,
    /// Resolved for the current level and rhythm.
    pub params: ClipParams,
    pub effects: Vec<EffectSlot>,
    pub transition: Option<ActiveTransition>,
    pub blend: BlendMode,
    pub opacity: f32,
    pub seed: u64,
}

/// The show's output for one frame.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct RenderPlan {
    pub layers: Vec<LayerState>,
    /// A full-frame flash is wanted; mosh-render's FlashLimiter decides whether it happens.
    pub flash_requested: bool,
    /// Events that drove this frame, for the stats overlay.
    pub events: EventFlags,
}
