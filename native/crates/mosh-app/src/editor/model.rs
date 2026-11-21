//! The editing model: a `Project` plus the commands that change it, with no window anywhere.
//!
//! One rule shapes the whole file: **a command either applies completely or not at all.** A clip
//! edit that would half-apply (a bad parameter index, a marker that does not exist) returns the
//! error and leaves the project exactly as it was, because the same project is what the stage is
//! drawing and what the next save writes to disk.
//!
//! A second rule is about the running show: nothing here talks to it. An applied edit marks the
//! model dirty, and the shell takes the whole project out with [`EditorModel::handoff`] and hands
//! it to `AppCore`, which swaps it in on the next bar line. So a failed command cannot leave the
//! stage half-edited, and a good one never lands mid-phrase. What *can* be shown at once (a
//! parameter of a clip that is on stage, a layer's opacity) is returned as control events in
//! [`EditorOutcome::live`], so a slider feels direct while the project still lands on the bar.
//!
//! Every edit to the project is one undo step ([`Command::Undo`]); a drag of one slider is one
//! step, not one per frame.

use std::sync::OnceLock;

use mosh_core::{
    ArrClip, BlendMode, Clip, ClipId, ClipParams, ControlEvent, EffectKind, EffectSlot,
    FollowAction, LaunchQuantize, Layer, LayerKind, ParamRange, Preset, Project, SceneRow,
    ShowMode, SigChange, SongMarker, Transition, TransitionKind,
};
use mosh_visuals::catalog::VisualizerEntry;

use crate::core::Status;

/// The most layers a project can hold: the compositor's one-pass stack, and the APC mini's
/// eight pad columns.
pub const MAX_LAYERS: usize = 8;

/// How many undo steps are kept.
const UNDO_DEPTH: usize = 100;

/// Two positions closer than this (in beats) are the same position.
const EPS: f64 = 1e-6;

/// Every visualizer a clip may use, in browser order (`mosh_visuals::catalog`, built once).
pub fn catalog() -> &'static [VisualizerEntry] {
    static CATALOG: OnceLock<Vec<VisualizerEntry>> = OnceLock::new();
    CATALOG.get_or_init(mosh_visuals::catalog::catalog)
}

/// The catalog entry of `visualizer`, when it is a known one.
pub fn catalog_entry(visualizer: &str) -> Option<&'static VisualizerEntry> {
    catalog().iter().find(|entry| entry.id == visualizer)
}

/// What a clip's `video` field holds, by visualizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaKind {
    /// The visualizer has no media source.
    None,
    /// `overlay`: a video file path.
    File,
    /// `camera`: `camera:<device name>`, or `camera:` for the default camera.
    Camera,
    /// `text`: `text:<the words>`, lines separated by a newline.
    Text,
}

impl MediaKind {
    pub fn of(visualizer: &str) -> MediaKind {
        match visualizer {
            "overlay" => MediaKind::File,
            "camera" => MediaKind::Camera,
            "text" => MediaKind::Text,
            _ => MediaKind::None,
        }
    }

    /// The prefix the source is stored with.
    pub fn prefix(self) -> &'static str {
        match self {
            MediaKind::Camera => "camera:",
            MediaKind::Text => "text:",
            MediaKind::None | MediaKind::File => "",
        }
    }
}

/// The control window's two main views. `Tab` toggles between them, as in Live.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum View {
    /// The timeline: bars, song markers, a lane per layer, clip blocks, the playhead.
    Arrangement,
    /// The clip grid: layers x scenes, launch, stop, what is playing and what is queued.
    #[default]
    Session,
}

impl View {
    pub fn label(self) -> &'static str {
        match self {
            View::Arrangement => "Arrangement",
            View::Session => "Session",
        }
    }

    pub fn other(self) -> View {
        match self {
            View::Arrangement => View::Session,
            View::Session => View::Arrangement,
        }
    }
}

/// The panels around the main view, and the shortcuts overlay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Panels {
    /// The browser on the left: visualizers, presets, the project file.
    pub browser: bool,
    /// The detail panel at the bottom: the selected clip.
    pub detail: bool,
    /// The `?` shortcuts overlay.
    pub help: bool,
}

impl Default for Panels {
    fn default() -> Panels {
        Panels {
            browser: true,
            detail: true,
            help: false,
        }
    }
}

/// Where a new clip comes from: the browser drags all three kinds onto a slot or a lane.
#[derive(Clone, Debug, PartialEq)]
pub enum ClipSource {
    /// A clip that is already in the project (shared, not copied).
    Existing(ClipId),
    /// A fresh clip of this visualizer.
    Visualizer(String),
    /// A fresh clip with a preset's visualizer, parameters and seed.
    Preset(usize),
}

/// One block on the arrangement: a lane and its position in that lane's list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpanRef {
    pub lane: usize,
    pub index: usize,
}

/// Everything the performer can ask the editor to do.
///
/// The UI layer builds these and applies nothing itself; the shell applies them and forwards the
/// outcome. Variants carry the values the widget collected, so a command is a complete
/// description of the edit (which is what makes replays and tests possible).
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    SelectView(View),
    /// `Tab`: the other main view.
    ToggleView,
    SetBrowser(bool),
    SetDetail(bool),
    SetHelp(bool),
    SelectClip(Option<ClipId>),
    SelectScene(usize),
    /// Select a block on the arrangement (and the clip it shows).
    SelectSpan(SpanRef),

    RenameClip {
        clip: ClipId,
        name: String,
    },
    /// One of the eight palette rows.
    SetPalette {
        clip: ClipId,
        palette: u16,
    },
    /// One of a clip's eight visualizer parameters, as a 0..1 position.
    SetClipParam {
        clip: ClipId,
        index: usize,
        value: f32,
    },
    SetVisualizer {
        clip: ClipId,
        visualizer: String,
    },
    /// The clip's fixed procedural seed (`ClipParams::seed`).
    SetSeed {
        clip: ClipId,
        seed: u64,
    },
    /// The value range a parameter moves in at intensity level 1..=4.
    SetRange {
        clip: ClipId,
        level: u8,
        index: usize,
        lo: f32,
        hi: f32,
    },
    /// How hard the beat pulse moves parameters, at intensity level 1..=4.
    SetPulseDepth {
        clip: ClipId,
        level: u8,
        depth: f32,
    },
    AddEffect {
        clip: ClipId,
        kind: EffectKind,
    },
    RemoveEffect {
        clip: ClipId,
        index: usize,
    },
    SetEffect {
        clip: ClipId,
        index: usize,
        amount: f32,
        param: f32,
    },
    SetTransition {
        clip: ClipId,
        kind: TransitionKind,
        beats: f32,
    },
    SetFollow {
        clip: ClipId,
        action: FollowAction,
        after_bars: u16,
    },
    SetReactToEvents {
        clip: ClipId,
        on: bool,
    },
    /// The clip's media source (`Clip::video`): a video path, `camera:<device>` or
    /// `text:<words>`. `None` or an empty source clears it.
    SetVideo {
        clip: ClipId,
        path: Option<String>,
    },

    /// A song marker, at an absolute position in beats (`mosh-als`'s unit, and the project's).
    AddMarker {
        name: String,
        ppq: f64,
    },
    RemoveMarker {
        index: usize,
    },
    MoveMarker {
        index: usize,
        ppq: f64,
        snap: bool,
    },
    RenameMarker {
        index: usize,
        name: String,
    },
    /// Read a Live set's locators and time signatures into the arrangement.
    ImportAls {
        path: String,
    },

    /// Put a block on a lane. `len_ppq: None` is four bars. Blocks under it are trimmed, split or
    /// replaced, as in Live.
    AddSpan {
        lane: usize,
        source: ClipSource,
        start_ppq: f64,
        len_ppq: Option<f64>,
        snap: bool,
    },
    /// Drag a block: to another start, and maybe another lane.
    MoveSpan {
        span: SpanRef,
        to_lane: usize,
        start_ppq: f64,
        snap: bool,
    },
    /// Drag a block's edge.
    ResizeSpan {
        span: SpanRef,
        start_ppq: f64,
        end_ppq: f64,
        snap: bool,
    },
    RemoveSpan(SpanRef),
    /// A copy of the block (and of its clip) right after it.
    DuplicateSpan(SpanRef),

    /// Fill a session slot. A clip that was there and is used nowhere else leaves the project.
    SetCell {
        layer: usize,
        scene: usize,
        source: ClipSource,
    },
    ClearCell {
        layer: usize,
        scene: usize,
    },
    /// Drag a slot onto another: the two swap.
    SwapCells {
        from: (usize, usize),
        to: (usize, usize),
    },
    AddScene,
    RenameScene {
        scene: usize,
        name: String,
    },
    RemoveScene(usize),
    AddLayer,
    RemoveLayer(usize),
    SetLayerOpacity {
        layer: usize,
        opacity: f32,
    },
    SetLayerBlend {
        layer: usize,
        blend: BlendMode,
    },
    /// Visualizer, Video (overlay), Adjustment or Mask.
    SetLayerKind {
        layer: usize,
        kind: LayerKind,
    },

    /// Save the selected clip's visualizer, parameters and seed as a preset.
    SavePreset {
        name: String,
    },
    ApplyPreset {
        index: usize,
    },
    DeletePreset {
        index: usize,
    },

    SetMode(ShowMode),
    SetQuantize(LaunchQuantize),
    /// Launch a session cell on the running show, for a look at the next bar.
    Launch {
        layer: usize,
        scene: usize,
    },
    LaunchScene(usize),
    /// Stop a layer on the running show, at the launch quantize.
    StopLayer(usize),
    /// Hand the layers a Session launch took over back to the arrangement.
    BackToArrangement,
    /// Freeze follow actions and intensity-driven launches (manual launches still work).
    Lock(bool),

    OpenProject {
        path: String,
    },
    SaveProject {
        path: String,
    },
    /// Save to the path the project was opened from or last saved to.
    Save,
    /// An empty project: one layer, one scene, nothing placed.
    NewProject,

    Undo,
    Redo,
}

/// What applying a command did, beyond changing the project.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EditorOutcome {
    /// The project changed; the shell should hand it to the running show.
    pub project_changed: bool,
    /// A launch, a stop or a lock for the running show, now.
    pub launch: Option<ControlEvent>,
    /// What the edit should look like on stage *now*, before the project lands on the next bar:
    /// a parameter or an opacity of a layer that is drawing the edited clip.
    pub live: Vec<ControlEvent>,
    /// The show's mode changed; the running show applies it at once, not at the next bar.
    pub set_mode: Option<ShowMode>,
}

impl EditorOutcome {
    fn changed() -> EditorOutcome {
        EditorOutcome {
            project_changed: true,
            ..EditorOutcome::default()
        }
    }

    fn control(event: ControlEvent) -> EditorOutcome {
        EditorOutcome {
            launch: Some(event),
            ..EditorOutcome::default()
        }
    }
}

/// Everything a command can be refused for.
///
/// Every variant means "nothing was changed"; the model reports the reason in its message line
/// and the project stays exactly as the stage is drawing it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EditorError {
    UnknownClip(ClipId),
    UnknownPreset(usize),
    UnknownMarker(usize),
    /// No clip is selected where one is needed (presets, mostly).
    NoClip,
    /// A visualizer parameter index (0..8).
    BadParam(usize),
    /// An intensity level (1..=4).
    BadLevel(u8),
    /// An effect slot index in this clip's list.
    BadEffect(usize),
    /// The clip already has this effect; the renderer merges duplicates by kind, so one slot each.
    DuplicateEffect(EffectKind),
    /// A session cell that does not exist.
    BadCell {
        layer: usize,
        scene: usize,
    },
    /// A scene row that does not exist.
    BadScene(usize),
    /// A visualizer id the renderer does not know: it would draw nothing at all.
    UnknownVisualizer(String),
    /// An arrangement lane that does not exist.
    BadLane(usize),
    /// A block that is not on that lane.
    BadSpan(SpanRef),
    /// A layer that does not exist.
    BadLayer(usize),
    /// The project already has [`MAX_LAYERS`] layers.
    TooManyLayers,
    /// The last layer or the last scene cannot be removed.
    LastOne(&'static str),
    /// `Save` on a project that has no file yet.
    NoPath,
    NothingToUndo,
    NothingToRedo,
    Io(String),
}

impl std::fmt::Display for EditorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EditorError::UnknownClip(id) => write!(f, "no clip {id} in this project"),
            EditorError::UnknownPreset(index) => write!(f, "no preset {}", index + 1),
            EditorError::UnknownMarker(index) => write!(f, "no song marker {}", index + 1),
            EditorError::NoClip => f.write_str("select a clip first"),
            EditorError::BadParam(index) => {
                write!(f, "parameter {index} does not exist (p0..p7)")
            }
            EditorError::BadLevel(level) => {
                write!(f, "intensity level {level} does not exist (1-4)")
            }
            EditorError::BadEffect(index) => write!(f, "no effect slot {}", index + 1),
            EditorError::DuplicateEffect(kind) => write!(f, "{kind:?} is already on this clip"),
            EditorError::BadCell { layer, scene } => {
                write!(f, "no clip in layer {}, scene {}", layer + 1, scene + 1)
            }
            EditorError::BadScene(scene) => write!(f, "no scene {}", scene + 1),
            EditorError::UnknownVisualizer(name) => write!(f, "no visualizer named {name}"),
            EditorError::BadLane(lane) => write!(f, "no lane {}", lane + 1),
            EditorError::BadSpan(span) => {
                write!(f, "no block {} on lane {}", span.index + 1, span.lane + 1)
            }
            EditorError::BadLayer(layer) => write!(f, "no layer {}", layer + 1),
            EditorError::TooManyLayers => write!(f, "a show has at most {MAX_LAYERS} layers"),
            EditorError::LastOne(what) => write!(f, "the last {what} stays"),
            EditorError::NoPath => f.write_str("this show has no file yet: type a path and save"),
            EditorError::NothingToUndo => f.write_str("nothing to undo"),
            EditorError::NothingToRedo => f.write_str("nothing to redo"),
            EditorError::Io(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for EditorError {}

/// Which slider a command is one tick of, so a whole drag is one undo step.
type UndoKey = (u8, u32, u32);

/// The control window's model: the project, what the performer has selected, and the last line of
/// status. Everything the editors do goes through [`EditorModel::apply`].
pub struct EditorModel {
    project: Project,
    view: View,
    panels: Panels,
    selected: Option<ClipId>,
    selected_scene: Option<usize>,
    selected_span: Option<SpanRef>,
    status: Status,
    message: Option<String>,
    /// The file the project was opened from or last saved to.
    path: Option<String>,
    /// The project has changed since the last [`EditorModel::handoff`].
    dirty: bool,
    edits: u64,
    /// `edits` at the last open or save.
    saved_at: u64,
    undo: Vec<Project>,
    redo: Vec<Project>,
    last_key: Option<UndoKey>,
    /// The project as it was before the command being applied (see `drop_orphans`).
    pending_before: Option<Project>,
}

impl EditorModel {
    /// An editor over `project`, with the first clip selected (so the detail panel is never blank
    /// on a project that has clips). It opens on the view the project is built in: the
    /// Arrangement when blocks are placed on it, the Session grid otherwise.
    pub fn new(project: Project) -> EditorModel {
        let selected = project.clips.first().map(|clip| clip.id);
        let view = opening_view(&project);
        EditorModel {
            project,
            view,
            panels: Panels::default(),
            selected,
            selected_scene: None,
            selected_span: None,
            status: Status::default(),
            message: None,
            path: None,
            dirty: false,
            edits: 0,
            saved_at: 0,
            undo: Vec::new(),
            redo: Vec::new(),
            last_key: None,
            pending_before: None,
        }
    }

    pub fn project(&self) -> &Project {
        &self.project
    }

    pub fn view(&self) -> View {
        self.view
    }

    pub fn panels(&self) -> Panels {
        self.panels
    }

    pub fn selected(&self) -> Option<ClipId> {
        self.selected
    }

    /// The clip the detail panel, the presets and the effect rows act on.
    pub fn selected_clip(&self) -> Option<&Clip> {
        self.clip(self.selected?)
    }

    pub fn clip(&self, id: ClipId) -> Option<&Clip> {
        self.project.clips.iter().find(|clip| clip.id == id)
    }

    /// The session row last clicked, when the grid has a selection.
    pub fn selected_scene(&self) -> Option<usize> {
        self.selected_scene
    }

    /// The arrangement block last clicked.
    pub fn selected_span(&self) -> Option<SpanRef> {
        self.selected_span
    }

    pub fn span(&self, span: SpanRef) -> Option<&ArrClip> {
        self.project.arrangement.lanes.get(span.lane)?.get(span.index)
    }

    /// Lanes the arrangement shows: one per layer, and never fewer than the file holds.
    pub fn lane_count(&self) -> usize {
        self.project
            .session
            .layers
            .len()
            .max(self.project.arrangement.lanes.len())
    }

    pub fn status(&self) -> &Status {
        &self.status
    }

    /// The last error or confirmation, for the status strip.
    pub fn message(&self) -> Option<&str> {
        self.message.as_deref()
    }

    /// The file the project was opened from or last saved to.
    pub fn path(&self) -> Option<&str> {
        self.path.as_deref()
    }

    /// Start with a file already attached (the project named on the command line).
    pub fn with_path(mut self, path: Option<String>) -> EditorModel {
        self.path = path;
        self
    }

    /// Commands applied since the model was built.
    pub fn edits(&self) -> u64 {
        self.edits
    }

    /// Whether the project has changed since the last handoff.
    pub fn dirty(&self) -> bool {
        self.dirty
    }

    /// Whether the project differs from what is on disk.
    pub fn modified(&self) -> bool {
        self.edits != self.saved_at
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Put a line in the status strip (an import summary, an I/O failure the shell saw).
    pub fn note(&mut self, text: impl Into<String>) {
        self.message = Some(text.into());
    }

    /// Fold one frame's status line in; returns whether anything a screen shows changed, so the
    /// window can leave an idle control window alone.
    pub fn set_status(&mut self, status: Status) -> bool {
        if self.status == status {
            return false;
        }
        self.status = status;
        true
    }

    /// Take the current project for the running show, or `None` when nothing changed since the
    /// last call.
    ///
    /// The project is cloned rather than borrowed because the caller sends it to another thread;
    /// an edit happens at most once per frame the performer is dragging something, so this is a
    /// handful of small clones a second, not a per-frame cost.
    pub fn handoff(&mut self) -> Option<Project> {
        if !self.dirty {
            return None;
        }
        self.dirty = false;
        Some(self.project.clone())
    }

    /// Apply one command. An error leaves the project untouched and is reported in the message.
    pub fn apply(&mut self, command: Command) -> Result<EditorOutcome, EditorError> {
        let key = undo_key(&command);
        let coalesce = key.is_some() && key == self.last_key;
        let recorded = undoable(&command);
        self.pending_before = (recorded && !coalesce).then(|| self.project.clone());
        let result = self.apply_command(command);
        let snapshot = self.pending_before.take();
        match result {
            Ok(outcome) => {
                if outcome.project_changed {
                    self.dirty = true;
                    self.edits += 1;
                    if recorded {
                        if let Some(before) = snapshot {
                            self.undo.push(before);
                            if self.undo.len() > UNDO_DEPTH {
                                self.undo.remove(0);
                            }
                            self.redo.clear();
                        }
                        self.last_key = key;
                    }
                }
                Ok(outcome)
            }
            Err(error) => {
                self.message = Some(error.to_string());
                Err(error)
            }
        }
    }

    /// Apply a batch (one frame of widgets), returning everything that reached the running show.
    ///
    /// The first error becomes the message line; the rest still apply, because one bad slider must
    /// not swallow the other nine edits of the same frame.
    pub fn apply_all(
        &mut self,
        commands: impl IntoIterator<Item = Command>,
    ) -> Vec<EditorOutcome> {
        let mut outcomes = Vec::new();
        for command in commands {
            if let Ok(outcome) = self.apply(command) {
                outcomes.push(outcome);
            }
        }
        outcomes
    }

    fn apply_command(&mut self, command: Command) -> Result<EditorOutcome, EditorError> {
        match command {
            Command::SelectView(view) => {
                self.view = view;
                Ok(EditorOutcome::default())
            }
            Command::ToggleView => {
                self.view = self.view.other();
                Ok(EditorOutcome::default())
            }
            Command::SetBrowser(open) => {
                self.panels.browser = open;
                Ok(EditorOutcome::default())
            }
            Command::SetDetail(open) => {
                self.panels.detail = open;
                Ok(EditorOutcome::default())
            }
            Command::SetHelp(open) => {
                self.panels.help = open;
                Ok(EditorOutcome::default())
            }
            Command::SelectClip(clip) => {
                if let Some(id) = clip
                    && self.clip(id).is_none()
                {
                    return Err(EditorError::UnknownClip(id));
                }
                self.selected = clip;
                if self
                    .selected_span
                    .and_then(|span| self.span(span))
                    .map(|span| span.clip)
                    != clip
                {
                    self.selected_span = None;
                }
                Ok(EditorOutcome::default())
            }
            Command::SelectScene(scene) => {
                self.selected_scene = Some(scene);
                Ok(EditorOutcome::default())
            }
            Command::SelectSpan(span) => {
                let clip = self.span(span).ok_or(EditorError::BadSpan(span))?.clip;
                self.selected_span = Some(span);
                self.selected = self.clip(clip).map(|clip| clip.id);
                Ok(EditorOutcome::default())
            }

            Command::RenameClip { clip, name } => {
                let name = name.trim().to_string();
                let clip = self.clip_mut(clip)?;
                if !name.is_empty() {
                    clip.name = name;
                }
                Ok(EditorOutcome::changed())
            }
            Command::SetPalette { clip, palette } => {
                self.clip_mut(clip)?.params.palette = palette.min(7);
                Ok(EditorOutcome::changed())
            }
            Command::SetClipParam { clip, index, value } => {
                let value = value.clamp(0.0, 1.0);
                let slot = self.clip_mut(clip)?.params.p.get_mut(index);
                let slot = slot.ok_or(EditorError::BadParam(index))?;
                *slot = value;
                let live = self.live_param(clip, index, value);
                Ok(EditorOutcome {
                    project_changed: true,
                    live,
                    ..EditorOutcome::default()
                })
            }
            Command::SetVisualizer { clip, visualizer } => {
                // A visualizer the renderer does not know would draw nothing at all, which on
                // stage looks like a dead layer: refuse it here instead.
                if !known_visualizer(&visualizer) {
                    return Err(EditorError::UnknownVisualizer(visualizer));
                }
                let clip = self.clip_mut(clip)?;
                // A media source belongs to the visualizer that reads it: a camera clip turned
                // into a tunnel must not keep "camera:" around for the next visualizer to misread.
                if MediaKind::of(&clip.visualizer) != MediaKind::of(&visualizer) {
                    clip.video = default_media(&visualizer);
                }
                clip.visualizer = visualizer;
                Ok(EditorOutcome::changed())
            }
            Command::SetSeed { clip, seed } => {
                self.clip_mut(clip)?.params.seed = seed;
                Ok(EditorOutcome::changed())
            }
            Command::SetRange {
                clip,
                level,
                index,
                lo,
                hi,
            } => {
                let slot = self.range_mut(clip, level, index)?;
                *slot = ParamRange {
                    lo: lo.min(hi),
                    hi: hi.max(lo),
                };
                Ok(EditorOutcome::changed())
            }
            Command::SetPulseDepth { clip, level, depth } => {
                let slot = self.pulse_mut(clip, level)?;
                *slot = depth.clamp(0.0, 1.0);
                Ok(EditorOutcome::changed())
            }
            Command::AddEffect { clip, kind } => {
                let clip = self.clip_mut(clip)?;
                if clip.effects.iter().any(|slot| slot.kind == kind) {
                    return Err(EditorError::DuplicateEffect(kind));
                }
                clip.effects.push(EffectSlot {
                    kind,
                    amount: 0.5,
                    param: if kind == EffectKind::Mosh { 0.0 } else { 0.5 },
                });
                Ok(EditorOutcome::changed())
            }
            Command::RemoveEffect { clip, index } => {
                let clip = self.clip_mut(clip)?;
                if index >= clip.effects.len() {
                    return Err(EditorError::BadEffect(index));
                }
                clip.effects.remove(index);
                Ok(EditorOutcome::changed())
            }
            Command::SetEffect {
                clip,
                index,
                amount,
                param,
            } => {
                let clip = self.clip_mut(clip)?;
                let slot = clip
                    .effects
                    .get_mut(index)
                    .ok_or(EditorError::BadEffect(index))?;
                slot.amount = amount.clamp(0.0, 1.0);
                slot.param = param.clamp(0.0, 1.0);
                Ok(EditorOutcome::changed())
            }
            Command::SetTransition { clip, kind, beats } => {
                self.clip_mut(clip)?.transition_in = Transition {
                    kind,
                    beats: beats.max(0.0),
                };
                Ok(EditorOutcome::changed())
            }
            Command::SetFollow {
                clip,
                action,
                after_bars,
            } => {
                let clip = self.clip_mut(clip)?;
                clip.follow = action;
                clip.follow_after_bars = after_bars.max(1);
                Ok(EditorOutcome::changed())
            }
            Command::SetReactToEvents { clip, on } => {
                self.clip_mut(clip)?.react_to_events = on;
                Ok(EditorOutcome::changed())
            }
            Command::SetVideo { clip, path } => {
                // Words are kept as typed (a lyric may start with a space or end with a newline);
                // a path or a device name is trimmed.
                let path = path
                    .map(|path| {
                        if path.starts_with("text:") {
                            path
                        } else {
                            path.trim().to_string()
                        }
                    })
                    .filter(|path| !path.is_empty());
                self.clip_mut(clip)?.video = path;
                Ok(EditorOutcome::changed())
            }

            Command::AddMarker { name, ppq } => {
                let name = match name.trim() {
                    "" => format!("marker {}", self.project.arrangement.songs.len() + 1),
                    name => name.to_string(),
                };
                let marker = SongMarker {
                    name,
                    start_ppq: ppq.max(0.0),
                };
                // One marker per position: adding on top of an existing one renames it, which is
                // what dragging a marker in Live does. The engine assumes a sorted list.
                let songs = &mut self.project.arrangement.songs;
                match songs
                    .iter_mut()
                    .find(|song| (song.start_ppq - marker.start_ppq).abs() < 1e-9)
                {
                    Some(slot) => *slot = marker,
                    None => songs.push(marker),
                }
                songs.sort_by(|a, b| a.start_ppq.total_cmp(&b.start_ppq));
                Ok(EditorOutcome::changed())
            }
            Command::RemoveMarker { index } => {
                let songs = &mut self.project.arrangement.songs;
                if index >= songs.len() {
                    return Err(EditorError::UnknownMarker(index));
                }
                songs.remove(index);
                Ok(EditorOutcome::changed())
            }
            Command::MoveMarker { index, ppq, snap } => {
                let at = self.snapped(ppq, snap);
                let songs = &mut self.project.arrangement.songs;
                let song = songs
                    .get_mut(index)
                    .ok_or(EditorError::UnknownMarker(index))?;
                song.start_ppq = at;
                songs.sort_by(|a, b| a.start_ppq.total_cmp(&b.start_ppq));
                Ok(EditorOutcome::changed())
            }
            Command::RenameMarker { index, name } => {
                let song = self
                    .project
                    .arrangement
                    .songs
                    .get_mut(index)
                    .ok_or(EditorError::UnknownMarker(index))?;
                let name = name.trim();
                if !name.is_empty() {
                    song.name = name.to_string();
                }
                Ok(EditorOutcome::changed())
            }
            Command::ImportAls { path } => {
                let info = mosh_als::read_als(&path)
                    .map_err(|error| EditorError::Io(format!("{path}: {error}")))?;
                let (markers, sigs) = (info.locators.len(), info.time_signatures.len());
                self.project.arrangement.songs = info.locators;
                self.project.arrangement.sig_map = info.time_signatures;
                self.message = Some(format!(
                    "imported {markers} song markers and {sigs} time signatures from {path}"
                ));
                Ok(EditorOutcome::changed())
            }

            Command::AddSpan {
                lane,
                source,
                start_ppq,
                len_ppq,
                snap,
            } => {
                if lane >= self.lane_count() {
                    return Err(EditorError::BadLane(lane));
                }
                let start = self.snapped(start_ppq, snap);
                let bar = bar_len_at(&self.project.arrangement.sig_map, start);
                let len = len_ppq.unwrap_or(bar * 4.0).max(min_len(bar, snap));
                let clip = self.resolve_source(source)?;
                self.ensure_lanes();
                let index = place_span(
                    &mut self.project.arrangement.lanes[lane],
                    ArrClip {
                        clip,
                        start_ppq: start,
                        end_ppq: start + len,
                    },
                );
                self.drop_orphans();
                self.selected = Some(clip);
                self.selected_span = Some(SpanRef { lane, index });
                Ok(EditorOutcome::changed())
            }
            Command::MoveSpan {
                span,
                to_lane,
                start_ppq,
                snap,
            } => {
                let old = *self.span(span).ok_or(EditorError::BadSpan(span))?;
                if to_lane >= self.lane_count() {
                    return Err(EditorError::BadLane(to_lane));
                }
                let start = self.snapped(start_ppq, snap);
                let moved = ArrClip {
                    clip: old.clip,
                    start_ppq: start,
                    end_ppq: start + (old.end_ppq - old.start_ppq),
                };
                self.ensure_lanes();
                self.project.arrangement.lanes[span.lane].remove(span.index);
                let index = place_span(&mut self.project.arrangement.lanes[to_lane], moved);
                self.drop_orphans();
                self.selected = Some(old.clip);
                self.selected_span = Some(SpanRef {
                    lane: to_lane,
                    index,
                });
                Ok(EditorOutcome::changed())
            }
            Command::ResizeSpan {
                span,
                start_ppq,
                end_ppq,
                snap,
            } => {
                let old = *self.span(span).ok_or(EditorError::BadSpan(span))?;
                let start = self.snapped(start_ppq, snap);
                let end = self.snapped(end_ppq, snap);
                let bar = bar_len_at(&self.project.arrangement.sig_map, start);
                // An edge dragged up to or past the other one does not flip the block: it leaves
                // the shortest block, at the edge that did not move.
                let least = min_len(bar, snap);
                let (start, end) = if end - start >= least - EPS {
                    (start, end)
                } else if (start_ppq - old.start_ppq).abs() > EPS {
                    ((old.end_ppq - least).max(0.0), old.end_ppq)
                } else {
                    (old.start_ppq, old.start_ppq + least)
                };
                let lane = &mut self.project.arrangement.lanes[span.lane];
                lane.remove(span.index);
                let index = place_span(
                    lane,
                    ArrClip {
                        clip: old.clip,
                        start_ppq: start,
                        end_ppq: end,
                    },
                );
                self.drop_orphans();
                self.selected = Some(old.clip);
                self.selected_span = Some(SpanRef {
                    lane: span.lane,
                    index,
                });
                Ok(EditorOutcome::changed())
            }
            Command::RemoveSpan(span) => {
                self.span(span).ok_or(EditorError::BadSpan(span))?;
                self.project.arrangement.lanes[span.lane].remove(span.index);
                self.selected_span = None;
                self.drop_orphans();
                Ok(EditorOutcome::changed())
            }
            Command::DuplicateSpan(span) => {
                let old = *self.span(span).ok_or(EditorError::BadSpan(span))?;
                let mut copy = self
                    .clip(old.clip)
                    .cloned()
                    .ok_or(EditorError::UnknownClip(old.clip))?;
                copy.id = self.next_clip_id();
                let id = copy.id;
                self.project.clips.push(copy);
                let index = place_span(
                    &mut self.project.arrangement.lanes[span.lane],
                    ArrClip {
                        clip: id,
                        start_ppq: old.end_ppq,
                        end_ppq: old.end_ppq + (old.end_ppq - old.start_ppq),
                    },
                );
                self.drop_orphans();
                self.selected = Some(id);
                self.selected_span = Some(SpanRef {
                    lane: span.lane,
                    index,
                });
                Ok(EditorOutcome::changed())
            }

            Command::SetCell {
                layer,
                scene,
                source,
            } => {
                self.check_slot(layer, scene)?;
                let clip = self.resolve_source(source)?;
                let row = &mut self.project.session.scenes[scene];
                row.cells[layer] = Some(clip);
                self.drop_orphans();
                self.selected = Some(clip);
                self.selected_scene = Some(scene);
                self.selected_span = None;
                Ok(EditorOutcome::changed())
            }
            Command::ClearCell { layer, scene } => {
                self.check_slot(layer, scene)?;
                self.project.session.scenes[scene].cells[layer] = None;
                self.drop_orphans();
                Ok(EditorOutcome::changed())
            }
            Command::SwapCells { from, to } => {
                self.check_slot(from.0, from.1)?;
                self.check_slot(to.0, to.1)?;
                let scenes = &mut self.project.session.scenes;
                let a = scenes[from.1].cells[from.0];
                let b = scenes[to.1].cells[to.0];
                scenes[from.1].cells[from.0] = b;
                scenes[to.1].cells[to.0] = a;
                self.selected_scene = Some(to.1);
                Ok(EditorOutcome::changed())
            }
            Command::AddScene => {
                let layers = self.project.session.layers.len();
                let scenes = &mut self.project.session.scenes;
                scenes.push(SceneRow {
                    name: format!("scene {}", scenes.len() + 1),
                    cells: vec![None; layers],
                });
                Ok(EditorOutcome::changed())
            }
            Command::RenameScene { scene, name } => {
                let row = self
                    .project
                    .session
                    .scenes
                    .get_mut(scene)
                    .ok_or(EditorError::BadScene(scene))?;
                let name = name.trim();
                if !name.is_empty() {
                    row.name = name.to_string();
                }
                Ok(EditorOutcome::changed())
            }
            Command::RemoveScene(scene) => {
                let scenes = &mut self.project.session.scenes;
                if scene >= scenes.len() {
                    return Err(EditorError::BadScene(scene));
                }
                if scenes.len() == 1 {
                    return Err(EditorError::LastOne("scene"));
                }
                scenes.remove(scene);
                self.selected_scene = None;
                self.drop_orphans();
                Ok(EditorOutcome::changed())
            }
            Command::AddLayer => {
                if self.project.session.layers.len() >= MAX_LAYERS {
                    return Err(EditorError::TooManyLayers);
                }
                // A new layer goes on top, and on top of a picture "add" is the blend that shows
                // both: a Normal layer at full opacity would hide everything under it.
                let first = self.project.session.layers.is_empty();
                self.project.session.layers.push(Layer {
                    blend: if first { BlendMode::Normal } else { BlendMode::Add },
                    opacity: if first { 1.0 } else { 0.6 },
                    ..Layer::default()
                });
                let layers = self.project.session.layers.len();
                for row in &mut self.project.session.scenes {
                    row.cells.resize(layers, None);
                }
                self.ensure_lanes();
                Ok(EditorOutcome::changed())
            }
            Command::RemoveLayer(layer) => {
                let layers = self.project.session.layers.len();
                if layer >= layers {
                    return Err(EditorError::BadLayer(layer));
                }
                if layers == 1 {
                    return Err(EditorError::LastOne("layer"));
                }
                self.project.session.layers.remove(layer);
                for row in &mut self.project.session.scenes {
                    if layer < row.cells.len() {
                        row.cells.remove(layer);
                    }
                }
                if layer < self.project.arrangement.lanes.len() {
                    self.project.arrangement.lanes.remove(layer);
                }
                self.selected_span = None;
                self.drop_orphans();
                Ok(EditorOutcome::changed())
            }
            Command::SetLayerOpacity { layer, opacity } => {
                let opacity = opacity.clamp(0.0, 1.0);
                self.layer_mut(layer)?.opacity = opacity;
                Ok(EditorOutcome {
                    project_changed: true,
                    live: vec![ControlEvent::SetOpacity {
                        layer,
                        value: opacity,
                    }],
                    ..EditorOutcome::default()
                })
            }
            Command::SetLayerBlend { layer, blend } => {
                self.layer_mut(layer)?.blend = blend;
                Ok(EditorOutcome::changed())
            }
            Command::SetLayerKind { layer, kind } => {
                self.layer_mut(layer)?.kind = kind;
                Ok(EditorOutcome::changed())
            }

            Command::SavePreset { name } => {
                let clip = self.selected_clip().ok_or(EditorError::NoClip)?;
                let name = match name.trim() {
                    "" => clip.name.clone(),
                    name => name.to_string(),
                };
                let preset = Preset {
                    name,
                    visualizer: clip.visualizer.clone(),
                    params: clip.params,
                };
                self.project.presets.push(preset);
                Ok(EditorOutcome::changed())
            }
            Command::ApplyPreset { index } => {
                let preset = self
                    .project
                    .presets
                    .get(index)
                    .cloned()
                    .ok_or(EditorError::UnknownPreset(index))?;
                let id = self.selected.ok_or(EditorError::NoClip)?;
                let clip = self.clip_mut(id)?;
                if MediaKind::of(&clip.visualizer) != MediaKind::of(&preset.visualizer) {
                    clip.video = default_media(&preset.visualizer);
                }
                clip.visualizer = preset.visualizer;
                clip.params = preset.params;
                Ok(EditorOutcome::changed())
            }
            Command::DeletePreset { index } => {
                if index >= self.project.presets.len() {
                    return Err(EditorError::UnknownPreset(index));
                }
                self.project.presets.remove(index);
                Ok(EditorOutcome::changed())
            }

            Command::SetMode(mode) => {
                self.project.mode = mode;
                Ok(EditorOutcome {
                    project_changed: true,
                    set_mode: Some(mode),
                    ..EditorOutcome::default()
                })
            }
            Command::SetQuantize(quantize) => {
                self.project.session.quantize = quantize;
                Ok(EditorOutcome::changed())
            }
            Command::Launch { layer, scene } => {
                self.check_cell(layer, scene)?;
                Ok(EditorOutcome::control(ControlEvent::LaunchClip {
                    layer,
                    scene,
                }))
            }
            Command::LaunchScene(scene) => {
                if scene >= self.project.session.scenes.len() {
                    return Err(EditorError::BadScene(scene));
                }
                Ok(EditorOutcome::control(ControlEvent::LaunchScene(scene)))
            }
            Command::StopLayer(layer) => {
                if layer >= self.lane_count() {
                    return Err(EditorError::BadLayer(layer));
                }
                Ok(EditorOutcome::control(ControlEvent::StopLayer(layer)))
            }
            Command::BackToArrangement => {
                Ok(EditorOutcome::control(ControlEvent::BackToArrangement))
            }
            Command::Lock(on) => Ok(EditorOutcome::control(ControlEvent::Lock(on))),

            Command::OpenProject { path } => {
                let project = mosh_show::io::load(&path)
                    .map_err(|error| EditorError::Io(error.to_string()))?;
                self.message = Some(format!(
                    "opened {path}: {} clips, {} scenes",
                    project.clips.len(),
                    project.session.scenes.len()
                ));
                self.path = Some(path);
                self.replace_project(project);
                Ok(EditorOutcome::changed())
            }
            Command::SaveProject { path } => {
                mosh_show::io::save(&self.project, &path)
                    .map_err(|error| EditorError::Io(error.to_string()))?;
                self.message = Some(format!("saved {path}"));
                self.path = Some(path);
                self.saved_at = self.edits;
                Ok(EditorOutcome::default())
            }
            Command::Save => {
                let path = self.path.clone().ok_or(EditorError::NoPath)?;
                mosh_show::io::save(&self.project, &path)
                    .map_err(|error| EditorError::Io(error.to_string()))?;
                self.message = Some(format!("saved {path}"));
                self.saved_at = self.edits;
                Ok(EditorOutcome::default())
            }
            Command::NewProject => {
                let project = Project {
                    mode: self.project.mode,
                    session: mosh_core::SessionGrid {
                        layers: vec![Layer::default()],
                        scenes: vec![SceneRow {
                            name: "scene 1".to_string(),
                            cells: vec![None],
                        }],
                        quantize: LaunchQuantize::Bar,
                    },
                    arrangement: mosh_core::Arrangement {
                        lanes: vec![Vec::new()],
                        ..mosh_core::Arrangement::default()
                    },
                    ..Project::default()
                };
                self.message = Some("new show: one layer, one scene".to_string());
                self.path = None;
                self.replace_project(project);
                Ok(EditorOutcome::changed())
            }

            Command::Undo => {
                let before = self.undo.pop().ok_or(EditorError::NothingToUndo)?;
                let now = std::mem::replace(&mut self.project, before);
                // The mode is the running show's state, not an edit: undo leaves it alone.
                self.project.mode = now.mode;
                self.redo.push(now);
                self.last_key = None;
                self.fix_selection();
                Ok(EditorOutcome::changed())
            }
            Command::Redo => {
                let after = self.redo.pop().ok_or(EditorError::NothingToRedo)?;
                let now = std::mem::replace(&mut self.project, after);
                self.project.mode = now.mode;
                self.undo.push(now);
                self.last_key = None;
                self.fix_selection();
                Ok(EditorOutcome::changed())
            }
        }
    }

    /// Swap the whole project (open, new): the history belongs to the old one.
    fn replace_project(&mut self, project: Project) {
        self.selected = project.clips.first().map(|clip| clip.id);
        self.selected_scene = None;
        self.selected_span = None;
        self.view = opening_view(&project);
        self.project = project;
        self.undo.clear();
        self.redo.clear();
        self.last_key = None;
        // `apply` counts this command after it returns: the project on disk is that count.
        self.saved_at = self.edits + 1;
    }

    /// After undo or redo: a selection that points at something that is gone is dropped.
    fn fix_selection(&mut self) {
        if self.selected.is_some_and(|id| self.clip(id).is_none()) {
            self.selected = self.project.clips.first().map(|clip| clip.id);
        }
        if self.selected_span.is_some_and(|span| {
            self.span(span).map(|span| span.clip) != self.selected
        }) {
            self.selected_span = None;
        }
        if self
            .selected_scene
            .is_some_and(|scene| scene >= self.project.session.scenes.len())
        {
            self.selected_scene = None;
        }
    }

    /// The control events that show a parameter edit on stage at once: one per layer that is
    /// drawing `clip`, with the value resolved in the current level's range.
    fn live_param(&self, clip: ClipId, index: usize, position: f32) -> Vec<ControlEvent> {
        let Some(edited) = self.clip(clip) else {
            return Vec::new();
        };
        let level = usize::from(self.status.level.clamp(1, 4) - 1);
        let range = edited.ranges[level][index];
        let value = range.lo + (range.hi - range.lo) * position;
        self.status
            .playing
            .iter()
            .enumerate()
            .filter(|(_, playing)| **playing == Some(clip))
            .map(|(layer, _)| ControlEvent::SetParam {
                layer,
                index,
                value,
            })
            .collect()
    }

    fn snapped(&self, ppq: f64, snap: bool) -> f64 {
        let ppq = ppq.max(0.0);
        if snap {
            snap_to_bar(&self.project.arrangement.sig_map, ppq)
        } else {
            ppq
        }
    }

    /// One lane per layer, so a block can be dropped on a layer that has none yet.
    fn ensure_lanes(&mut self) {
        let count = self.lane_count();
        self.project.arrangement.lanes.resize(count, Vec::new());
    }

    fn next_clip_id(&self) -> ClipId {
        self.project
            .clips
            .iter()
            .map(|clip| clip.id)
            .max()
            .map_or(1, |id| id + 1)
    }

    /// The clip a source names, creating it when the source is a visualizer or a preset.
    fn resolve_source(&mut self, source: ClipSource) -> Result<ClipId, EditorError> {
        match source {
            ClipSource::Existing(id) => {
                self.clip(id).ok_or(EditorError::UnknownClip(id))?;
                Ok(id)
            }
            ClipSource::Visualizer(visualizer) => {
                if !known_visualizer(&visualizer) {
                    return Err(EditorError::UnknownVisualizer(visualizer));
                }
                let id = self.next_clip_id();
                self.project.clips.push(new_clip(id, &visualizer));
                Ok(id)
            }
            ClipSource::Preset(index) => {
                let preset = self
                    .project
                    .presets
                    .get(index)
                    .cloned()
                    .ok_or(EditorError::UnknownPreset(index))?;
                let id = self.next_clip_id();
                let mut clip = new_clip(id, &preset.visualizer);
                clip.name = preset.name;
                clip.params = preset.params;
                self.project.clips.push(clip);
                Ok(id)
            }
        }
    }

    /// Remove every clip no slot and no block shows, if this command is what orphaned it.
    ///
    /// A project read from disk may hold clips nothing references (a hand-written file); those
    /// were not placed by this editor and are left alone, which is why the set of orphans is
    /// compared with the one before the edit rather than swept.
    fn drop_orphans(&mut self) {
        let used = |id: ClipId, project: &Project| {
            project
                .session
                .scenes
                .iter()
                .any(|row| row.cells.contains(&Some(id)))
                || project
                    .arrangement
                    .lanes
                    .iter()
                    .flatten()
                    .any(|span| span.clip == id)
        };
        let Some(before) = self.undo_reference() else {
            return;
        };
        let orphaned: Vec<ClipId> = self
            .project
            .clips
            .iter()
            .map(|clip| clip.id)
            .filter(|id| !used(*id, &self.project) && used(*id, before))
            .collect();
        if orphaned.is_empty() {
            return;
        }
        self.project.clips.retain(|clip| !orphaned.contains(&clip.id));
        if self.selected.is_some_and(|id| orphaned.contains(&id)) {
            self.selected = None;
            self.selected_span = None;
        }
    }

    /// The project as it was before the command being applied: `apply` clones it for the undo
    /// stack before calling `apply_command`, but only hands it over afterwards, so the copy is
    /// kept here for the duration of the command.
    fn undo_reference(&self) -> Option<&Project> {
        self.pending_before.as_ref()
    }

    fn clip_mut(&mut self, id: ClipId) -> Result<&mut Clip, EditorError> {
        self.project
            .clips
            .iter_mut()
            .find(|clip| clip.id == id)
            .ok_or(EditorError::UnknownClip(id))
    }

    fn layer_mut(&mut self, layer: usize) -> Result<&mut Layer, EditorError> {
        self.project
            .session
            .layers
            .get_mut(layer)
            .ok_or(EditorError::BadLayer(layer))
    }

    /// One parameter range: `level` is 1..=4, as the design names the intensity levels.
    fn range_mut(
        &mut self,
        id: ClipId,
        level: u8,
        index: usize,
    ) -> Result<&mut ParamRange, EditorError> {
        if !(1..=4).contains(&level) {
            return Err(EditorError::BadLevel(level));
        }
        if index >= 8 {
            return Err(EditorError::BadParam(index));
        }
        let clip = self.clip_mut(id)?;
        Ok(&mut clip.ranges[usize::from(level - 1)][index])
    }

    fn pulse_mut(&mut self, id: ClipId, level: u8) -> Result<&mut f32, EditorError> {
        if !(1..=4).contains(&level) {
            return Err(EditorError::BadLevel(level));
        }
        let clip = self.clip_mut(id)?;
        Ok(&mut clip.pulse_depth[usize::from(level - 1)])
    }

    /// A slot that exists (filled or not), with its row as wide as the layer stack.
    fn check_slot(&mut self, layer: usize, scene: usize) -> Result<(), EditorError> {
        let layers = self.project.session.layers.len();
        let row = self
            .project
            .session
            .scenes
            .get_mut(scene)
            .ok_or(EditorError::BadCell { layer, scene })?;
        if layer >= layers {
            return Err(EditorError::BadCell { layer, scene });
        }
        if row.cells.len() < layers {
            row.cells.resize(layers, None);
        }
        Ok(())
    }

    /// A slot with a clip in it.
    fn check_cell(&self, layer: usize, scene: usize) -> Result<(), EditorError> {
        let exists = self
            .project
            .session
            .scenes
            .get(scene)
            .is_some_and(|row| row.cells.get(layer).is_some_and(Option::is_some));
        if exists {
            Ok(())
        } else {
            Err(EditorError::BadCell { layer, scene })
        }
    }
}

/// The view a project opens on: where its content is.
fn opening_view(project: &Project) -> View {
    if project.arrangement.lanes.iter().any(|lane| !lane.is_empty()) {
        View::Arrangement
    } else {
        View::Session
    }
}

/// Commands that change the project and are one step of history.
fn undoable(command: &Command) -> bool {
    !matches!(
        command,
        Command::SelectView(_)
            | Command::ToggleView
            | Command::SetBrowser(_)
            | Command::SetDetail(_)
            | Command::SetHelp(_)
            | Command::SelectClip(_)
            | Command::SelectScene(_)
            | Command::SelectSpan(_)
            | Command::SetMode(_)
            | Command::Launch { .. }
            | Command::LaunchScene(_)
            | Command::StopLayer(_)
            | Command::BackToArrangement
            | Command::Lock(_)
            | Command::OpenProject { .. }
            | Command::SaveProject { .. }
            | Command::Save
            | Command::NewProject
            | Command::Undo
            | Command::Redo
    )
}

/// The slider a command is one tick of. Consecutive commands with the same key are one undo step.
fn undo_key(command: &Command) -> Option<UndoKey> {
    Some(match command {
        Command::SetClipParam { clip, index, .. } => (1, *clip, *index as u32),
        Command::SetRange {
            clip, level, index, ..
        } => (2, *clip, u32::from(*level) * 8 + *index as u32),
        Command::SetPulseDepth { clip, level, .. } => (3, *clip, u32::from(*level)),
        Command::SetEffect { clip, index, .. } => (4, *clip, *index as u32),
        Command::SetTransition { clip, kind, .. } => (5, *clip, *kind as u32),
        Command::SetFollow { clip, action, .. } => (6, *clip, *action as u32),
        Command::SetLayerOpacity { layer, .. } => (7, *layer as u32, 0),
        Command::SetSeed { clip, .. } => (8, *clip, 0),
        Command::RenameClip { clip, .. } => (9, *clip, 0),
        Command::SetVideo { clip, .. } => (10, *clip, 0),
        Command::RenameScene { scene, .. } => (11, *scene as u32, 0),
        Command::RenameMarker { index, .. } => (12, *index as u32, 0),
        _ => return None,
    })
}

/// A visualizer id a clip may carry.
pub fn known_visualizer(visualizer: &str) -> bool {
    visualizer.starts_with("custom:") || catalog_entry(visualizer).is_some()
}

/// The media source a fresh clip of `visualizer` starts with.
fn default_media(visualizer: &str) -> Option<String> {
    match MediaKind::of(visualizer) {
        MediaKind::Camera => Some("camera:".to_string()),
        MediaKind::Text => Some("text:".to_string()),
        MediaKind::None | MediaKind::File => None,
    }
}

/// A fresh clip of `visualizer`: the look in the middle of its range, a light beat pulse, and the
/// web app's `newBlock` default datamosh (`mosh:.6, trans:2`).
pub fn new_clip(id: ClipId, visualizer: &str) -> Clip {
    let name = catalog_entry(visualizer)
        .map(|entry| entry.name.to_string())
        .unwrap_or_else(|| visualizer.to_string());
    // A scene is a datamosh from the moment it is dragged in, as the web app's `newBlock` is.
    // Media clips (a video file, the webcam, words) are left clean on purpose: a mosh smears them
    // into mush, so that stays a choice the user makes in the Effects list. The fluid gets a
    // moderate 0.5: it stirs the mosh buffer with its own velocity instead of being smeared by it.
    let (effects, transition_in) = if MediaKind::of(visualizer) == MediaKind::None {
        let amount = if visualizer == "fluid" { 0.5 } else { 0.6 };
        (
            vec![EffectSlot {
                kind: EffectKind::Mosh,
                amount,
                param: 0.0,
            }],
            Transition {
                kind: TransitionKind::Datamosh,
                beats: 2.0,
            },
        )
    } else {
        (Vec::new(), Transition::default())
    };
    Clip {
        id,
        name,
        visualizer: visualizer.to_string(),
        // p0..p3 are the look (speed, density, hue, variant): start in the middle. p4..p7 are
        // post amounts on the classic scenes: start with none.
        params: ClipParams {
            p: [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
            palette: 0,
            seed: 0,
        },
        ranges: [[ParamRange::default(); 8]; 4],
        pulse_depth: [0.25; 4],
        effects,
        transition_in,
        follow: FollowAction::None,
        follow_after_bars: 8,
        react_to_events: true,
        video: default_media(visualizer),
    }
}

/// The shortest block: a bar when snapping, a sixteenth when not.
fn min_len(bar: f64, snap: bool) -> f64 {
    if snap { bar } else { 0.25 }
}

/// Put `span` on `lane`. Whatever it covers gives way, as in Live: a block wholly under it is
/// removed, one it overlaps is trimmed, one it lands inside is split. Returns the block's index
/// in the lane, which stays sorted by start.
pub fn place_span(lane: &mut Vec<ArrClip>, span: ArrClip) -> usize {
    let mut out = Vec::with_capacity(lane.len() + 2);
    for other in lane.drain(..) {
        if other.end_ppq <= span.start_ppq + EPS || other.start_ppq >= span.end_ppq - EPS {
            out.push(other);
            continue;
        }
        if other.start_ppq < span.start_ppq - EPS {
            out.push(ArrClip {
                end_ppq: span.start_ppq,
                ..other
            });
        }
        if other.end_ppq > span.end_ppq + EPS {
            out.push(ArrClip {
                start_ppq: span.end_ppq,
                ..other
            });
        }
    }
    out.push(span);
    out.sort_by(|a, b| a.start_ppq.total_cmp(&b.start_ppq));
    let index = out
        .iter()
        .position(|other| *other == span)
        .unwrap_or_default();
    *lane = out;
    index
}

/// The parameter names of `visualizer` as `p0..p7`, or the generic slot names where the catalog
/// has none (a custom shader, a stub that has not named its parameters yet).
pub fn param_labels(visualizer: &str) -> [&'static str; 8] {
    const GENERIC: [&str; 8] = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"];
    let Some(entry) = catalog_entry(visualizer) else {
        return GENERIC;
    };
    let mut labels = entry.params;
    for (label, generic) in labels.iter_mut().zip(GENERIC) {
        if label.is_empty() {
            *label = generic;
        }
    }
    labels
}

/// Where `ppq` falls on a bar axis, counted through the signature map.
///
/// The arrangement's stored positions are in beats (`ppq`), which is what `mosh-als` imports and
/// what the show engine reads. The performers think in bars, and a set is in 4/4 until it is not,
/// so the axis and the marker list convert through the signature changes:
///
/// ```text
/// 4/4 for 64 beats -> bar 16; then 6/8 (3 beats a bar) -> 67 beats is bar 17
/// ```
pub fn bars_at(sig_map: &[SigChange], ppq: f64) -> f64 {
    let mut bars = 0.0;
    let mut from = 0.0;
    let mut len = 4.0;
    for change in sig_map {
        let at = change.ppq.max(0.0);
        if at <= from {
            len = bar_len(*change);
            continue;
        }
        if at >= ppq {
            break;
        }
        bars += (at - from) / len;
        from = at;
        len = bar_len(*change);
    }
    bars + (ppq.max(from) - from) / len
}

/// The inverse of [`bars_at`]: the position in beats of a point on the bar axis.
pub fn ppq_at_bar(sig_map: &[SigChange], bars: f64) -> f64 {
    let mut counted = 0.0;
    let mut from = 0.0;
    let mut len = 4.0;
    for change in sig_map {
        let at = change.ppq.max(0.0);
        if at <= from {
            len = bar_len(*change);
            continue;
        }
        let segment = (at - from) / len;
        if counted + segment >= bars {
            break;
        }
        counted += segment;
        from = at;
        len = bar_len(*change);
    }
    from + (bars - counted).max(0.0) * len
}

/// The bar line nearest to `ppq`.
pub fn snap_to_bar(sig_map: &[SigChange], ppq: f64) -> f64 {
    ppq_at_bar(sig_map, bars_at(sig_map, ppq.max(0.0)).round())
}

/// Beats in the bar at `ppq` (4 when the map says nothing).
pub fn bar_len_at(sig_map: &[SigChange], ppq: f64) -> f64 {
    let mut len = 4.0;
    for change in sig_map {
        if change.ppq <= ppq + 1e-9 {
            len = bar_len(*change);
        }
    }
    len
}

fn bar_len(change: SigChange) -> f64 {
    let len = f64::from(change.num) * 4.0 / f64::from(change.den.max(1));
    if len > 0.0 { len } else { 4.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bars_follow_the_signature_map() {
        assert_eq!(bars_at(&[], 0.0), 0.0);
        assert_eq!(bars_at(&[], 4.0), 1.0);
        assert_eq!(bars_at(&[], 6.0), 1.5);

        let map = [
            SigChange {
                ppq: 0.0,
                num: 4,
                den: 4,
            },
            SigChange {
                ppq: 16.0,
                num: 3,
                den: 4,
            },
        ];
        assert_eq!(bars_at(&map, 16.0), 4.0);
        assert_eq!(bars_at(&map, 19.0), 5.0);
        // Backwards of the first change: no bars were counted yet.
        assert_eq!(bars_at(&map, 8.0), 2.0);

        // And back: a bar line is a position in beats, through the same map.
        assert_eq!(ppq_at_bar(&[], 1.5), 6.0);
        assert_eq!(ppq_at_bar(&map, 4.0), 16.0);
        assert_eq!(ppq_at_bar(&map, 5.0), 19.0);
        assert_eq!(ppq_at_bar(&map, 2.0), 8.0);
        // Snapping takes the nearest bar line on either side of the signature change.
        assert_eq!(snap_to_bar(&map, 17.0), 16.0);
        assert_eq!(snap_to_bar(&map, 18.0), 19.0);
        assert_eq!(snap_to_bar(&[], 9.9), 8.0);
        assert_eq!(snap_to_bar(&[], 10.1), 12.0);
        assert_eq!(bar_len_at(&map, 20.0), 3.0);
        assert_eq!(bar_len_at(&map, 2.0), 4.0);
    }

    #[test]
    fn parameter_labels_come_from_the_catalog() {
        assert_eq!(param_labels("fluid")[0], "viscosity");
        assert_eq!(param_labels("tunnel")[0], "speed");
        assert_eq!(param_labels("camera")[0], "zoom");
        assert_eq!(param_labels("text")[1], "font");
        assert_eq!(
            param_labels("custom:mine"),
            ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]
        );
        // A catalog entry that has not named its parameters still gets a readable row.
        assert_eq!(param_labels("overlay")[0], "p0");
        assert!(known_visualizer("overlay") && known_visualizer("particles"));
    }

    #[test]
    fn a_placed_block_trims_splits_and_replaces_what_it_covers() {
        let span = |clip, start_ppq, end_ppq| ArrClip {
            clip,
            start_ppq,
            end_ppq,
        };
        // Inside one long block: it is split around the new one.
        let mut lane = vec![span(1, 0.0, 32.0)];
        assert_eq!(place_span(&mut lane, span(2, 8.0, 16.0)), 1);
        assert_eq!(lane, vec![span(1, 0.0, 8.0), span(2, 8.0, 16.0), span(1, 16.0, 32.0)]);
        // Over the join of two: both are trimmed.
        assert_eq!(place_span(&mut lane, span(3, 4.0, 12.0)), 1);
        assert_eq!(
            lane,
            vec![
                span(1, 0.0, 4.0),
                span(3, 4.0, 12.0),
                span(2, 12.0, 16.0),
                span(1, 16.0, 32.0)
            ]
        );
        // Over a whole block: it is gone. A block that only touches is left alone.
        assert_eq!(place_span(&mut lane, span(4, 12.0, 16.0)), 2);
        assert_eq!(lane[2], span(4, 12.0, 16.0));
        assert_eq!(lane.len(), 4);
    }

    #[test]
    fn a_marker_added_twice_at_one_position_renames_it() {
        let mut model = EditorModel::new(Project::default());
        model
            .apply(Command::AddMarker {
                name: "Intro".to_string(),
                ppq: 0.0,
            })
            .expect("first");
        model
            .apply(Command::AddMarker {
                name: "Start".to_string(),
                ppq: 0.0,
            })
            .expect("second");
        assert_eq!(model.project().arrangement.songs.len(), 1);
        assert_eq!(model.project().arrangement.songs[0].name, "Start");
        model
            .apply(Command::AddMarker {
                name: "Later".to_string(),
                ppq: 32.0,
            })
            .expect("later");
        // The list stays sorted, which the show engine's `song_at` relies on.
        assert_eq!(model.project().arrangement.songs[1].name, "Later");
        model.apply(Command::RemoveMarker { index: 0 }).expect("remove");
        assert_eq!(model.project().arrangement.songs.len(), 1);
        assert_eq!(model.project().arrangement.songs[0].start_ppq, 32.0);
    }
}
