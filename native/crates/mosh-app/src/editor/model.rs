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
//! stage half-edited, and a good one never lands mid-phrase.

use mosh_core::{
    Clip, ClipId, ControlEvent, EffectKind, EffectSlot, FollowAction, LaunchQuantize, ParamRange,
    Preset, Project, ShowMode, SigChange, SongMarker, Transition, TransitionKind,
};
use mosh_shaders::VISUALIZERS;

use crate::core::Status;

/// The control window's screens, in the order the tabs show them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Screen {
    /// Layers x scenes: pick a clip, launch one for a look.
    #[default]
    Session,
    /// Lanes per layer in bars, with the song markers and the `.als` import.
    Arrangement,
    /// One clip: visualizer, parameters, ranges, effects, transition, follow action.
    Clip,
    /// Saved looks taken from a clip's parameters and seed.
    Presets,
    /// The project file and the show's mode.
    Project,
}

impl Screen {
    pub const ALL: [Screen; 5] = [
        Screen::Session,
        Screen::Arrangement,
        Screen::Clip,
        Screen::Presets,
        Screen::Project,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Screen::Session => "Session",
            Screen::Arrangement => "Arrangement",
            Screen::Clip => "Clip",
            Screen::Presets => "Presets",
            Screen::Project => "Project",
        }
    }
}

/// Everything the performer can ask the editor to do.
///
/// The UI layer builds these and applies nothing itself; the shell applies them and forwards the
/// outcome. Variants carry the values the widget collected, so a command is a complete
/// description of the edit (which is what makes replays and tests possible).
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    SelectScreen(Screen),
    SelectClip(Option<ClipId>),
    SelectScene(usize),

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
    /// The overlay video path; `None` or an empty path clears it.
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
    /// Read a Live set's locators and time signatures into the arrangement.
    ImportAls {
        path: String,
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

    OpenProject {
        path: String,
    },
    SaveProject {
        path: String,
    },
}

/// What applying a command did, beyond changing the project.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EditorOutcome {
    /// The project changed; the shell should hand it to the running show.
    pub project_changed: bool,
    /// A preview launch for the running show, now.
    pub launch: Option<ControlEvent>,
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
            EditorError::BadLevel(level) => write!(f, "intensity level {level} does not exist (1-4)"),
            EditorError::BadEffect(index) => write!(f, "no effect slot {}", index + 1),
            EditorError::DuplicateEffect(kind) => write!(f, "{kind:?} is already on this clip"),
            EditorError::BadCell { layer, scene } => {
                write!(f, "no clip cell at layer {layer}, scene {}", scene + 1)
            }
            EditorError::BadScene(scene) => write!(f, "no scene {}", scene + 1),
            EditorError::UnknownVisualizer(name) => write!(f, "no visualizer named {name}"),
            EditorError::Io(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for EditorError {}

/// The control window's model: the project, what the performer has selected, and the last line of
/// status. Everything the editors do goes through [`EditorModel::apply`].
pub struct EditorModel {
    project: Project,
    screen: Screen,
    selected: Option<ClipId>,
    selected_scene: Option<usize>,
    status: Status,
    message: Option<String>,
    /// The project has changed since the last [`EditorModel::handoff`].
    dirty: bool,
    edits: u64,
}

impl EditorModel {
    /// An editor over `project`, with the first clip selected (so the clip screen is never blank
    /// on a project that has clips) and the Session screen showing.
    pub fn new(project: Project) -> EditorModel {
        let selected = project.clips.first().map(|clip| clip.id);
        EditorModel {
            project,
            screen: Screen::default(),
            selected,
            selected_scene: None,
            status: Status::default(),
            message: None,
            dirty: false,
            edits: 0,
        }
    }

    pub fn project(&self) -> &Project {
        &self.project
    }

    pub fn screen(&self) -> Screen {
        self.screen
    }

    pub fn selected(&self) -> Option<ClipId> {
        self.selected
    }

    /// The clip the Clip, Presets and effect rows act on.
    pub fn selected_clip(&self) -> Option<&Clip> {
        let id = self.selected?;
        self.project.clips.iter().find(|clip| clip.id == id)
    }

    /// The session cell last clicked, when the grid has a selection.
    pub fn selected_scene(&self) -> Option<usize> {
        self.selected_scene
    }

    pub fn status(&self) -> &Status {
        &self.status
    }

    /// The last error or confirmation, for the status strip.
    pub fn message(&self) -> Option<&str> {
        self.message.as_deref()
    }

    /// Commands applied since the model was built.
    pub fn edits(&self) -> u64 {
        self.edits
    }

    /// Whether the project has changed since the last handoff.
    pub fn dirty(&self) -> bool {
        self.dirty
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
        match self.apply_command(command) {
            Ok(outcome) => {
                if outcome.project_changed {
                    self.dirty = true;
                    self.edits += 1;
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
            Command::SelectScreen(screen) => {
                self.screen = screen;
                Ok(EditorOutcome::default())
            }
            Command::SelectClip(clip) => {
                if let Some(id) = clip
                    && !self.project.clips.iter().any(|clip| clip.id == id)
                {
                    return Err(EditorError::UnknownClip(id));
                }
                self.selected = clip;
                Ok(EditorOutcome::default())
            }
            Command::SelectScene(scene) => {
                self.selected_scene = Some(scene);
                Ok(EditorOutcome::default())
            }

            Command::SetClipParam { clip, index, value } => {
                let slot = self.clip_mut(clip)?.params.p.get_mut(index);
                let slot = slot.ok_or(EditorError::BadParam(index))?;
                *slot = value.clamp(0.0, 1.0);
                Ok(EditorOutcome::changed())
            }
            Command::SetVisualizer { clip, visualizer } => {
                // A visualizer the renderer does not know would draw nothing at all, which on
                // stage looks like a dead layer: refuse it here instead.
                let known = visualizer.starts_with("custom:")
                    || VISUALIZERS.iter().any(|info| info.id == visualizer);
                if !known {
                    return Err(EditorError::UnknownVisualizer(visualizer));
                }
                self.clip_mut(clip)?.visualizer = visualizer;
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
                    param: 0.5,
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
                let path = path
                    .map(|path| path.trim().to_string())
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
                    launch: None,
                    set_mode: Some(mode),
                })
            }
            Command::SetQuantize(quantize) => {
                self.project.session.quantize = quantize;
                Ok(EditorOutcome::changed())
            }
            Command::Launch { layer, scene } => {
                self.check_cell(layer, scene)?;
                Ok(EditorOutcome {
                    launch: Some(ControlEvent::LaunchClip { layer, scene }),
                    ..EditorOutcome::default()
                })
            }
            Command::LaunchScene(scene) => {
                if scene >= self.project.session.scenes.len() {
                    return Err(EditorError::BadScene(scene));
                }
                Ok(EditorOutcome {
                    launch: Some(ControlEvent::LaunchScene(scene)),
                    ..EditorOutcome::default()
                })
            }

            Command::OpenProject { path } => {
                let project = mosh_show::io::load(&path)
                    .map_err(|error| EditorError::Io(error.to_string()))?;
                self.message = Some(format!(
                    "opened {path}: {} clips, {} scenes",
                    project.clips.len(),
                    project.session.scenes.len()
                ));
                self.selected = project.clips.first().map(|clip| clip.id);
                self.selected_scene = None;
                self.project = project;
                Ok(EditorOutcome::changed())
            }
            Command::SaveProject { path } => {
                mosh_show::io::save(&self.project, &path)
                    .map_err(|error| EditorError::Io(error.to_string()))?;
                self.message = Some(format!("saved {path}"));
                Ok(EditorOutcome::default())
            }
        }
    }

    fn clip_mut(&mut self, id: ClipId) -> Result<&mut Clip, EditorError> {
        self.project
            .clips
            .iter_mut()
            .find(|clip| clip.id == id)
            .ok_or(EditorError::UnknownClip(id))
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

/// The parameter names of `visualizer` as `p0..p7`, or the generic slot names for an unknown one.
pub fn param_labels(visualizer: &str) -> [&'static str; 8] {
    match VISUALIZERS.iter().find(|info| info.id == visualizer) {
        Some(info) => info.params,
        None => ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"],
    }
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

fn bar_len(change: SigChange) -> f64 {
    f64::from(change.num) * 4.0 / f64::from(change.den.max(1))
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
    }

    #[test]
    fn parameter_labels_come_from_the_shader_crate() {
        assert_eq!(param_labels("fluid")[0], "viscosity");
        assert_eq!(param_labels("tunnel")[0], "speed");
        assert_eq!(param_labels("custom:mine"), ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]);
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
