//! The show engine: one `AnalysisFrame` plus controller input in, one `RenderPlan` out.
//!
//! The order of the frame is the behaviour:
//!
//! 1. transitions advance with the playhead, before anything can replace them,
//! 2. control input is handled: manual launches are queued, lock, back to arrangement, parameters,
//! 3. how long each clip has played is counted on bar lines,
//! 4. builds are counted; `BUILD_START` starts a hold and `DROP` (or eight bars) ends it,
//! 5. automation runs unless it is locked: timed follow actions, `LEVEL_UP`/`DROP` launches, and
//!    autopilot's idle start,
//! 6. launches that came due are applied, before the arrangement is read, so a manual launch on a
//!    bar line wins over the clip the arrangement would have shown,
//! 7. the arrangement is read for every layer no Session launch has taken over,
//! 8. the plan: parameters resolved for this frame's level and beat pulse, plus the flash request.

use mosh_core::{
    ActiveTransition, AnalysisFrame, Clip, ClipId, ClipParams, ControlEvent, EventFlags,
    FollowAction, LaunchQuantize, LayerState, Project, RenderPlan, ShowMode, TransitionKind,
};
use rand_pcg::Pcg64;
use rand_pcg::rand_core::{Rng, SeedableRng};

use crate::arrangement;
use crate::mapping;
use crate::procedural;
use crate::session;

/// ppq tolerance for "the playhead reached a boundary". Frame phases are `f32`, so a boundary
/// derived from one carries about 1e-7 of a beat; 1e-4 beats (50 us at 120 BPM) absorbs that and
/// still fires a launch well inside one frame.
const EPS: f64 = 1e-4;

/// How many bars a `BUILD_START` holds automation before releasing it anyway.
pub const BUILD_HOLD_BARS: u32 = 8;

/// What a queued launch will do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Cell { layer: usize, scene: usize },
    Scene(usize),
    Stop(usize),
}

impl Action {
    /// The layer this action owns, if it owns one; `None` for a scene launch.
    fn layer(self) -> Option<usize> {
        match self {
            Action::Cell { layer, .. } => Some(layer),
            Action::Stop(layer) => Some(layer),
            Action::Scene(_) => None,
        }
    }

    fn is_scene(self) -> bool {
        matches!(self, Action::Scene(_))
    }
}

/// A launch waiting for its quantize boundary.
#[derive(Clone, Copy, Debug)]
struct Pending {
    action: Action,
    /// ppq the launch is due at.
    target_ppq: f64,
    /// ppq the launch was requested at, so a jump backwards can be noticed.
    since_ppq: f64,
    quantize: LaunchQuantize,
    manual: bool,
}

/// An open build: automated launches wait for the drop, for at most `BUILD_HOLD_BARS` bars.
#[derive(Clone, Copy, Debug)]
struct BuildHold {
    bars: u32,
}

/// What one layer is doing.
#[derive(Clone, Debug)]
struct LayerRt {
    clip: Option<ClipId>,
    /// The scene the clip was launched from; `None` for an arrangement clip.
    scene: Option<usize>,
    /// Parameter positions 0..1 inside the current level's ranges.
    base: [f32; 8],
    /// The procedural seed behind `base` (`Clip.params.seed` for a fixed look).
    seed: u64,
    transition: Option<ActiveTransition>,
    /// Bar boundaries the clip has been playing.
    bars_playing: u32,
    /// A Session launch holds this layer against the arrangement.
    overridden: bool,
    manual_p: [Option<f32>; 8],
    opacity: Option<f32>,
}

impl Default for LayerRt {
    fn default() -> Self {
        LayerRt {
            clip: None,
            scene: None,
            base: [0.0; 8],
            seed: 0,
            transition: None,
            bars_playing: 0,
            overridden: false,
            manual_p: [None; 8],
            opacity: None,
        }
    }
}

/// Which boundaries this frame crossed.
#[derive(Clone, Copy, Debug, Default)]
struct Boundary {
    bar: bool,
}

impl Boundary {
    /// Bar lines come from the frame itself: the analysis' `BAR` flag, or the bar counter or phase
    /// moving, which keeps working when frames are dropped or flags are late. A frame that did not
    /// move forward (a stalled or repeated position) crosses nothing, so a repeated `BAR` flag
    /// cannot count the same bar twice.
    fn detect(prev: Option<&AnalysisFrame>, f: &AnalysisFrame) -> Boundary {
        let Some(prev) = prev else {
            return Boundary { bar: f.events.contains(EventFlags::BAR) };
        };
        if f.ppq <= prev.ppq + EPS {
            return Boundary::default();
        }
        Boundary {
            bar: f.events.contains(EventFlags::BAR)
                || f.bar_index != prev.bar_index
                || f.bar_phase < prev.bar_phase,
        }
    }
}

/// The show: the project plus everything it has decided so far.
pub struct Show {
    project: Project,
    mode: ShowMode,
    rng: Pcg64,
    layers: Vec<LayerRt>,
    pending: Vec<Pending>,
    last: Option<AnalysisFrame>,
    build: Option<BuildHold>,
    locked: bool,
    bias: f32,
    /// The song under the playhead (Arrangement) or picked by `SelectSong`.
    song: Option<usize>,
    /// The scene the grid is on, for autopilot's advance.
    scene: Option<usize>,
    /// A strobe transition asked for a full-frame flash this frame.
    flash: bool,
}

impl Show {
    /// A show from `p`, with the default procedural seed.
    pub fn new(p: Project) -> Show {
        Show::with_seed(p, procedural::DEFAULT_SEED)
    }

    /// A show whose procedural draws start from `seed`, so a set can be replayed exactly.
    pub fn with_seed(p: Project, seed: u64) -> Show {
        let layers = p.session.layers.len().max(p.arrangement.lanes.len());
        Show {
            mode: p.mode,
            project: p,
            rng: Pcg64::seed_from_u64(seed),
            layers: vec![LayerRt::default(); layers],
            pending: Vec::new(),
            last: None,
            build: None,
            locked: false,
            bias: 0.0,
            song: None,
            scene: None,
            flash: false,
        }
    }

    pub fn mode(&self) -> ShowMode {
        self.mode
    }

    /// Change mode. Leaving Arrangement drops every Session override, and a queued launch is
    /// dropped with the mode it was requested in.
    pub fn set_mode(&mut self, mode: ShowMode) {
        if mode == self.mode {
            return;
        }
        self.mode = mode;
        self.project.mode = mode;
        self.pending.clear();
        if mode == ShowMode::Arrangement {
            for layer in &mut self.layers {
                layer.overridden = false;
            }
        }
    }

    pub fn project(&self) -> &Project {
        &self.project
    }

    /// The song the show is in: the arrangement marker under the playhead in Arrangement mode,
    /// or the song last selected with `SelectSong`.
    pub fn song(&self) -> Option<usize> {
        self.song
    }

    pub fn locked(&self) -> bool {
        self.locked
    }

    /// One frame: what to draw, and whether a flash is wanted.
    pub fn update(&mut self, f: &AnalysisFrame, input: &[ControlEvent]) -> RenderPlan {
        let prev = self.last;
        let boundary = Boundary::detect(prev.as_ref(), f);
        let bar_len = arrangement::bar_len_at(&self.project.arrangement.sig_map, f.ppq);
        self.flash = false;

        // 1. Transitions advance with the playhead, before anything can replace them, so a
        //    transition created this frame starts at zero instead of jumping by the frame delta.
        self.advance_transitions(f, prev.as_ref());

        // 2. Control input next: a manual launch at this frame's bar line has to win over the
        //    automation and the arrangement that are about to run.
        for event in input {
            self.handle_control(*event, f, bar_len);
        }
        let level = self.effective_level(f);

        // 3. Per-bar bookkeeping: how long each clip has played.
        if boundary.bar {
            for layer in &mut self.layers {
                if layer.clip.is_some() {
                    layer.bars_playing = layer.bars_playing.saturating_add(1);
                }
            }
        }

        // 4. Builds: a build counts the bars that complete after it started, is released after
        //    `BUILD_HOLD_BARS` of them, and ends early on a drop. A drop ends a build before the
        //    automation below runs, so the change the build held lands on the drop's downbeat.
        if boundary.bar
            && let Some(build) = self.build.as_mut()
        {
            build.bars = build.bars.saturating_add(1);
            if build.bars >= BUILD_HOLD_BARS {
                self.build = None;
            }
        }
        if f.events.contains(EventFlags::BUILD_START) {
            self.build = Some(BuildHold { bars: 0 });
        }
        if f.events.contains(EventFlags::DROP) {
            self.build = None;
        }

        // 5. Automation, unless the human has locked it.
        if !self.locked {
            if boundary.bar {
                self.timed_follows(f, bar_len);
            }
            self.event_launches(f, bar_len);
            self.autopilot_start(f, bar_len);
        }

        // 6. Launches that came due, before the arrangement is read.
        self.apply_pending(f, bar_len, level);

        // 7. The arrangement, for the layers nobody has taken over.
        self.sync_arrangement(f, level);

        // 8. The plan.
        if self.mode == ShowMode::Arrangement {
            self.song = arrangement::song_at(&self.project.arrangement.songs, f.ppq);
        }
        let layers = (0..self.layers.len()).map(|i| self.layer_state(i, f, level)).collect();
        // A full-frame flash is wanted on a drop's downbeat, and when a strobe transition starts.
        // mosh-render's FlashLimiter decides whether the flash happens.
        let drop = f.events.contains(EventFlags::DROP) && f.events.contains(EventFlags::BAR);
        let flash_requested = drop || self.flash;
        self.last = Some(*f);
        RenderPlan { layers, flash_requested, events: f.events }
    }

    /// The level the ranges use: the analysis' level, or one from the biased intensity when the
    /// analysis has none.
    fn effective_level(&self, f: &AnalysisFrame) -> u8 {
        if f.level == 0 {
            mapping::level_from_intensity((f.intensity + self.bias).clamp(0.0, 1.0))
        } else {
            f.level.min(4)
        }
    }

    fn handle_control(&mut self, event: ControlEvent, f: &AnalysisFrame, bar_len: f64) {
        let quantize = self.project.session.quantize;
        match event {
            ControlEvent::LaunchClip { layer, scene } => {
                if layer < self.layers.len() && scene < self.project.session.scenes.len() {
                    self.queue(Action::Cell { layer, scene }, true, quantize, f, bar_len);
                }
            }
            ControlEvent::LaunchScene(scene) => {
                if scene < self.project.session.scenes.len() {
                    self.queue(Action::Scene(scene), true, quantize, f, bar_len);
                }
            }
            ControlEvent::StopLayer(layer) => {
                if layer < self.layers.len() {
                    self.queue(Action::Stop(layer), true, quantize, f, bar_len);
                }
            }
            ControlEvent::Lock(on) => self.locked = on,
            ControlEvent::BackToArrangement => {
                for layer in &mut self.layers {
                    layer.overridden = false;
                }
                // Automation that was queued for a layer which is no longer overridden is stale.
                self.pending.retain(|p| p.manual);
            }
            ControlEvent::SetParam { layer, index, value } => {
                if let Some(state) = self.layers.get_mut(layer)
                    && index < state.manual_p.len()
                {
                    state.manual_p[index] = Some(value);
                }
            }
            ControlEvent::SetOpacity { layer, value } => {
                if let Some(state) = self.layers.get_mut(layer) {
                    state.opacity = Some(value.clamp(0.0, 1.0));
                }
            }
            ControlEvent::IntensityBias(bias) => self.bias = bias.clamp(-1.0, 1.0),
            ControlEvent::SelectSong(index) => {
                self.song = Some(index);
                // Outside Arrangement mode the song is picked by launching its scene.
                if self.mode != ShowMode::Arrangement && index < self.project.session.scenes.len() {
                    self.queue(Action::Scene(index), true, quantize, f, bar_len);
                }
            }
        }
    }

    /// Queue a launch. A manual action replaces whatever was queued for its layer, and a manual
    /// scene launch replaces everything; automation never displaces a manual launch.
    fn queue(
        &mut self,
        action: Action,
        manual: bool,
        quantize: LaunchQuantize,
        f: &AnalysisFrame,
        bar_len: f64,
    ) {
        match action.layer() {
            Some(layer) => {
                self.pending
                    .retain(|p| p.action.layer() != Some(layer) || (!manual && p.manual));
                if !manual && self.pending.iter().any(|p| p.action.layer() == Some(layer)) {
                    return;
                }
            }
            None => {
                if manual {
                    self.pending.clear();
                } else {
                    if self.pending.iter().any(|p| p.action.is_scene() && !p.manual) {
                        return;
                    }
                    self.pending.retain(|p| !(p.action.is_scene() && !p.manual));
                }
            }
        }
        self.pending.push(Pending {
            action,
            target_ppq: session::next_boundary(quantize, f, bar_len),
            since_ppq: f.ppq,
            quantize,
            manual,
        });
    }

    /// Apply every queued launch that is due. A launch the playhead jumped behind is retargeted
    /// from the new position instead of waiting for a boundary that already went past.
    fn apply_pending(&mut self, f: &AnalysisFrame, bar_len: f64, level: u8) {
        let hold = self.build.is_some();
        let mut due: Vec<Pending> = Vec::new();
        let mut keep: Vec<Pending> = Vec::new();
        for mut pending in std::mem::take(&mut self.pending) {
            if f.ppq < pending.since_ppq - EPS {
                pending.since_ppq = f.ppq;
                pending.target_ppq = session::next_boundary(pending.quantize, f, bar_len);
            }
            let reached = f.ppq >= pending.target_ppq - EPS;
            let held = hold && !pending.manual;
            if reached && !held {
                due.push(pending);
            } else {
                keep.push(pending);
            }
        }
        self.pending = keep;
        for pending in due {
            match pending.action {
                Action::Cell { layer, scene } => self.launch_cell(layer, scene, pending.manual, level),
                Action::Scene(scene) => self.launch_scene(scene, pending.manual, level),
                Action::Stop(layer) => self.stop_layer(layer, pending.manual),
            }
        }
    }

    /// Launch the clip in a cell. An empty cell leaves the layer playing: Live's scene and clip
    /// slots behave the same way, and stopping is `StopLayer`'s job.
    fn launch_cell(&mut self, layer: usize, scene: usize, manual: bool, level: u8) {
        let Some(id) = session::cell(&self.project.session, layer, scene) else { return };
        self.set_clip(layer, id, Some(scene), manual, level);
    }

    fn launch_scene(&mut self, scene: usize, manual: bool, level: u8) {
        for layer in 0..self.layers.len() {
            self.launch_cell(layer, scene, manual, level);
        }
        self.scene = Some(scene);
    }

    /// Point a layer at a clip. Procedural clips draw a fresh seed here, so a manual launch of one
    /// gives a new look too, which is what makes them useful while building.
    fn set_clip(&mut self, layer: usize, id: ClipId, scene: Option<usize>, manual: bool, level: u8) {
        let Some(clip) = self.project.clips.iter().find(|c| c.id == id).cloned() else { return };
        let (seed, base) = if clip.follow == FollowAction::RandomProcedural {
            let seed = self.rng.next_u64();
            (seed, procedural::positions(&clip, level, seed))
        } else {
            (clip.params.seed, authored_positions(&clip))
        };
        let transition = match self.layers[layer].clip {
            Some(from) if cuts_into(&clip) => Some(ActiveTransition {
                kind: clip.transition_in.kind,
                progress: 0.0,
                from: Some(from),
            }),
            _ => None,
        };
        if transition.is_some_and(|t| t.kind == TransitionKind::Strobe) {
            self.flash = true;
        }

        let state = &mut self.layers[layer];
        state.clip = Some(id);
        state.scene = scene;
        state.base = base;
        state.seed = seed;
        state.transition = transition;
        state.bars_playing = 0;
        state.manual_p = [None; 8];
        state.overridden = state.overridden || (manual && self.mode == ShowMode::Arrangement);
        if scene.is_some() {
            self.scene = scene;
        }
    }

    fn stop_layer(&mut self, layer: usize, manual: bool) {
        let state = &mut self.layers[layer];
        state.clip = None;
        state.scene = None;
        state.base = [0.0; 8];
        state.seed = 0;
        state.transition = None;
        state.bars_playing = 0;
        state.manual_p = [None; 8];
        state.overridden = state.overridden || (manual && self.mode == ShowMode::Arrangement);
    }

    /// Can automation change this layer? Not in Arrangement mode unless a Session launch holds it.
    fn automation_allowed(&self, layer: usize) -> bool {
        self.mode != ShowMode::Arrangement || self.layers[layer].overridden
    }

    /// Is something already queued for this layer? Automation must not queue a second launch on
    /// top of a queued one, and must never displace a manual one.
    fn queued_for(&self, layer: usize) -> bool {
        self.pending.iter().any(|p| p.action.layer() == Some(layer))
    }

    fn clip_meta(&self, layer: usize) -> Option<(FollowAction, u16, bool)> {
        let id = self.layers[layer].clip?;
        self.project
            .clips
            .iter()
            .find(|c| c.id == id)
            .map(|c| (c.follow, c.follow_after_bars, c.react_to_events))
    }

    /// Follow actions due by bar count, on the bar line that completes them.
    fn timed_follows(&mut self, f: &AnalysisFrame, bar_len: f64) {
        if self.build.is_some() {
            return;
        }
        let quantize = self.project.session.quantize;
        for layer in 0..self.layers.len() {
            if self.queued_for(layer) || !self.automation_allowed(layer) {
                continue;
            }
            let Some((follow, after_bars, _)) = self.clip_meta(layer) else { continue };
            if follow == FollowAction::None {
                continue;
            }
            let interval = u32::from(after_bars.max(1));
            if self.layers[layer].bars_playing >= interval {
                self.schedule_follow(layer, follow, quantize, f, bar_len);
            }
        }
    }

    /// Launches driven by intensity: autopilot advances the scene, Session clips that asked for it
    /// fire their follow action. Both land on the next downbeat.
    fn event_launches(&mut self, f: &AnalysisFrame, bar_len: f64) {
        let events = f.events;
        if !(events.contains(EventFlags::LEVEL_UP) || events.contains(EventFlags::DROP)) {
            return;
        }
        // A build holds the change back so it lands on the drop.
        if self.build.is_some() {
            return;
        }
        let level_event = LaunchQuantize::Bar;
        match self.mode {
            ShowMode::Autopilot => {
                if let Some(scene) = self.scene_advance() {
                    self.queue(Action::Scene(scene), false, level_event, f, bar_len);
                }
            }
            ShowMode::Session | ShowMode::Arrangement => {
                for layer in 0..self.layers.len() {
                    if self.queued_for(layer) || !self.automation_allowed(layer) {
                        continue;
                    }
                    let Some((follow, _, react)) = self.clip_meta(layer) else { continue };
                    if !react || follow == FollowAction::None {
                        continue;
                    }
                    self.schedule_follow(layer, follow, level_event, f, bar_len);
                }
            }
        }
    }

    /// Autopilot with nothing playing starts the show by itself: hands-free from the first bar.
    fn autopilot_start(&mut self, f: &AnalysisFrame, bar_len: f64) {
        if self.mode != ShowMode::Autopilot || !self.pending.is_empty() {
            return;
        }
        if self.layers.iter().any(|l| l.clip.is_some()) {
            return;
        }
        if let Some(scene) = self.scene_advance() {
            self.queue(Action::Scene(scene), false, LaunchQuantize::Bar, f, bar_len);
        }
    }

    /// The next scene autopilot should play, and therefore the scene a level change advances to.
    fn scene_advance(&self) -> Option<usize> {
        match self.scene {
            Some(current) => session::next_scene_with_content(&self.project.session, current),
            None => session::first_scene_with_content(&self.project.session),
        }
    }

    fn schedule_follow(
        &mut self,
        layer: usize,
        follow: FollowAction,
        quantize: LaunchQuantize,
        f: &AnalysisFrame,
        bar_len: f64,
    ) {
        let Some(scene) = self.pick_follow_scene(layer, follow) else { return };
        self.queue(Action::Cell { layer, scene }, false, quantize, f, bar_len);
    }

    /// Where a follow action goes: another cell of the same column, never the clip that is playing
    /// while the column has a choice, so `Random` cannot look like a stuck loop.
    fn pick_follow_scene(&mut self, layer: usize, follow: FollowAction) -> Option<usize> {
        let grid = &self.project.session;
        match follow {
            FollowAction::None => None,
            FollowAction::Next => match self.layers[layer].scene {
                Some(current) => session::next_scene_in_column(grid, layer, current)
                    .or_else(|| session::first_scene_in_column(grid, layer)),
                None => session::first_scene_in_column(grid, layer),
            },
            FollowAction::Random | FollowAction::RandomProcedural => {
                let candidates = session::column(grid, layer);
                if candidates.is_empty() {
                    return None;
                }
                let playing = self.layers[layer].clip;
                let mut pool: Vec<usize> =
                    candidates.iter().filter(|(_, id)| Some(*id) != playing).map(|(s, _)| *s).collect();
                if pool.is_empty() {
                    pool = candidates.iter().map(|(s, _)| *s).collect();
                }
                let pick = (self.rng.next_u64() % pool.len() as u64) as usize;
                pool.get(pick).copied()
            }
        }
    }

    /// The arrangement: the clip under the playhead, every frame, with no memory of the last one.
    fn sync_arrangement(&mut self, f: &AnalysisFrame, level: u8) {
        if self.mode != ShowMode::Arrangement {
            return;
        }
        for layer in 0..self.layers.len() {
            if self.locked || self.layers[layer].overridden {
                continue;
            }
            let wanted = self
                .project
                .arrangement
                .lanes
                .get(layer)
                .and_then(|lane| arrangement::clip_at(lane, f.ppq));
            match wanted {
                Some(id) if self.layers[layer].clip != Some(id) => {
                    self.set_clip(layer, id, None, false, level)
                }
                Some(_) => {}
                None if self.layers[layer].clip.is_some() => self.stop_layer(layer, false),
                None => {}
            }
        }
    }

    /// A transition advances with the playhead in beats and ends when it is spent.
    fn advance_transitions(&mut self, f: &AnalysisFrame, prev: Option<&AnalysisFrame>) {
        let delta = match prev {
            Some(prev) => (f.ppq - prev.ppq).max(0.0),
            None => 0.0,
        };
        for layer in 0..self.layers.len() {
            if self.layers[layer].transition.is_none() {
                continue;
            }
            let beats = self.layers[layer]
                .clip
                .and_then(|id| self.project.clips.iter().find(|c| c.id == id))
                .map(|c| c.transition_in.beats)
                .unwrap_or(0.0);
            let done = if beats <= 0.0 {
                true
            } else {
                match self.layers[layer].transition.as_mut() {
                    Some(transition) => {
                        transition.progress = (transition.progress + (delta / f64::from(beats)) as f32)
                            .clamp(0.0, 1.0);
                        transition.progress >= 1.0
                    }
                    None => true,
                }
            };
            if done {
                self.layers[layer].transition = None;
            }
        }
    }

    fn layer_state(&self, layer: usize, f: &AnalysisFrame, level: u8) -> LayerState {
        let config = self.project.session.layers.get(layer).copied().unwrap_or_default();
        let state = &self.layers[layer];
        let opacity = state.opacity.unwrap_or(config.opacity);
        let Some(clip) = state.clip.and_then(|id| self.project.clips.iter().find(|c| c.id == id))
        else {
            return LayerState {
                layer,
                clip: None,
                visualizer: None,
                params: ClipParams::default(),
                effects: Vec::new(),
                transition: None,
                blend: config.blend,
                opacity,
                seed: state.seed,
            };
        };
        // The bias may push the pulse by up to half again, never below silence.
        let pulse = mapping::beat_pulse(f.beat_phase) * (1.0 + 0.5 * self.bias).max(0.0);
        let mut p = mapping::resolve_params(clip, level, pulse, &state.base);
        for (slot, manual) in p.iter_mut().zip(state.manual_p.iter()) {
            if let Some(value) = manual {
                *slot = *value;
            }
        }
        LayerState {
            layer,
            clip: Some(clip.id),
            visualizer: Some(clip.visualizer.clone()),
            params: ClipParams { p, palette: clip.params.palette, seed: state.seed },
            effects: clip.effects.clone(),
            transition: state.transition,
            blend: config.blend,
            opacity,
            seed: state.seed,
        }
    }
}

/// The authored parameter positions, as positions inside the level's range.
fn authored_positions(clip: &Clip) -> [f32; 8] {
    clip.params.p.map(|value| value.clamp(0.0, 1.0))
}

/// Does this clip's `transition_in` ask for something to cross from?
fn cuts_into(clip: &Clip) -> bool {
    clip.transition_in.kind != TransitionKind::Cut && clip.transition_in.beats > 0.0
}
