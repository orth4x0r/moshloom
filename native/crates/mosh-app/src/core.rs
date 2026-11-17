//! `AppCore`: one video frame of the live show, minus the pixels.
//!
//! The window shell (`shell.rs`) owns the render thread; this module owns everything that decides
//! *what* is drawn, so all of it is testable with fake inputs and no window:
//!
//! ```text
//! audio source  -> InputMux -> Analyzer (one push per block) -> AnalysisFrame
//! control input -> Controls -> ControlEvent                    \
//! song markers  -> Analyzer::song_start                         > Show::update -> RenderPlan
//! ```
//!
//! Inputs (`Review Focus` items 1 and 4):
//!
//! - The plugin writer is optional and may appear or disappear at any moment, including mid-set.
//!   With no writer the app runs on the WASAPI loopback fallback; a writer that has been quiet for
//!   [`PLUGIN_GRACE`] hands over to the loopback, and the first fresh heartbeat hands back.
//! - Crossing a song marker restarts the intensity history and puts `SONG_START` on that frame's
//!   events, so a quiet intro is judged against itself and not the previous song's drop.

use std::time::{Duration, Instant};

use mosh_analysis::{AnalysisConfig, Analyzer};
use mosh_capture::Loopback;
use mosh_core::{
    ArrClip, AudioBlock, BlendMode, Clip, ClipParams, ControlEvent, FollowAction, LaunchQuantize,
    Layer, MidiEvent, ParamRange, Project, RenderPlan, SceneRow, SessionGrid, ShowMode,
    SongMarker, Transition,
};
use mosh_ipc::IpcReader;
use mosh_show::Show;

use crate::monitor::{MonitorInfo, StagePlacement};

/// How long the plugin writer may be silent before the loopback takes over.
///
/// The plugin stops writing when Live is closed, when the plugin is removed, or when Live's audio
/// engine is stopped; all three mean "no more transport", and two seconds is long enough not to
/// flap on a stalled audio callback and short enough that a set does not sit on stale audio.
pub const PLUGIN_GRACE: Duration = Duration::from_secs(2);

/// How often a missing plugin writer or controller is looked for again.
pub const REOPEN_INTERVAL: Duration = Duration::from_secs(5);

/// A writer that has been dead this long is let go, so the plugin can be removed and loaded again
/// mid-set (a fresh writer may publish a fresh region and the old mapping would never see it).
pub const PLUGIN_REOPEN_AFTER: Duration = Duration::from_secs(30);

/// Which input the app is analysing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SourceKind {
    /// The tap plugin's shared memory, with Ableton's transport.
    Plugin,
    /// WASAPI loopback of the render endpoint, with no transport.
    Loopback,
    /// No audio at all (neither a writer nor a capture device): the show keeps drawing.
    #[default]
    Silent,
}

impl SourceKind {
    pub fn label(self) -> &'static str {
        match self {
            SourceKind::Plugin => "plugin",
            SourceKind::Loopback => "loopback",
            SourceKind::Silent => "silent",
        }
    }
}

/// One audio input.
///
/// `poll` must never block: the render thread calls it once per frame, and the writer / capture
/// thread owns the other side of the ring.
pub trait AudioSource: Send {
    /// Append the blocks that arrived since the last poll. Returns whether this source is alive:
    /// a fresh plugin heartbeat, or a capture thread that is still running.
    fn poll(&mut self, out: &mut Vec<AudioBlock>) -> bool;
    /// A short name for the control window.
    fn label(&self) -> &'static str;
}

/// One control input: the Moshloom Control plugin's MIDI ring, or an APC mini.
pub trait ControlSource: Send {
    /// Append the events that arrived since the last poll. Never blocks.
    fn poll(&mut self, out: &mut Vec<ControlEvent>);
    fn label(&self) -> &'static str;
}

/// The tap plugin's audio ring, with Ableton's transport.
pub struct PluginInput {
    reader: IpcReader,
    midi: Vec<MidiEvent>,
}

impl PluginInput {
    /// Attach to the master tap. Fails while no plugin instance is running, which is the normal
    /// state before Live starts (Review Focus item 1).
    pub fn open() -> Option<PluginInput> {
        IpcReader::open(mosh_core::ShowRole::Master)
            .ok()
            .map(|reader| PluginInput {
                reader,
                midi: Vec::new(),
            })
    }

    pub fn sample_rate(&self) -> u32 {
        self.reader.sample_rate()
    }
}

impl AudioSource for PluginInput {
    fn poll(&mut self, out: &mut Vec<AudioBlock>) -> bool {
        // The master tap carries no control MIDI; the Control instance has its own ring.
        self.midi.clear();
        let stats = self.reader.poll(out, &mut self.midi);
        stats.writer_alive
    }

    fn label(&self) -> &'static str {
        "plugin"
    }
}

/// WASAPI loopback of the default render endpoint: the fallback audio input.
pub struct LoopbackInput {
    capture: Loopback,
}

impl LoopbackInput {
    pub fn open() -> Option<LoopbackInput> {
        Loopback::start(None).ok().map(|capture| LoopbackInput { capture })
    }
}

impl AudioSource for LoopbackInput {
    fn poll(&mut self, out: &mut Vec<AudioBlock>) -> bool {
        self.capture.poll(out);
        self.capture.is_running()
    }

    fn label(&self) -> &'static str {
        "loopback"
    }
}

/// The Moshloom Control plugin's MIDI ring: notes and CCs the Live set sends.
pub struct ControlPlugin {
    reader: IpcReader,
    blocks: Vec<AudioBlock>,
    midi: Vec<MidiEvent>,
}

impl ControlPlugin {
    pub fn open() -> Option<ControlPlugin> {
        IpcReader::open(mosh_core::ShowRole::Control)
            .ok()
            .map(|reader| ControlPlugin {
                reader,
                blocks: Vec::new(),
                midi: Vec::new(),
            })
    }
}

impl ControlSource for ControlPlugin {
    fn poll(&mut self, out: &mut Vec<ControlEvent>) {
        self.blocks.clear();
        self.midi.clear();
        // The Control instance writes no audio; the blocks are drained so its ring cannot fill.
        self.reader.poll(&mut self.blocks, &mut self.midi);
        for event in &self.midi {
            if let Some(control) = mosh_midi::decode_control_note(event) {
                out.push(control);
            }
        }
    }

    fn label(&self) -> &'static str {
        "control"
    }
}

/// An APC mini, for building and rehearsing a show.
pub struct ApcInput {
    controller: mosh_midi::Controller,
}

impl ApcInput {
    pub fn open() -> Option<ApcInput> {
        mosh_midi::Controller::open_apc_mini()
            .ok()
            .map(|controller| ApcInput {
                controller,
            })
    }

    /// Light a pad, if a controller with an output port is attached.
    pub fn set_pad(&mut self, x: u8, y: u8, colour: mosh_midi::PadColor) {
        self.controller.set_pad(x, y, colour);
    }
}

impl ControlSource for ApcInput {
    fn poll(&mut self, out: &mut Vec<ControlEvent>) {
        out.extend(self.controller.poll());
    }

    fn label(&self) -> &'static str {
        "apc"
    }
}

/// The audio inputs, with the loopback fallback policy.
pub struct InputMux {
    plugin: Option<Box<dyn AudioSource>>,
    loopback: Option<Box<dyn AudioSource>>,
    plugin_scratch: Vec<AudioBlock>,
    loopback_scratch: Vec<AudioBlock>,
    active: SourceKind,
    plugin_dead_since: Option<Instant>,
    grace: Duration,
    /// The shell's mux opens its sources itself; a test's mux is handed them.
    auto_open: bool,
    plugin_retry_at: Option<Instant>,
    loopback_retry_at: Option<Instant>,
    fallbacks: u32,
    switches: u32,
}

impl InputMux {
    /// A mux over sources the caller owns (tests, and the shell's explicit fallbacks).
    pub fn new(plugin: Option<Box<dyn AudioSource>>, loopback: Option<Box<dyn AudioSource>>) -> Self {
        InputMux {
            plugin,
            loopback,
            plugin_scratch: Vec::new(),
            loopback_scratch: Vec::new(),
            active: SourceKind::Silent,
            plugin_dead_since: None,
            grace: PLUGIN_GRACE,
            auto_open: false,
            plugin_retry_at: None,
            loopback_retry_at: None,
            fallbacks: 0,
            switches: 0,
        }
    }

    /// The app's mux: attaches to the plugin when it can and opens loopback capture when the
    /// plugin is absent or quiet for [`PLUGIN_GRACE`].
    pub fn open() -> Self {
        let mut mux = InputMux::new(None, None);
        mux.auto_open = true;
        mux
    }

    /// Change the grace period (tests use a short one; the app uses [`PLUGIN_GRACE`]).
    pub fn with_grace(mut self, grace: Duration) -> Self {
        self.grace = grace;
        self
    }

    pub fn active(&self) -> SourceKind {
        self.active
    }

    /// Times the active source changed: a writer appearing, dying, or coming back.
    pub fn switches(&self) -> u32 {
        self.switches
    }

    /// Times the app fell back to loopback capture.
    pub fn fallbacks(&self) -> u32 {
        self.fallbacks
    }

    /// Which source the policy wants, given the plugin's liveness at `now`.
    ///
    /// Pure apart from the switch counters: this is the state machine
    /// `switches_to_loopback_and_back` drives directly.
    pub fn decide(&mut self, plugin_alive: bool, now: Instant) -> SourceKind {
        let next = if self.plugin.is_some() && plugin_alive {
            self.plugin_dead_since = None;
            SourceKind::Plugin
        } else if self.plugin.is_some() {
            let since = *self.plugin_dead_since.get_or_insert(now);
            if now.saturating_duration_since(since) >= self.grace {
                if self.loopback.is_some() {
                    SourceKind::Loopback
                } else {
                    SourceKind::Silent
                }
            } else {
                // Inside the grace period the plugin stays selected: a single late heartbeat is
                // not a reason to switch inputs mid-phrase.
                SourceKind::Plugin
            }
        } else if self.loopback.is_some() {
            SourceKind::Loopback
        } else {
            SourceKind::Silent
        };

        if next != self.active {
            self.switches += 1;
            if next == SourceKind::Loopback {
                self.fallbacks += 1;
            }
        }
        self.active = next;
        next
    }

    /// Poll every source and append the blocks of the active one to `out`.
    pub fn poll(&mut self, now: Instant, out: &mut Vec<AudioBlock>) -> SourceKind {
        if self.plugin.is_none() && self.auto_open && retry_due(&mut self.plugin_retry_at, now) {
            self.plugin = PluginInput::open().map(|input| Box::new(input) as Box<dyn AudioSource>);
        }

        let plugin_alive = match self.plugin.as_mut() {
            Some(plugin) => {
                self.plugin_scratch.clear();
                plugin.poll(&mut self.plugin_scratch)
            }
            None => false,
        };

        let kind = self.decide(plugin_alive, now);

        // A writer that has been gone for a while is released: if the plugin was removed and
        // loaded again, a fresh reader is the only way back to its transport.
        if self.auto_open
            && !plugin_alive
            && let Some(since) = self.plugin_dead_since
            && now.saturating_duration_since(since) >= PLUGIN_REOPEN_AFTER
        {
            self.plugin = None;
            self.plugin_dead_since = None;
            self.plugin_scratch.clear();
        }

        // The fallback is opened on demand: the app must not touch the output device while the
        // plugin is working, and opening WASAPI takes a moment.
        if self.loopback.is_none()
            && self.auto_open
            && kind != SourceKind::Plugin
            && retry_due(&mut self.loopback_retry_at, now)
        {
            self.loopback =
                LoopbackInput::open().map(|input| Box::new(input) as Box<dyn AudioSource>);
        }
        let loopback_alive = match self.loopback.as_mut() {
            Some(capture) => {
                self.loopback_scratch.clear();
                capture.poll(&mut self.loopback_scratch)
            }
            None => false,
        };
        if kind == SourceKind::Loopback && !loopback_alive {
            // The capture thread died (endpoint removed, device reset): drop the handle so the
            // next fallback opens a fresh one instead of polling a corpse every frame.
            self.loopback = None;
            self.loopback_scratch.clear();
        }

        match kind {
            SourceKind::Plugin => out.append(&mut self.plugin_scratch),
            SourceKind::Loopback => {
                // The plugin's stale blocks are not audio that is still playing.
                self.plugin_scratch.clear();
                out.append(&mut self.loopback_scratch)
            }
            SourceKind::Silent => {
                self.plugin_scratch.clear();
                self.loopback_scratch.clear();
            }
        }
        kind
    }
}

/// `true` when a deferred open may be tried again at `now`, and records the attempt.
fn retry_due(stamp: &mut Option<Instant>, now: Instant) -> bool {
    match *stamp {
        Some(at) if now.saturating_duration_since(at) < REOPEN_INTERVAL => false,
        _ => {
            *stamp = Some(now);
            true
        }
    }
}

/// The control inputs: the Control plugin's MIDI ring and an APC mini, both optional.
pub struct Controls {
    sources: Vec<Box<dyn ControlSource>>,
    auto_open: bool,
    retry_at: Option<Instant>,
}

impl Controls {
    /// Controls over sources the caller owns (tests).
    pub fn new(sources: Vec<Box<dyn ControlSource>>) -> Self {
        Controls {
            sources,
            auto_open: false,
            retry_at: None,
        }
    }

    /// The app's controls: both inputs are looked for now and again every
    /// [`REOPEN_INTERVAL`] if one is missing.
    pub fn open() -> Self {
        let mut controls = Controls::new(Vec::new());
        controls.auto_open = true;
        controls.try_open_controls();
        controls
    }

    /// Append every control event that arrived since the last call.
    pub fn poll(&mut self, out: &mut Vec<ControlEvent>, now: Instant) {
        if self.auto_open
            && retry_due(&mut self.retry_at, now)
            && (self.sources.len() < 2 || self.missing("apc") || self.missing("control"))
        {
            self.try_open_controls();
        }
        for source in &mut self.sources {
            source.poll(out);
        }
    }

    fn missing(&self, label: &str) -> bool {
        !self.sources.iter().any(|source| source.label() == label)
    }

    fn try_open_controls(&mut self) {
        if self.missing("control")
            && let Some(input) = ControlPlugin::open()
        {
            self.sources.push(Box::new(input));
        }
        if self.missing("apc")
            && let Some(input) = ApcInput::open()
        {
            self.sources.push(Box::new(input));
        }
    }
}

/// Everything one video frame produced.
#[derive(Clone, Debug)]
pub struct Frame {
    pub analysis: mosh_core::AnalysisFrame,
    pub plan: RenderPlan,
    pub source: SourceKind,
    pub mode: ShowMode,
    /// The arrangement song the playhead is in, once a marker has been crossed.
    pub song: Option<usize>,
}

/// The show without its pixels: audio in, one plan out.
pub struct AppCore {
    analyzer: Analyzer,
    show: Show,
    inputs: InputMux,
    controls: Controls,
    blocks: Vec<AudioBlock>,
    events: Vec<ControlEvent>,
    song: Option<usize>,
    song_starts: u64,
    frames: u64,
}

impl AppCore {
    pub fn new(project: Project, inputs: InputMux, controls: Controls) -> Self {
        AppCore {
            analyzer: Analyzer::new(AnalysisConfig::default()),
            show: Show::new(project),
            inputs,
            controls,
            blocks: Vec::new(),
            events: Vec::new(),
            song: None,
            song_starts: 0,
            frames: 0,
        }
    }

    /// The app's core: real inputs, best effort, no failure is fatal.
    pub fn open(project: Project) -> Self {
        AppCore::new(project, InputMux::open(), Controls::open())
    }

    pub fn show(&self) -> &Show {
        &self.show
    }

    pub fn show_mut(&mut self) -> &mut Show {
        &mut self.show
    }

    /// Song markers crossed since the app started.
    pub fn song_starts(&self) -> u64 {
        self.song_starts
    }

    /// Frames produced since the app started.
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// One video frame: poll the inputs, analyse the audio, let the show decide.
    ///
    /// Never blocks and never fails: a missing input, a stalled writer or a silent device yields
    /// a frame at the last tempo, which is what keeps the stage alive mid-set.
    pub fn step(&mut self, now: Instant) -> Frame {
        self.blocks.clear();
        let source = self.inputs.poll(now, &mut self.blocks);

        self.events.clear();
        self.controls.poll(&mut self.events, now);

        for index in 0..self.blocks.len() {
            if let Some(transport) = self.blocks[index].transport {
                self.note_playhead(transport.ppq);
            }
            self.analyzer.push(&self.blocks[index]);
        }
        self.blocks.clear();

        let analysis = self.analyzer.take_frame();
        let plan = self.show.update(&analysis, &self.events);
        self.frames += 1;
        Frame {
            analysis,
            plan,
            source,
            mode: self.show.mode(),
            song: self.song,
        }
    }

    /// A transport block moved the playhead: crossing into another song restarts the intensity
    /// history and flags `SONG_START` on this frame.
    fn note_playhead(&mut self, ppq: f64) {
        let index = mosh_show::arrangement::song_at(&self.show.project().arrangement.songs, ppq);
        if index == self.song {
            return;
        }
        self.song = index;
        if index.is_some() {
            self.analyzer.song_start();
            self.song_starts += 1;
        }
    }
}

/// The project the app runs with when it is started without one: an empty show in Autopilot over
/// a 16-scene grid of the 16 ported scenes, so pressing play on nothing still draws something.
pub fn default_project() -> Project {
    // `VISUALIZERS` order, without `fluid` (a visualizer of its own, not a scene).
    const SCENES: [&str; 16] = [
        "tunnel",
        "loom",
        "numogram",
        "flesh",
        "rotor",
        "moire",
        "spectrum",
        "cells",
        "punch_cards",
        "vortex",
        "kaleido",
        "reaction",
        "grid",
        "glyph_rain",
        "scan",
        "swarm",
    ];

    let layers = vec![
        Layer::default(),
        Layer {
            kind: mosh_core::LayerKind::Visualizer,
            blend: BlendMode::Add,
            opacity: 0.6,
            internal_scale: 0.5,
        },
    ];

    let mut clips = Vec::with_capacity(SCENES.len() * layers.len());
    let mut scenes = Vec::with_capacity(SCENES.len());
    for (row, _scene) in SCENES.iter().enumerate() {
        let mut cells = Vec::with_capacity(layers.len());
        for layer in 0..layers.len() {
            let id = (row * layers.len() + layer) as u32 + 1;
            let visualizer = SCENES[(row + layer * 5) % SCENES.len()];
            clips.push(Clip {
                id,
                name: format!("{visualizer} {row}"),
                visualizer: visualizer.to_string(),
                params: ClipParams::default(),
                ranges: [[ParamRange::default(); 8]; 4],
                pulse_depth: [0.25; 4],
                effects: Vec::new(),
                transition_in: Transition::default(),
                follow: FollowAction::None,
                follow_after_bars: 8,
                react_to_events: true,
                video: None,
            });
            cells.push(Some(id));
        }
        scenes.push(SceneRow {
            name: format!("scene {}", row + 1),
            cells,
        });
    }

    Project {
        mode: ShowMode::Autopilot,
        clips,
        session: SessionGrid {
            layers,
            scenes,
            quantize: LaunchQuantize::Bar,
        },
        ..Project::default()
    }
}

/// A clip with nothing set but its visualizer, for tests and defaults.
pub fn simple_clip(id: u32, visualizer: &str) -> Clip {
    Clip {
        id,
        name: format!("clip {id}"),
        visualizer: visualizer.to_string(),
        params: ClipParams::default(),
        ranges: [[ParamRange::default(); 8]; 4],
        pulse_depth: [0.0; 4],
        effects: Vec::new(),
        transition_in: Transition::default(),
        follow: FollowAction::None,
        follow_after_bars: 4,
        react_to_events: false,
        video: None,
    }
}

/// An arrangement with one lane and one clip per span, for tests.
pub fn arrangement_project(spans: &[(u32, f64, f64)], songs: &[(&str, f64)]) -> Project {
    Project {
        mode: ShowMode::Arrangement,
        clips: spans.iter().map(|(id, _, _)| simple_clip(*id, "tunnel")).collect(),
        arrangement: mosh_core::Arrangement {
            songs: songs
                .iter()
                .map(|(name, start_ppq)| SongMarker {
                    name: (*name).to_string(),
                    start_ppq: *start_ppq,
                })
                .collect(),
            lanes: vec![
                spans
                    .iter()
                    .map(|(id, start_ppq, end_ppq)| ArrClip {
                        clip: *id,
                        start_ppq: *start_ppq,
                        end_ppq: *end_ppq,
                    })
                    .collect(),
            ],
            sig_map: Vec::new(),
        },
        session: SessionGrid {
            layers: vec![Layer::default()],
            scenes: Vec::new(),
            quantize: LaunchQuantize::Bar,
        },
        ..Project::default()
    }
}

/// What the control window's title says, given one frame's state.
///
/// Pure, so the title's content (and the "redraw only on change" rule) is testable without a
/// window. `None` means the title is unchanged and must not be set again.
pub fn title_for(previous: Option<&str>, status: &Status) -> Option<String> {
    let next = format!(
        "ilanaVisual · {} · {} · {:.1} BPM · bar {} ({:.0}%) · L{} · GPU {:.2} ms · CPU {:.2} ms",
        status.mode,
        status.source,
        status.tempo_bpm,
        status.bar + 1,
        status.bar_phase * 100.0,
        status.level,
        status.gpu_ms,
        status.cpu_ms,
    );
    if previous == Some(next.as_str()) {
        None
    } else {
        Some(next)
    }
}

/// The numbers the control window shows.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Status {
    pub mode: ModeName,
    pub source: SourceName,
    pub tempo_bpm: f32,
    pub bar: i64,
    pub bar_phase: f32,
    pub level: u8,
    pub gpu_ms: f32,
    pub cpu_ms: f32,
}

/// `ShowMode` as the title spells it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModeName {
    #[default]
    Arrangement,
    Session,
    Autopilot,
}

impl From<ShowMode> for ModeName {
    fn from(mode: ShowMode) -> Self {
        match mode {
            ShowMode::Arrangement => ModeName::Arrangement,
            ShowMode::Session => ModeName::Session,
            ShowMode::Autopilot => ModeName::Autopilot,
        }
    }
}

impl std::fmt::Display for ModeName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ModeName::Arrangement => "arrangement",
            ModeName::Session => "session",
            ModeName::Autopilot => "autopilot",
        })
    }
}

/// `SourceKind` as the title spells it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SourceName(pub SourceKind);

impl std::fmt::Display for SourceName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0.label())
    }
}

/// The stage placement of a monitor list, as the shell reports it in the startup log.
pub fn stage_placement_label(placement: StagePlacement, monitors: &[MonitorInfo]) -> String {
    let name_of = |index: u32, fallback: &str| {
        monitors
            .iter()
            .find(|monitor| monitor.index == index)
            .map(|monitor| monitor.name.clone())
            .unwrap_or_else(|| fallback.to_string())
    };
    match placement {
        StagePlacement::Fullscreen(index) => {
            format!("stage on {} (fullscreen)", name_of(index, "monitor"))
        }
        StagePlacement::Laptop(index) => {
            format!("stage on {} (borderless)", name_of(index, "laptop screen"))
        }
        StagePlacement::Hidden => "stage hidden (no monitor)".to_string(),
    }
}
