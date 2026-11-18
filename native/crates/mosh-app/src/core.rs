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
    ArrClip, AudioBlock, BlendMode, Clip, ClipId, ClipParams, ControlEvent, EventFlags,
    FollowAction, LaunchQuantize, Layer, MidiEvent, ParamRange, Project, RenderPlan, SceneRow,
    SessionGrid, ShowMode, ShowRole, SongMarker, Transition,
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
    /// Light a pad at `(x, y)`, `y` from the top row. Only a pad controller does anything here.
    fn set_pad(&mut self, _x: u8, _y: u8, _colour: mosh_midi::PadColor) {}
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
        PluginInput::open_role(mosh_core::ShowRole::Master)
    }

    /// Attach to a Tap instance set to `role` (a drum or bass stem). Fails while none is running.
    pub fn open_role(role: mosh_core::ShowRole) -> Option<PluginInput> {
        IpcReader::open(role)
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

    fn set_pad(&mut self, x: u8, y: u8, colour: mosh_midi::PadColor) {
        self.controller.set_pad(x, y, colour);
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

/// The stem taps the analysis can use: a Tap instance on the drum bus and one on the bass bus
/// (their `role` parameter set to Drums / Bass). Both are optional and found on their own, like
/// the master tap; a show without them analyses the master alone.
///
/// Stems are read only while the master comes from the plugin: they are the same Live set's
/// audio, and next to a loopback fallback they would be another clock.
pub struct Stems {
    taps: Vec<StemInput>,
    auto_open: bool,
    /// Stem blocks handed to the analyzer, for the status line and the tests.
    fed: u64,
}

struct StemInput {
    role: ShowRole,
    source: Option<Box<dyn AudioSource>>,
    retry_at: Option<Instant>,
    dead_since: Option<Instant>,
    blocks: Vec<AudioBlock>,
}

/// The stem roles the analyzer reads (`Analyzer::push_stem` ignores the others).
const STEM_ROLES: [ShowRole; 2] = [ShowRole::Drums, ShowRole::Bass];

impl Stems {
    /// No stems and no attempt to find any (tests, and a mux over caller-owned sources).
    pub fn none() -> Stems {
        Stems {
            taps: STEM_ROLES
                .iter()
                .map(|&role| StemInput {
                    role,
                    source: None,
                    retry_at: None,
                    dead_since: None,
                    blocks: Vec::new(),
                })
                .collect(),
            auto_open: false,
            fed: 0,
        }
    }

    /// The app's stems: attaches to each stem tap when one appears.
    pub fn open() -> Stems {
        Stems {
            auto_open: true,
            ..Stems::none()
        }
    }

    /// Use `source` for `role` (tests). A role the analyzer does not read is ignored.
    pub fn set_source(&mut self, role: ShowRole, source: Box<dyn AudioSource>) {
        if let Some(tap) = self.taps.iter_mut().find(|tap| tap.role == role) {
            tap.source = Some(source);
        }
    }

    /// Stem blocks handed to the analyzer so far.
    pub fn fed(&self) -> u64 {
        self.fed
    }

    /// Read every stem tap's new blocks. `master_from_plugin` false drops them (see the type).
    fn poll(&mut self, now: Instant, master_from_plugin: bool) {
        for tap in &mut self.taps {
            tap.blocks.clear();
            if tap.source.is_none()
                && self.auto_open
                && master_from_plugin
                && retry_due(&mut tap.retry_at, now)
            {
                tap.source = PluginInput::open_role(tap.role)
                    .map(|input| Box::new(input) as Box<dyn AudioSource>);
            }
            let Some(source) = tap.source.as_mut() else {
                continue;
            };
            let alive = source.poll(&mut tap.blocks);
            if alive {
                tap.dead_since = None;
            } else {
                tap.blocks.clear();
                let since = *tap.dead_since.get_or_insert(now);
                // A stem tap removed from the set is let go, so a new instance can be found.
                if self.auto_open && now.saturating_duration_since(since) >= PLUGIN_REOPEN_AFTER {
                    tap.source = None;
                    tap.dead_since = None;
                }
            }
            if !master_from_plugin {
                tap.blocks.clear();
            }
        }
    }

    /// Hand the analyzer the stem blocks that go with master block `index` of this frame.
    fn feed(&mut self, analyzer: &mut Analyzer, index: usize) {
        for tap in &self.taps {
            if let Some(block) = tap.blocks.get(index) {
                analyzer.push_stem(tap.role, block);
                self.fed += 1;
            }
        }
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

    /// Whether a pad controller (the APC mini) is attached.
    pub fn has_pads(&self) -> bool {
        !self.missing("apc")
    }

    /// Light a pad on every pad controller.
    pub fn set_pad(&mut self, x: u8, y: u8, colour: mosh_midi::PadColor) {
        for source in &mut self.sources {
            source.set_pad(x, y, colour);
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
    /// What the session grid is doing, layer by layer (the control window and the pad LEDs).
    pub grid: GridState,
    /// Follow actions and intensity-driven launches are frozen.
    pub locked: bool,
    /// An APC mini is attached.
    pub pads: bool,
}

/// The show without its pixels: audio in, one plan out.
pub struct AppCore {
    analyzer: Analyzer,
    show: Show,
    inputs: InputMux,
    /// Drum and bass stem taps, when the set has them.
    stems: Stems,
    controls: Controls,
    blocks: Vec<AudioBlock>,
    events: Vec<ControlEvent>,
    /// Control events handed in by the control window (a preview launch), not from a device.
    queued: Vec<ControlEvent>,
    /// The project the editor handed over, waiting for the next bar line.
    pending_project: Option<Project>,
    /// The clip each layer drew last frame, so a project swap can put the show back where it was.
    playing: Vec<Option<ClipId>>,
    /// The last analysis frame, for the bar-line rule (the same one the show engine uses).
    last: Option<mosh_core::AnalysisFrame>,
    song: Option<usize>,
    song_starts: u64,
    frames: u64,
    /// The input the analyzer was last fed from; a change starts a fresh analyzer.
    analyzed_source: Option<SourceKind>,
    analyzer_resets: u64,
    /// Launches and stops on their way to a bar line, and the scene each layer plays.
    grid: GridState,
    /// Bar lines a queued launch has waited, so one the show dropped cannot blink forever.
    queued_bars: [u8; GRID_LAYERS],
    /// The pad LEDs as last sent, and when the whole grid was last sent.
    leds: Option<crate::leds::PadGrid>,
    leds_sent_at: Option<Instant>,
}

impl AppCore {
    pub fn new(project: Project, inputs: InputMux, controls: Controls) -> Self {
        AppCore {
            analyzer: Analyzer::new(AnalysisConfig::default()),
            show: Show::new(project),
            stems: if inputs.auto_open {
                Stems::open()
            } else {
                Stems::none()
            },
            inputs,
            controls,
            blocks: Vec::new(),
            events: Vec::new(),
            queued: Vec::new(),
            pending_project: None,
            playing: Vec::new(),
            last: None,
            song: None,
            song_starts: 0,
            frames: 0,
            analyzed_source: None,
            analyzer_resets: 0,
            grid: GridState::default(),
            queued_bars: [0; GRID_LAYERS],
            leds: None,
            leds_sent_at: None,
        }
    }

    /// Change the show's mode now. The show drops its queued launches with the mode they were
    /// asked in, so the grid's "queued" marks go with them.
    pub fn set_mode(&mut self, mode: ShowMode) {
        if mode != self.show.mode() {
            self.grid.queued = [None; GRID_LAYERS];
            self.grid.stopping = 0;
        }
        self.show.set_mode(mode);
    }

    /// The stem taps: how many blocks they fed, and a way to hand one in (tests).
    pub fn stems_mut(&mut self) -> &mut Stems {
        &mut self.stems
    }

    /// Times the analyzer was started fresh because the input changed.
    pub fn analyzer_resets(&self) -> u64 {
        self.analyzer_resets
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
        // A different input is a different stream: the loopback the app listened to before Live
        // pressed play (or the plugin's last song before a fallback) must not become the history
        // the new stream's intensity is judged against, so the analyzer starts over.
        if self.analyzed_source.is_some_and(|previous| previous != source) {
            self.analyzer = Analyzer::new(AnalysisConfig::default());
            self.analyzer_resets += 1;
        }
        self.analyzed_source = Some(source);
        self.stems.poll(now, source == SourceKind::Plugin);

        self.events.clear();
        self.controls.poll(&mut self.events, now);
        // The control window's preview launches arrive here, on the same path as a controller's.
        self.events.append(&mut self.queued);

        for index in 0..self.blocks.len() {
            if let Some(transport) = self.blocks[index].transport {
                self.note_playhead(transport.ppq);
            }
            // A stem block goes in before the master block it belongs to.
            self.stems.feed(&mut self.analyzer, index);
            self.analyzer.push(&self.blocks[index]);
        }
        self.blocks.clear();

        let analysis = self.analyzer.take_frame();
        // What the performer asked for this frame is "queued" until the show plays it.
        self.grid
            .note_requests(&self.show.project().session, &self.events);
        let bar = crossed_bar(self.last.as_ref(), &analysis);
        // An edited project lands on a bar line, before the show reads the analysis, so the new
        // project's first frame is the one the playhead is on.
        self.apply_pending_project(&analysis);
        let plan = self.show.update(&analysis, &self.events);
        self.playing = plan.layers.iter().map(|layer| layer.clip).collect();
        self.grid
            .settle(&self.show.project().session, &self.playing);
        self.expire_queued(bar);
        self.last = Some(analysis);
        self.frames += 1;
        self.send_leds(now);
        Frame {
            analysis,
            plan,
            source,
            mode: self.show.mode(),
            song: self.song,
            grid: self.grid,
            locked: self.show.locked(),
            pads: self.controls.has_pads(),
        }
    }

    /// A launch the show never played (a scene that was removed, a mode change inside the show)
    /// stops blinking after [`QUEUE_PATIENCE_BARS`] bar lines.
    fn expire_queued(&mut self, bar: bool) {
        for layer in 0..GRID_LAYERS {
            let waiting =
                self.grid.queued[layer].is_some() || self.grid.stopping & (1 << layer) != 0;
            if !waiting {
                self.queued_bars[layer] = 0;
                continue;
            }
            if bar {
                self.queued_bars[layer] = self.queued_bars[layer].saturating_add(1);
            }
            if self.queued_bars[layer] > QUEUE_PATIENCE_BARS {
                self.grid.queued[layer] = None;
                self.grid.stopping &= !(1 << layer);
                self.queued_bars[layer] = 0;
            }
        }
    }

    /// Mirror the session grid on the APC mini's pads: only the pads that changed, and the whole
    /// grid every [`LED_REFRESH`] so a controller plugged in mid-rehearsal lights up.
    fn send_leds(&mut self, now: Instant) {
        if !self.controls.has_pads() {
            self.leds = None;
            return;
        }
        let next = crate::leds::pad_grid(&self.show.project().session, &self.grid);
        let refresh = self
            .leds_sent_at
            .is_none_or(|at| now.saturating_duration_since(at) >= LED_REFRESH);
        let changes = match (&self.leds, refresh) {
            (Some(previous), false) => crate::leds::pad_changes(previous, &next),
            _ => {
                self.leds_sent_at = Some(now);
                crate::leds::pad_changes(&crate::leds::unknown_grid(), &next)
            }
        };
        for (x, y, colour) in changes {
            self.controls.set_pad(x, y, colour);
        }
        self.leds = Some(next);
    }

    /// Give the show an edited project. It is swapped in on the next bar line, never mid-phrase.
    ///
    /// A newer project replaces one that has not been applied yet: only the last edit matters,
    /// and the performer may well drag a slider across a whole bar.
    pub fn queue_project(&mut self, project: Project) {
        self.pending_project = Some(project);
    }

    /// Is an edited project waiting for the next bar line?
    pub fn pending_project(&self) -> bool {
        self.pending_project.is_some()
    }

    /// A control event from the control window, for the next frame.
    pub fn queue_control(&mut self, event: ControlEvent) {
        self.queued.push(event);
    }

    /// Swap in the queued project if this frame is a bar line.
    ///
    /// `mosh-show` owns `Show` and has no way to replace its project, so the show is rebuilt from
    /// the edited one here. The clock, the analyzer, the render thread and the stage are
    /// untouched, which is what "without restarting" means for a set. What a rebuild costs:
    ///
    /// * **Arrangement** mode re-reads the playhead on this same frame, so the right clip is
    ///   drawn immediately.
    /// * **Session** and **Autopilot** get their layers' clips back through ordinary launch
    ///   events, so the project's own quantize applies to them too (on the bar line itself that
    ///   is this frame).
    /// * Procedural clips draw again from the default seed, and a transition in progress ends.
    fn apply_pending_project(&mut self, f: &mosh_core::AnalysisFrame) {
        if self.pending_project.is_none() || !crossed_bar(self.last.as_ref(), f) {
            return;
        }
        let Some(project) = self.pending_project.take() else {
            return;
        };
        let mode = self.show.mode();
        let mut show = Show::new(project);
        show.set_mode(mode);
        if mode != ShowMode::Arrangement {
            self.events.extend(restore_cells(&show, &self.playing));
        }
        self.show = show;
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

/// The layers the grid state tracks: the compositor's stack, and the APC mini's pad columns.
pub const GRID_LAYERS: usize = 8;

/// Bar lines a queued launch may wait before its mark is dropped (a phrase is four bars).
pub const QUEUE_PATIENCE_BARS: u8 = 8;

/// How often the whole pad grid is sent again.
pub const LED_REFRESH: Duration = Duration::from_secs(2);

/// What the session grid is doing, per layer: for the control window's cells and the pad LEDs.
///
/// `mosh-show` keeps its queue to itself, so this is rebuilt from what goes in (the launch and
/// stop events of a frame) and what comes out (the clip each layer draws): a launch is *queued*
/// from the frame it is asked for until the layer draws that slot's clip.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GridState {
    /// The clip each layer draws.
    pub playing: [Option<ClipId>; GRID_LAYERS],
    /// The scene row each layer's clip was launched from, when the grid has it.
    pub scene: [Option<u16>; GRID_LAYERS],
    /// The scene row waiting for its bar line on each layer.
    pub queued: [Option<u16>; GRID_LAYERS],
    /// Bit per layer: a stop is waiting for its bar line.
    pub stopping: u8,
}

impl GridState {
    /// Record this frame's launch and stop requests.
    pub fn note_requests(&mut self, grid: &SessionGrid, events: &[ControlEvent]) {
        let cell = |layer: usize, scene: usize| {
            grid.scenes
                .get(scene)
                .and_then(|row| row.cells.get(layer).copied().flatten())
        };
        for event in events {
            match *event {
                ControlEvent::LaunchClip { layer, scene }
                    if layer < GRID_LAYERS && cell(layer, scene).is_some() =>
                {
                    self.queued[layer] = Some(scene as u16);
                    self.stopping &= !(1 << layer);
                }
                ControlEvent::LaunchScene(scene) => {
                    for layer in 0..GRID_LAYERS {
                        if cell(layer, scene).is_some() {
                            self.queued[layer] = Some(scene as u16);
                            self.stopping &= !(1 << layer);
                        }
                    }
                }
                ControlEvent::StopLayer(layer) if layer < GRID_LAYERS => {
                    self.queued[layer] = None;
                    if self.playing[layer].is_some() {
                        self.stopping |= 1 << layer;
                    }
                }
                _ => {}
            }
        }
    }

    /// Fold in what each layer drew this frame: a queued slot whose clip is now on its layer is
    /// playing, a stopped layer that is empty has stopped.
    pub fn settle(&mut self, grid: &SessionGrid, playing: &[Option<ClipId>]) {
        let cell = |layer: usize, scene: u16| {
            grid.scenes
                .get(usize::from(scene))
                .and_then(|row| row.cells.get(layer).copied().flatten())
        };
        for layer in 0..GRID_LAYERS {
            let now = playing.get(layer).copied().flatten();
            self.playing[layer] = now;
            if let Some(scene) = self.queued[layer]
                && now.is_some()
                && cell(layer, scene) == now
            {
                self.scene[layer] = Some(scene);
                self.queued[layer] = None;
            }
            if now.is_none() {
                self.stopping &= !(1 << layer);
                self.scene[layer] = None;
                continue;
            }
            // A clip the show launched by itself (a follow action, Autopilot, the arrangement):
            // the row it came from is the first one that holds it on this layer, if any does.
            if self.scene[layer].is_none_or(|scene| cell(layer, scene) != now) {
                self.scene[layer] = grid
                    .scenes
                    .iter()
                    .position(|row| row.cells.get(layer).copied().flatten() == now)
                    .map(|scene| scene as u16);
            }
        }
    }
}

/// The three event lamps (BUILD, DROP, BREAKDOWN): an event lights its lamp, which then fades,
/// so a flag that lasts one frame is visible in a window that paints ten times a second.
#[derive(Clone, Copy, Debug, Default)]
pub struct Lamps {
    lit_at: [Option<Instant>; 3],
}

impl Lamps {
    /// How long a lamp takes to fade out.
    pub const FADE: Duration = Duration::from_millis(1_500);

    /// Light the lamps of this frame's events. A drop ends the build it resolves.
    pub fn note(&mut self, events: EventFlags, now: Instant) {
        if events.contains(EventFlags::BUILD_START) {
            self.lit_at[0] = Some(now);
        }
        if events.contains(EventFlags::DROP) {
            self.lit_at[1] = Some(now);
            self.lit_at[0] = None;
        }
        if events.contains(EventFlags::BREAKDOWN) {
            self.lit_at[2] = Some(now);
        }
    }

    /// Each lamp's brightness, 1 when just lit down to 0, in steps of a tenth so an idle window
    /// is not repainted for a change nobody can see.
    pub fn levels(&self, now: Instant) -> [f32; 3] {
        self.lit_at.map(|at| match at {
            Some(at) => {
                let age = now.saturating_duration_since(at).as_secs_f32();
                let level = (1.0 - age / Self::FADE.as_secs_f32()).clamp(0.0, 1.0);
                (level * 10.0).ceil() / 10.0
            }
            None => 0.0,
        })
    }
}

/// A bar line, by the same rule the show engine uses: the analysis' `BAR` flag, or the bar counter
/// or phase moving, and never twice for a position that did not move forward.
fn crossed_bar(prev: Option<&mosh_core::AnalysisFrame>, f: &mosh_core::AnalysisFrame) -> bool {
    /// Frame phases are `f32`, so a boundary derived from one carries about 1e-7 of a beat.
    const EPS: f64 = 1e-4;
    let Some(prev) = prev else {
        return f.events.contains(EventFlags::BAR);
    };
    if f.ppq <= prev.ppq + EPS {
        return false;
    }
    f.events.contains(EventFlags::BAR) || f.bar_index != prev.bar_index || f.bar_phase < prev.bar_phase
}

/// Launch events that put the clips a layer was drawing back on the rebuilt show, in their own
/// column of the session grid. A clip that the edit removed is simply not restored.
fn restore_cells(show: &Show, playing: &[Option<ClipId>]) -> Vec<ControlEvent> {
    let grid = &show.project().session;
    let mut events = Vec::new();
    for (layer, clip) in playing.iter().enumerate() {
        let Some(id) = *clip else { continue };
        let scene = grid.scenes.iter().enumerate().find_map(|(scene, row)| {
            (row.cells.get(layer) == Some(&Some(id))).then_some(scene)
        });
        if let Some(scene) = scene {
            events.push(ControlEvent::LaunchClip { layer, scene });
        }
    }
    events
}

/// The project the app runs with when it is started without one: an empty show in Autopilot over
/// a 16-scene grid, so pressing play on nothing still draws something. Twelve web scenes and four
/// abstract style scenes; the retro web scenes (`catalog::LEGACY`) are left out.
pub fn default_project() -> Project {
    const SCENES: [&str; 16] = [
        "tunnel",
        "loom",
        "numogram",
        "flesh",
        "rotor",
        "moire",
        "spectrum",
        "cells",
        "aura-mesh",
        "vortex",
        "kaleido",
        "reaction",
        "contour-field",
        "glass-refract",
        "silk-ribbon",
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
    /// The playhead in beats: Live's when the plugin is the source, the tracker's otherwise.
    pub ppq: f64,
    /// Live's transport is running.
    pub transport: bool,
    /// Continuous intensity, 0..1.
    pub intensity: f32,
    /// BUILD, DROP, BREAKDOWN lamps, 0..1 each (see [`Lamps`]).
    pub lamps: [f32; 3],
    /// The song marker the playhead is in.
    pub song: Option<u16>,
    /// The clip each layer draws.
    pub playing: [Option<ClipId>; GRID_LAYERS],
    /// The scene row each layer's clip came from.
    pub scene: [Option<u16>; GRID_LAYERS],
    /// The scene row queued on each layer.
    pub queued: [Option<u16>; GRID_LAYERS],
    /// Bit per layer: a stop is queued.
    pub stopping: u8,
    /// Automation is frozen (`ControlEvent::Lock`).
    pub locked: bool,
    /// An APC mini is attached.
    pub pads: bool,
}

impl Status {
    /// The status line of one frame. The frame times are rounded to a tenth of a millisecond and
    /// the playhead to a sixteenth of a beat, so two frames that would paint the same window
    /// compare equal and the control window is left alone.
    pub fn from_frame(frame: &Frame, gpu_ms: f32, cpu_ms: f32, lamps: [f32; 3]) -> Status {
        let round = |value: f32, steps: f32| (value * steps).round() / steps;
        Status {
            mode: frame.mode.into(),
            source: SourceName(frame.source),
            tempo_bpm: round(frame.analysis.tempo_bpm, 10.0),
            bar: frame.analysis.bar_index,
            bar_phase: round(frame.analysis.bar_phase, 16.0),
            level: frame.analysis.level,
            gpu_ms: round(gpu_ms, 10.0),
            cpu_ms: round(cpu_ms, 10.0),
            ppq: (frame.analysis.ppq * 16.0).round() / 16.0,
            transport: frame.analysis.playing,
            intensity: round(frame.analysis.intensity, 50.0),
            lamps,
            song: frame.song.map(|song| song as u16),
            playing: frame.grid.playing,
            scene: frame.grid.scene,
            queued: frame.grid.queued,
            stopping: frame.grid.stopping,
            locked: frame.locked,
            pads: frame.pads,
        }
    }
}

/// The control window's title: the app, the show's file and whether it has unsaved edits.
pub fn window_title(path: Option<&str>, modified: bool) -> String {
    let name = path
        .map(|path| path.rsplit(['/', '\\']).next().unwrap_or(path))
        .filter(|name| !name.is_empty())
        .unwrap_or("untitled show");
    format!("ilanaVisual - {name}{}", if modified { " *" } else { "" })
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
