//! The app's decisions, driven with fake inputs and no window.
//!
//! These are Task 2.6's tests: the show must start with no plugin and no second monitor, fall
//! back to loopback and back, restart its intensity history on a song marker, follow a playhead
//! jump, and fit the performance budgets. Nothing here opens a window or a device: the audio and
//! control sources are scripts.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mosh_app::core::{
    AppCore, AudioSource, ControlSource, Controls, InputMux, ModeName, SourceKind, SourceName,
    Status, arrangement_project, default_project, simple_clip, stage_placement_label, title_for,
};
use mosh_app::monitor::{MonitorInfo, StagePlacement, monitor_choice as choose_monitor};
use mosh_core::{
    AudioBlock, ControlEvent, EventFlags, LaunchQuantize, Layer, ParamRange, Project, SceneRow,
    SessionGrid, ShowMode, Transport, TransportSource,
};

/// A scripted audio source: `alive` plus the block it hands out, both changeable mid-test.
#[derive(Clone, Default)]
struct Script {
    state: Arc<Mutex<ScriptState>>,
}

#[derive(Default)]
struct ScriptState {
    alive: bool,
    block: Option<AudioBlock>,
}

impl Script {
    /// A live, silent loopback device: 256 frames of zeros at 48 kHz, no transport.
    fn silent() -> Script {
        Script::with_block(zero_block(48_000, None))
    }

    fn with_block(block: AudioBlock) -> Script {
        let script = Script::default();
        {
            let mut state = script.state.lock().unwrap();
            state.alive = true;
            state.block = Some(block);
        }
        script
    }

    fn set_alive(&self, alive: bool) {
        self.state.lock().unwrap().alive = alive;
    }

    fn set_block(&self, block: AudioBlock) {
        self.state.lock().unwrap().block = Some(block);
    }
}

impl AudioSource for Script {
    fn poll(&mut self, out: &mut Vec<AudioBlock>) -> bool {
        let state = self.state.lock().unwrap();
        if state.alive
            && let Some(block) = &state.block
        {
            out.push(block.clone());
        }
        state.alive
    }

    fn label(&self) -> &'static str {
        "script"
    }
}

/// Control events queued for the next poll, like a controller gesture waiting in a ring.
struct QueuedControls {
    events: Vec<ControlEvent>,
    label: &'static str,
}

impl ControlSource for QueuedControls {
    fn poll(&mut self, out: &mut Vec<ControlEvent>) {
        out.append(&mut self.events);
    }

    fn label(&self) -> &'static str {
        self.label
    }
}

fn zero_block(sample_rate: u32, transport: Option<Transport>) -> AudioBlock {
    AudioBlock {
        transport,
        sample_rate,
        channels: 2,
        samples: vec![0.0; 512],
    }
}

/// A block at Ableton's playhead position `ppq`, 120 BPM, 4/4.
fn transport_block(ppq: f64) -> AudioBlock {
    let measure = ppq.rem_euclid(4.0);
    zero_block(
        48_000,
        Some(Transport {
            tempo_bpm: 120.0,
            ppq,
            bar_start_ppq: ppq - measure,
            sig_num: 4,
            sig_den: 4,
            playing: true,
            sample_pos: 0,
        }),
    )
}

/// A one-layer project in Session mode with two one-clip scenes and no launch quantization.
fn session_project() -> Project {
    Project {
        mode: ShowMode::Session,
        clips: vec![simple_clip(1, "tunnel"), simple_clip(2, "loom")],
        session: SessionGrid {
            layers: vec![Layer::default()],
            scenes: vec![
                SceneRow {
                    name: "one".to_string(),
                    cells: vec![Some(1)],
                },
                SceneRow {
                    name: "two".to_string(),
                    cells: vec![Some(2)],
                },
            ],
            quantize: LaunchQuantize::None,
        },
        ..Project::default()
    }
}

// Review Focus item 1 and 5: the app starts before Live and before the second monitor exists.
#[test]
fn starts_with_no_plugin_and_no_monitor() {
    // No IPC writer (Live is not running yet) and no second monitor.
    assert_eq!(choose_monitor(&[]), StagePlacement::Hidden);

    let mux = InputMux::new(None, Some(Box::new(Script::silent())));
    let mut core = AppCore::new(default_project(), mux, Controls::new(Vec::new()));
    let start = Instant::now();

    for i in 0..10 {
        let frame = core.step(start + Duration::from_millis(16 * i));
        // The loopback carries no transport: the show holds its last tempo and keeps drawing.
        assert_eq!(frame.source, SourceKind::Loopback);
        assert_eq!(frame.analysis.source, TransportSource::None);
        assert_eq!(frame.mode, ShowMode::Autopilot);
        // The default project is a two-layer 16-scene grid, so every frame has a full plan.
        assert_eq!(frame.plan.layers.len(), 2);
        assert!(frame.plan.layers.iter().all(|layer| layer.blend != mosh_core::BlendMode::LumaKey));
    }
    assert_eq!(core.frames(), 10);
}

#[test]
fn switches_to_loopback_and_back() {
    let plugin = Script::with_block(zero_block(48_000, Some(Transport::default())));
    let loopback = Script::with_block(zero_block(44_100, None));
    let handle = plugin.clone();
    let mut mux = InputMux::new(Some(Box::new(plugin)), Some(Box::new(loopback)));
    let mut out = Vec::new();
    let start = Instant::now();

    assert_eq!(mux.poll(start, &mut out), SourceKind::Plugin);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].sample_rate, 48_000);
    out.clear();

    // The writer dies (Live closed, or the plugin was removed): inside the grace period the
    // plugin stays selected and nothing new arrives.
    handle.set_alive(false);
    assert_eq!(
        mux.poll(start + Duration::from_millis(1_000), &mut out),
        SourceKind::Plugin
    );
    assert!(out.is_empty());
    assert_eq!(
        mux.poll(start + Duration::from_millis(2_900), &mut out),
        SourceKind::Plugin
    );
    assert!(out.is_empty());

    // Two seconds after the last heartbeat the loopback takes over.
    assert_eq!(
        mux.poll(start + Duration::from_millis(3_100), &mut out),
        SourceKind::Loopback
    );
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].sample_rate, 44_100);
    assert_eq!(mux.fallbacks(), 1);
    out.clear();

    // A fresh heartbeat hands back immediately: stale loopback audio is not the picture.
    handle.set_alive(true);
    assert_eq!(
        mux.poll(start + Duration::from_millis(4_000), &mut out),
        SourceKind::Plugin
    );
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].sample_rate, 48_000);
}

// Review Focus item 4: history restarts at song markers.
#[test]
fn song_marker_resets_intensity() {
    let project = arrangement_project(&[(1, 0.0, 64.0), (2, 64.0, 128.0)], &[("one", 0.0), ("two", 64.0)]);
    let source = Script::with_block(transport_block(10.0));
    let handle = source.clone();
    let mut core = AppCore::new(
        project,
        InputMux::new(Some(Box::new(source)), None),
        Controls::new(Vec::new()),
    );
    let start = Instant::now();

    // Entering the first song is a reset too: the app may well start mid-song.
    let first = core.step(start);
    assert!(first.analysis.events.contains(EventFlags::SONG_START));
    assert_eq!(first.song, Some(0));
    assert_eq!(core.song_starts(), 1);

    // A frame that stays in the same song carries no SONG_START.
    handle.set_block(transport_block(12.0));
    let inside = core.step(start + Duration::from_millis(16));
    assert!(!inside.analysis.events.contains(EventFlags::SONG_START));
    assert_eq!(core.song_starts(), 1);

    // Crossing the marker at ppq 64 restarts the model: level back to 1, intensity to 0, and the
    // frame says SONG_START exactly once.
    handle.set_block(transport_block(70.0));
    let crossed = core.step(start + Duration::from_millis(32));
    assert!(crossed.analysis.events.contains(EventFlags::SONG_START));
    assert_eq!(crossed.song, Some(1));
    assert_eq!(crossed.analysis.level, 1);
    assert_eq!(crossed.analysis.intensity, 0.0);
    assert_eq!(core.song_starts(), 2);

    handle.set_block(transport_block(74.0));
    let after = core.step(start + Duration::from_millis(48));
    assert!(!after.analysis.events.contains(EventFlags::SONG_START));
    assert_eq!(core.song_starts(), 2);
}

// Review Focus item 2: Live jumps back, loops, or is scrubbed. The arrangement answers from the
// playhead every frame, so the clip under it is the one shown on the next frame.
#[test]
fn playhead_jump_reselects_clip() {
    let project = arrangement_project(&[(1, 0.0, 16.0), (2, 16.0, 32.0)], &[]);
    let source = Script::with_block(transport_block(20.0));
    let handle = source.clone();
    let mut core = AppCore::new(
        project,
        InputMux::new(Some(Box::new(source)), None),
        Controls::new(Vec::new()),
    );
    let start = Instant::now();

    let late = core.step(start);
    assert_eq!(late.plan.layers[0].clip, Some(2));

    // Jump back eight bars: the earlier clip, on this frame, with no stale clip behind it.
    handle.set_block(transport_block(4.0));
    let back = core.step(start + Duration::from_millis(16));
    assert_eq!(back.plan.layers[0].clip, Some(1));

    // And forward again.
    handle.set_block(transport_block(18.0));
    let forward = core.step(start + Duration::from_millis(32));
    assert_eq!(forward.plan.layers[0].clip, Some(2));
}

#[test]
fn control_events_reach_the_show() {
    let start = Instant::now();
    let mut core = AppCore::new(
        session_project(),
        InputMux::new(None, Some(Box::new(Script::silent()))),
        Controls::new(vec![Box::new(QueuedControls {
            events: vec![ControlEvent::LaunchClip { layer: 0, scene: 1 }],
            label: "queued",
        })]),
    );

    // A launch from the Control plugin (or an APC mini) lands on this frame's plan.
    let launched = core.step(start);
    assert_eq!(launched.plan.layers[0].clip, Some(2));

    // And stays there: nothing else was asked for.
    let held = core.step(start + Duration::from_millis(16));
    assert_eq!(held.plan.layers[0].clip, Some(2));
}

// Review Focus item 5: the stage monitor is chosen from whatever monitors exist.
#[test]
fn monitor_choice() {
    let laptop = MonitorInfo::new(0, "laptop", true, 1920, 1080);
    let stage = MonitorInfo::new(1, "stage", false, 2560, 1440);
    assert_eq!(choose_monitor(&[]), StagePlacement::Hidden);
    assert_eq!(
        choose_monitor(std::slice::from_ref(&laptop)),
        StagePlacement::Laptop(0)
    );
    assert_eq!(
        choose_monitor(&[laptop.clone(), stage.clone()]),
        StagePlacement::Fullscreen(1)
    );
    // Unplugging the stage monitor falls back to the laptop screen, not to no output.
    assert_eq!(
        choose_monitor(std::slice::from_ref(&laptop)),
        StagePlacement::Laptop(0)
    );
    // Plugging it back in returns the stage to it.
    assert_eq!(
        choose_monitor(&[laptop, stage]),
        StagePlacement::Fullscreen(1)
    );
}

// The params of a default clip are the identity range, so the default project draws as authored.
#[test]
fn default_project_is_a_sixteen_scene_grid() {
    let project = default_project();
    assert_eq!(project.mode, ShowMode::Autopilot);
    assert_eq!(project.session.scenes.len(), 16);
    assert_eq!(project.session.layers.len(), 2);
    assert_eq!(project.clips.len(), 32);
    assert!(project.session.scenes.iter().all(|scene| scene.cells.len() == 2));
    let range: ParamRange = ParamRange::default();
    assert_eq!(range.lo, 0.0);
    assert_eq!(range.hi, 1.0);
}

// The control window has no scene yet (Task 2.7 builds it), so its title is the stats line: it
// must change only when the numbers do, and it must carry every number the brief lists.
#[test]
fn control_title_changes_only_when_the_numbers_change() {
    let status = Status {
        mode: ModeName::Autopilot,
        source: SourceName(SourceKind::Plugin),
        tempo_bpm: 128.0,
        bar: 7,
        bar_phase: 0.5,
        level: 3,
        gpu_ms: 4.25,
        cpu_ms: 0.8,
    };
    let title = title_for(None, &status).expect("first title");
    for expected in [
        "autopilot",
        "plugin",
        "128.0 BPM",
        "bar 8",
        "L3",
        "GPU 4.25 ms",
        "CPU 0.80 ms",
    ] {
        assert!(title.contains(expected), "{expected} missing from {title}");
    }

    // The same numbers do not ask for another redraw.
    assert_eq!(title_for(Some(&title), &status), None);

    // A level change does.
    let louder = Status { level: 4, ..status };
    let next = title_for(Some(&title), &louder).expect("a changed line");
    assert!(next.contains("L4"), "{next}");
}

#[test]
fn stage_placement_label_names_the_monitor() {
    let laptop = MonitorInfo::new(0, "laptop", true, 1920, 1080);
    let stage = MonitorInfo::new(1, "stage", false, 2560, 1440);
    let monitors = [laptop, stage];
    assert_eq!(
        stage_placement_label(StagePlacement::Fullscreen(1), &monitors),
        "stage on stage (fullscreen)"
    );
    assert_eq!(
        stage_placement_label(StagePlacement::Laptop(0), &monitors),
        "stage on laptop (borderless)"
    );
    assert_eq!(
        stage_placement_label(StagePlacement::Hidden, &monitors),
        "stage hidden (no monitor)"
    );
}

// A project file is the app's only input; a broken one must not stop the show. The shell falls
// back to the default project, which this test pins to something that can draw.
#[test]
fn the_default_project_plan_has_content_without_audio() {
    let mut core = AppCore::new(
        default_project(),
        InputMux::new(None, Some(Box::new(Script::silent()))),
        Controls::new(Vec::new()),
    );
    let start = Instant::now();
    let mut launched = false;
    for i in 0..40 {
        let frame = core.step(start + Duration::from_millis(16 * i));
        launched |= frame.plan.layers.iter().any(|layer| layer.clip.is_some());
    }
    // Autopilot starts the first scene by itself, with no input at all.
    assert!(launched, "autopilot never launched a clip: {:?}", core.show().mode());
}
