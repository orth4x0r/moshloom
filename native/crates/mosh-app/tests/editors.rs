//! Task 2.7's tests: the control window's editors, driven with no window, no GPU and no Chromium.
//!
//! The editing model (`mosh_app::editor`) owns the `Project` and applies commands; the egui layer
//! only maps widgets to those commands. So everything the performer can do to a show is checked
//! here as data, and the only thing left for the coordinator's windowed run is that the window
//! appears and paints.
//!
//! What is checked:
//!
//! * `editing_a_clip_updates_the_project` — every clip-editor command lands in the project, and
//!   an edit is handed to the running show exactly once.
//! * `import_als_adds_markers_and_sig_map` — a gzipped Live set built here fills the arrangement's
//!   song markers and signature map (no real `.als` is committed, same rule as `mosh-als`).
//! * `save_and_load_roundtrip` — RON through `mosh_show::io`, and a failed open leaves the open
//!   project alone.
//! * `preset_save_and_apply` — params and seed go into a preset and come back.
//! * `edits_reach_the_running_show_at_next_bar` — `AppCore` keeps drawing the old project mid-bar
//!   and switches on the bar line, without restarting.
//! * the views themselves (layout, shortcuts, hit-testing, frame cost) are in `tests/ui.rs`, and
//!   the commands behind them in `tests/commands.rs`.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use flate2::Compression;
use flate2::write::GzEncoder;

use mosh_app::core::{
    AppCore, AudioSource, Controls, InputMux, arrangement_project, simple_clip,
};
use mosh_app::editor::{Command, EditorError, EditorModel, View};
use mosh_core::{
    AudioBlock, EffectKind, FollowAction, LaunchQuantize, Layer, ParamRange, Project, SceneRow,
    SessionGrid, ShowMode, SigChange, Transport, Transition, TransitionKind,
};

/// A file in the OS temp directory that deletes itself.
struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn path_for(suffix: &str) -> TempFile {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "mosh-app-editor-test-{}-{}.{suffix}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        TempFile { path }
    }

    /// An empty file, for a path a command writes to.
    fn empty(suffix: &str) -> TempFile {
        let file = TempFile::path_for(suffix);
        File::create(&file.path).expect("create temp file");
        file
    }

    /// A text file, plain or gzipped.
    fn text(suffix: &str, body: &str, gzipped: bool) -> TempFile {
        let file = TempFile::path_for(suffix);
        let handle = File::create(&file.path).expect("create temp file");
        if gzipped {
            let mut gz = GzEncoder::new(handle, Compression::fast());
            gz.write_all(body.as_bytes()).expect("gzip temp file");
            gz.finish().expect("finish gzip");
        } else {
            let mut handle = handle;
            handle.write_all(body.as_bytes()).expect("write temp file");
        }
        file
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn text_path(&self) -> String {
        self.path.to_string_lossy().into_owned()
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// A Live 11-shaped set with three locators and a 4/4 -> 6/8 change at beat 64.
///
/// Deliberately smaller than `mosh-als`'s fixture: this test is about the editor filling the
/// project's arrangement, not about the parser (that crate's own tests pin the parser).
const ALS_SET: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Ableton MajorVersion="5" MinorVersion="11.0_433" Creator="Ableton Live 11.0.1d1">
  <LiveSet>
    <Locators>
      <Locators>
        <Locator Id="0">
          <Time Value="128" />
          <Name Value="Outro" />
        </Locator>
        <Locator Id="1">
          <Time Value="0" />
          <Name Value="Intro" />
        </Locator>
        <Locator Id="2">
          <Time Value="64" />
          <Name Value="Song &amp; Chorus" />
        </Locator>
      </Locators>
    </Locators>
    <MasterTrack>
      <AutomationEnvelopes>
        <Envelopes>
          <AutomationEnvelope Id="0">
            <EnvelopeTarget>
              <PointeeId Value="7001" />
            </EnvelopeTarget>
            <Automation>
              <Events>
                <EnumEvent Id="0" Time="-63072000" Value="201" />
                <EnumEvent Id="1" Time="64" Value="302" />
              </Events>
            </Automation>
          </AutomationEnvelope>
        </Envelopes>
      </AutomationEnvelopes>
      <Tempo>
        <Manual Value="124.000252" />
        <AutomationTarget Id="7002">
          <LockEnvelope Value="0" />
        </AutomationTarget>
      </Tempo>
      <TimeSignature>
        <Manual Value="201" />
        <AutomationTarget Id="7001">
          <LockEnvelope Value="0" />
        </AutomationTarget>
      </TimeSignature>
    </MasterTrack>
  </LiveSet>
</Ableton>
"#;

/// A one-layer project with one clip, in Session mode, quantized to the bar.
fn clip_project() -> Project {
    Project {
        mode: ShowMode::Session,
        clips: vec![simple_clip(7, "tunnel")],
        session: SessionGrid {
            layers: vec![Layer::default()],
            scenes: vec![SceneRow {
                name: "one".to_string(),
                cells: vec![Some(7)],
            }],
            quantize: LaunchQuantize::Bar,
        },
        ..Project::default()
    }
}

/// A scripted audio source: the block it hands out can change mid-test.
#[derive(Clone, Default)]
struct Script {
    state: Arc<Mutex<Option<AudioBlock>>>,
}

impl Script {
    fn with_block(block: AudioBlock) -> Script {
        Script {
            state: Arc::new(Mutex::new(Some(block))),
        }
    }

    fn set_block(&self, block: AudioBlock) {
        *self.state.lock().unwrap() = Some(block);
    }
}

impl AudioSource for Script {
    fn poll(&mut self, out: &mut Vec<AudioBlock>) -> bool {
        if let Some(block) = self.state.lock().unwrap().clone() {
            out.push(block);
        }
        true
    }

    fn label(&self) -> &'static str {
        "script"
    }
}

/// Control events waiting for the next poll — the shape a controller's ring has.
/// A one-layer project in Session mode with two scenes: clip 1 and clip 2.
fn two_clip_session() -> Project {
    let mut project = clip_project();
    project.clips.push(simple_clip(2, "loom"));
    project.session.scenes.push(SceneRow {
        name: "two".to_string(),
        cells: vec![Some(2)],
    });
    project
}

/// A 120 BPM 4/4 block at Ableton's playhead position `ppq`.
fn transport_block(ppq: f64) -> AudioBlock {
    let measure = ppq.rem_euclid(4.0);
    AudioBlock {
        transport: Some(Transport {
            tempo_bpm: 120.0,
            ppq,
            bar_start_ppq: ppq - measure,
            sig_num: 4,
            sig_den: 4,
            playing: true,
            sample_pos: 0,
        }),
        sample_rate: 48_000,
        channels: 2,
        samples: vec![0.0; 512],
    }
}

/// Every clip-editor command lands in the project, and the edit is handed over exactly once.
#[test]
fn editing_a_clip_updates_the_project() {
    let mut model = EditorModel::new(clip_project());
    assert_eq!(model.view(), View::Session);
    assert_eq!(model.selected(), Some(7));
    model.apply(Command::SelectClip(Some(7))).expect("select");

    model
        .apply(Command::SetClipParam {
            clip: 7,
            index: 0,
            value: 0.75,
        })
        .expect("p0");
    model
        .apply(Command::SetVisualizer {
            clip: 7,
            visualizer: "fluid".to_string(),
        })
        .expect("visualizer");
    model
        .apply(Command::SetSeed { clip: 7, seed: 42 })
        .expect("seed");
    model
        .apply(Command::SetRange {
            clip: 7,
            level: 2,
            index: 3,
            lo: 0.2,
            hi: 0.9,
        })
        .expect("range");
    model
        .apply(Command::SetPulseDepth {
            clip: 7,
            level: 2,
            depth: 0.6,
        })
        .expect("pulse depth");
    model
        .apply(Command::AddEffect {
            clip: 7,
            kind: EffectKind::Mirror,
        })
        .expect("effect");
    model
        .apply(Command::SetTransition {
            clip: 7,
            kind: TransitionKind::Datamosh,
            beats: 2.0,
        })
        .expect("transition");
    model
        .apply(Command::SetFollow {
            clip: 7,
            action: FollowAction::Next,
            after_bars: 8,
        })
        .expect("follow action");
    model
        .apply(Command::SetReactToEvents { clip: 7, on: true })
        .expect("react to events");
    model
        .apply(Command::SetVideo {
            clip: 7,
            path: Some("C:/video/a.mov".to_string()),
        })
        .expect("overlay video");

    let clip = &model.project().clips[0];
    assert_eq!(clip.params.p[0], 0.75);
    assert_eq!(clip.params.seed, 42);
    assert_eq!(clip.visualizer, "fluid");
    assert_eq!(
        clip.ranges[1][3],
        ParamRange {
            lo: 0.2,
            hi: 0.9
        }
    );
    assert_eq!(clip.pulse_depth[1], 0.6);
    assert_eq!(clip.effects.len(), 1);
    assert_eq!(clip.effects[0].kind, EffectKind::Mirror);
    assert_eq!(
        clip.transition_in,
        Transition {
            kind: TransitionKind::Datamosh,
            beats: 2.0
        }
    );
    assert_eq!(clip.follow, FollowAction::Next);
    assert_eq!(clip.follow_after_bars, 8);
    assert!(clip.react_to_events);
    assert_eq!(clip.video.as_deref(), Some("C:/video/a.mov"));

    // The running show is handed the whole project once, not once per command.
    let handed = model.handoff().expect("a changed project");
    assert_eq!(handed.clips[0].params.p[0], 0.75);
    assert!(model.handoff().is_none(), "a second handoff of no change");
    assert!(model.edits() >= 10);

    // A command that cannot be applied is an error and changes nothing: the show is never
    // silently left with half an edit.
    let before = model.project().clone();
    assert_eq!(
        model.apply(Command::SetClipParam {
            clip: 99,
            index: 0,
            value: 1.0
        }),
        Err(EditorError::UnknownClip(99))
    );
    assert_eq!(
        model.apply(Command::SetClipParam {
            clip: 7,
            index: 8,
            value: 1.0
        }),
        Err(EditorError::BadParam(8))
    );
    assert_eq!(
        model.apply(Command::SetRange {
            clip: 7,
            level: 5,
            index: 0,
            lo: 0.0,
            hi: 1.0
        }),
        Err(EditorError::BadLevel(5))
    );
    // A visualizer id the renderer does not know would be a dead layer on stage.
    assert_eq!(
        model.apply(Command::SetVisualizer {
            clip: 7,
            visualizer: "not-a-visualizer".to_string()
        }),
        Err(EditorError::UnknownVisualizer(
            "not-a-visualizer".to_string()
        ))
    );
    assert_eq!(model.project(), &before);
    assert!(model.handoff().is_none(), "an error must not mark the project dirty");
}

/// Importing a Live set fills the arrangement: markers and the signature map.
#[test]
fn import_als_adds_markers_and_sig_map() {
    let set = TempFile::text("als", ALS_SET, true);
    let mut model = EditorModel::new(clip_project());

    model
        .apply(Command::ImportAls {
            path: set.text_path(),
        })
        .expect("import");

    let arrangement = &model.project().arrangement;
    let names: Vec<&str> = arrangement.songs.iter().map(|song| song.name.as_str()).collect();
    assert_eq!(names, ["Intro", "Song & Chorus", "Outro"]);
    assert_eq!(arrangement.songs[0].start_ppq, 0.0);
    assert_eq!(arrangement.songs[1].start_ppq, 64.0);
    assert_eq!(arrangement.songs[2].start_ppq, 128.0);
    assert_eq!(
        arrangement.sig_map,
        vec![
            SigChange {
                ppq: 0.0,
                num: 4,
                den: 4
            },
            SigChange {
                ppq: 64.0,
                num: 6,
                den: 8
            },
        ]
    );

    // The arrangement's axis is in bars, so the editors can label it: 16 bars of 4/4, then 6/8.
    assert_eq!(mosh_app::editor::bars_at(&arrangement.sig_map, 0.0), 0.0);
    assert_eq!(mosh_app::editor::bars_at(&arrangement.sig_map, 64.0), 16.0);
    let after = mosh_app::editor::bars_at(&arrangement.sig_map, 67.0);
    assert!((after - 17.0).abs() < 1e-9, "one 6/8 bar later: {after}");

    // A file that is not a set is an error, and the markers already imported survive it.
    let junk = TempFile::text("als", "this is not a Live set", false);
    assert!(
        model
            .apply(Command::ImportAls {
                path: junk.text_path()
            })
            .is_err()
    );
    assert_eq!(model.project().arrangement.songs.len(), 3);
}

/// A project saves to RON and loads back identical, and a failed open keeps the open project.
#[test]
fn save_and_load_roundtrip() {
    let file = TempFile::empty("ron");
    let mut model = EditorModel::new(clip_project());
    model
        .apply(Command::SetClipParam {
            clip: 7,
            index: 5,
            value: 0.125,
        })
        .expect("edit");
    model
        .apply(Command::AddMarker {
            name: "Intro".to_string(),
            ppq: 0.0,
        })
        .expect("marker");
    model.apply(Command::SetMode(ShowMode::Arrangement)).expect("mode");
    model
        .apply(Command::SaveProject {
            path: file.text_path(),
        })
        .expect("save");
    assert!(file.path().metadata().expect("saved file").len() > 0);

    let mut fresh = EditorModel::new(Project::default());
    fresh
        .apply(Command::OpenProject {
            path: file.text_path(),
        })
        .expect("open");
    assert_eq!(fresh.project(), model.project());
    assert_eq!(fresh.project().mode, ShowMode::Arrangement);
    assert_eq!(fresh.selected(), Some(7));

    // A missing project is an error, not an empty show: the open one is untouched.
    let before = fresh.project().clone();
    assert!(matches!(
        fresh.apply(Command::OpenProject {
            path: "Z:/no/such/show.ron".to_string()
        }),
        Err(EditorError::Io(_))
    ));
    assert_eq!(fresh.project(), &before);
}

/// A preset carries a clip's params and seed, and putting it back restores both.
#[test]
fn preset_save_and_apply() {
    let mut model = EditorModel::new(clip_project());
    model.apply(Command::SelectClip(Some(7))).expect("select");
    model
        .apply(Command::SetClipParam {
            clip: 7,
            index: 2,
            value: 0.4,
        })
        .expect("param");
    model
        .apply(Command::SetSeed { clip: 7, seed: 7_777 })
        .expect("seed");
    model
        .apply(Command::SetVisualizer {
            clip: 7,
            visualizer: "swarm".to_string(),
        })
        .expect("visualizer");
    model
        .apply(Command::SavePreset {
            name: "swarm look".to_string(),
        })
        .expect("save preset");

    assert_eq!(model.project().presets.len(), 1);
    assert_eq!(model.project().presets[0].name, "swarm look");
    assert_eq!(model.project().presets[0].params.seed, 7_777);
    assert_eq!(model.project().presets[0].visualizer, "swarm");

    // Change the clip away from the preset, then put the preset back.
    model
        .apply(Command::SetClipParam {
            clip: 7,
            index: 2,
            value: 0.9,
        })
        .expect("param");
    model
        .apply(Command::SetSeed { clip: 7, seed: 1 })
        .expect("seed");
    model
        .apply(Command::SetVisualizer {
            clip: 7,
            visualizer: "scan".to_string(),
        })
        .expect("visualizer");
    model.apply(Command::ApplyPreset { index: 0 }).expect("apply");

    let clip = &model.project().clips[0];
    assert_eq!(clip.params.p[2], 0.4);
    assert_eq!(clip.params.seed, 7_777);
    assert_eq!(clip.visualizer, "swarm");

    // An unnamed preset takes the clip's name, so the list never has a blank row.
    model
        .apply(Command::SavePreset {
            name: String::new(),
        })
        .expect("save preset");
    assert_eq!(model.project().presets[1].name, "clip 7");

    // Both a bad preset index and applying one with no clip selected are errors.
    assert_eq!(
        model.apply(Command::ApplyPreset { index: 9 }),
        Err(EditorError::UnknownPreset(9))
    );
    model.apply(Command::SelectClip(None)).expect("deselect");
    assert_eq!(
        model.apply(Command::ApplyPreset { index: 0 }),
        Err(EditorError::NoClip)
    );
}

/// The show keeps drawing the project it has and takes the edited one on the next bar line.
#[test]
fn edits_reach_the_running_show_at_next_bar() {
    let project = arrangement_project(&[(1, 0.0, 64.0)], &[]);
    let source = Script::with_block(transport_block(0.0));
    let handle = source.clone();
    let mut core = AppCore::new(
        project,
        InputMux::new(Some(Box::new(source)), None),
        Controls::new(Vec::new()),
    );
    let start = Instant::now();

    let first = core.step(start);
    assert_eq!(first.plan.layers[0].visualizer.as_deref(), Some("tunnel"));

    // The performer edits the clip that is on stage and hands the project over.
    let mut model = EditorModel::new(core.show().project().clone());
    model
        .apply(Command::SetVisualizer {
            clip: 1,
            visualizer: "loom".to_string(),
        })
        .expect("edit");
    core.queue_project(model.handoff().expect("a changed project"));
    assert!(core.pending_project());

    // Mid-bar the old project is still what is drawn: an edit never lands mid-phrase.
    handle.set_block(transport_block(1.0));
    let mid = core.step(start + Duration::from_millis(16));
    assert_eq!(mid.plan.layers[0].visualizer.as_deref(), Some("tunnel"));
    assert_eq!(core.show().project().clips[0].visualizer, "tunnel");
    assert!(core.pending_project());

    // On the bar line the show switches, in the same frame, without a restart.
    handle.set_block(transport_block(4.0));
    let bar = core.step(start + Duration::from_millis(32));
    assert_eq!(bar.plan.layers[0].visualizer.as_deref(), Some("loom"));
    assert_eq!(core.show().project().clips[0].visualizer, "loom");
    assert!(!core.pending_project());
    // Three frames, one show: the clock and the render thread were never restarted.
    assert_eq!(core.frames(), 3);
}

/// A preview launch from the Session grid goes to the running show at once, as a control event.
#[test]
fn preview_launch_reaches_the_show() {
    let source = Script::with_block(transport_block(0.0));
    let mut core = AppCore::new(
        two_clip_session(),
        InputMux::new(Some(Box::new(source)), None),
        Controls::new(Vec::new()),
    );
    let start = Instant::now();
    assert_eq!(core.step(start).plan.layers[0].clip, None);

    let mut model = EditorModel::new(core.show().project().clone());
    let outcome = model
        .apply(Command::Launch {
            layer: 0,
            scene: 1,
        })
        .expect("a cell that has a clip");
    assert!(!outcome.project_changed, "a launch is not an edit");
    assert!(model.handoff().is_none());
    core.queue_control(outcome.launch.expect("a launch event"));

    let launched = core.step(start + Duration::from_millis(16));
    assert_eq!(launched.plan.layers[0].clip, Some(2));
    assert_eq!(launched.plan.layers[0].visualizer.as_deref(), Some("loom"));

    // An empty cell is refused before it reaches the show.
    assert_eq!(
        model.apply(Command::Launch {
            layer: 4,
            scene: 0
        }),
        Err(EditorError::BadCell { layer: 4, scene: 0 })
    );
}
