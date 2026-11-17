//! What the Session grid is doing, as the control window and the APC mini see it: a launch is
//! queued until its bar line, then playing; a stop is queued, then the slot is just loaded again.
//!
//! Driven through `AppCore::step` with a scripted transport and a fake pad controller, so the
//! whole path is covered without a window, a device or a GPU: control event -> grid state ->
//! status -> pad LEDs.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mosh_app::core::{
    AppCore, AudioSource, ControlSource, Controls, InputMux, Lamps, SourceKind, Status,
    simple_clip, window_title,
};
use mosh_core::{
    AudioBlock, ControlEvent, EventFlags, LaunchQuantize, Layer, Project, SceneRow, SessionGrid,
    ShowMode, Transport,
};
use mosh_midi::PadColor;

/// A transport the test moves by hand.
#[derive(Clone)]
struct Playhead(Arc<Mutex<f64>>);

impl AudioSource for Playhead {
    fn poll(&mut self, out: &mut Vec<AudioBlock>) -> bool {
        let ppq = *self.0.lock().unwrap();
        out.push(AudioBlock {
            transport: Some(Transport {
                tempo_bpm: 120.0,
                ppq,
                bar_start_ppq: ppq - ppq.rem_euclid(4.0),
                sig_num: 4,
                sig_den: 4,
                playing: true,
                sample_pos: 0,
            }),
            sample_rate: 48_000,
            channels: 2,
            samples: vec![0.0; 512],
        });
        true
    }

    fn label(&self) -> &'static str {
        "script"
    }
}

/// A pad controller that remembers what every pad was last set to.
#[derive(Clone, Default)]
struct Pads {
    lit: Arc<Mutex<Vec<(u8, u8, PadColor)>>>,
}

impl Pads {
    fn at(&self, x: u8, y: u8) -> Option<PadColor> {
        self.lit
            .lock()
            .unwrap()
            .iter()
            .rev()
            .find(|(px, py, _)| (*px, *py) == (x, y))
            .map(|(_, _, colour)| *colour)
    }

    fn messages(&self) -> usize {
        self.lit.lock().unwrap().len()
    }
}

impl ControlSource for Pads {
    fn poll(&mut self, _out: &mut Vec<ControlEvent>) {}

    fn label(&self) -> &'static str {
        "apc"
    }

    fn set_pad(&mut self, x: u8, y: u8, colour: PadColor) {
        self.lit.lock().unwrap().push((x, y, colour));
    }
}

/// One layer, two scenes: clip 1 and clip 2, launches quantized to the bar.
fn project() -> Project {
    Project {
        mode: ShowMode::Session,
        clips: vec![simple_clip(1, "tunnel"), simple_clip(2, "loom")],
        session: SessionGrid {
            layers: vec![Layer::default()],
            scenes: vec![
                SceneRow {
                    name: "one".into(),
                    cells: vec![Some(1)],
                },
                SceneRow {
                    name: "two".into(),
                    cells: vec![Some(2)],
                },
            ],
            quantize: LaunchQuantize::Bar,
        },
        ..Project::default()
    }
}

#[test]
fn a_launch_is_queued_then_playing_and_the_pads_follow() {
    let playhead = Playhead(Arc::new(Mutex::new(1.0)));
    let pads = Pads::default();
    let mut core = AppCore::new(
        project(),
        InputMux::new(Some(Box::new(playhead.clone())), None),
        Controls::new(vec![Box::new(pads.clone())]),
    );
    let start = Instant::now();
    let mut tick = 0u64;
    let mut step = |core: &mut AppCore, ppq: f64| {
        *playhead.0.lock().unwrap() = ppq;
        tick += 1;
        core.step(start + Duration::from_millis(16 * tick))
    };

    // Nothing playing: the whole grid is sent once, loaded slots yellow.
    let frame = step(&mut core, 1.0);
    assert!(frame.pads, "a pad controller is attached");
    assert_eq!(frame.grid.playing[0], None);
    assert_eq!(pads.messages(), 64);
    assert_eq!(pads.at(0, 0), Some(PadColor::Yellow));
    assert_eq!(pads.at(0, 1), Some(PadColor::Yellow));
    assert_eq!(pads.at(1, 0), Some(PadColor::Off));
    // A frame in which nothing changed lights nothing again.
    step(&mut core, 1.1);
    assert_eq!(pads.messages(), 64);

    // Mid-bar: the launch waits for the bar line, and the slot blinks.
    core.queue_control(ControlEvent::LaunchClip { layer: 0, scene: 1 });
    let frame = step(&mut core, 1.5);
    assert_eq!(frame.grid.queued[0], Some(1));
    assert_eq!(frame.plan.layers[0].clip, None);
    assert_eq!(pads.at(0, 1), Some(PadColor::GreenBlink));
    assert_eq!(pads.messages(), 65, "one pad changed, one message");

    // The bar line: playing, from row 2.
    let frame = step(&mut core, 4.0);
    assert_eq!(frame.plan.layers[0].clip, Some(2));
    assert_eq!(frame.grid.playing[0], Some(2));
    assert_eq!(frame.grid.scene[0], Some(1));
    assert_eq!(frame.grid.queued[0], None);
    assert_eq!(pads.at(0, 1), Some(PadColor::Green));

    // The status the control window gets says the same thing.
    let status = Status::from_frame(&frame, 4.83, 0.71, [0.0; 3]);
    assert_eq!(status.playing[0], Some(2));
    assert_eq!(status.scene[0], Some(1));
    assert_eq!(status.queued[0], None);
    assert_eq!(status.source.0, SourceKind::Plugin);
    assert!(status.transport && status.pads && !status.locked);
    assert_eq!(status.ppq, 4.0);
    assert_eq!((status.gpu_ms, status.cpu_ms), (4.8, 0.7));

    // A stop: red blinking until the bar line, then the slot is just loaded again.
    core.queue_control(ControlEvent::StopLayer(0));
    let frame = step(&mut core, 5.0);
    assert_eq!(frame.grid.stopping, 1);
    assert_eq!(pads.at(0, 1), Some(PadColor::RedBlink));
    let frame = step(&mut core, 8.0);
    assert_eq!(frame.plan.layers[0].clip, None);
    assert_eq!(frame.grid.stopping, 0);
    assert_eq!(pads.at(0, 1), Some(PadColor::Yellow));

    // A launch followed by a mode change: the show drops it, and so does the grid.
    core.queue_control(ControlEvent::LaunchClip { layer: 0, scene: 0 });
    let frame = step(&mut core, 9.0);
    assert_eq!(frame.grid.queued[0], Some(0));
    core.set_mode(ShowMode::Autopilot);
    core.set_mode(ShowMode::Session);
    let frame = step(&mut core, 9.5);
    assert_eq!(frame.grid.queued[0], None);
    assert_eq!(pads.at(0, 0), Some(PadColor::Yellow));

    // Lock reaches the status.
    core.queue_control(ControlEvent::Lock(true));
    let frame = step(&mut core, 10.0);
    assert!(Status::from_frame(&frame, 0.0, 0.0, [0.0; 3]).locked);

    // The whole grid is sent again every two seconds, so a controller plugged in mid-rehearsal
    // lights up without anyone touching it.
    let before = pads.messages();
    *playhead.0.lock().unwrap() = 10.5;
    core.step(start + Duration::from_secs(5));
    assert_eq!(pads.messages(), before + 64);
}

#[test]
fn event_lamps_light_and_fade() {
    let mut lamps = Lamps::default();
    let start = Instant::now();
    assert_eq!(lamps.levels(start), [0.0; 3]);

    lamps.note(EventFlags::BUILD_START | EventFlags::BAR, start);
    assert_eq!(lamps.levels(start), [1.0, 0.0, 0.0]);
    // Half-way through the fade, in steps of a tenth.
    let half = lamps.levels(start + Lamps::FADE / 2);
    assert_eq!(half, [0.5, 0.0, 0.0]);

    // The drop ends the build it resolves.
    let drop_at = start + Duration::from_millis(600);
    lamps.note(EventFlags::DROP, drop_at);
    assert_eq!(lamps.levels(drop_at), [0.0, 1.0, 0.0]);
    lamps.note(EventFlags::BREAKDOWN, drop_at);
    assert_eq!(lamps.levels(drop_at + Lamps::FADE * 2), [0.0; 3], "all dark again");
}

#[test]
fn the_title_names_the_show_and_marks_unsaved_edits() {
    assert_eq!(window_title(None, false), "ilanaVisual - untitled show");
    assert_eq!(
        window_title(Some(r"C:\shows\spring tour.ron"), false),
        "ilanaVisual - spring tour.ron"
    );
    assert_eq!(
        window_title(Some("shows/spring tour.ron"), true),
        "ilanaVisual - spring tour.ron *"
    );
}
