//! The control window's views, built without a window.
//!
//! `egui` runs headless here: the same `theme::apply` and `ui::draw` the window calls, fed with
//! synthetic input, so layout, hit-testing, shortcuts and cost are all checked on a machine with
//! no display (AGENTS.md: workers open no windows).
//!
//! * both views build at 800x600, 1280x800 and 1920x1080 with the panels open and closed, emit
//!   nothing when untouched and ask for no repaint when nothing moves;
//! * the keyboard and the mouse do what the shortcuts overlay says (Tab, 1-9, a click on a
//!   slot's launch button, a block dragged one bar);
//! * a frame's cost (layout + tessellation) is measured idle and busy for each view;
//! * `screenshots` rasterises the tessellated frames into PNGs under the target directory, which
//!   is how the look was reviewed before any window existed. Run it with
//!   `cargo test -p mosh-app --test ui screenshots -- --nocapture` to see where they went.

use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

use egui::{Event, Key, Modifiers, PointerButton, Pos2, Rect, Vec2};
use mosh_app::core::{SourceKind, SourceName, Status, default_project};
use mosh_app::editor::model::{ClipSource, SpanRef};
use mosh_app::editor::ui::{self, EditorUi, UiAction};
use mosh_app::editor::{Command, EditorModel, View, theme};
use mosh_app::preview::Picture;
use mosh_core::{BlendMode, EffectKind, LayerKind, ShowMode, TransitionKind};

const SIZES: [(f32, f32); 3] = [(800.0, 600.0), (1280.0, 800.0), (1920.0, 1080.0)];

/// A show with something of everything: three layers (one of them an adjustment layer, one a
/// mask), a filled grid, an arrangement with song markers, effects, a text clip and a preset.
fn demo_project() -> EditorModel {
    let mut model = EditorModel::new(default_project());
    let apply = |model: &mut EditorModel, command: Command| {
        model.apply(command.clone()).unwrap_or_else(|error| panic!("{command:?}: {error}"));
    };
    apply(&mut model, Command::AddLayer);
    apply(&mut model, Command::AddLayer);
    apply(
        &mut model,
        Command::SetLayerKind {
            layer: 2,
            kind: LayerKind::Adjustment,
        },
    );
    apply(
        &mut model,
        Command::SetLayerKind {
            layer: 3,
            kind: LayerKind::Mask,
        },
    );
    apply(
        &mut model,
        Command::SetLayerBlend {
            layer: 1,
            blend: BlendMode::Screen,
        },
    );
    for (scene, name) in ["intro", "verse", "build", "drop", "break"].iter().enumerate() {
        apply(
            &mut model,
            Command::RenameScene {
                scene,
                name: (*name).to_string(),
            },
        );
    }
    apply(
        &mut model,
        Command::SetCell {
            layer: 2,
            scene: 3,
            source: ClipSource::Visualizer("tunnel".to_string()),
        },
    );
    apply(
        &mut model,
        Command::SetCell {
            layer: 3,
            scene: 0,
            source: ClipSource::Visualizer("text".to_string()),
        },
    );

    // The arrangement: two songs, blocks on three lanes.
    let blocks: [(usize, &str, f64, f64); 9] = [
        (0, "tunnel", 0.0, 32.0),
        (0, "fluid", 32.0, 32.0),
        (0, "kaleido", 64.0, 48.0),
        (0, "swarm", 128.0, 64.0),
        (1, "spectrum", 16.0, 48.0),
        (1, "glyph_rain", 96.0, 32.0),
        (1, "moire", 144.0, 48.0),
        (2, "vortex", 48.0, 16.0),
        (3, "text", 0.0, 16.0),
    ];
    for (lane, visualizer, start, len) in blocks {
        apply(
            &mut model,
            Command::AddSpan {
                lane,
                source: ClipSource::Visualizer(visualizer.to_string()),
                start_ppq: start,
                len_ppq: Some(len),
                snap: true,
            },
        );
    }
    for (name, ppq) in [("Opening", 0.0), ("Second song", 128.0)] {
        apply(
            &mut model,
            Command::AddMarker {
                name: name.to_string(),
                ppq,
            },
        );
    }

    // A clip with a look worth a detail panel. The mosh is already on it: `new_clip` starts a
    // scene clip with the web's `mosh:.6` default, so adding it here would be a duplicate.
    let span = SpanRef { lane: 0, index: 1 };
    apply(&mut model, Command::SelectSpan(span));
    let clip = model.selected().expect("the fluid block");
    for kind in [EffectKind::Mirror, EffectKind::Echo] {
        apply(&mut model, Command::AddEffect { clip, kind });
    }
    apply(
        &mut model,
        Command::SetTransition {
            clip,
            kind: TransitionKind::Datamosh,
            beats: 2.0,
        },
    );
    apply(
        &mut model,
        Command::SetRange {
            clip,
            level: 3,
            index: 0,
            lo: 0.2,
            hi: 0.8,
        },
    );
    apply(
        &mut model,
        Command::SavePreset {
            name: "slow fluid".to_string(),
        },
    );
    apply(&mut model, Command::SelectScene(1));
    assert!(model.handoff().is_some());
    model
}

/// A show in full swing: Live connected, bar 9, level 3, a build lit, clips playing and queued.
fn busy_status(model: &EditorModel, ppq: f64) -> Status {
    let project = model.project();
    let cell = |layer: usize, scene: usize| project.session.scenes[scene].cells[layer];
    let mut status = Status {
        source: SourceName(SourceKind::Plugin),
        tempo_bpm: 128.0,
        bar: (ppq / 4.0) as i64,
        bar_phase: ((ppq / 4.0).fract()) as f32,
        level: 3,
        gpu_ms: 4.8,
        cpu_ms: 0.7,
        ppq,
        transport: true,
        intensity: 0.68,
        lamps: [0.8, 0.0, 0.0],
        song: Some(0),
        pads: true,
        ..Status::default()
    };
    status.playing[0] = cell(0, 1);
    status.scene[0] = Some(1);
    status.playing[1] = cell(1, 1);
    status.scene[1] = Some(1);
    status.queued[0] = Some(2);
    status.queued[1] = Some(2);
    status
}

/// A headless control window.
struct Harness {
    ctx: egui::Context,
    state: EditorUi,
    size: Vec2,
    time: f64,
    modifiers: Modifiers,
    /// Every texture egui has allocated, for the rasteriser.
    textures: HashMap<egui::TextureId, (usize, usize, Vec<egui::Color32>)>,
}

struct FrameOut {
    commands: Vec<Command>,
    primitives: Vec<egui::ClippedPrimitive>,
    repaint_after: std::time::Duration,
}

impl Harness {
    fn new(size: (f32, f32)) -> Harness {
        let ctx = egui::Context::default();
        theme::apply(&ctx);
        Harness {
            ctx,
            state: EditorUi::default(),
            size: Vec2::new(size.0, size.1),
            time: 0.0,
            modifiers: Modifiers::NONE,
            textures: HashMap::new(),
        }
    }

    fn frame(&mut self, model: &EditorModel, events: Vec<Event>, modifiers: Modifiers) -> FrameOut {
        self.time += 1.0 / 30.0;
        // Modifiers travel as an event of their own, before the keys they modify, and only
        // when they change (any event at all makes egui ask for another frame).
        let mut all = Vec::new();
        if modifiers != self.modifiers {
            self.modifiers = modifiers;
            all.push(Event::ModifiersChanged(modifiers));
        }
        all.extend(events);
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, self.size)),
            time: Some(self.time),
            max_texture_side: Some(8192),
            focused: true,
            events: all,
            ..Default::default()
        };
        let mut commands = Vec::new();
        let state = &mut self.state;
        let output = self.ctx.run_ui(input, |root| {
            commands = ui::draw(root, state, model);
        });
        let mut output = output;
        let mut textures_delta = std::mem::take(&mut output.textures_delta);
        for (id, delta) in textures_delta
            .set
            .iter()
            .flat_map(|(id, deltas)| deltas.iter().map(move |delta| (id, delta)))
        {
            let egui::ImageData::Color(image) = &delta.image;
            let [width, height] = image.size;
            match delta.pos {
                None => {
                    self.textures.insert(*id, (width, height, image.pixels.clone()));
                }
                Some([x0, y0]) => {
                    if let Some((full_w, _, pixels)) = self.textures.get_mut(id) {
                        for y in 0..height {
                            for x in 0..width {
                                pixels[(y0 + y) * *full_w + x0 + x] = image.pixels[y * width + x];
                            }
                        }
                    }
                }
            }
        }
        // Applied above: the delta refuses to be dropped unhandled.
        textures_delta.clear();
        let repaint_after = output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .map_or(std::time::Duration::MAX, |viewport| viewport.repaint_delay);
        let primitives = self.ctx.tessellate(output.shapes, output.pixels_per_point);
        FrameOut {
            commands,
            primitives,
            repaint_after,
        }
    }

    fn idle(&mut self, model: &EditorModel) -> FrameOut {
        self.frame(model, Vec::new(), Modifiers::NONE)
    }

    /// Answer the thumbnail requests of the last frames with synthetic pictures, as the render
    /// thread would with real ones.
    fn deliver_thumbs(&mut self) -> usize {
        let requests = self.state.thumbs.take_requests();
        let count = requests.len();
        for request in requests {
            let picture = synthetic_picture(96, 54, request.key);
            self.state.thumbs.deliver(&self.ctx, request.key, Some(picture));
        }
        count
    }

    fn key(&mut self, model: &EditorModel, key: Key, modifiers: Modifiers) -> FrameOut {
        self.frame(
            model,
            vec![Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers,
            }],
            modifiers,
        )
    }

    fn pointer(&mut self, model: &EditorModel, pos: Pos2, pressed: Option<bool>) -> FrameOut {
        let mut events = vec![Event::PointerMoved(pos)];
        if let Some(pressed) = pressed {
            events.push(Event::PointerButton {
                pos,
                button: PointerButton::Primary,
                pressed,
                modifiers: Modifiers::NONE,
            });
        }
        self.frame(model, events, Modifiers::NONE)
    }
}

/// A picture that looks like *something*: rings in a colour taken from `key`.
fn synthetic_picture(width: u32, height: u32, key: u64) -> Picture {
    let hue = [(key >> 8) as u8, (key >> 24) as u8, (key >> 40) as u8];
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 / width as f32 - 0.5;
            let dy = (y as f32 / height as f32 - 0.5) * height as f32 / width as f32;
            let ring = ((dx * dx + dy * dy).sqrt() * 40.0 + (key & 7) as f32).sin() * 0.5 + 0.5;
            for channel in hue {
                rgba.push((f32::from(channel) * (0.25 + 0.75 * ring)) as u8);
            }
            rgba.push(255);
        }
    }
    Picture {
        width,
        height,
        rgba,
    }
}

#[test]
fn both_views_build_at_every_size_with_panels_open_and_closed() {
    let mut model = demo_project();
    for size in SIZES {
        for view in [View::Arrangement, View::Session] {
            for (browser, detail) in [(true, true), (false, false), (true, false), (false, true)] {
                model.apply(Command::SelectView(view)).expect("view");
                model.apply(Command::SetBrowser(browser)).expect("browser");
                model.apply(Command::SetDetail(detail)).expect("detail");
                let what = format!("{size:?} {view:?} browser={browser} detail={detail}");
                let mut harness = Harness::new(size);
                let mut last = None;
                for _ in 0..12 {
                    let out = harness.idle(&model);
                    assert!(
                        out.commands.is_empty(),
                        "{what}: an untouched window emitted {:?}",
                        out.commands
                    );
                    assert!(!out.primitives.is_empty(), "{what}: nothing was drawn");
                    harness.deliver_thumbs();
                    last = Some(out);
                }
                let out = last.expect("frames");
                // Nothing is moving (no transport, nothing queued): the window must go quiet,
                // which is what "costs nothing during a set" rests on.
                assert!(
                    out.repaint_after >= std::time::Duration::from_secs(1),
                    "{what}: an idle window asked for a repaint in {:?}",
                    out.repaint_after
                );
                // Everything is clipped to the window: no panel lays itself out past the edge.
                let screen = Rect::from_min_size(Pos2::ZERO, harness.size).expand(0.5);
                for primitive in &out.primitives {
                    assert!(
                        screen.contains_rect(primitive.clip_rect.intersect(screen)),
                        "{what}: clip {:?}",
                        primitive.clip_rect
                    );
                }
            }
        }
    }

    // The empty state: a new show has no clip, and both views and the detail panel say what to do.
    model.apply(Command::NewProject).expect("new");
    for view in [View::Arrangement, View::Session] {
        model.apply(Command::SelectView(view)).expect("view");
        let mut harness = Harness::new((1280.0, 800.0));
        let out = harness.idle(&model);
        assert!(out.commands.is_empty() && !out.primitives.is_empty());
    }
}

#[test]
fn the_keyboard_does_what_the_overlay_says() {
    let mut model = demo_project();
    model.apply(Command::SelectView(View::Arrangement)).expect("arrangement");
    let mut harness = Harness::new((1280.0, 800.0));
    harness.idle(&model);

    let out = harness.key(&model, Key::Tab, Modifiers::NONE);
    assert_eq!(out.commands, vec![Command::ToggleView]);
    let out = harness.key(&model, Key::Tab, Modifiers::SHIFT);
    assert_eq!(out.commands, vec![Command::SetDetail(false)]);
    let out = harness.key(&model, Key::B, Modifiers::NONE);
    assert_eq!(out.commands, vec![Command::SetBrowser(false)]);
    let out = harness.key(&model, Key::Num3, Modifiers::NONE);
    assert_eq!(out.commands, vec![Command::LaunchScene(2)]);
    let out = harness.key(&model, Key::Num0, Modifiers::NONE);
    assert_eq!(
        out.commands,
        (0..4).map(Command::StopLayer).collect::<Vec<_>>(),
        "0 stops every layer"
    );
    let out = harness.key(&model, Key::M, Modifiers::NONE);
    assert_eq!(out.commands, vec![Command::SetMode(ShowMode::Arrangement)]);
    let out = harness.key(&model, Key::Z, Modifiers::COMMAND);
    assert_eq!(out.commands, vec![Command::Undo]);
    let out = harness.key(&model, Key::Y, Modifiers::COMMAND);
    assert_eq!(out.commands, vec![Command::Redo]);
    let out = harness.key(&model, Key::D, Modifiers::COMMAND);
    assert_eq!(
        out.commands,
        vec![Command::DuplicateSpan(SpanRef { lane: 0, index: 1 })]
    );
    let out = harness.key(&model, Key::Delete, Modifiers::NONE);
    assert_eq!(
        out.commands,
        vec![Command::RemoveSpan(SpanRef { lane: 0, index: 1 })],
        "Delete removes the selected block in the Arrangement view"
    );
    let out = harness.frame(&model, vec![Event::Text("?".to_string())], Modifiers::SHIFT);
    assert_eq!(out.commands, vec![Command::SetHelp(true)]);

    // F11 is the shell's: it reaches it as an action, not as an edit.
    let out = harness.key(&model, Key::F11, Modifiers::NONE);
    assert!(out.commands.is_empty());
    assert_eq!(
        std::mem::take(&mut harness.state.actions),
        vec![UiAction::ToggleStageFullscreen]
    );

    // With the overlay open the window still builds, and Esc closes it.
    let mut open = demo_project();
    open.apply(Command::SetHelp(true)).expect("help");
    let out = harness.idle(&open);
    assert!(!out.primitives.is_empty());
    let out = harness.key(&open, Key::Escape, Modifiers::NONE);
    assert_eq!(out.commands, vec![Command::SetHelp(false)]);
}

#[test]
fn a_click_on_a_slot_selects_it_and_its_triangle_launches_it() {
    let mut model = demo_project();
    model.apply(Command::SelectView(View::Session)).expect("session");
    let mut harness = Harness::new((1280.0, 800.0));
    harness.idle(&model);
    harness.idle(&model);

    // Layout, from the constants the views use: browser 212 wide, top bar 56 tall, the grid's
    // header 58, rows of 46, a 30 px launch button at the left of each slot.
    // Four layers share what is left of 1280 after the browser, the scene column (148), the
    // add-layer button (30) and a margin (8).
    let column = (1280.0 - 212.0 - 148.0 - 30.0 - 8.0) / 4.0;
    let slot = |layer: f32, scene: f32, dx: f32| {
        Pos2::new(212.0 + column * layer + dx, 56.0 + 58.0 + 46.0 * scene + 22.0)
    };
    let clip = model.project().session.scenes[2].cells[0].expect("a clip");

    // The name: select, do not launch.
    let at = slot(0.0, 2.0, 90.0);
    harness.pointer(&model, at, None);
    harness.pointer(&model, at, Some(true));
    let out = harness.pointer(&model, at, Some(false));
    assert_eq!(
        out.commands,
        vec![Command::SelectClip(Some(clip)), Command::SelectScene(2)]
    );

    // The triangle: launch.
    let at = slot(0.0, 2.0, 14.0);
    harness.pointer(&model, at, None);
    harness.pointer(&model, at, Some(true));
    let out = harness.pointer(&model, at, Some(false));
    assert_eq!(
        out.commands,
        vec![
            Command::SelectClip(Some(clip)),
            Command::SelectScene(2),
            Command::Launch { layer: 0, scene: 2 }
        ]
    );
}

#[test]
fn a_block_dragged_one_bar_is_one_move_on_release() {
    let mut model = demo_project();
    model.apply(Command::SelectView(View::Arrangement)).expect("arrangement");
    let mut harness = Harness::new((1280.0, 800.0));
    harness.state.follow = false;
    harness.idle(&model);
    harness.idle(&model);

    // Lane 0's first block is beats 0..32 at 14 px a beat; the timeline starts after the
    // browser (212) and the lane headers (178), under the top bar (56) and the ruler rows (44).
    let left = 212.0 + 178.0;
    let y = 56.0 + 44.0 + 31.0;
    let grab = Pos2::new(left + 14.0 * 10.0, y);
    let drop = Pos2::new(grab.x + 14.0 * 4.0, y);

    let mut commands = Vec::new();
    commands.extend(harness.pointer(&model, grab, None).commands);
    commands.extend(harness.pointer(&model, grab, Some(true)).commands);
    for step in 1..=4 {
        let pos = grab + (drop - grab) * (step as f32 / 4.0);
        let out = harness.pointer(&model, pos, None);
        assert!(
            !out.commands.iter().any(|command| matches!(command, Command::MoveSpan { .. })),
            "nothing is edited while the button is down"
        );
        commands.extend(out.commands);
    }
    assert!(harness.state.drag.is_some(), "the ghost is being dragged");
    commands.extend(harness.pointer(&model, drop, Some(false)).commands);
    assert!(harness.state.drag.is_none());

    let span = SpanRef { lane: 0, index: 0 };
    assert_eq!(
        commands,
        vec![
            Command::SelectSpan(span),
            Command::MoveSpan {
                span,
                to_lane: 0,
                start_ppq: 4.0,
                snap: true,
            }
        ]
    );
    // And the model takes it: the block now starts on bar 2 and trimmed its neighbour.
    for command in commands {
        model.apply(command).expect("the drag's commands apply");
    }
    let lane = &model.project().arrangement.lanes[0];
    assert_eq!((lane[0].start_ppq, lane[0].end_ppq), (4.0, 36.0));
    assert_eq!(lane[1].start_ppq, 36.0);
}

/// The central gesture of building a show: drag a visualizer out of the browser and drop it
/// on a session slot, or on an arrangement lane.
#[test]
fn a_visualizer_dragged_from_the_browser_becomes_a_clip() {
    // The first row of the browser ("Tunnel"), under the filter field and the two headings.
    let tunnel = Pos2::new(100.0, 149.0);
    let drag = |harness: &mut Harness, model: &EditorModel, to: Pos2| -> Vec<Command> {
        let mut commands = Vec::new();
        commands.extend(harness.pointer(model, tunnel, None).commands);
        commands.extend(harness.pointer(model, tunnel, Some(true)).commands);
        for step in 1..=5 {
            let pos = tunnel + (to - tunnel) * (step as f32 / 5.0);
            commands.extend(harness.pointer(model, pos, None).commands);
        }
        commands.extend(harness.pointer(model, to, None).commands);
        commands.extend(harness.pointer(model, to, Some(false)).commands);
        commands
    };

    // Onto the empty slot of layer 3, scene 1.
    let mut model = demo_project();
    model.apply(Command::SelectView(View::Session)).expect("session");
    let mut harness = Harness::new((1280.0, 800.0));
    harness.idle(&model);
    harness.idle(&model);
    let column = (1280.0 - 212.0 - 148.0 - 30.0 - 8.0) / 4.0;
    let slot = Pos2::new(212.0 + column * 2.0 + 100.0, 56.0 + 58.0 + 22.0);
    let commands = drag(&mut harness, &model, slot);
    assert_eq!(
        commands,
        vec![Command::SetCell {
            layer: 2,
            scene: 0,
            source: ClipSource::Visualizer("tunnel".to_string()),
        }]
    );
    let clips = model.project().clips.len();
    for command in commands {
        model.apply(command).expect("the drop applies");
    }
    assert_eq!(model.project().clips.len(), clips + 1);

    // Onto lane 2 of the arrangement, in bar 11: a four-bar block starting on that bar line.
    model.apply(Command::SelectView(View::Arrangement)).expect("arrangement");
    let mut harness = Harness::new((1280.0, 800.0));
    harness.state.follow = false;
    harness.idle(&model);
    harness.idle(&model);
    let lane = Pos2::new(
        212.0 + 178.0 + 14.0 * (10.0 * 4.0 + 1.5),
        56.0 + 44.0 + 62.0 + 31.0,
    );
    let commands = drag(&mut harness, &model, lane);
    assert_eq!(
        commands,
        vec![Command::AddSpan {
            lane: 1,
            source: ClipSource::Visualizer("tunnel".to_string()),
            start_ppq: 40.0,
            len_ppq: None,
            snap: false,
        }]
    );
}

/// A layer header's controls stay inside their column and off each other, with any number of
/// layers, in both views, at the sizes the window is actually used at. 833 points wide is the
/// control window on the reference laptop (58% of a 2560 px panel at 175%), where the first
/// windowed run showed the opacity control cut off by the stop button.
#[test]
fn layer_headers_fit_their_columns_from_two_to_eight_layers() {
    use mosh_app::editor::ui::{
        MIN_COLUMN, TWO_ROW_COLUMN, column_width, header_height, header_layout,
    };

    // The layout itself: the narrowest column (three rows), just under and at the width where
    // it becomes two rows, and a roomy one.
    for width in [
        MIN_COLUMN - 2.0,
        118.0,
        TWO_ROW_COLUMN - 3.0,
        TWO_ROW_COLUMN - 2.0,
        178.0,
        238.0,
    ] {
        let column = Rect::from_min_size(
            Pos2::new(40.0, 60.0),
            Vec2::new(width, header_height(width) - 2.0),
        );
        let rects = header_layout(column);
        let all = rects.all();
        for (index, rect) in all.iter().enumerate() {
            assert!(column.contains_rect(*rect), "{width}: control {index} {rect:?}");
            assert!(rect.width() >= 20.0 && rect.height() >= 20.0, "{width}: {rect:?}");
            for other in &all[index + 1..] {
                assert!(!rect.intersects(*other), "{width}: {rect:?} overlaps {other:?}");
            }
        }
        // The pickers get what the fixed controls leave, and it is enough for a short label.
        assert!(rects.kind.width() >= 64.0, "{width}: kind {:?}", rects.kind);
        assert!(rects.blend.width() >= 64.0, "{width}: blend {:?}", rects.blend);
        assert!(rects.opacity.width() >= 48.0, "{width}: room for \"100%\"");
        // Three rows below the threshold, two from it on.
        let rows = if rects.stop.top() > rects.blend.top() { 3 } else { 2 };
        assert_eq!(rows, if width < TWO_ROW_COLUMN - 2.0 { 3 } else { 2 }, "{width}");
    }

    for layers in 2..=8usize {
        let mut model = EditorModel::new(default_project());
        while model.project().session.layers.len() < layers {
            model.apply(Command::AddLayer).expect("layer");
        }
        // The widest labels there are: every kind and blend mode appears.
        for layer in 0..layers {
            let kinds = [
                LayerKind::Visualizer,
                LayerKind::Adjustment,
                LayerKind::Overlay,
                LayerKind::Mask,
            ];
            let blends = [
                BlendMode::LumaKey,
                BlendMode::Multiply,
                BlendMode::Normal,
                BlendMode::Screen,
            ];
            model
                .apply(Command::SetLayerKind {
                    layer,
                    kind: kinds[layer % 4],
                })
                .expect("kind");
            model
                .apply(Command::SetLayerBlend {
                    layer,
                    blend: blends[layer % 4],
                })
                .expect("blend");
        }
        for width in [833.0, 1280.0, 1920.0] {
            for view in [View::Session, View::Arrangement] {
                model.apply(Command::SelectView(view)).expect("view");
                let mut harness = Harness::new((width, 827.0));
                harness.idle(&model);
                harness.idle(&model);
                let what = format!("{layers} layers, {width} wide, {view:?}");
                let drawn = harness.state.header_probe.clone();
                assert_eq!(drawn.len(), layers, "{what}: one header per layer");
                if view == View::Session {
                    let column = column_width(width - 212.0, layers, layers < 8);
                    assert!(column >= MIN_COLUMN, "{what}: column {column}");
                    assert!(
                        (drawn[0].column.width() - (column - 2.0)).abs() < 0.01,
                        "{what}: {:?}",
                        drawn[0].column
                    );
                }
                for header in &drawn {
                    let column = header.column.expand(0.5);
                    // What was planned is in the column...
                    for rect in header.planned.all() {
                        assert!(column.contains_rect(rect), "{what}: {rect:?} in {column:?}");
                    }
                    // ...and what the pickers actually drew stayed in their rectangles: a long
                    // label ("Adjustment", "Luma key") is shortened, not allowed to push the
                    // opacity bar or the stop button out.
                    for (name, actual, planned) in [
                        ("kind", header.kind, header.planned.kind),
                        ("blend", header.blend, header.planned.blend),
                    ] {
                        assert!(
                            planned.expand(1.0).contains_rect(actual),
                            "{what}: layer {} {name} drew {actual:?}, planned {planned:?}",
                            header.layer + 1
                        );
                    }
                    assert!(!header.kind.intersects(header.planned.stop), "{what}");
                    assert!(!header.blend.intersects(header.planned.opacity), "{what}");
                }
            }
        }
    }
}

/// The second windowed run: the stress project's four layers in the control window as it opens
/// on the reference laptop (1485x1516 px at 175% is 833 points wide, the browser open). Only
/// three layer columns were in view and the video overlay's needed a sideways scroll. Every
/// layer has to be in view at this width, with the scene column beside them.
#[test]
fn four_layers_are_all_in_view_beside_the_browser_on_the_laptop() {
    use mosh_app::editor::ui::{MIN_COLUMN, column_width, scene_width};

    let view = 833.0 - 212.0;
    let scenes = scene_width(view);
    for layers in 1..=4usize {
        let column = column_width(view, layers, true);
        // The columns, the add-layer button and the scene column fit the view: no scroll.
        assert!(
            column * layers as f32 + 30.0 + scenes <= view,
            "{layers} layers: {column} each + {scenes}"
        );
        assert!(column >= MIN_COLUMN);
    }
    assert_eq!(scenes, 104.0, "the scene column gives way at this width");
    assert!((column_width(view, 4, true) - 119.75).abs() < 0.01);
    // A wide view keeps the wide columns and the roomy scene column it had.
    assert_eq!(scene_width(1068.0), 148.0);
    assert_eq!(column_width(1068.0, 4, true), 220.5);

    let mut model = EditorModel::new(default_project());
    model.apply(Command::AddLayer).expect("layer 3");
    model.apply(Command::AddLayer).expect("layer 4");
    model
        .apply(Command::SetLayerKind {
            layer: 3,
            kind: LayerKind::Overlay,
        })
        .expect("the video overlay");
    model
        .apply(Command::SetLayerBlend {
            layer: 3,
            blend: BlendMode::LumaKey,
        })
        .expect("luma key");
    model
        .apply(Command::SetCell {
            layer: 3,
            scene: 0,
            source: ClipSource::Visualizer("overlay".to_string()),
        })
        .expect("the overlay clip");
    let mut harness = Harness::new((833.0, 827.0));
    harness.idle(&model);
    harness.idle(&model);
    let drawn = harness.state.header_probe.clone();
    assert_eq!(drawn.len(), 4);
    let scenes_left = 833.0 - scenes;
    for header in &drawn {
        let column = header.column;
        assert!(
            column.left() >= 212.0 && column.right() <= scenes_left,
            "layer {} is not wholly in view: {column:?} (browser to 212, scenes from {scenes_left})",
            header.layer + 1
        );
        // In its narrow, three-row form every control still has its place and stays in it.
        for rect in header.planned.all() {
            assert!(column.expand(0.5).contains_rect(rect), "{rect:?} in {column:?}");
        }
        assert!(header.planned.kind.expand(1.0).contains_rect(header.kind));
        assert!(header.planned.blend.expand(1.0).contains_rect(header.blend));
    }
    // Launching the fourth layer's slot works where it now is: its triangle is in view.
    let column = column_width(view, 4, true);
    let header = mosh_app::editor::ui::header_height(column - 2.0);
    let at = Pos2::new(212.0 + column * 3.0 + 14.0, 56.0 + header + 22.0);
    harness.pointer(&model, at, None);
    harness.pointer(&model, at, Some(true));
    let out = harness.pointer(&model, at, Some(false));
    assert!(
        out.commands.contains(&Command::Launch { layer: 3, scene: 0 }),
        "{:?}",
        out.commands
    );
}

/// The detail panel never cuts a column off: wide enough, all four columns side by side;
/// narrower, the clip column and one page, each of which fits.
#[test]
fn the_detail_panel_pages_instead_of_cutting_its_effects_off() {
    use mosh_app::editor::ui::{DetailPage, detail_plan};

    // The panel's inner width in the laptop's control window, browser open: 833 - 212 - 16.
    let laptop = detail_plan(605.0);
    assert!(laptop.paged);
    for page in [DetailPage::Parameters, DetailPage::Effects] {
        assert!(laptop.used(page) <= 605.0, "{page:?}: {} of 605", laptop.used(page));
    }
    // From the smallest window (800 wide, browser open: 572) up, whatever is shown fits.
    for width in (572..=2000).step_by(4) {
        let plan = detail_plan(width as f32);
        for page in [DetailPage::Parameters, DetailPage::Effects] {
            assert!(
                plan.used(page) <= width as f32 + 0.01,
                "{width}: {page:?} needs {}",
                plan.used(page)
            );
        }
        assert_eq!(plan.paged, width < 1052, "{width}");
    }

    // Both pages build at the laptop's size, the switch is in the clip column, and a click on
    // "Effects" turns the page (it is widget state: no command, nothing lands on the show).
    let model = demo_project();
    let mut harness = Harness::new((833.0, 827.0));
    harness.idle(&model);
    assert_eq!(harness.state.detail_page, DetailPage::Parameters);
    let before = harness.idle(&model).primitives.len();
    harness.state.detail_page = DetailPage::Effects;
    let out = harness.idle(&model);
    assert!(out.commands.is_empty());
    assert!(!out.primitives.is_empty() && out.primitives.len() != before);
}

/// Eight groups and some forty scenes: a group folds away on a click on its heading, and a
/// filter shows matches in every group, folded or not.
#[test]
fn browser_groups_fold_away() {
    let model = demo_project();
    let groups: Vec<&str> = {
        let mut names: Vec<&str> = mosh_app::editor::model::catalog()
            .iter()
            .map(|entry| entry.group)
            .collect();
        names.dedup();
        names
    };
    println!("catalog groups: {groups:?}");
    assert!(groups.len() >= 3, "{groups:?}");
    let vertices = |out: &FrameOut| -> usize {
        out.primitives
            .iter()
            .map(|primitive| match &primitive.primitive {
                egui::epaint::Primitive::Mesh(mesh) => mesh.vertices.len(),
                egui::epaint::Primitive::Callback(_) => 0,
            })
            .sum()
    };
    // Tall enough to fit every open browser group without scroll-clipping the bottom packs
    // (otherwise folding a top group pulls clipped rows into view).
    let mut harness = Harness::new((1280.0, 2200.0));
    // The retro looks start folded away; everything else starts open.
    assert_eq!(
        harness.state.folded,
        vec![mosh_visuals::catalog::LEGACY.to_string()]
    );
    assert_eq!(groups.last().copied(), Some(mosh_visuals::catalog::LEGACY));
    harness.idle(&model);
    let open = vertices(&harness.idle(&model));

    // A click on the first group's heading ("Classic", under the filter field and the section
    // heading) folds it: its rows are gone and the next group moves up.
    let heading = Pos2::new(100.0, 124.0);
    harness.pointer(&model, heading, None);
    harness.pointer(&model, heading, Some(true));
    harness.pointer(&model, heading, Some(false));
    assert_eq!(
        harness.state.folded,
        vec![mosh_visuals::catalog::LEGACY.to_string(), groups[0].to_string()]
    );
    harness.idle(&model);
    let folded = vertices(&harness.idle(&model));
    assert!(folded < open, "{folded} vertices folded, {open} open");

    // A filter looks inside folded groups too.
    harness.state.filter = "tunnel".to_string();
    harness.idle(&model);
    let filtered = vertices(&harness.idle(&model));
    harness.state.filter = "no such scene".to_string();
    harness.idle(&model);
    let nothing = vertices(&harness.idle(&model));
    assert!(filtered > nothing, "the folded group's match is shown");
}

/// What one frame of the control window costs on this machine, with no window: layout and
/// tessellation, no GL, no swapchain. "Idle" is a frame in which nothing changed; "busy" is a
/// frame of a running show (the playhead moved, the meters changed, a slot is blinking).
/// Run with `--release -- --nocapture` for the numbers the README quotes.
#[test]
fn a_frame_costs_a_number() {
    let mut model = demo_project();
    let mut worst: f64 = 0.0;
    for size in [(1280.0, 800.0), (1920.0, 1080.0)] {
        for view in [View::Arrangement, View::Session] {
            model.apply(Command::SelectView(view)).expect("view");
            let mut harness = Harness::new(size);
            for _ in 0..8 {
                harness.idle(&model);
                harness.deliver_thumbs();
            }
            let runs = 120;
            let started = Instant::now();
            for _ in 0..runs {
                let out = harness.idle(&model);
                std::hint::black_box(out.primitives.len());
            }
            let idle = started.elapsed().as_secs_f64() * 1_000.0 / f64::from(runs);

            let started = Instant::now();
            for frame in 0..runs {
                let status = busy_status(&model, 32.0 + f64::from(frame) * 0.07);
                model.set_status(status);
                let out = harness.idle(&model);
                std::hint::black_box(out.primitives.len());
            }
            let busy = started.elapsed().as_secs_f64() * 1_000.0 / f64::from(runs);
            model.set_status(Status::default());
            println!(
                "control window frame, {}x{} {view:?}, browser + detail open, headless egui: idle {idle:.3} ms, busy {busy:.3} ms",
                size.0, size.1
            );
            worst = worst.max(idle).max(busy);
        }
    }
    // Ten frames a second of a running show must stay a rounding error next to the stage: the
    // bound is loose because this also runs unoptimised.
    assert!(worst < 12.0, "one frame took {worst:.3} ms");
}

// ---------------------------------------------------------------------------------------------
// Screenshots without a window: rasterise egui's triangles on the CPU and write PNGs.
// ---------------------------------------------------------------------------------------------

struct Canvas {
    width: usize,
    height: usize,
    /// Premultiplied RGBA, 0..1.
    pixels: Vec<[f32; 4]>,
}

impl Canvas {
    fn new(width: usize, height: usize) -> Canvas {
        let background = theme::BG;
        Canvas {
            width,
            height,
            pixels: vec![
                [
                    f32::from(background.r()) / 255.0,
                    f32::from(background.g()) / 255.0,
                    f32::from(background.b()) / 255.0,
                    1.0
                ];
                width * height
            ],
        }
    }

    fn draw(
        &mut self,
        primitives: &[egui::ClippedPrimitive],
        textures: &HashMap<egui::TextureId, (usize, usize, Vec<egui::Color32>)>,
    ) {
        for clipped in primitives {
            let egui::epaint::Primitive::Mesh(mesh) = &clipped.primitive else {
                continue;
            };
            let Some((tex_w, tex_h, texels)) = textures.get(&mesh.texture_id) else {
                continue;
            };
            let sample = |uv: Pos2| -> [f32; 4] {
                let x = (uv.x * *tex_w as f32 - 0.5).clamp(0.0, *tex_w as f32 - 1.0);
                let y = (uv.y * *tex_h as f32 - 0.5).clamp(0.0, *tex_h as f32 - 1.0);
                let (x0, y0) = (x.floor() as usize, y.floor() as usize);
                let (x1, y1) = ((x0 + 1).min(tex_w - 1), (y0 + 1).min(tex_h - 1));
                let (fx, fy) = (x - x0 as f32, y - y0 as f32);
                let texel = |x: usize, y: usize| {
                    let c = texels[y * tex_w + x];
                    [
                        f32::from(c.r()) / 255.0,
                        f32::from(c.g()) / 255.0,
                        f32::from(c.b()) / 255.0,
                        f32::from(c.a()) / 255.0,
                    ]
                };
                let (a, b, c, d) = (texel(x0, y0), texel(x1, y0), texel(x0, y1), texel(x1, y1));
                std::array::from_fn(|i| {
                    (a[i] * (1.0 - fx) + b[i] * fx) * (1.0 - fy) + (c[i] * (1.0 - fx) + d[i] * fx) * fy
                })
            };
            let clip = clipped.clip_rect;
            for triangle in mesh.indices.as_chunks::<3>().0 {
                let v: [&egui::epaint::Vertex; 3] =
                    std::array::from_fn(|i| &mesh.vertices[triangle[i] as usize]);
                let area = edge(v[0].pos, v[1].pos, v[2].pos);
                if area.abs() < 1e-6 {
                    continue;
                }
                let min_x = v.iter().map(|v| v.pos.x).fold(f32::MAX, f32::min).max(clip.left()).max(0.0);
                let max_x = v.iter().map(|v| v.pos.x).fold(f32::MIN, f32::max).min(clip.right());
                let min_y = v.iter().map(|v| v.pos.y).fold(f32::MAX, f32::min).max(clip.top()).max(0.0);
                let max_y = v.iter().map(|v| v.pos.y).fold(f32::MIN, f32::max).min(clip.bottom());
                let x_range = (min_x.floor() as usize)..(max_x.ceil().max(0.0) as usize).min(self.width);
                let y_range = (min_y.floor() as usize)..(max_y.ceil().max(0.0) as usize).min(self.height);
                let colors: [[f32; 4]; 3] = std::array::from_fn(|i| {
                    let c = v[i].color;
                    [
                        f32::from(c.r()) / 255.0,
                        f32::from(c.g()) / 255.0,
                        f32::from(c.b()) / 255.0,
                        f32::from(c.a()) / 255.0,
                    ]
                });
                for y in y_range {
                    for x in x_range.clone() {
                        let p = Pos2::new(x as f32 + 0.5, y as f32 + 0.5);
                        if !clip.contains(p) {
                            continue;
                        }
                        let w0 = edge(v[1].pos, v[2].pos, p) / area;
                        let w1 = edge(v[2].pos, v[0].pos, p) / area;
                        let w2 = edge(v[0].pos, v[1].pos, p) / area;
                        if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                            continue;
                        }
                        let uv = Pos2::new(
                            v[0].uv.x * w0 + v[1].uv.x * w1 + v[2].uv.x * w2,
                            v[0].uv.y * w0 + v[1].uv.y * w1 + v[2].uv.y * w2,
                        );
                        let texel = sample(uv);
                        let pixel = &mut self.pixels[y * self.width + x];
                        let src: [f32; 4] = std::array::from_fn(|i| {
                            (colors[0][i] * w0 + colors[1][i] * w1 + colors[2][i] * w2) * texel[i]
                        });
                        for i in 0..4 {
                            pixel[i] = src[i] + pixel[i] * (1.0 - src[3]);
                        }
                    }
                }
            }
        }
    }

    fn png(&self) -> Vec<u8> {
        let mut raw = Vec::with_capacity((self.width * 3 + 1) * self.height);
        for row in self.pixels.chunks_exact(self.width) {
            raw.push(0);
            for pixel in row {
                for channel in &pixel[..3] {
                    raw.push((channel.clamp(0.0, 1.0) * 255.0).round() as u8);
                }
            }
        }
        let mut encoder =
            flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
        encoder.write_all(&raw).expect("deflate");
        let data = encoder.finish().expect("deflate");
        let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        let mut chunk = |kind: &[u8; 4], body: &[u8]| {
            out.extend_from_slice(&(body.len() as u32).to_be_bytes());
            let mut crc = flate2::Crc::new();
            crc.update(kind);
            crc.update(body);
            out.extend_from_slice(kind);
            out.extend_from_slice(body);
            out.extend_from_slice(&crc.sum().to_be_bytes());
        };
        let mut header = Vec::new();
        header.extend_from_slice(&(self.width as u32).to_be_bytes());
        header.extend_from_slice(&(self.height as u32).to_be_bytes());
        header.extend_from_slice(&[8, 2, 0, 0, 0]);
        chunk(b"IHDR", &header);
        chunk(b"IDAT", &data);
        chunk(b"IEND", &[]);
        out
    }
}

fn edge(a: Pos2, b: Pos2, c: Pos2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Render the control window to PNG files, with a running show and synthetic thumbnails.
#[test]
fn screenshots() {
    let folder = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("ui-shots");
    std::fs::create_dir_all(&folder).expect("shots folder");
    /// One screenshot: file name, window size, view, browser open, detail open, help open.
    type Shot = (&'static str, (f32, f32), View, bool, bool, bool);
    let shots: [Shot; 9] = [
        ("session-833", (833.0, 827.0), View::Session, true, true, false),
        ("session-833-effects", (833.0, 827.0), View::Session, true, true, false),
        ("browser-packs", (1280.0, 1500.0), View::Session, true, true, false),
        ("session-1280", (1280.0, 800.0), View::Session, true, true, false),
        ("arrangement-1280", (1280.0, 800.0), View::Arrangement, true, true, false),
        ("arrangement-1920", (1920.0, 1080.0), View::Arrangement, true, true, false),
        ("session-800", (800.0, 600.0), View::Session, false, false, false),
        ("arrangement-800", (800.0, 600.0), View::Arrangement, true, true, false),
        ("help-1280", (1280.0, 800.0), View::Session, true, true, true),
    ];
    for (name, size, view, browser, detail, help) in shots {
        let mut model = demo_project();
        model.apply(Command::SelectView(view)).expect("view");
        model.apply(Command::SetBrowser(browser)).expect("browser");
        model.apply(Command::SetDetail(detail)).expect("detail");
        model.apply(Command::SetHelp(help)).expect("help");
        model.set_status(busy_status(&model, 37.5));
        let mut harness = Harness::new(size);
        if name.ends_with("-effects") {
            harness.state.detail_page = mosh_app::editor::ui::DetailPage::Effects;
        }
        harness.state.recent = vec!["C:/shows/spring tour.ron".into(), "C:/shows/demo.ron".into()];
        let preview = synthetic_picture(96, 54, 0x5e_e6_c8_40_90_c0_11);
        let ctx = harness.ctx.clone();
        harness.state.set_preview(&ctx, preview);
        let mut out = harness.idle(&model);
        for _ in 0..14 {
            harness.deliver_thumbs();
            out = harness.idle(&model);
        }
        let mut canvas = Canvas::new(size.0 as usize, size.1 as usize);
        canvas.draw(&out.primitives, &harness.textures);
        let path = folder.join(format!("{name}.png"));
        std::fs::write(&path, canvas.png()).expect("write the screenshot");
        println!("screenshot: {}", path.display());
        // The picture is not a flat rectangle: the views drew, in more than the background.
        let distinct: std::collections::HashSet<[u8; 3]> = canvas
            .pixels
            .iter()
            .step_by(97)
            .map(|pixel| std::array::from_fn(|i| (pixel[i] * 255.0) as u8))
            .collect();
        assert!(distinct.len() > 40, "{name}: only {} colours", distinct.len());
    }
}
