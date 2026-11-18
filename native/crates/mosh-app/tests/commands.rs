//! The editing commands behind the two views, with no window: every one either applies
//! completely or leaves the project exactly as it was.
//!
//! * blocks on the arrangement: create (the browser's drag-to-create), move, resize, duplicate,
//!   remove, with bar snap through the signature map and Live's "what it covers gives way";
//! * the session grid: fill a slot from a visualizer or a preset, empty it, swap two, scenes and
//!   layers added, renamed and removed;
//! * layers: opacity, blend, kind (Visualizer / Video / Adjustment / Mask);
//! * the running show: launch, scene launch, stop, lock, back to the arrangement, and the live
//!   control events that make a slider feel direct;
//! * media sources (`camera:`, `text:`), undo / redo, the panels and the two views.

use mosh_app::core::{Status, arrangement_project, default_project, simple_clip};
use mosh_app::editor::model::{MAX_LAYERS, MediaKind, Panels, new_clip};
use mosh_app::editor::{ClipSource, Command, EditorError, EditorModel, SpanRef, View};
use mosh_core::{
    ArrClip, BlendMode, ControlEvent, EffectKind, EffectSlot, LayerKind, ParamRange, Project,
    ShowMode, SigChange, Transition, TransitionKind,
};

fn span(clip: u32, start_ppq: f64, end_ppq: f64) -> ArrClip {
    ArrClip {
        clip,
        start_ppq,
        end_ppq,
    }
}

/// One lane: clip 1 on bars 1-4, clip 2 on bars 5-8.
fn two_blocks() -> EditorModel {
    EditorModel::new(arrangement_project(&[(1, 0.0, 16.0), (2, 16.0, 32.0)], &[]))
}

fn lane(model: &EditorModel, index: usize) -> &[ArrClip] {
    &model.project().arrangement.lanes[index]
}

#[test]
fn a_block_moves_with_bar_snap_and_takes_its_place() {
    let mut model = two_blocks();
    assert_eq!(model.view(), View::Arrangement, "a show with blocks opens on its timeline");

    // Dragged to "a bit before bar 11": it snaps to the bar line (beat 40).
    model
        .apply(Command::MoveSpan {
            span: SpanRef { lane: 0, index: 0 },
            to_lane: 0,
            start_ppq: 39.2,
            snap: true,
        })
        .expect("move");
    assert_eq!(lane(&model, 0), [span(2, 16.0, 32.0), span(1, 40.0, 56.0)]);
    assert_eq!(model.selected_span(), Some(SpanRef { lane: 0, index: 1 }));
    assert_eq!(model.selected(), Some(1));

    // Alt (no snap): exactly where it was dropped.
    model
        .apply(Command::MoveSpan {
            span: SpanRef { lane: 0, index: 1 },
            to_lane: 0,
            start_ppq: 41.5,
            snap: false,
        })
        .expect("free move");
    assert_eq!(lane(&model, 0)[1], span(1, 41.5, 57.5));

    // Dropped onto the other block: what it covers gives way (trimmed, as in Live).
    model
        .apply(Command::MoveSpan {
            span: SpanRef { lane: 0, index: 1 },
            to_lane: 0,
            start_ppq: 24.0,
            snap: true,
        })
        .expect("overlap");
    assert_eq!(lane(&model, 0), [span(2, 16.0, 24.0), span(1, 24.0, 40.0)]);

    // Never before the start of the timeline.
    model
        .apply(Command::MoveSpan {
            span: SpanRef { lane: 0, index: 0 },
            to_lane: 0,
            start_ppq: -30.0,
            snap: true,
        })
        .expect("clamped");
    assert_eq!(lane(&model, 0)[0], span(2, 0.0, 8.0));

    // A block that does not exist, a lane that does not exist: refused, nothing changed.
    let before = model.project().clone();
    assert_eq!(
        model.apply(Command::MoveSpan {
            span: SpanRef { lane: 0, index: 9 },
            to_lane: 0,
            start_ppq: 0.0,
            snap: true,
        }),
        Err(EditorError::BadSpan(SpanRef { lane: 0, index: 9 }))
    );
    assert_eq!(
        model.apply(Command::MoveSpan {
            span: SpanRef { lane: 0, index: 0 },
            to_lane: 5,
            start_ppq: 0.0,
            snap: true,
        }),
        Err(EditorError::BadLane(5))
    );
    assert_eq!(model.project(), &before);
}

#[test]
fn a_block_moves_to_another_lane() {
    let mut model = two_blocks();
    model.apply(Command::AddLayer).expect("a second layer");
    assert_eq!(model.lane_count(), 2);
    model
        .apply(Command::MoveSpan {
            span: SpanRef { lane: 0, index: 1 },
            to_lane: 1,
            start_ppq: 8.0,
            snap: true,
        })
        .expect("to lane 2");
    assert_eq!(lane(&model, 0), [span(1, 0.0, 16.0)]);
    assert_eq!(lane(&model, 1), [span(2, 8.0, 24.0)]);
    assert_eq!(model.selected_span(), Some(SpanRef { lane: 1, index: 0 }));
}

#[test]
fn a_block_resizes_by_either_edge_and_keeps_a_bar() {
    let mut model = two_blocks();
    let first = SpanRef { lane: 0, index: 0 };

    // The right edge, dragged to beat 22.3: snaps to the bar line at beat 24, and trims its
    // neighbour.
    model
        .apply(Command::ResizeSpan {
            span: first,
            start_ppq: 0.0,
            end_ppq: 22.3,
            snap: true,
        })
        .expect("longer");
    assert_eq!(lane(&model, 0), [span(1, 0.0, 24.0), span(2, 24.0, 32.0)]);

    // The left edge, dragged in by five beats: snaps to bar 2.
    model
        .apply(Command::ResizeSpan {
            span: first,
            start_ppq: 5.0,
            end_ppq: 24.0,
            snap: true,
        })
        .expect("shorter from the left");
    assert_eq!(lane(&model, 0)[0], span(1, 4.0, 24.0));

    // The right edge dragged past the left one: the block keeps one bar, at the edge that
    // stayed.
    model
        .apply(Command::ResizeSpan {
            span: first,
            start_ppq: 4.0,
            end_ppq: 1.0,
            snap: true,
        })
        .expect("collapsed");
    assert_eq!(lane(&model, 0)[0], span(1, 4.0, 8.0));

    // Without snap the shortest block is a sixteenth.
    model
        .apply(Command::ResizeSpan {
            span: first,
            start_ppq: 4.0,
            end_ppq: 4.01,
            snap: false,
        })
        .expect("tiny");
    assert_eq!(lane(&model, 0)[0], span(1, 4.0, 4.25));
}

#[test]
fn bar_snap_follows_the_signature_map() {
    // 4/4 for four bars, then 3/4: bar lines at 16, 19, 22, ...
    let mut project = arrangement_project(&[(1, 0.0, 16.0)], &[]);
    project.arrangement.sig_map = vec![
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
    let mut model = EditorModel::new(project);
    model
        .apply(Command::MoveSpan {
            span: SpanRef { lane: 0, index: 0 },
            to_lane: 0,
            start_ppq: 20.9,
            snap: true,
        })
        .expect("move into 3/4");
    assert_eq!(lane(&model, 0)[0].start_ppq, 22.0, "the nearest 3/4 bar line");

    // A new block is four bars long *there*: 12 beats in 3/4.
    model
        .apply(Command::AddSpan {
            lane: 0,
            source: ClipSource::Visualizer("loom".to_string()),
            start_ppq: 40.0,
            len_ppq: None,
            snap: true,
        })
        .expect("new block");
    let added = lane(&model, 0)[1];
    assert_eq!((added.start_ppq, added.end_ppq), (40.0, 52.0));
}

#[test]
fn dragging_from_the_browser_makes_a_clip() {
    let mut model = EditorModel::new(default_project());
    let clips = model.project().clips.len();

    // A visualizer dropped on lane 2, a bit after bar 3.
    model
        .apply(Command::AddSpan {
            lane: 1,
            source: ClipSource::Visualizer("fluid".to_string()),
            start_ppq: 9.0,
            len_ppq: None,
            snap: true,
        })
        .expect("drop on a lane");
    let placed = lane(&model, 1)[0];
    assert_eq!((placed.start_ppq, placed.end_ppq), (8.0, 24.0), "four bars from bar 3");
    assert_eq!(model.project().clips.len(), clips + 1);
    let clip = model.selected_clip().expect("the new clip is selected");
    assert_eq!(clip.id, placed.clip);
    assert_eq!((clip.visualizer.as_str(), clip.name.as_str()), ("fluid", "Fluid"));
    assert_eq!(clip.params.p[..4], [0.5; 4], "a new clip starts in the middle of its look");
    assert_eq!(model.selected_span(), Some(SpanRef { lane: 1, index: 0 }));

    // The same visualizer dropped on an empty session slot.
    model.apply(Command::AddScene).expect("a row");
    let scene = model.project().session.scenes.len() - 1;
    model
        .apply(Command::SetCell {
            layer: 0,
            scene,
            source: ClipSource::Visualizer("camera".to_string()),
        })
        .expect("drop on a slot");
    let id = model.project().session.scenes[scene].cells[0].expect("filled");
    let clip = model.clip(id).expect("the clip");
    assert_eq!(clip.video.as_deref(), Some("camera:"), "the default camera");
    assert_eq!(model.selected(), Some(id));
    assert_eq!(model.selected_scene(), Some(scene));

    // A preset dropped on a slot: a clip with the preset's name, look and seed.
    model
        .apply(Command::SetClipParam {
            clip: id,
            index: 2,
            value: 0.9,
        })
        .expect("p2");
    model.apply(Command::SetSeed { clip: id, seed: 77 }).expect("seed");
    model
        .apply(Command::SavePreset {
            name: "close up".to_string(),
        })
        .expect("preset");
    model
        .apply(Command::SetCell {
            layer: 1,
            scene,
            source: ClipSource::Preset(0),
        })
        .expect("drop a preset");
    let from_preset = model.selected_clip().expect("the preset's clip");
    assert_ne!(from_preset.id, id);
    assert_eq!(from_preset.name, "close up");
    assert_eq!(from_preset.visualizer, "camera");
    assert_eq!((from_preset.params.p[2], from_preset.params.seed), (0.9, 77));

    // Things that cannot be dropped: an unknown visualizer, a lane or a slot that is not there.
    let before = model.project().clone();
    assert_eq!(
        model.apply(Command::AddSpan {
            lane: 0,
            source: ClipSource::Visualizer("nope".to_string()),
            start_ppq: 0.0,
            len_ppq: None,
            snap: true,
        }),
        Err(EditorError::UnknownVisualizer("nope".to_string()))
    );
    assert_eq!(
        model.apply(Command::AddSpan {
            lane: 7,
            source: ClipSource::Visualizer("fluid".to_string()),
            start_ppq: 0.0,
            len_ppq: None,
            snap: true,
        }),
        Err(EditorError::BadLane(7))
    );
    assert_eq!(
        model.apply(Command::SetCell {
            layer: 5,
            scene: 0,
            source: ClipSource::Preset(0),
        }),
        Err(EditorError::BadCell { layer: 5, scene: 0 })
    );
    assert_eq!(
        model.apply(Command::SetCell {
            layer: 0,
            scene: 0,
            source: ClipSource::Preset(4),
        }),
        Err(EditorError::UnknownPreset(4))
    );
    assert_eq!(model.project(), &before, "a refused drop leaves no clip behind");
}

#[test]
fn removing_and_duplicating_blocks() {
    let mut model = two_blocks();
    model
        .apply(Command::DuplicateSpan(SpanRef { lane: 0, index: 0 }))
        .expect("duplicate");
    // The copy sits right after the original (over what was there) and is its own clip.
    let blocks = lane(&model, 0).to_vec();
    assert_eq!(blocks.len(), 2);
    assert_eq!((blocks[1].start_ppq, blocks[1].end_ppq), (16.0, 32.0));
    assert_ne!(blocks[1].clip, 1);
    assert_eq!(model.clip(blocks[1].clip).expect("copy").visualizer, "tunnel");
    // Clip 2 was wholly covered: it left the arrangement and, used nowhere else, the project.
    assert!(model.clip(2).is_none());

    model
        .apply(Command::RemoveSpan(SpanRef { lane: 0, index: 0 }))
        .expect("remove");
    assert_eq!(lane(&model, 0).len(), 1);
    assert!(model.clip(1).is_none(), "an unused clip does not stay in the file");
    assert_eq!(model.selected_span(), None);
    assert_eq!(model.selected(), Some(blocks[1].clip), "the copy is still selected");

    // A clip that a session slot still shows survives the removal of its block.
    let mut model = EditorModel::new(default_project());
    let shared = model.project().session.scenes[0].cells[0].expect("a clip");
    model
        .apply(Command::AddSpan {
            lane: 0,
            source: ClipSource::Existing(shared),
            start_ppq: 0.0,
            len_ppq: Some(8.0),
            snap: true,
        })
        .expect("the same clip on the timeline");
    model
        .apply(Command::RemoveSpan(SpanRef { lane: 0, index: 0 }))
        .expect("remove");
    assert!(model.clip(shared).is_some());
}

#[test]
fn the_session_grid_is_edited_slot_by_slot() {
    let mut model = EditorModel::new(default_project());
    let a = model.project().session.scenes[0].cells[0].expect("a");
    let b = model.project().session.scenes[1].cells[1].expect("b");

    model
        .apply(Command::SwapCells {
            from: (0, 0),
            to: (1, 1),
        })
        .expect("swap");
    assert_eq!(model.project().session.scenes[0].cells[0], Some(b));
    assert_eq!(model.project().session.scenes[1].cells[1], Some(a));

    model
        .apply(Command::ClearCell { layer: 0, scene: 0 })
        .expect("clear");
    assert_eq!(model.project().session.scenes[0].cells[0], None);
    assert!(model.clip(b).is_none(), "the emptied slot's clip is gone");
    assert_eq!(
        model.apply(Command::Launch { layer: 0, scene: 0 }),
        Err(EditorError::BadCell { layer: 0, scene: 0 }),
        "an empty slot cannot be launched"
    );

    // Scenes: add, rename, remove; the last one stays.
    let scenes = model.project().session.scenes.len();
    model.apply(Command::AddScene).expect("add");
    model
        .apply(Command::RenameScene {
            scene: scenes,
            name: "  encore ".to_string(),
        })
        .expect("rename");
    let row = &model.project().session.scenes[scenes];
    assert_eq!(row.name, "encore");
    assert_eq!(row.cells, vec![None, None], "one empty slot per layer");
    model.apply(Command::RemoveScene(scenes)).expect("remove");
    assert_eq!(model.project().session.scenes.len(), scenes);
    assert_eq!(
        model.apply(Command::RemoveScene(99)),
        Err(EditorError::BadScene(99))
    );
    while model.project().session.scenes.len() > 1 {
        model.apply(Command::RemoveScene(0)).expect("remove");
    }
    assert_eq!(
        model.apply(Command::RemoveScene(0)),
        Err(EditorError::LastOne("scene"))
    );
}

#[test]
fn layers_have_opacity_blend_and_kind() {
    let mut model = EditorModel::new(default_project());

    // Opacity: clamped, in the project for the next bar, and on stage at once.
    let outcome = model
        .apply(Command::SetLayerOpacity {
            layer: 1,
            opacity: 1.4,
        })
        .expect("opacity");
    assert_eq!(model.project().session.layers[1].opacity, 1.0);
    assert!(outcome.project_changed);
    assert_eq!(
        outcome.live,
        vec![ControlEvent::SetOpacity {
            layer: 1,
            value: 1.0
        }]
    );

    model
        .apply(Command::SetLayerBlend {
            layer: 1,
            blend: BlendMode::Multiply,
        })
        .expect("blend");
    assert_eq!(model.project().session.layers[1].blend, BlendMode::Multiply);

    for kind in [
        LayerKind::Adjustment,
        LayerKind::Mask,
        LayerKind::Overlay,
        LayerKind::Visualizer,
    ] {
        model
            .apply(Command::SetLayerKind { layer: 0, kind })
            .expect("kind");
        assert_eq!(model.project().session.layers[0].kind, kind);
    }

    assert_eq!(
        model.apply(Command::SetLayerOpacity {
            layer: 9,
            opacity: 0.5
        }),
        Err(EditorError::BadLayer(9))
    );

    // Adding layers: a slot in every scene, a lane, and a blend that does not hide the rest.
    model.apply(Command::AddLayer).expect("third layer");
    let grid = &model.project().session;
    assert_eq!(grid.layers.len(), 3);
    assert_eq!(grid.layers[2].blend, BlendMode::Add);
    assert!(grid.scenes.iter().all(|row| row.cells.len() == 3));
    assert_eq!(model.project().arrangement.lanes.len(), 3);
    while model.project().session.layers.len() < MAX_LAYERS {
        model.apply(Command::AddLayer).expect("more layers");
    }
    assert_eq!(model.apply(Command::AddLayer), Err(EditorError::TooManyLayers));

    // Removing one takes its column, its lane and the clips only it showed.
    let gone = model.project().session.scenes[0].cells[1].expect("layer 2's clip");
    model.apply(Command::RemoveLayer(1)).expect("remove");
    let grid = &model.project().session;
    assert_eq!(grid.layers.len(), MAX_LAYERS - 1);
    assert!(grid.scenes.iter().all(|row| row.cells.len() == MAX_LAYERS - 1));
    assert!(model.clip(gone).is_none());
    while model.project().session.layers.len() > 1 {
        model.apply(Command::RemoveLayer(0)).expect("remove");
    }
    assert_eq!(
        model.apply(Command::RemoveLayer(0)),
        Err(EditorError::LastOne("layer"))
    );
}

#[test]
fn launch_stop_and_lock_reach_the_show_and_change_no_file() {
    let mut model = EditorModel::new(default_project());
    let before = model.project().clone();

    let outcome = model.apply(Command::LaunchScene(3)).expect("scene");
    assert_eq!(outcome.launch, Some(ControlEvent::LaunchScene(3)));
    assert_eq!(
        model.apply(Command::LaunchScene(99)),
        Err(EditorError::BadScene(99))
    );
    let outcome = model
        .apply(Command::Launch { layer: 1, scene: 2 })
        .expect("slot");
    assert_eq!(
        outcome.launch,
        Some(ControlEvent::LaunchClip { layer: 1, scene: 2 })
    );
    let outcome = model.apply(Command::StopLayer(1)).expect("stop");
    assert_eq!(outcome.launch, Some(ControlEvent::StopLayer(1)));
    assert_eq!(
        model.apply(Command::StopLayer(4)),
        Err(EditorError::BadLayer(4))
    );
    let outcome = model.apply(Command::Lock(true)).expect("lock");
    assert_eq!(outcome.launch, Some(ControlEvent::Lock(true)));
    let outcome = model.apply(Command::BackToArrangement).expect("back");
    assert_eq!(outcome.launch, Some(ControlEvent::BackToArrangement));

    assert_eq!(model.project(), &before);
    assert!(model.handoff().is_none(), "a launch is not an edit");
    assert!(!model.can_undo(), "and not a step of history");
}

#[test]
fn a_slider_of_a_clip_on_stage_moves_the_stage_at_once() {
    let mut model = EditorModel::new(default_project());
    let clip = model.project().session.scenes[0].cells[1].expect("a clip");
    model
        .apply(Command::SetRange {
            clip,
            level: 3,
            index: 0,
            lo: 0.2,
            hi: 0.6,
        })
        .expect("range");

    // Not on stage: the edit waits for the bar like everything else.
    let outcome = model
        .apply(Command::SetClipParam {
            clip,
            index: 0,
            value: 0.5,
        })
        .expect("p0");
    assert!(outcome.live.is_empty());

    // On layer 2 at level 3: the value inside level 3's range goes to that layer now.
    let mut status = Status {
        level: 3,
        ..Status::default()
    };
    status.playing[1] = Some(clip);
    model.set_status(status);
    let outcome = model
        .apply(Command::SetClipParam {
            clip,
            index: 0,
            value: 0.5,
        })
        .expect("p0 live");
    match outcome.live.as_slice() {
        [ControlEvent::SetParam {
            layer: 1,
            index: 0,
            value,
        }] => assert!((value - 0.4).abs() < 1e-6, "{value}"),
        other => panic!("expected one live parameter on layer 2, got {other:?}"),
    }
    assert!(outcome.project_changed);
}

#[test]
fn presets_are_browsed_applied_and_deleted() {
    let mut model = EditorModel::new(default_project());
    let source = model.selected().expect("a clip");
    model
        .apply(Command::SetVisualizer {
            clip: source,
            visualizer: "text".to_string(),
        })
        .expect("text");
    model
        .apply(Command::SetVideo {
            clip: source,
            path: Some("text: hello\nworld ".to_string()),
        })
        .expect("words");
    assert_eq!(
        model.clip(source).expect("clip").video.as_deref(),
        Some("text: hello\nworld "),
        "words are kept as typed, spaces and line breaks included"
    );
    model
        .apply(Command::SavePreset {
            name: String::new(),
        })
        .expect("save");
    let presets = &model.project().presets;
    assert_eq!(presets.len(), 1);
    assert_eq!(presets[0].visualizer, "text");

    // Applied to a tunnel clip: the look changes, and the clip gets a text source to fill.
    let target = model.project().session.scenes[3].cells[0].expect("another clip");
    model.apply(Command::SelectClip(Some(target))).expect("select");
    model.apply(Command::ApplyPreset { index: 0 }).expect("apply");
    let clip = model.clip(target).expect("clip");
    assert_eq!(clip.visualizer, "text");
    assert_eq!(clip.video.as_deref(), Some("text:"));
    assert_eq!(MediaKind::of(&clip.visualizer), MediaKind::Text);

    // Back to a scene: the media source goes with the visualizer that read it.
    model
        .apply(Command::SetVisualizer {
            clip: target,
            visualizer: "tunnel".to_string(),
        })
        .expect("tunnel");
    assert_eq!(model.clip(target).expect("clip").video, None);

    model.apply(Command::DeletePreset { index: 0 }).expect("delete");
    assert!(model.project().presets.is_empty());
    assert_eq!(
        model.apply(Command::ApplyPreset { index: 0 }),
        Err(EditorError::UnknownPreset(0))
    );
    model.apply(Command::SelectClip(None)).expect("deselect");
    assert_eq!(
        model.apply(Command::SavePreset {
            name: "x".to_string()
        }),
        Err(EditorError::NoClip)
    );
}

#[test]
fn undo_steps_back_one_edit_and_one_drag_is_one_edit() {
    let mut model = two_blocks();
    let start = model.project().clone();
    assert_eq!(model.apply(Command::Undo), Err(EditorError::NothingToUndo));

    // A slider dragged across thirty frames is one step.
    for frame in 0..30 {
        model
            .apply(Command::SetClipParam {
                clip: 1,
                index: 0,
                value: frame as f32 / 30.0,
            })
            .expect("drag");
    }
    model
        .apply(Command::MoveSpan {
            span: SpanRef { lane: 0, index: 0 },
            to_lane: 0,
            start_ppq: 64.0,
            snap: true,
        })
        .expect("move");
    let moved = model.project().clone();

    model.apply(Command::Undo).expect("undo the move");
    assert_eq!(lane(&model, 0)[0], span(1, 0.0, 16.0));
    assert!(model.clip(1).expect("clip").params.p[0] > 0.9, "the drag is still there");
    model.apply(Command::Undo).expect("undo the drag");
    assert_eq!(model.project(), &start);
    assert_eq!(model.apply(Command::Undo), Err(EditorError::NothingToUndo));

    model.apply(Command::Redo).expect("redo the drag");
    model.apply(Command::Redo).expect("redo the move");
    assert_eq!(model.project(), &moved);
    assert_eq!(model.apply(Command::Redo), Err(EditorError::NothingToRedo));

    // A new edit after an undo drops the redo.
    model.apply(Command::Undo).expect("undo");
    model
        .apply(Command::RenameClip {
            clip: 1,
            name: "opener".to_string(),
        })
        .expect("rename");
    assert!(!model.can_redo());
    assert_eq!(model.clip(1).expect("clip").name, "opener");

    // The mode is the running show's state: undo does not flip it back.
    model.apply(Command::SetMode(ShowMode::Autopilot)).expect("mode");
    model.apply(Command::Undo).expect("undo the rename");
    assert_eq!(model.project().mode, ShowMode::Autopilot);
    // Every undo and redo reaches the running show like any other edit.
    assert!(model.handoff().is_some());
}

#[test]
fn views_panels_and_the_empty_show() {
    let mut model = EditorModel::new(default_project());
    assert_eq!(model.view(), View::Session, "a grid-only show opens on the grid");
    assert_eq!(
        model.panels(),
        Panels {
            browser: true,
            detail: true,
            help: false
        }
    );
    model.apply(Command::ToggleView).expect("tab");
    assert_eq!(model.view(), View::Arrangement);
    model.apply(Command::ToggleView).expect("tab");
    assert_eq!(model.view(), View::Session);
    model.apply(Command::SelectView(View::Arrangement)).expect("view");
    model.apply(Command::SetBrowser(false)).expect("browser");
    model.apply(Command::SetDetail(false)).expect("detail");
    model.apply(Command::SetHelp(true)).expect("help");
    assert_eq!(
        model.panels(),
        Panels {
            browser: false,
            detail: false,
            help: true
        }
    );
    assert!(model.handoff().is_none(), "looking around is not an edit");
    assert!(!model.modified());

    // Save needs a file; an edit marks the show as modified.
    assert_eq!(model.apply(Command::Save), Err(EditorError::NoPath));
    model.apply(Command::AddScene).expect("edit");
    assert!(model.modified());

    // A new show: one layer, one empty scene, one empty lane, no history, nothing selected.
    model.apply(Command::NewProject).expect("new");
    let project = model.project();
    assert_eq!(project.clips.len(), 0);
    assert_eq!(project.session.layers.len(), 1);
    assert_eq!(project.session.scenes.len(), 1);
    assert_eq!(project.arrangement.lanes, vec![Vec::new()]);
    assert_eq!(model.selected(), None);
    assert!(!model.can_undo());
    assert!(!model.modified());
    assert!(model.handoff().is_some(), "the running show gets the empty project");
}

#[test]
fn song_markers_move_and_rename() {
    let mut model = EditorModel::new(arrangement_project(
        &[(1, 0.0, 16.0)],
        &[("one", 0.0), ("two", 32.0)],
    ));
    model
        .apply(Command::MoveMarker {
            index: 0,
            ppq: 45.0,
            snap: true,
        })
        .expect("move");
    // The list stays sorted (the show engine relies on it): "one" is now after "two".
    let songs = &model.project().arrangement.songs;
    assert_eq!((songs[0].name.as_str(), songs[0].start_ppq), ("two", 32.0));
    assert_eq!((songs[1].name.as_str(), songs[1].start_ppq), ("one", 44.0));
    model
        .apply(Command::RenameMarker {
            index: 1,
            name: "encore".to_string(),
        })
        .expect("rename");
    assert_eq!(model.project().arrangement.songs[1].name, "encore");
    assert_eq!(
        model.apply(Command::MoveMarker {
            index: 5,
            ppq: 0.0,
            snap: true
        }),
        Err(EditorError::UnknownMarker(5))
    );
}

#[test]
fn a_new_clip_is_ready_to_draw() {
    for entry in mosh_app::editor::model::catalog() {
        let clip = new_clip(9, entry.id);
        assert_eq!(clip.visualizer, entry.id);
        assert_eq!(clip.name, entry.name);
        assert_eq!(clip.ranges, [[ParamRange::default(); 8]; 4]);
        assert!(clip.params.p.iter().all(|value| (0.0..=1.0).contains(value)));
        // The media source matches the visualizer that reads it.
        let prefix = MediaKind::of(entry.id).prefix();
        match clip.video.as_deref() {
            Some(source) => assert_eq!(source, prefix, "{}", entry.id),
            None => assert!(prefix.is_empty(), "{}", entry.id),
        }
    }
    // The catalog has the layer visualizers the browser groups under "Layers".
    let layers: Vec<&str> = mosh_app::editor::model::catalog()
        .iter()
        .filter(|entry| entry.group == "Layers")
        .map(|entry| entry.id)
        .collect();
    for id in ["fluid", "particles", "camera", "text", "overlay"] {
        assert!(layers.contains(&id), "{id} is missing from {layers:?}");
    }
    let _ = (Project::default(), simple_clip(1, "tunnel"));
}

/// A scene dragged in from the browser is a datamosh from its first frame, as the web app's
/// `newBlock` is (`mosh:.6`, `trans:2`). A media clip (a video file, the webcam, words) is left
/// clean: a mosh smears those into mush, so that stays a choice in the Effects list.
#[test]
fn a_new_clip_starts_with_the_web_mosh_default() {
    for entry in mosh_app::editor::model::catalog() {
        let clip = new_clip(9, entry.id);
        match MediaKind::of(entry.id) {
            MediaKind::None => {
                let amount = if entry.id == "fluid" { 0.5 } else { 0.6 };
                assert_eq!(
                    clip.effects,
                    vec![EffectSlot {
                        kind: EffectKind::Mosh,
                        amount,
                        param: 0.5,
                    }],
                    "{}",
                    entry.id
                );
                assert_eq!(
                    clip.transition_in,
                    Transition {
                        kind: TransitionKind::Datamosh,
                        beats: 2.0,
                    },
                    "{}",
                    entry.id
                );
            }
            MediaKind::File | MediaKind::Camera | MediaKind::Text => {
                assert!(clip.effects.is_empty(), "{}: {:?}", entry.id, clip.effects);
                assert_eq!(clip.transition_in, Transition::default(), "{}", entry.id);
            }
        }
    }
    // Both branches above are populated by the catalog, and the fluid is a scene visualizer.
    for id in ["overlay", "camera", "text"] {
        assert_ne!(MediaKind::of(id), MediaKind::None, "{id} is not media");
    }
    assert_eq!(MediaKind::of("fluid"), MediaKind::None);
}
