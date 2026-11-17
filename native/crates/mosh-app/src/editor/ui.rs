//! The control window's widgets: they read the model and return commands, and never change it.
//!
//! Keeping the egui tree in its own module is what makes the control window's cost measurable
//! without a display (`tests/editors.rs` runs this against a headless `egui::Context`), and it is
//! also the rule the task asks for: the layer here maps widgets to [`Command`]s; applying them,
//! and every decision about the running show, stays in [`super::model`] and the shell.
//!
//! Drafts (paths, names, a beat position) are widget state and live in [`EditorUi`]: typing a path
//! is not an edit to the show, and only the button that reads it turns it into a command.

use mosh_core::{EffectKind, FollowAction, LaunchQuantize, ShowMode, TransitionKind};
use mosh_shaders::VISUALIZERS;

use crate::editor::model::{Command, EditorModel, Screen, bars_at, param_labels};

/// The nine effects the renderer has a slot for, in `mosh_core::EffectKind` order.
fn effect_kinds() -> [EffectKind; 9] {
    [
        EffectKind::Fold,
        EffectKind::Mirror,
        EffectKind::Streak,
        EffectKind::Crush,
        EffectKind::Zoom,
        EffectKind::Spin,
        EffectKind::Echo,
        EffectKind::Negative,
        EffectKind::Mosh,
    ]
}

fn transition_kinds() -> [TransitionKind; 5] {
    [
        TransitionKind::Cut,
        TransitionKind::Crossfade,
        TransitionKind::Datamosh,
        TransitionKind::Melt,
        TransitionKind::Strobe,
    ]
}

fn follow_actions() -> [FollowAction; 4] {
    [
        FollowAction::None,
        FollowAction::Next,
        FollowAction::Random,
        FollowAction::RandomProcedural,
    ]
}

fn quantize_label(quantize: LaunchQuantize) -> &'static str {
    match quantize {
        LaunchQuantize::None => "off",
        LaunchQuantize::Beat => "beat",
        LaunchQuantize::Bar => "bar",
        LaunchQuantize::Phrase => "phrase",
    }
}

/// Widget state that is not part of the project: what the performer is typing.
#[derive(Clone, Debug)]
pub struct EditorUi {
    /// The `.als` to import.
    pub als_path: String,
    /// The next song marker's name (`""` takes `marker N`).
    pub marker_name: String,
    /// The next song marker's position in beats.
    pub marker_beats: f64,
    /// The name for a preset saved from the selected clip (`""` takes the clip's name).
    pub preset_name: String,
    /// Where the project is saved to, or read from.
    pub project_path: String,
    /// A path for the selected clip's overlay video.
    pub video_path: String,
}

impl Default for EditorUi {
    fn default() -> EditorUi {
        EditorUi {
            als_path: String::new(),
            marker_name: String::new(),
            marker_beats: 0.0,
            preset_name: String::new(),
            project_path: "show.ron".to_string(),
            video_path: String::new(),
        }
    }
}

/// Draw the whole control window and return what the performer asked for.
///
/// One frame of widgets, one batch of commands: the caller applies them in order, so a screen that
/// emits several (a clip selected and its parameter dragged in the same frame) cannot interleave.
pub fn draw(root: &mut egui::Ui, state: &mut EditorUi, model: &EditorModel) -> Vec<Command> {
    let mut out = Vec::new();

    egui::Panel::top("editor_tabs").show(root, |ui| tabs(ui, model, &mut out));
    egui::Panel::bottom("editor_status").show(root, |ui| status_strip(ui, model));
    egui::CentralPanel::default().show(root, |ui| match model.screen() {
        Screen::Session => session(ui, model, &mut out),
        Screen::Arrangement => arrangement(ui, state, model, &mut out),
        Screen::Clip => clip_editor(ui, state, model, &mut out),
        Screen::Presets => presets(ui, state, model, &mut out),
        Screen::Project => project(ui, state, model, &mut out),
    });

    out
}

/// The screen tabs, and one line about the project.
fn tabs(ui: &mut egui::Ui, model: &EditorModel, out: &mut Vec<Command>) {
    ui.horizontal_wrapped(|ui| {
        for screen in Screen::ALL {
            if ui
                .selectable_label(model.screen() == screen, screen.label())
                .clicked()
            {
                out.push(Command::SelectScreen(screen));
            }
        }
        ui.separator();
        let project = model.project();
        ui.label(format!(
            "{} clips · {} scenes · {} presets{}",
            project.clips.len(),
            project.session.scenes.len(),
            project.presets.len(),
            if model.dirty() { " · edited" } else { "" }
        ));
    });
}

/// Source, tempo, bar, level and the two frame times, always visible.
fn status_strip(ui: &mut egui::Ui, model: &EditorModel) {
    let status = model.status();
    ui.horizontal_wrapped(|ui| {
        ui.label(format!(
            "{} · {} · {:.1} BPM · bar {} ({:.0}%) · L{} · GPU {:.2} ms · CPU {:.2} ms",
            status.mode,
            status.source,
            status.tempo_bpm,
            status.bar + 1,
            status.bar_phase * 100.0,
            status.level,
            status.gpu_ms,
            status.cpu_ms,
        ));
    });
    let message = model
        .message()
        .unwrap_or("1/2/3 mode · Esc quits · editing lands on the next bar");
    ui.label(message);
}

/// Layers x scenes: click a clip to edit it, launch a cell to look at it on stage next bar.
fn session(ui: &mut egui::Ui, model: &EditorModel, out: &mut Vec<Command>) {
    let grid = &model.project().session;
    ui.label(format!(
        "{} layers x {} scenes · launches quantized to {}",
        grid.layers.len(),
        grid.scenes.len(),
        quantize_label(grid.quantize)
    ));
    egui::ScrollArea::both().show(ui, |ui| {
        egui::Grid::new("session_grid")
            .striped(true)
            .min_col_width(96.0)
            .show(ui, |ui| {
                ui.label("");
                for (layer, config) in grid.layers.iter().enumerate() {
                    ui.label(format!(
                        "layer {layer} · {:?} {:.0}%",
                        config.blend,
                        config.opacity * 100.0
                    ));
                }
                ui.end_row();

                for (scene, row) in grid.scenes.iter().enumerate() {
                    if ui.button(format!("> {}", row.name)).clicked() {
                        out.push(Command::LaunchScene(scene));
                    }
                    for (layer, cell) in row.cells.iter().enumerate() {
                        let name = cell
                            .and_then(|id| {
                                model
                                    .project()
                                    .clips
                                    .iter()
                                    .find(|clip| clip.id == id)
                                    .map(|clip| format!("{} · {}", clip.name, clip.visualizer))
                            })
                            .unwrap_or_else(|| "·".to_string());
                        let selected = cell.is_some() && *cell == model.selected();
                        let clicked = ui.selectable_label(selected, name).clicked();
                        if clicked
                            && let Some(id) = cell
                        {
                            out.push(Command::SelectClip(Some(*id)));
                            out.push(Command::SelectScene(scene));
                        }
                        if ui
                            .add(egui::Button::new(">").small())
                            .on_hover_text("launch this cell on the next bar")
                            .clicked()
                        {
                            out.push(Command::Launch { layer, scene });
                        }
                    }
                    ui.end_row();
                }
            });
    });
}

/// The timeline in bars: lanes per layer, the song markers and the Live set import.
fn arrangement(
    ui: &mut egui::Ui,
    state: &mut EditorUi,
    model: &EditorModel,
    out: &mut Vec<Command>,
) {
    let project = model.project();
    let arrangement = &project.arrangement;
    let sig_map = &arrangement.sig_map;

    ui.horizontal(|ui| {
        ui.label(".als");
        ui.text_edit_singleline(&mut state.als_path);
        let ready = !state.als_path.trim().is_empty();
        if ui
            .add_enabled(ready, egui::Button::new("Import Live set (.als)"))
            .clicked()
        {
            out.push(Command::ImportAls {
                path: state.als_path.trim().to_string(),
            });
        }
    });
    ui.label("fills the markers and the signature map from Live's locators and master signature");

    ui.separator();
    ui.horizontal(|ui| {
        ui.label("marker");
        ui.text_edit_singleline(&mut state.marker_name);
        ui.label("at beat");
        ui.add(
            egui::DragValue::new(&mut state.marker_beats)
                .speed(0.5)
                .range(0.0..=8192.0),
        );
        if ui.button("Add marker").clicked() {
            out.push(Command::AddMarker {
                name: state.marker_name.clone(),
                ppq: state.marker_beats,
            });
            state.marker_name.clear();
        }
    });

    ui.separator();
    let total = total_beats(arrangement);
    if arrangement.songs.is_empty() {
        ui.label("no song markers: import a Live set, or add one");
    }
    for (index, song) in arrangement.songs.iter().enumerate() {
        ui.horizontal(|ui| {
            ui.label(format!(
                "{}. {} · bar {:.2} · beat {:.0}",
                index + 1,
                song.name,
                bars_at(sig_map, song.start_ppq),
                song.start_ppq
            ));
            if ui.button("remove").clicked() {
                out.push(Command::RemoveMarker { index });
            }
        });
    }

    ui.separator();
    ui.label(format!(
        "0 to {:.0} bars · click a clip to edit it",
        bars_at(sig_map, total)
    ));
    for (layer, lane) in arrangement.lanes.iter().enumerate() {
        ui.label(format!("layer {layer}"));
        let (response, painter) = ui.allocate_painter(
            egui::vec2(ui.available_width().max(240.0), 20.0),
            egui::Sense::click(),
        );
        let rect = response.rect;
        painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
        for span in lane {
            let x0 = rect.left() + (span.start_ppq / total) as f32 * rect.width();
            let x1 = rect.left() + (span.end_ppq / total) as f32 * rect.width();
            let bar = egui::Rect::from_min_max(
                egui::pos2(x0, rect.top()),
                egui::pos2(x1.max(x0 + 2.0), rect.bottom()),
            );
            let selected = Some(span.clip) == model.selected();
            let fill = if selected {
                ui.visuals().selection.bg_fill
            } else {
                ui.visuals().widgets.active.bg_fill
            };
            painter.rect_filled(bar, 2.0, fill);
            let name = project
                .clips
                .iter()
                .find(|clip| clip.id == span.clip)
                .map(|clip| clip.name.clone())
                .unwrap_or_else(|| format!("clip {}", span.clip));
            painter.text(
                egui::pos2(x0 + 3.0, rect.center().y),
                egui::Align2::LEFT_CENTER,
                name,
                egui::FontId::proportional(10.0),
                ui.visuals().text_color(),
            );
        }
        for song in &arrangement.songs {
            let x = rect.left() + (song.start_ppq / total) as f32 * rect.width();
            painter.line_segment(
                [
                    egui::pos2(x, rect.top()),
                    egui::pos2(x, rect.bottom()),
                ],
                egui::Stroke::new(1.0, ui.visuals().warn_fg_color),
            );
        }
        if response.clicked()
            && let Some(pointer) = response.interact_pointer_pos()
        {
            let ppq = f64::from((pointer.x - rect.left()) / rect.width().max(1.0)) * total;
            if let Some(span) = lane.iter().find(|span| span.start_ppq <= ppq && ppq < span.end_ppq) {
                out.push(Command::SelectClip(Some(span.clip)));
            }
        }
    }
}

/// One clip: visualizer, parameters, ranges per level, effects, transition, follow action, video.
fn clip_editor(
    ui: &mut egui::Ui,
    state: &mut EditorUi,
    model: &EditorModel,
    out: &mut Vec<Command>,
) {
    let Some(clip) = model.selected_clip() else {
        ui.label("select a clip in Session or Arrangement first");
        return;
    };
    let id = clip.id;

    ui.horizontal_wrapped(|ui| {
        ui.heading(clip.name.clone());
        ui.label(format!(
            "#{} · {} · seed {}",
            clip.id, clip.visualizer, clip.params.seed
        ));
    });

    ui.horizontal_wrapped(|ui| {
        ui.label("visualizer");
        for info in VISUALIZERS {
            if ui
                .selectable_label(clip.visualizer == info.id, info.name)
                .clicked()
            {
                out.push(Command::SetVisualizer {
                    clip: id,
                    visualizer: info.id.to_string(),
                });
            }
        }
    });
    if let Some(info) = VISUALIZERS.iter().find(|info| info.id == clip.visualizer) {
        ui.label(format!("p0..p7: {}", info.params.join(", ")));
    }

    egui::CollapsingHeader::new("parameters")
        .default_open(true)
        .show(ui, |ui| {
            let labels = param_labels(&clip.visualizer);
            for (index, label) in labels.iter().enumerate() {
                let mut value = clip.params.p[index];
                let changed = ui
                    .add(
                        egui::Slider::new(&mut value, 0.0..=1.0)
                            .text(format!("p{index} {label}")),
                    )
                    .changed();
                if changed {
                    out.push(Command::SetClipParam {
                        clip: id,
                        index,
                        value,
                    });
                }
            }
            let mut seed = clip.params.seed;
            if ui
                .add(egui::DragValue::new(&mut seed).speed(1.0).prefix("seed "))
                .changed()
            {
                out.push(Command::SetSeed { clip: id, seed });
            }
            ui.label("the seed fixes a procedural look; a preset carries it");
        });

    egui::CollapsingHeader::new("ranges per intensity level")
        .default_open(false)
        .show(ui, |ui| {
            for level in 1..=4u8 {
                let slot = usize::from(level - 1);
                ui.label(format!("level {level}"));
                let mut depth = clip.pulse_depth[slot];
                if ui
                    .add(egui::Slider::new(&mut depth, 0.0..=1.0).text("pulse depth"))
                    .changed()
                {
                    out.push(Command::SetPulseDepth {
                        clip: id,
                        level,
                        depth,
                    });
                }
                let labels = param_labels(&clip.visualizer);
                for (index, label) in labels.iter().enumerate() {
                    let range = clip.ranges[slot][index];
                    let (mut lo, mut hi) = (range.lo, range.hi);
                    ui.horizontal(|ui| {
                        ui.label(format!("p{index} {label}"));
                        let a = ui.add(
                            egui::DragValue::new(&mut lo)
                                .speed(0.01)
                                .range(0.0..=1.0)
                                .prefix("lo "),
                        );
                        let b = ui.add(
                            egui::DragValue::new(&mut hi)
                                .speed(0.01)
                                .range(0.0..=1.0)
                                .prefix("hi "),
                        );
                        if a.changed() || b.changed() {
                            out.push(Command::SetRange {
                                clip: id,
                                level,
                                index,
                                lo,
                                hi,
                            });
                        }
                    });
                }
            }
        });

    egui::CollapsingHeader::new(format!("effects ({})", clip.effects.len()))
        .default_open(true)
        .show(ui, |ui| {
            for (index, slot) in clip.effects.iter().enumerate() {
                let (mut amount, mut param) = (slot.amount, slot.param);
                ui.horizontal(|ui| {
                    ui.label(format!("{:?}", slot.kind));
                    let a = ui.add(
                        egui::Slider::new(&mut amount, 0.0..=1.0)
                            .text("amount")
                            .show_value(false),
                    );
                    let b = ui.add(
                        egui::Slider::new(&mut param, 0.0..=1.0)
                            .text("param")
                            .show_value(false),
                    );
                    if a.changed() || b.changed() {
                        out.push(Command::SetEffect {
                            clip: id,
                            index,
                            amount,
                            param,
                        });
                    }
                    if ui.button("remove").clicked() {
                        out.push(Command::RemoveEffect { clip: id, index });
                    }
                });
            }
            ui.horizontal_wrapped(|ui| {
                ui.label("add");
                for kind in effect_kinds() {
                    let present = clip.effects.iter().any(|slot| slot.kind == kind);
                    if ui
                        .add_enabled(!present, egui::Button::new(format!("{kind:?}")))
                        .clicked()
                    {
                        out.push(Command::AddEffect { clip: id, kind });
                    }
                }
            });
        });

    ui.separator();
    ui.horizontal_wrapped(|ui| {
        ui.label("transition in");
        for kind in transition_kinds() {
            if ui
                .selectable_label(clip.transition_in.kind == kind, format!("{kind:?}"))
                .clicked()
            {
                out.push(Command::SetTransition {
                    clip: id,
                    kind,
                    beats: clip.transition_in.beats,
                });
            }
        }
        let mut beats = clip.transition_in.beats;
        if ui
            .add(
                egui::DragValue::new(&mut beats)
                    .speed(0.25)
                    .range(0.0..=64.0)
                    .prefix("beats "),
            )
            .changed()
        {
            out.push(Command::SetTransition {
                clip: id,
                kind: clip.transition_in.kind,
                beats,
            });
        }
    });

    ui.horizontal_wrapped(|ui| {
        ui.label("follow");
        for action in follow_actions() {
            if ui
                .selectable_label(clip.follow == action, format!("{action:?}"))
                .clicked()
            {
                out.push(Command::SetFollow {
                    clip: id,
                    action,
                    after_bars: clip.follow_after_bars,
                });
            }
        }
        let mut bars = u32::from(clip.follow_after_bars);
        if ui
            .add(
                egui::DragValue::new(&mut bars)
                    .speed(0.25)
                    .range(1..=64)
                    .prefix("after "),
            )
            .changed()
        {
            out.push(Command::SetFollow {
                clip: id,
                action: clip.follow,
                after_bars: bars as u16,
            });
        }
        let mut react = clip.react_to_events;
        if ui.checkbox(&mut react, "on LEVEL_UP / DROP").changed() {
            out.push(Command::SetReactToEvents {
                clip: id,
                on: react,
            });
        }
    });

    ui.horizontal(|ui| {
        ui.label("overlay video");
        ui.label(match clip.video.as_deref() {
            Some(path) => path.to_string(),
            None => "(none)".to_string(),
        });
        ui.text_edit_singleline(&mut state.video_path);
        if ui.button("set").clicked() {
            out.push(Command::SetVideo {
                clip: id,
                path: Some(state.video_path.clone()),
            });
        }
        if ui.button("clear").clicked() {
            out.push(Command::SetVideo { clip: id, path: None });
            state.video_path.clear();
        }
    });
}

/// Saved looks: save the selected clip's parameters and seed, or put one back.
fn presets(ui: &mut egui::Ui, state: &mut EditorUi, model: &EditorModel, out: &mut Vec<Command>) {
    let ready = model.selected_clip().is_some();
    ui.horizontal(|ui| {
        ui.label("save the selected clip as");
        ui.text_edit_singleline(&mut state.preset_name);
        if ui
            .add_enabled(ready, egui::Button::new("Save preset"))
            .clicked()
        {
            out.push(Command::SavePreset {
                name: state.preset_name.clone(),
            });
            state.preset_name.clear();
        }
    });
    if !ready {
        ui.label("select a clip first");
    }
    ui.separator();
    if model.project().presets.is_empty() {
        ui.label("no presets yet");
    }
    for (index, preset) in model.project().presets.iter().enumerate() {
        ui.horizontal(|ui| {
            let applies = model.selected_clip().is_some();
            if ui
                .add_enabled(applies, egui::Button::new("Apply"))
                .clicked()
            {
                out.push(Command::ApplyPreset { index });
            }
            ui.label(format!(
                "{} · {} · seed {}",
                preset.name, preset.visualizer, preset.params.seed
            ));
            if ui.button("delete").clicked() {
                out.push(Command::DeletePreset { index });
            }
        });
    }
}

/// The project file and the show's mode.
fn project(ui: &mut egui::Ui, state: &mut EditorUi, model: &EditorModel, out: &mut Vec<Command>) {
    let project = model.project();
    ui.horizontal(|ui| {
        ui.label("show mode");
        for mode in [ShowMode::Arrangement, ShowMode::Session, ShowMode::Autopilot] {
            if ui
                .selectable_label(project.mode == mode, format!("{mode:?}"))
                .clicked()
            {
                out.push(Command::SetMode(mode));
            }
        }
    });
    ui.horizontal(|ui| {
        ui.label("launch quantize");
        for quantize in [
            LaunchQuantize::None,
            LaunchQuantize::Beat,
            LaunchQuantize::Bar,
            LaunchQuantize::Phrase,
        ] {
            if ui
                .selectable_label(
                    project.session.quantize == quantize,
                    quantize_label(quantize),
                )
                .clicked()
            {
                out.push(Command::SetQuantize(quantize));
            }
        }
    });

    ui.separator();
    ui.horizontal(|ui| {
        ui.label("project (.ron)");
        ui.text_edit_singleline(&mut state.project_path);
        let ready = !state.project_path.trim().is_empty();
        if ui
            .add_enabled(ready, egui::Button::new("Open"))
            .clicked()
        {
            out.push(Command::OpenProject {
                path: state.project_path.trim().to_string(),
            });
        }
        let ready = !state.project_path.trim().is_empty();
        if ui
            .add_enabled(ready, egui::Button::new("Save"))
            .clicked()
        {
            out.push(Command::SaveProject {
                path: state.project_path.trim().to_string(),
            });
        }
    });

    ui.separator();
    ui.label(format!(
        "{} clips · {} layers · {} scenes · {} presets · {} song markers · {} signature changes",
        project.clips.len(),
        project.session.layers.len(),
        project.session.scenes.len(),
        project.presets.len(),
        project.arrangement.songs.len(),
        project.arrangement.sig_map.len(),
    ));
    ui.label(format!(
        "arrangement: {} lanes, {} clips placed",
        project.arrangement.lanes.len(),
        project
            .arrangement
            .lanes
            .iter()
            .map(Vec::len)
            .sum::<usize>()
    ));
}

/// The span the timeline draws: the last clip, the last marker plus a phrase, at least 4 bars.
fn total_beats(arrangement: &mosh_core::Arrangement) -> f64 {
    let clips = arrangement
        .lanes
        .iter()
        .flatten()
        .map(|span| span.end_ppq)
        .fold(0.0, f64::max);
    let markers = arrangement
        .songs
        .last()
        .map(|song| song.start_ppq + 16.0)
        .unwrap_or(0.0);
    clips.max(markers).max(16.0)
}
