//! The Session view: the clip grid. Columns are layers, rows are scenes.
//!
//! ```text
//! | layer 1      | layer 2      | + |  SCENES        |
//! | kind  blend  | kind  blend  |   |                |
//! |--------------|--------------|---|----------------|
//! | > tunnel     | > fluid      |   | > 1  intro     |   click the triangle: launch
//! | > loom       |   (empty)    |   | > 2  verse     |   click the name: select and edit
//! |--------------|--------------|---|----------------|
//! |   stop       |   stop       |   |   stop all     |
//! ```
//!
//! A slot shows what it is doing in the one accent colour: a playing slot has a lit launch
//! button and an accent frame, a queued one blinks until its bar line, a stopping one blinks
//! red. The same states light the APC mini's pads (`crate::leds`).

use egui::{Align2, Pos2, Rect, Sense, Stroke, StrokeKind, UiBuilder, Vec2};
use mosh_core::{BlendMode, Layer, LayerKind};

use crate::editor::model::{Command, EditorModel, MAX_LAYERS};
use crate::editor::theme;
use crate::editor::ui::widgets::{self, clipped_text};
use crate::editor::ui::{DragPayload, EditorUi, clip_slot, thumb_params};

const HEADER: f32 = 58.0;
const CELL: f32 = 46.0;
const SCENES: f32 = 148.0;
const ADD: f32 = 30.0;
const LAUNCH: f32 = 30.0;
const MIN_COLUMN: f32 = 132.0;
const MAX_COLUMN: f32 = 240.0;

const BLENDS: [(BlendMode, &str); 5] = [
    (BlendMode::Normal, "Normal"),
    (BlendMode::Add, "Add"),
    (BlendMode::Screen, "Screen"),
    (BlendMode::Multiply, "Multiply"),
    (BlendMode::LumaKey, "Luma key"),
];

/// Blink phase for queued slots: on for half of every 0.4 s, and ask for the next frame.
fn blink(ui: &egui::Ui) -> bool {
    ui.ctx()
        .request_repaint_after(std::time::Duration::from_millis(200));
    ui.input(|input| (input.time * 2.5).fract() < 0.5)
}

/// A layer's controls: its number, kind, blend, opacity and a stop button, in `rect`.
///
/// Shared by the Session column headers and the Arrangement lane headers, so a layer is edited
/// the same way in both views.
pub(super) fn layer_header(
    ui: &mut egui::Ui,
    rect: Rect,
    index: usize,
    layer: &Layer,
    model: &EditorModel,
    out: &mut Vec<Command>,
) {
    let status = model.status();
    let playing = status.playing.get(index).copied().flatten().is_some();
    // The header's own click area goes down first, so the controls on it stay on top of it.
    let header = ui.interact(rect, ui.scope_id().with(("layer_menu", index)), Sense::click());
    ui.painter().rect_filled(rect, 0.0, theme::PANEL);
    widgets::kind_texture(ui.painter(), rect, layer.kind);

    let inner = rect.shrink2(Vec2::new(6.0, 5.0));
    let row = 22.0;
    // Line 1: the layer's number and its kind.
    let badge = Rect::from_min_size(inner.min, Vec2::new(20.0, row));
    ui.painter().rect_filled(
        badge,
        theme::RADIUS,
        if playing { theme::ACCENT } else { theme::RAISED },
    );
    ui.painter().text(
        badge.center(),
        Align2::CENTER_CENTER,
        index + 1,
        theme::strong(),
        if playing { theme::ON_ACCENT } else { theme::TEXT },
    );
    let line1 = Rect::from_min_max(
        Pos2::new(badge.right() + 6.0, inner.top()),
        Pos2::new(inner.right(), inner.top() + row),
    );
    let line2 = Rect::from_min_max(
        Pos2::new(inner.left(), inner.top() + row + 4.0),
        Pos2::new(inner.right(), inner.top() + row * 2.0 + 4.0),
    );
    let stop = Rect::from_min_size(
        Pos2::new(line2.right() - row, line2.top()),
        Vec2::splat(row),
    );

    ui.scope_builder(UiBuilder::new().max_rect(line1).id_salt(("layer_kind", index)), |ui| {
        egui::ComboBox::from_id_salt("kind")
            .width(line1.width() - 8.0)
            .selected_text(widgets::kind_label(layer.kind))
            .show_ui(ui, |ui| {
                for kind in widgets::LAYER_KINDS {
                    if ui
                        .selectable_label(layer.kind == kind, widgets::kind_label(kind))
                        .on_hover_text(widgets::kind_tip(kind))
                        .clicked()
                    {
                        out.push(Command::SetLayerKind { layer: index, kind });
                    }
                }
            })
            .response
            .on_hover_text(widgets::kind_tip(layer.kind));
    });

    ui.scope_builder(UiBuilder::new().max_rect(line2).id_salt(("layer_mix", index)), |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            let blend_width = (line2.width() - row - 4.0 - 46.0 - 12.0).max(40.0);
            let name = BLENDS
                .iter()
                .find(|(mode, _)| *mode == layer.blend)
                .map_or("Normal", |(_, name)| *name);
            egui::ComboBox::from_id_salt("blend")
                .width(blend_width)
                .selected_text(name)
                .show_ui(ui, |ui| {
                    for (mode, name) in BLENDS {
                        if ui.selectable_label(layer.blend == mode, name).clicked() {
                            out.push(Command::SetLayerBlend {
                                layer: index,
                                blend: mode,
                            });
                        }
                    }
                })
                .response
                .on_hover_text("how this layer is mixed onto the ones below");
            let mut percent = (layer.opacity * 100.0).round();
            let response = ui
                .add(
                    egui::DragValue::new(&mut percent)
                        .range(0.0..=100.0)
                        .speed(0.5)
                        .fixed_decimals(0)
                        .suffix("%"),
                )
                .on_hover_text("opacity (drag, or double-click to type)");
            if response.changed() {
                out.push(Command::SetLayerOpacity {
                    layer: index,
                    opacity: percent / 100.0,
                });
            }
        });
    });

    let stopping = status.stopping & (1 << index.min(7)) != 0;
    let ink = if stopping && blink(ui) {
        theme::DANGER
    } else if playing {
        theme::TEXT
    } else {
        theme::FAINT
    };
    if widgets::icon_button(
        ui,
        stop,
        ("layer_stop", index),
        theme::RAISED,
        ink,
        widgets::stop_icon,
    )
    .on_hover_text("stop this layer at the next launch point")
    .clicked()
    {
        out.push(Command::StopLayer(index));
    }

    header.context_menu(|ui| {
        if ui.button("Remove this layer").clicked() {
            out.push(Command::RemoveLayer(index));
            ui.close();
        }
    });
}

pub fn draw(ui: &mut egui::Ui, state: &mut EditorUi, model: &EditorModel, out: &mut Vec<Command>) {
    let project = model.project();
    let grid = &project.session;
    let status = *model.status();
    let area = ui.max_rect();
    let layers = grid.layers.len();
    let can_add = layers < MAX_LAYERS;
    let add = if can_add { ADD } else { 0.0 };
    let column = ((area.width() - SCENES - add - 8.0) / layers.max(1) as f32)
        .clamp(MIN_COLUMN, MAX_COLUMN);
    let grid_width = column * layers as f32 + add + SCENES;

    egui::ScrollArea::both()
        .id_salt("session_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let height = HEADER + CELL * grid.scenes.len() as f32 + CELL + 8.0;
            let (rect, _) =
                ui.allocate_exact_size(Vec2::new(grid_width, height), Sense::hover());
            let column_x = |layer: usize| rect.left() + column * layer as f32;
            let scenes_x = column_x(layers) + add;

            // Column headers.
            for (index, layer) in grid.layers.iter().enumerate() {
                let header = Rect::from_min_size(
                    Pos2::new(column_x(index), rect.top()),
                    Vec2::new(column - 2.0, HEADER - 2.0),
                );
                layer_header(ui, header, index, layer, model, out);
            }
            if can_add {
                let plus = Rect::from_min_size(
                    Pos2::new(column_x(layers) + 2.0, rect.top() + 6.0),
                    Vec2::splat(ADD - 6.0),
                );
                if widgets::icon_button(
                    ui,
                    plus,
                    "add_layer",
                    theme::RAISED,
                    theme::DIM,
                    widgets::plus_icon,
                )
                .on_hover_text("add a layer on top")
                .clicked()
                {
                    out.push(Command::AddLayer);
                }
            }
            ui.painter().text(
                Pos2::new(scenes_x + 8.0, rect.top() + HEADER - 12.0),
                Align2::LEFT_BOTTOM,
                "SCENES   keys 1-9 launch",
                theme::small(),
                theme::FAINT,
            );

            // The grid.
            for (scene, row) in grid.scenes.iter().enumerate() {
                let y = rect.top() + HEADER + CELL * scene as f32;
                for layer in 0..layers {
                    let cell = Rect::from_min_size(
                        Pos2::new(column_x(layer), y),
                        Vec2::new(column - 2.0, CELL - 2.0),
                    );
                    if !ui.is_rect_visible(cell) {
                        continue;
                    }
                    let clip = row.cells.get(layer).copied().flatten();
                    slot(ui, state, model, cell, layer, scene, clip, out);
                }

                // The scene's launch button.
                let button = Rect::from_min_size(
                    Pos2::new(scenes_x, y),
                    Vec2::new(SCENES - 2.0, CELL - 2.0),
                );
                if !ui.is_rect_visible(button) {
                    continue;
                }
                let response =
                    ui.interact(button, ui.scope_id().with(("scene", scene)), Sense::click());
                let on = (0..layers.min(8)).any(|layer| {
                    status.scene[layer] == Some(scene as u16) && status.playing[layer].is_some()
                });
                let queued =
                    (0..layers.min(8)).any(|layer| status.queued[layer] == Some(scene as u16));
                let lit = on || (queued && blink(ui));
                ui.painter().rect_filled(
                    button,
                    theme::RADIUS,
                    if response.hovered() {
                        theme::HOVER
                    } else {
                        theme::RAISED
                    },
                );
                let launch = Rect::from_min_size(button.min, Vec2::new(LAUNCH, button.height()));
                if lit {
                    ui.painter().rect_filled(launch, theme::RADIUS, theme::ACCENT);
                }
                widgets::play_icon(
                    ui.painter(),
                    launch,
                    if lit { theme::ON_ACCENT } else { theme::TEXT },
                );
                let label = Rect::from_min_max(
                    Pos2::new(launch.right() + 6.0, button.top()),
                    Pos2::new(button.right() - 6.0, button.bottom()),
                );
                clipped_text(
                    ui.painter(),
                    label,
                    Align2::LEFT_CENTER,
                    &format!("{}   {}", scene + 1, row.name),
                    theme::body(),
                    if model.selected_scene() == Some(scene) {
                        theme::TEXT
                    } else {
                        theme::DIM
                    },
                );
                if response.clicked() {
                    out.push(Command::LaunchScene(scene));
                    out.push(Command::SelectScene(scene));
                }
                response
                    .on_hover_text("launch every clip of this row (right-click: rename, remove)")
                    .context_menu(|ui| {
                        ui.label("scene name");
                        if let Some(name) = widgets::text_field(
                            ui,
                            &mut state.draft,
                            widgets::field_key(("scene_name", scene)),
                            &row.name,
                            140.0,
                            "name",
                            1,
                        ) {
                            out.push(Command::RenameScene { scene, name });
                        }
                        if ui.button("Remove this scene").clicked() {
                            out.push(Command::RemoveScene(scene));
                            ui.close();
                        }
                    });
            }

            // The bottom row: stop buttons, and a new scene.
            let y = rect.top() + HEADER + CELL * grid.scenes.len() as f32 + 4.0;
            let stop_all = Rect::from_min_size(
                Pos2::new(scenes_x, y),
                Vec2::new((SCENES - 6.0) * 0.5, CELL - 12.0),
            );
            if widgets::flat_button(ui, stop_all, "stop_all", "Stop all", false)
                .on_hover_text("stop every layer (0)")
                .clicked()
            {
                for layer in 0..layers {
                    out.push(Command::StopLayer(layer));
                }
            }
            let add_scene = Rect::from_min_size(
                Pos2::new(stop_all.right() + 4.0, y),
                Vec2::new((SCENES - 6.0) * 0.5, CELL - 12.0),
            );
            if widgets::flat_button(ui, add_scene, "add_scene", "+ Scene", false)
                .on_hover_text("add a scene row")
                .clicked()
            {
                out.push(Command::AddScene);
            }
            if grid.scenes.is_empty() || project.clips.is_empty() {
                ui.painter().text(
                    Pos2::new(rect.left() + 8.0, y + 6.0),
                    Align2::LEFT_TOP,
                    "Drag a visualizer from the browser onto a slot to make a clip.",
                    theme::body(),
                    theme::DIM,
                );
            }
        });
}

/// One slot of the grid.
#[allow(clippy::too_many_arguments)]
fn slot(
    ui: &mut egui::Ui,
    state: &mut EditorUi,
    model: &EditorModel,
    rect: Rect,
    layer: usize,
    scene: usize,
    clip: Option<mosh_core::ClipId>,
    out: &mut Vec<Command>,
) {
    let status = model.status();
    let kind = model
        .project()
        .session
        .layers
        .get(layer)
        .map_or(LayerKind::Visualizer, |layer| layer.kind);
    let response = ui.interact(
        rect,
        ui.scope_id().with(("slot", layer, scene)),
        Sense::click_and_drag(),
    );
    let dropping = response.dnd_hover_payload::<DragPayload>().is_some();
    let clip = clip.and_then(|id| model.clip(id));

    let Some(clip) = clip else {
        // An empty slot: a quiet well, lit while something is dragged over it.
        ui.painter().rect_filled(rect, theme::RADIUS, theme::SLOT);
        widgets::kind_texture(ui.painter(), rect, kind);
        if dropping {
            ui.painter().rect_stroke(
                rect,
                theme::RADIUS,
                Stroke::new(2.0, theme::ACCENT),
                StrokeKind::Inside,
            );
        }
        drop_on(&response, layer, scene, out);
        return;
    };

    let column = layer.min(7);
    let here = status.scene[column] == Some(scene as u16);
    let playing = here && status.playing[column] == Some(clip.id);
    let queued = status.queued[column] == Some(scene as u16);
    let stopping = playing && status.stopping & (1 << column) != 0;
    let selected = model.selected() == Some(clip.id) && model.selected_scene() == Some(scene);
    let flash = (queued || stopping) && blink(ui);

    let color = theme::clip_color(&clip.visualizer);
    let painter = ui.painter().clone();
    painter.rect_filled(
        rect,
        theme::RADIUS,
        if response.hovered() {
            theme::HOVER
        } else {
            theme::RAISED
        },
    );
    widgets::kind_texture(&painter, rect, kind);

    // The launch button.
    let launch = Rect::from_min_size(rect.min, Vec2::new(LAUNCH, rect.height()));
    let (fill, ink) = if stopping {
        (if flash { theme::DANGER } else { theme::ACCENT }, theme::ON_ACCENT)
    } else if playing || (queued && flash) {
        (theme::ACCENT, theme::ON_ACCENT)
    } else {
        (theme::dimmed(color, 0.35), theme::TEXT)
    };
    painter.rect_filled(launch, theme::RADIUS, fill);
    widgets::play_icon(&painter, launch, ink);

    // The picture: a thumbnail of the clip's look, or its colour. A narrow column keeps the
    // name readable and shows the colour as a strip instead.
    let roomy = rect.width() >= 170.0;
    let picture = Rect::from_min_size(
        Pos2::new(launch.right() + 4.0, rect.top() + 4.0),
        Vec2::new(
            if roomy {
                (rect.height() - 8.0) * 16.0 / 9.0
            } else {
                5.0
            },
            rect.height() - 8.0,
        ),
    );
    painter.rect_filled(picture, 2.0, color);
    if roomy
        && let Some(texture) = state.thumbs.get(
            ui.ctx(),
            clip_slot(clip.id),
            &clip.visualizer,
            &thumb_params(clip),
        )
    {
        painter.image(
            texture,
            picture,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            egui::Color32::WHITE,
        );
    }

    let text = Rect::from_min_max(
        Pos2::new(picture.right() + 6.0, rect.top() + 4.0),
        Pos2::new(rect.right() - 4.0, rect.bottom() - 4.0),
    );
    clipped_text(
        &painter,
        text,
        Align2::LEFT_TOP,
        &clip.name,
        theme::body(),
        theme::TEXT,
    );
    let state_line = if stopping {
        "stopping"
    } else if playing {
        "playing"
    } else if queued {
        "queued"
    } else {
        crate::editor::model::catalog_entry(&clip.visualizer)
            .map_or(clip.visualizer.as_str(), |entry| entry.name)
    };
    clipped_text(
        &painter,
        text,
        Align2::LEFT_BOTTOM,
        state_line,
        theme::small(),
        if playing || queued {
            theme::ACCENT
        } else {
            theme::FAINT
        },
    );

    if playing || selected || dropping {
        painter.rect_stroke(
            rect,
            theme::RADIUS,
            Stroke::new(
                if dropping { 2.0 } else { 1.0 },
                if playing || dropping {
                    theme::ACCENT
                } else {
                    theme::TEXT
                },
            ),
            StrokeKind::Inside,
        );
    }

    // Click the triangle to launch; click the rest to select; double-click launches too.
    let on_launch = response
        .interact_pointer_pos()
        .is_some_and(|pos| launch.contains(pos));
    if response.clicked() {
        out.push(Command::SelectClip(Some(clip.id)));
        out.push(Command::SelectScene(scene));
        if on_launch {
            out.push(Command::Launch { layer, scene });
        }
    }
    if response.double_clicked() && !on_launch {
        out.push(Command::Launch { layer, scene });
    }
    if response.drag_started() {
        response.dnd_set_drag_payload(DragPayload::Cell { layer, scene });
    }
    drop_on(&response, layer, scene, out);
    response.context_menu(|ui| {
        if ui.button("Launch").clicked() {
            out.push(Command::Launch { layer, scene });
            ui.close();
        }
        if ui.button("Empty this slot").clicked() {
            out.push(Command::ClearCell { layer, scene });
            ui.close();
        }
    });
}

/// Something dragged was released on this slot.
fn drop_on(response: &egui::Response, layer: usize, scene: usize, out: &mut Vec<Command>) {
    let Some(payload) = response.dnd_release_payload::<DragPayload>() else {
        return;
    };
    match payload.as_ref() {
        DragPayload::Source(source, _) => out.push(Command::SetCell {
            layer,
            scene,
            source: source.clone(),
        }),
        DragPayload::Cell {
            layer: from_layer,
            scene: from_scene,
        } => {
            if (*from_layer, *from_scene) != (layer, scene) {
                out.push(Command::SwapCells {
                    from: (*from_layer, *from_scene),
                    to: (layer, scene),
                });
            }
        }
    }
}
