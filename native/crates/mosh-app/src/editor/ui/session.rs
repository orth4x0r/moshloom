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

const CELL: f32 = 46.0;
const ADD: f32 = 30.0;
const LAUNCH: f32 = 30.0;
/// A column is never narrower than this: below it a clip's name is a few letters and the
/// header's pickers cannot say what they are. Only with more layers than fit at this width do
/// the layer columns scroll sideways.
pub const MIN_COLUMN: f32 = 104.0;
const MAX_COLUMN: f32 = 240.0;
/// A column at least this wide has its header on two rows; a narrower one stacks it on three.
pub const TWO_ROW_COLUMN: f32 = 150.0;
/// A view narrower than this gets the narrow scene column.
const NARROW_VIEW: f32 = 760.0;

/// The scene column's width in a view `width` wide: room for a scene's name where there is
/// room, the launch button and a short name where the layers need the space.
pub fn scene_width(width: f32) -> f32 {
    if width >= NARROW_VIEW { 148.0 } else { 104.0 }
}

/// The width of a session column for `layers` layers in a view `width` wide: the layers share
/// what the scene column leaves, so that all of them are in view whenever each can have
/// [`MIN_COLUMN`].
pub fn column_width(width: f32, layers: usize, can_add: bool) -> f32 {
    let add = if can_add { ADD } else { 0.0 };
    ((width - scene_width(width) - add - 8.0) / layers.max(1) as f32)
        .clamp(MIN_COLUMN, MAX_COLUMN)
}

/// The height of a layer header in a column `column` wide.
pub fn header_height(column: f32) -> f32 {
    if column >= TWO_ROW_COLUMN { 58.0 } else { 84.0 }
}

/// Blend modes: the name, and the short one a narrow header shows.
const BLENDS: [(BlendMode, &str, &str); 5] = [
    (BlendMode::Normal, "Normal", "Normal"),
    (BlendMode::Add, "Add", "Add"),
    (BlendMode::Screen, "Screen", "Screen"),
    (BlendMode::Multiply, "Multiply", "Mult"),
    (BlendMode::LumaKey, "Luma key", "Luma"),
];

/// Where a layer header's five controls go.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeaderRects {
    /// The layer's number.
    pub badge: Rect,
    /// The kind picker.
    pub kind: Rect,
    /// The stop button.
    pub stop: Rect,
    /// The blend picker.
    pub blend: Rect,
    /// The opacity bar.
    pub opacity: Rect,
}

impl HeaderRects {
    pub fn all(&self) -> [Rect; 5] {
        [self.badge, self.kind, self.stop, self.blend, self.opacity]
    }
}

/// Lay a layer header's controls out in `rect`. A column of [`TWO_ROW_COLUMN`] or more gets
/// two rows, a narrower one three, so that four layers fit beside the browser on a laptop:
///
/// ```text
/// | 1 | Visualizer      v | [] |        | 1 | Visual     v |     number, kind
/// | Normal        v | 100%     |        | Luma key       v |     blend
///                                       | 100%        | [] |     opacity, stop
/// ```
///
/// The number, the stop button and (on two rows) the opacity bar have fixed widths; the
/// pickers take what is left of their row, so nothing is ever pushed out of the column.
pub fn header_layout(rect: Rect) -> HeaderRects {
    const ROW: f32 = 22.0;
    const GAP: f32 = 4.0;
    const BADGE: f32 = 20.0;
    const OPACITY: f32 = 48.0;
    let inner = rect.shrink2(Vec2::new(6.0, 5.0));
    let row = |index: f32, left: f32, right: f32| {
        Rect::from_min_max(
            Pos2::new(left, inner.top() + (ROW + GAP) * index),
            Pos2::new(right.max(left), inner.top() + (ROW + GAP) * index + ROW),
        )
    };
    let badge = row(0.0, inner.left(), inner.left() + BADGE);
    if rect.width() < TWO_ROW_COLUMN - 2.0 {
        let stop = row(2.0, inner.right() - ROW, inner.right());
        return HeaderRects {
            badge,
            kind: row(0.0, badge.right() + GAP, inner.right()),
            stop,
            blend: row(1.0, inner.left(), inner.right()),
            opacity: row(2.0, inner.left(), stop.left() - GAP),
        };
    }
    let stop = row(0.0, inner.right() - ROW, inner.right());
    let kind = row(0.0, badge.right() + GAP, stop.left() - GAP);
    let opacity = row(1.0, inner.right() - OPACITY, inner.right());
    let blend = row(1.0, inner.left(), opacity.left() - GAP);
    HeaderRects {
        badge,
        kind,
        stop,
        blend,
        opacity,
    }
}

/// What a layer header actually drew, for the layout tests: the column and the rectangle each
/// control ended up in (a picker reports its own, which is where a too-long label would show).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeaderDrawn {
    pub layer: usize,
    pub column: Rect,
    pub planned: HeaderRects,
    pub kind: Rect,
    pub blend: Rect,
}

/// A layer kind's short name, for a header too narrow for the long one.
fn kind_short(kind: LayerKind) -> &'static str {
    match kind {
        LayerKind::Visualizer => "Visual",
        LayerKind::Overlay => "Video",
        LayerKind::Adjustment => "Adjust",
        LayerKind::Mask => "Mask",
    }
}

/// The opacity control: a thin bar that fills with the value and carries the percentage. Drag
/// sideways to change it (a pixel is a percent), double-click for 100%.
fn opacity_bar(ui: &mut egui::Ui, rect: Rect, layer: usize, opacity: f32) -> Option<f32> {
    let response = ui
        .interact(rect, ui.scope_id().with(("opacity", layer)), Sense::click_and_drag())
        .on_hover_text(format!(
            "opacity {:.0}%\ndrag left or right, double-click for 100%",
            opacity * 100.0
        ));
    let painter = ui.painter();
    painter.rect_filled(
        rect,
        theme::RADIUS,
        if response.hovered() || response.dragged() {
            theme::HOVER
        } else {
            theme::RAISED
        },
    );
    let fill = Rect::from_min_max(
        rect.min,
        Pos2::new(
            rect.left() + rect.width() * opacity.clamp(0.0, 1.0),
            rect.bottom(),
        ),
    );
    painter.rect_filled(fill, theme::RADIUS, theme::ACCENT_DIM);
    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        format!("{:.0}%", opacity * 100.0),
        theme::small(),
        theme::TEXT,
    );
    if response.double_clicked() {
        return Some(1.0);
    }
    let delta = response.drag_delta().x;
    (response.dragged() && delta != 0.0).then(|| (opacity + delta / 100.0).clamp(0.0, 1.0))
}

/// Blink phase for queued slots: on for half of every 0.4 s, and ask for the next frame.
fn blink(ui: &egui::Ui) -> bool {
    ui.ctx()
        .request_repaint_after(std::time::Duration::from_millis(200));
    ui.input(|input| (input.time * 2.5).fract() < 0.5)
}

/// A layer's controls: its number, kind, blend, opacity and a stop button, in `rect`.
///
/// Shared by the Session column headers and the Arrangement lane headers, so a layer is edited
/// the same way in both views. Every control is given its rectangle by [`header_layout`] and
/// stays in it: a picker's label is shortened, then truncated, never allowed to push the
/// control beside it out of the column.
pub(super) fn layer_header(
    ui: &mut egui::Ui,
    rect: Rect,
    index: usize,
    layer: &Layer,
    model: &EditorModel,
    probe: &mut Vec<HeaderDrawn>,
    out: &mut Vec<Command>,
) {
    let status = model.status();
    let playing = status.playing.get(index).copied().flatten().is_some();
    // The header's own click area goes down first, so the controls on it stay on top of it.
    let header = ui.interact(rect, ui.scope_id().with(("layer_menu", index)), Sense::click());
    ui.painter().rect_filled(rect, 0.0, theme::PANEL);
    widgets::kind_texture(ui.painter(), rect, layer.kind);
    let rects = header_layout(rect);

    ui.painter().rect_filled(
        rects.badge,
        theme::RADIUS,
        if playing { theme::ACCENT } else { theme::RAISED },
    );
    ui.painter().text(
        rects.badge.center(),
        Align2::CENTER_CENTER,
        index + 1,
        theme::strong(),
        if playing { theme::ON_ACCENT } else { theme::TEXT },
    );

    // A picker in exactly its rectangle: the button is that wide, and its label is cut to fit.
    let picker = |ui: &mut egui::Ui,
                  area: Rect,
                  salt: &'static str,
                  label: &str,
                  tip: String,
                  list: &mut dyn FnMut(&mut egui::Ui)|
     -> Rect {
        ui.scope_builder(
            UiBuilder::new().max_rect(area).id_salt((salt, index)),
            |ui| {
                ui.spacing_mut().button_padding.x = 5.0;
                egui::ComboBox::from_id_salt(salt)
                    .width(area.width())
                    .truncate()
                    .selected_text(label.to_string())
                    .show_ui(ui, |ui| list(ui))
                    .response
                    .on_hover_text(tip)
                    .rect
            },
        )
        .inner
    };

    let kind_label = if rects.kind.width() >= 104.0 {
        widgets::kind_label(layer.kind)
    } else {
        kind_short(layer.kind)
    };
    let kind_rect = picker(
        ui,
        rects.kind,
        "kind",
        kind_label,
        format!(
            "{}: {}",
            widgets::kind_label(layer.kind),
            widgets::kind_tip(layer.kind)
        ),
        &mut |ui| {
            for kind in widgets::LAYER_KINDS {
                if ui
                    .selectable_label(layer.kind == kind, widgets::kind_label(kind))
                    .on_hover_text(widgets::kind_tip(kind))
                    .clicked()
                {
                    out.push(Command::SetLayerKind { layer: index, kind });
                }
            }
        },
    );

    let (_, blend_name, blend_short) = BLENDS
        .iter()
        .find(|(mode, _, _)| *mode == layer.blend)
        .copied()
        .unwrap_or(BLENDS[0]);
    let blend_rect = picker(
        ui,
        rects.blend,
        "blend",
        if rects.blend.width() >= 96.0 {
            blend_name
        } else {
            blend_short
        },
        format!("blend: {blend_name}\nhow this layer is mixed onto the ones below"),
        &mut |ui| {
            for (mode, name, _) in BLENDS {
                if ui.selectable_label(layer.blend == mode, name).clicked() {
                    out.push(Command::SetLayerBlend {
                        layer: index,
                        blend: mode,
                    });
                }
            }
        },
    );

    if let Some(opacity) = opacity_bar(ui, rects.opacity, index, layer.opacity) {
        out.push(Command::SetLayerOpacity {
            layer: index,
            opacity,
        });
    }

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
        rects.stop,
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
    probe.push(HeaderDrawn {
        layer: index,
        column: rect,
        planned: rects,
        kind: kind_rect,
        blend: blend_rect,
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
    let column = column_width(area.width(), layers, can_add);
    let columns_width = column * layers as f32 + add;
    let header_h = header_height(column - 2.0);
    let scenes_w = scene_width(area.width());
    let narrow = scenes_w < 130.0;

    // The rows scroll up and down as one. The layer columns scroll sideways on their own when
    // there are more of them than fit, so the scene column (the launch buttons) stays in view.
    egui::ScrollArea::vertical()
        .id_salt("session_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let height = header_h + CELL * grid.scenes.len() as f32 + CELL + 20.0;
            let (rect, _) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), height), Sense::hover());
            let scenes_x = (rect.left() + columns_width).min(rect.right() - scenes_w);
            let columns = Rect::from_min_max(
                rect.min,
                Pos2::new(scenes_x - 4.0, rect.top() + header_h + CELL * grid.scenes.len() as f32 + 12.0),
            );
            ui.scope_builder(UiBuilder::new().max_rect(columns).id_salt("columns"), |ui| {
                egui::ScrollArea::horizontal()
                    .id_salt("session_columns")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let (rect, _) = ui.allocate_exact_size(
                            Vec2::new(columns_width, columns.height() - 12.0),
                            Sense::hover(),
                        );
                        let column_x = |layer: usize| rect.left() + column * layer as f32;
                        // Column headers.
                        for (index, layer) in grid.layers.iter().enumerate() {
                            let header = Rect::from_min_size(
                                Pos2::new(column_x(index), rect.top()),
                                Vec2::new(column - 2.0, header_h - 2.0),
                            );
                            layer_header(
                                ui,
                                header,
                                index,
                                layer,
                                model,
                                &mut state.header_probe,
                                out,
                            );
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
                        // The slots.
                        for (scene, row) in grid.scenes.iter().enumerate() {
                            let y = rect.top() + header_h + CELL * scene as f32;
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
                        }
                    });
            });
            ui.painter().text(
                Pos2::new(scenes_x + 8.0, rect.top() + header_h - 12.0),
                Align2::LEFT_BOTTOM,
                if narrow {
                    "SCENES  1-9"
                } else {
                    "SCENES   keys 1-9 launch"
                },
                theme::small(),
                theme::FAINT,
            );

            // The scene column.
            for (scene, row) in grid.scenes.iter().enumerate() {
                let y = rect.top() + header_h + CELL * scene as f32;
                // The scene's launch button.
                let button = Rect::from_min_size(
                    Pos2::new(scenes_x, y),
                    Vec2::new(scenes_w - 2.0, CELL - 2.0),
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

            // Under the scene column: stop everything, and a new scene (side by side where the
            // column is wide enough for both labels, one under the other where it is not).
            let y = rect.top() + header_h + CELL * grid.scenes.len() as f32 + 4.0;
            let (stop_all, add_scene) = if narrow {
                let size = Vec2::new(scenes_w - 2.0, 24.0);
                (
                    Rect::from_min_size(Pos2::new(scenes_x, y), size),
                    Rect::from_min_size(Pos2::new(scenes_x, y + 28.0), size),
                )
            } else {
                let size = Vec2::new((scenes_w - 6.0) * 0.5, CELL - 12.0);
                (
                    Rect::from_min_size(Pos2::new(scenes_x, y), size),
                    Rect::from_min_size(Pos2::new(scenes_x + size.x + 4.0, y), size),
                )
            };
            if widgets::flat_button(ui, stop_all, "stop_all", "Stop all", false)
                .on_hover_text("stop every layer (0)")
                .clicked()
            {
                for layer in 0..layers {
                    out.push(Command::StopLayer(layer));
                }
            }
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
    // A narrow column cuts a long name short: the whole of it is one hover away.
    let response = match clip {
        Some(clip) => response.on_hover_text(format!(
            "{}\n{}",
            clip.name,
            crate::editor::model::catalog_entry(&clip.visualizer)
                .map_or(clip.visualizer.as_str(), |entry| entry.name)
        )),
        None => response,
    };

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
