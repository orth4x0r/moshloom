//! The browser on the left: everything a show is built from, and the show's file.
//!
//! * **Visualizers**, grouped as `mosh_visuals::catalog` groups them (the web scenes, the layer
//!   visualizers, the style packs), so a scene another worker adds appears here with no change.
//!   Drag one onto a session slot or an arrangement lane to make a clip; double-click to use it
//!   for the selected clip.
//! * **Presets**: saved looks. Drag one to make a clip of it; double-click to put it on the
//!   selected clip.
//! * **Show**: the project file (new, open, save, the recent list) and the Live-set import.

use egui::{Align2, Pos2, Rect, Sense, Vec2};

use crate::editor::model::{ClipSource, Command, EditorModel, catalog, new_clip};
use crate::editor::theme;
use crate::editor::ui::widgets::clipped_text;
use crate::editor::ui::{DragPayload, EditorUi, thumb_params};

pub const WIDTH: f32 = 212.0;
const ROW: f32 = 24.0;
/// Thumbnail slots of browser rows live far above any clip id.
const SLOT_BASE: u64 = 1 << 40;

pub fn draw(ui: &mut egui::Ui, state: &mut EditorUi, model: &EditorModel, out: &mut Vec<Command>) {
    ui.spacing_mut().item_spacing.y = 2.0;
    ui.add(
        egui::TextEdit::singleline(&mut state.filter)
            .hint_text("filter visualizers and presets")
            .desired_width(f32::INFINITY),
    );
    ui.add_space(theme::GAP);
    let filter = state.filter.trim().to_lowercase();
    let matches = |name: &str| filter.is_empty() || name.to_lowercase().contains(&filter);

    egui::ScrollArea::vertical()
        .id_salt("browser_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            if section(ui, &mut state.sections[0], "VISUALIZERS", "drag to a slot or a lane") {
                let mut group = "";
                for (index, entry) in catalog().iter().enumerate() {
                    if !matches(entry.name) && !matches(entry.group) {
                        continue;
                    }
                    if entry.group != group {
                        group = entry.group;
                        let (rect, _) = ui.allocate_exact_size(
                            Vec2::new(ui.available_width(), 18.0),
                            Sense::hover(),
                        );
                        ui.painter().text(
                            rect.left_bottom() + Vec2::new(2.0, -2.0),
                            Align2::LEFT_BOTTOM,
                            group,
                            theme::small(),
                            theme::DIM,
                        );
                    }
                    let sample = new_clip(0, entry.id);
                    let texture = state.thumbs.get(
                        ui.ctx(),
                        SLOT_BASE + index as u64,
                        entry.id,
                        &thumb_params(&sample),
                    );
                    let used = model
                        .selected_clip()
                        .is_some_and(|clip| clip.visualizer == entry.id);
                    let response = row(ui, entry.id, entry.name, None, texture, used);
                    if response.drag_started() {
                        response.dnd_set_drag_payload(DragPayload::Source(
                            ClipSource::Visualizer(entry.id.to_string()),
                            entry.name.to_string(),
                        ));
                    }
                    if response.double_clicked()
                        && let Some(clip) = model.selected()
                    {
                        out.push(Command::SetVisualizer {
                            clip,
                            visualizer: entry.id.to_string(),
                        });
                    }
                    response.on_hover_text(
                        "drag to a slot or a lane to make a clip\ndouble-click: use it for the selected clip",
                    );
                }
                ui.add_space(theme::GAP * 2.0);
            }

            if section(ui, &mut state.sections[1], "PRESETS", "saved looks") {
                let presets = &model.project().presets;
                if presets.is_empty() {
                    hint(ui, "Save a clip's look as a preset in the detail panel.");
                }
                for (index, preset) in presets.iter().enumerate() {
                    if !matches(&preset.name) {
                        continue;
                    }
                    let name = catalog()
                        .iter()
                        .find(|entry| entry.id == preset.visualizer)
                        .map_or(preset.visualizer.as_str(), |entry| entry.name);
                    let texture = state.thumbs.get(
                        ui.ctx(),
                        SLOT_BASE * 2 + index as u64,
                        &preset.visualizer,
                        &preset.params,
                    );
                    let response = row(
                        ui,
                        &preset.visualizer,
                        &preset.name,
                        Some(name),
                        texture,
                        false,
                    );
                    if response.drag_started() {
                        response.dnd_set_drag_payload(DragPayload::Source(
                            ClipSource::Preset(index),
                            preset.name.clone(),
                        ));
                    }
                    if response.double_clicked() {
                        out.push(Command::ApplyPreset { index });
                    }
                    response.context_menu(|ui| {
                        if ui.button("Use for the selected clip").clicked() {
                            out.push(Command::ApplyPreset { index });
                            ui.close();
                        }
                        if ui.button("Delete preset").clicked() {
                            out.push(Command::DeletePreset { index });
                            ui.close();
                        }
                    });
                }
                ui.add_space(theme::GAP * 2.0);
            }

            if section(ui, &mut state.sections[2], "SHOW", "the project file") {
                show_file(ui, state, model, out);
            }
        });
}

/// A collapsible section header. Returns whether the section is open.
fn section(ui: &mut egui::Ui, open: &mut bool, title: &str, about: &str) -> bool {
    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 22.0), Sense::click());
    if response.clicked() {
        *open = !*open;
    }
    let painter = ui.painter();
    painter.hline(rect.x_range(), rect.top(), theme::line());
    // An open section's marker points down, a closed one's to the right.
    let c = Pos2::new(rect.left() + 6.0, rect.center().y + 1.0);
    let points = if *open {
        vec![c + Vec2::new(-4.0, -2.5), c + Vec2::new(4.0, -2.5), c + Vec2::new(0.0, 3.0)]
    } else {
        vec![c + Vec2::new(-2.5, -4.0), c + Vec2::new(3.0, 0.0), c + Vec2::new(-2.5, 4.0)]
    };
    painter.add(egui::Shape::convex_polygon(
        points,
        theme::DIM,
        egui::Stroke::NONE,
    ));
    painter.text(
        Pos2::new(rect.left() + 16.0, rect.center().y + 1.0),
        Align2::LEFT_CENTER,
        title,
        theme::small(),
        if response.hovered() {
            theme::TEXT
        } else {
            theme::DIM
        },
    );
    response.on_hover_text(about);
    *open
}

/// One draggable row: a swatch or thumbnail, a name, an optional second word.
fn row(
    ui: &mut egui::Ui,
    visualizer: &str,
    name: &str,
    detail: Option<&str>,
    texture: Option<egui::TextureId>,
    marked: bool,
) -> egui::Response {
    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), ROW), Sense::click_and_drag());
    let painter = ui.painter();
    if response.hovered() || response.dragged() {
        painter.rect_filled(rect, theme::RADIUS, theme::HOVER);
    }
    let swatch = Rect::from_min_size(
        Pos2::new(rect.left() + 2.0, rect.top() + 3.0),
        Vec2::new(32.0, ROW - 6.0),
    );
    painter.rect_filled(swatch, 2.0, theme::clip_color(visualizer));
    if let Some(texture) = texture {
        painter.image(
            texture,
            swatch,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            egui::Color32::WHITE,
        );
    }
    let text = Rect::from_min_max(
        Pos2::new(swatch.right() + 8.0, rect.top()),
        Pos2::new(rect.right() - 4.0, rect.bottom()),
    );
    clipped_text(
        painter,
        text,
        Align2::LEFT_CENTER,
        name,
        theme::body(),
        if marked { theme::ACCENT } else { theme::TEXT },
    );
    if let Some(detail) = detail {
        clipped_text(
            painter,
            text,
            Align2::RIGHT_CENTER,
            detail,
            theme::small(),
            theme::FAINT,
        );
    }
    response
}

fn hint(ui: &mut egui::Ui, text: &str) {
    ui.add(egui::Label::new(
        egui::RichText::new(text).font(theme::small()).color(theme::FAINT),
    ));
}

/// New / open / save, the recent list and the Live-set import.
fn show_file(ui: &mut egui::Ui, state: &mut EditorUi, model: &EditorModel, out: &mut Vec<Command>) {
    let current = model.path().unwrap_or("not saved yet");
    hint(
        ui,
        &format!(
            "{current}{}",
            if model.modified() { "  (edited)" } else { "" }
        ),
    );
    ui.add(
        egui::TextEdit::singleline(&mut state.project_path)
            .hint_text("path to a show (.ron)")
            .desired_width(f32::INFINITY),
    );
    let path = state.project_path.trim().to_string();
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = theme::GAP;
        if ui
            .add_enabled(!path.is_empty(), egui::Button::new("Open"))
            .on_hover_text("open the show at this path")
            .clicked()
        {
            out.push(Command::OpenProject { path: path.clone() });
        }
        if ui
            .add_enabled(!path.is_empty(), egui::Button::new("Save"))
            .on_hover_text("save the show to this path (Ctrl+S saves to its own file)")
            .clicked()
        {
            out.push(Command::SaveProject { path: path.clone() });
        }
        if ui
            .button("New")
            .on_hover_text("an empty show: one layer, one scene")
            .clicked()
        {
            out.push(Command::NewProject);
        }
    });

    if !state.recent.is_empty() {
        ui.add_space(theme::GAP);
        hint(ui, "recent");
        let mut open = None;
        for path in &state.recent {
            let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
            let (rect, response) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 20.0), Sense::click());
            if response.hovered() {
                ui.painter().rect_filled(rect, theme::RADIUS, theme::HOVER);
            }
            clipped_text(
                ui.painter(),
                rect.shrink2(Vec2::new(4.0, 0.0)),
                Align2::LEFT_CENTER,
                name,
                theme::body(),
                theme::TEXT,
            );
            if response.on_hover_text(path.as_str()).clicked() {
                open = Some(path.clone());
            }
        }
        if let Some(path) = open {
            state.project_path = path.clone();
            out.push(Command::OpenProject { path });
        }
    }

    ui.add_space(theme::GAP * 2.0);
    hint(ui, "song markers and time signatures from a Live set");
    ui.add(
        egui::TextEdit::singleline(&mut state.als_path)
            .hint_text("path to a Live set (.als)")
            .desired_width(f32::INFINITY),
    );
    let als = state.als_path.trim().to_string();
    if ui
        .add_enabled(!als.is_empty(), egui::Button::new("Import markers"))
        .on_hover_text("replaces the song markers and the signature map with the Live set's")
        .clicked()
    {
        out.push(Command::ImportAls { path: als });
    }
}
