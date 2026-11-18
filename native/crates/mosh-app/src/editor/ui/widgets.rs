//! Small painted controls shared by every panel, so a toggle looks the same wherever it is.
//!
//! The bundled fonts have no play triangle, stop square or dice, so those are drawn as shapes.

use egui::{Align2, Color32, Pos2, Rect, Response, Sense, Shape, Stroke, StrokeKind, Ui, Vec2};
use mosh_core::LayerKind;

use crate::editor::theme;

/// A tiny grey heading over a control group.
pub fn caption(painter: &egui::Painter, pos: Pos2, text: &str) {
    painter.text(
        pos,
        Align2::LEFT_TOP,
        text,
        egui::FontId::proportional(9.5),
        theme::FAINT,
    );
}

/// One line of text in `rect`. Text that is too long for it ends in an ellipsis instead of
/// running over its neighbour.
pub fn clipped_text(
    painter: &egui::Painter,
    rect: Rect,
    anchor: Align2,
    text: &str,
    font: egui::FontId,
    color: Color32,
) {
    let mut job = egui::text::LayoutJob::simple_singleline(text.to_string(), font, color);
    job.wrap = egui::text::TextWrapping {
        max_width: rect.width().max(0.0),
        max_rows: 1,
        break_anywhere: true,
        overflow_character: Some('\u{2026}'),
    };
    let galley = painter.layout_job(job);
    let pos = anchor.align_size_within_rect(galley.size(), rect).min;
    painter
        .with_clip_rect(rect.intersect(painter.clip_rect()))
        .galley(pos, galley, color);
}

/// A row of mutually exclusive choices in `rect`. Returns the one that was clicked.
pub fn segmented(
    ui: &mut Ui,
    rect: Rect,
    id: impl std::hash::Hash + std::fmt::Debug,
    labels: &[&str],
    selected: Option<usize>,
    tips: &[&str],
) -> Option<usize> {
    let mut clicked = None;
    let width = rect.width() / labels.len().max(1) as f32;
    ui.painter().rect_filled(rect, theme::RADIUS, theme::RAISED);
    // Words in narrow cells are all set smaller, rather than some of them running into their
    // neighbours (numbers and three-letter labels fit as they are).
    let small = width < 40.0 && labels.iter().any(|label| label.len() > 3);
    for (index, label) in labels.iter().enumerate() {
        let cell = Rect::from_min_size(
            Pos2::new(rect.left() + width * index as f32, rect.top()),
            Vec2::new(width, rect.height()),
        );
        let mut response = ui.interact(cell, ui.scope_id().with((&id, index)), Sense::click());
        if let Some(tip) = tips.get(index) {
            response = response.on_hover_text(*tip);
        }
        let on = selected == Some(index);
        let fill = if on {
            theme::ACCENT
        } else if response.hovered() {
            theme::HOVER
        } else {
            Color32::TRANSPARENT
        };
        ui.painter().rect_filled(cell.shrink(1.0), theme::RADIUS - 1.0, fill);
        ui.painter().text(
            cell.center(),
            Align2::CENTER_CENTER,
            *label,
            if small { theme::small() } else { theme::body() },
            if on { theme::ON_ACCENT } else { theme::TEXT },
        );
        if response.clicked() {
            clicked = Some(index);
        }
    }
    clicked
}

/// A pill that is on or off. Takes its width from the label.
pub fn chip(ui: &mut Ui, label: &str, on: bool) -> Response {
    let galley = ui
        .painter()
        .layout_no_wrap(label.to_string(), theme::body(), theme::TEXT);
    let size = Vec2::new(galley.size().x + theme::GAP * 4.0, 22.0);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    let fill = if on {
        theme::ACCENT_DIM
    } else if response.hovered() {
        theme::HOVER
    } else {
        theme::RAISED
    };
    ui.painter().rect_filled(rect, 11.0, fill);
    if on {
        ui.painter().rect_stroke(
            rect,
            11.0,
            Stroke::new(1.0, theme::ACCENT),
            StrokeKind::Inside,
        );
    }
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        label,
        theme::body(),
        if on { theme::TEXT } else { theme::DIM },
    );
    response
}

/// A flat button in `rect` with a text label; `on` lights it.
pub fn flat_button(
    ui: &mut Ui,
    rect: Rect,
    id: impl std::hash::Hash + std::fmt::Debug,
    label: &str,
    on: bool,
) -> Response {
    let response = ui.interact(rect, ui.scope_id().with(id), Sense::click());
    let fill = if on {
        theme::ACCENT_DIM
    } else if response.hovered() {
        theme::HOVER
    } else {
        theme::RAISED
    };
    ui.painter().rect_filled(rect, theme::RADIUS, fill);
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        label,
        theme::body(),
        theme::TEXT,
    );
    response
}

/// A play triangle centred in `rect`.
pub fn play_icon(painter: &egui::Painter, rect: Rect, color: Color32) {
    let c = rect.center();
    let r = rect.width().min(rect.height()) * 0.26;
    painter.add(Shape::convex_polygon(
        vec![
            Pos2::new(c.x - r * 0.8, c.y - r),
            Pos2::new(c.x + r, c.y),
            Pos2::new(c.x - r * 0.8, c.y + r),
        ],
        color,
        Stroke::NONE,
    ));
}

/// A stop square centred in `rect`.
pub fn stop_icon(painter: &egui::Painter, rect: Rect, color: Color32) {
    let r = rect.width().min(rect.height()) * 0.2;
    painter.rect_filled(Rect::from_center_size(rect.center(), Vec2::splat(r * 2.0)), 1.0, color);
}

/// A die showing five, centred in `rect`: "roll a new seed".
pub fn dice_icon(painter: &egui::Painter, rect: Rect, color: Color32) {
    let side = rect.width().min(rect.height()) * 0.62;
    let body = Rect::from_center_size(rect.center(), Vec2::splat(side));
    painter.rect_stroke(body, 3.0, Stroke::new(1.5, color), StrokeKind::Inside);
    let d = side * 0.24;
    for (dx, dy) in [(-d, -d), (d, -d), (0.0, 0.0), (-d, d), (d, d)] {
        painter.circle_filled(body.center() + Vec2::new(dx, dy), side * 0.075, color);
    }
}

/// A plus centred in `rect`.
pub fn plus_icon(painter: &egui::Painter, rect: Rect, color: Color32) {
    let c = rect.center();
    let r = rect.width().min(rect.height()) * 0.22;
    let stroke = Stroke::new(1.5, color);
    painter.line_segment([Pos2::new(c.x - r, c.y), Pos2::new(c.x + r, c.y)], stroke);
    painter.line_segment([Pos2::new(c.x, c.y - r), Pos2::new(c.x, c.y + r)], stroke);
}

/// A square icon button; `paint` draws the icon.
pub fn icon_button(
    ui: &mut Ui,
    rect: Rect,
    id: impl std::hash::Hash + std::fmt::Debug,
    fill: Color32,
    ink: Color32,
    paint: fn(&egui::Painter, Rect, Color32),
) -> Response {
    let response = ui.interact(rect, ui.scope_id().with(id), Sense::click());
    let fill = if response.hovered() && fill != theme::ACCENT {
        theme::HOVER
    } else {
        fill
    };
    ui.painter().rect_filled(rect, theme::RADIUS, fill);
    paint(ui.painter(), rect, ink);
    response
}

/// The short badge of a layer kind, for headers and lanes.
pub fn kind_label(kind: LayerKind) -> &'static str {
    match kind {
        LayerKind::Visualizer => "Visualizer",
        LayerKind::Overlay => "Video",
        LayerKind::Adjustment => "Adjustment",
        LayerKind::Mask => "Mask",
    }
}

/// What a layer kind does, for the kind picker's hover text.
pub fn kind_tip(kind: LayerKind) -> &'static str {
    match kind {
        LayerKind::Visualizer => "draws its clip's visualizer",
        LayerKind::Overlay => "draws its clip's video on top",
        LayerKind::Adjustment => {
            "draws nothing itself: its clip's effects apply to every layer below"
        }
        LayerKind::Mask => "not shown itself: its brightness masks the layer above",
    }
}

pub const LAYER_KINDS: [LayerKind; 4] = [
    LayerKind::Visualizer,
    LayerKind::Overlay,
    LayerKind::Adjustment,
    LayerKind::Mask,
];

/// Mark `rect` as belonging to an adjustment or a mask layer, so the two read differently from
/// a layer that draws: an adjustment is hatched (it lies *over* what is below), a mask gets a
/// dashed outline (a stencil).
pub fn kind_texture(painter: &egui::Painter, rect: Rect, kind: LayerKind) {
    match kind {
        LayerKind::Visualizer | LayerKind::Overlay => {}
        LayerKind::Adjustment => {
            let painter = painter.with_clip_rect(rect.intersect(painter.clip_rect()));
            let stroke = Stroke::new(1.0, theme::alpha(theme::TEXT, 0.07));
            let step = 8.0;
            let mut x = rect.left() - rect.height();
            while x < rect.right() {
                painter.line_segment(
                    [
                        Pos2::new(x, rect.bottom()),
                        Pos2::new(x + rect.height(), rect.top()),
                    ],
                    stroke,
                );
                x += step;
            }
        }
        LayerKind::Mask => {
            let stroke = Stroke::new(1.0, theme::alpha(theme::TEXT, 0.22));
            let inner = rect.shrink(2.0);
            let dash = 5.0;
            let mut x = inner.left();
            while x < inner.right() {
                let end = (x + dash).min(inner.right());
                painter.line_segment([Pos2::new(x, inner.top()), Pos2::new(end, inner.top())], stroke);
                painter.line_segment(
                    [Pos2::new(x, inner.bottom()), Pos2::new(end, inner.bottom())],
                    stroke,
                );
                x += dash * 2.0;
            }
            let mut y = inner.top();
            while y < inner.bottom() {
                let end = (y + dash).min(inner.bottom());
                painter.line_segment([Pos2::new(inner.left(), y), Pos2::new(inner.left(), end)], stroke);
                painter.line_segment(
                    [Pos2::new(inner.right(), y), Pos2::new(inner.right(), end)],
                    stroke,
                );
                y += dash * 2.0;
            }
        }
    }
}

/// A text field over a value the model owns.
///
/// Typing is not an edit until it is one: while the field has focus it shows its own draft (so a
/// trailing space or an emptied field is not "corrected" by the model mid-word), and every change
/// is returned for the caller to turn into a command.
pub struct TextDraft {
    key: u64,
    text: String,
}

pub fn text_field(
    ui: &mut Ui,
    draft: &mut Option<TextDraft>,
    key: u64,
    current: &str,
    width: f32,
    hint: &str,
    rows: usize,
) -> Option<String> {
    let mut text = match draft {
        Some(draft) if draft.key == key => draft.text.clone(),
        _ => current.to_string(),
    };
    let edit = if rows > 1 {
        egui::TextEdit::multiline(&mut text).desired_rows(rows)
    } else {
        egui::TextEdit::singleline(&mut text)
    };
    let response = ui.add(
        edit.id_salt(("field", key))
            .desired_width(width)
            .hint_text(hint.to_string()),
    );
    let mut changed = None;
    if response.changed() {
        *draft = Some(TextDraft {
            key,
            text: text.clone(),
        });
        changed = Some(text);
    } else if response.gained_focus() {
        *draft = Some(TextDraft { key, text });
    }
    if response.lost_focus()
        && draft.as_ref().is_some_and(|draft| draft.key == key)
    {
        *draft = None;
    }
    changed
}

/// A stable key for a [`text_field`].
pub fn field_key(parts: impl std::hash::Hash) -> u64 {
    use std::hash::Hasher;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    parts.hash(&mut hasher);
    hasher.finish()
}
