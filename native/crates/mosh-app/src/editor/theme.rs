//! The control window's look: one dark palette, one accent, one spacing scale.
//!
//! Everything a view paints takes its colours and sizes from here, so the two views, the browser
//! and the detail panel cannot drift apart. The rules:
//!
//! * **Dark, three surfaces.** The window (`BG`), panels and lanes (`PANEL`), things that sit on
//!   them (`RAISED`). Lines are one colour.
//! * **One accent** ([`ACCENT`]) means "live": what is playing, what is selected for launch, the
//!   playhead, the lit lamps, the level meter. Nothing else uses it.
//! * **Clip colours are data, not decoration**: a clip's hue comes from its visualizer
//!   ([`clip_color`]), so the same scene is the same colour in the browser, the grid and the
//!   timeline.
//! * **A 4 px grid**: gaps are [`GAP`] or multiples; rows are [`ROW`] tall.
//! * **Readable from a step back**: body text is 13 px, the tempo and the bar counter are 20 px.

use egui::{Color32, FontId, Stroke};

pub const BG: Color32 = Color32::from_rgb(0x10, 0x12, 0x16);
pub const PANEL: Color32 = Color32::from_rgb(0x17, 0x1a, 0x20);
pub const RAISED: Color32 = Color32::from_rgb(0x21, 0x25, 0x2e);
pub const HOVER: Color32 = Color32::from_rgb(0x2c, 0x32, 0x3d);
pub const LINE: Color32 = Color32::from_rgb(0x2e, 0x34, 0x3f);
pub const SLOT: Color32 = Color32::from_rgb(0x13, 0x15, 0x1a);

pub const TEXT: Color32 = Color32::from_rgb(0xe6, 0xe9, 0xee);
pub const DIM: Color32 = Color32::from_rgb(0x9a, 0xa3, 0xb2);
pub const FAINT: Color32 = Color32::from_rgb(0x66, 0x70, 0x80);

/// "Live": playing, the playhead, lit lamps, the level meter.
pub const ACCENT: Color32 = Color32::from_rgb(0x5e, 0xe6, 0xc8);
pub const ACCENT_DIM: Color32 = Color32::from_rgb(0x24, 0x5c, 0x52);
/// Text on an accent fill.
pub const ON_ACCENT: Color32 = Color32::from_rgb(0x06, 0x1a, 0x16);
/// Errors, and a queued stop.
pub const DANGER: Color32 = Color32::from_rgb(0xff, 0x6b, 0x6b);
/// Over budget.
pub const WARN: Color32 = Color32::from_rgb(0xff, 0xc8, 0x57);

/// The spacing unit.
pub const GAP: f32 = 4.0;
/// A row of controls.
pub const ROW: f32 = 26.0;
/// The top bar.
pub const TOP_BAR: f32 = 56.0;
/// Corner radius of blocks, cells and buttons.
pub const RADIUS: f32 = 4.0;

pub fn small() -> FontId {
    FontId::proportional(11.0)
}

pub fn body() -> FontId {
    FontId::proportional(13.0)
}

pub fn strong() -> FontId {
    FontId::proportional(14.0)
}

/// The numbers read from a step back: tempo, bar.beat.
pub fn big() -> FontId {
    FontId::proportional(20.0)
}

pub fn line() -> Stroke {
    Stroke::new(1.0, LINE)
}

/// A clip's colour: a hue per visualizer id, the same everywhere the clip appears.
///
/// The hue is a hash of the id (FNV-1a) spread with the golden angle, so neighbours in the
/// catalog do not get neighbouring hues and a new scene gets a colour without a table to edit.
pub fn clip_color(visualizer: &str) -> Color32 {
    // A catalog entry takes its hue from its position, stepped by the golden ratio: every entry
    // is a different colour however many style packs land, and the same entry is always the same
    // colour. A hash (the old rule) put several of 37 entries on one RGB colour.
    let hue = match crate::editor::model::catalog()
        .iter()
        .position(|entry| entry.id == visualizer)
    {
        Some(index) => (index as f32 * 0.618_034).fract(),
        None => {
            let mut hash: u32 = 0x811c_9dc5;
            for byte in visualizer.bytes() {
                hash ^= u32::from(byte);
                hash = hash.wrapping_mul(0x0100_0193);
            }
            (f64::from(hash) / f64::from(u32::MAX)) as f32
        }
    };
    egui::ecolor::Hsva::new(hue, 0.52, 0.78, 1.0).into()
}

/// `color` towards the panel colour, for a block's body under its title strip.
pub fn dimmed(color: Color32, keep: f32) -> Color32 {
    let mix = |a: u8, b: u8| (f32::from(a) * keep + f32::from(b) * (1.0 - keep)).round() as u8;
    Color32::from_rgb(
        mix(color.r(), PANEL.r()),
        mix(color.g(), PANEL.g()),
        mix(color.b(), PANEL.b()),
    )
}

/// `color` at `alpha` (0..1) over whatever is under it.
pub fn alpha(color: Color32, alpha: f32) -> Color32 {
    Color32::from_rgba_unmultiplied(
        color.r(),
        color.g(),
        color.b(),
        (alpha.clamp(0.0, 1.0) * 255.0).round() as u8,
    )
}

/// Set the egui style for the control window: called once, when the window is built (and by the
/// headless tests, so they lay out what the window lays out).
pub fn apply(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = PANEL;
    visuals.window_fill = PANEL;
    visuals.extreme_bg_color = SLOT;
    visuals.faint_bg_color = RAISED;
    visuals.override_text_color = Some(TEXT);
    visuals.selection.bg_fill = ACCENT_DIM;
    visuals.selection.stroke = Stroke::new(1.0, ACCENT);
    visuals.window_stroke = line();
    visuals.slider_trailing_fill = true;
    visuals.widgets.noninteractive.bg_fill = PANEL;
    visuals.widgets.noninteractive.bg_stroke = line();
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, DIM);
    visuals.widgets.inactive.bg_fill = RAISED;
    visuals.widgets.inactive.weak_bg_fill = RAISED;
    visuals.widgets.inactive.bg_stroke = Stroke::NONE;
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT);
    visuals.widgets.hovered.bg_fill = HOVER;
    visuals.widgets.hovered.weak_bg_fill = HOVER;
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, FAINT);
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, TEXT);
    visuals.widgets.active.bg_fill = ACCENT_DIM;
    visuals.widgets.active.weak_bg_fill = ACCENT_DIM;
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
    visuals.widgets.active.fg_stroke = Stroke::new(1.0, TEXT);
    visuals.widgets.open.bg_fill = HOVER;
    visuals.widgets.open.weak_bg_fill = HOVER;
    for widget in [
        &mut visuals.widgets.noninteractive,
        &mut visuals.widgets.inactive,
        &mut visuals.widgets.hovered,
        &mut visuals.widgets.active,
        &mut visuals.widgets.open,
    ] {
        widget.corner_radius = RADIUS.into();
    }
    ctx.set_visuals(visuals);
    ctx.all_styles_mut(|style| {
        style.spacing.item_spacing = egui::vec2(GAP * 2.0, GAP);
        style.spacing.button_padding = egui::vec2(GAP * 2.0, GAP);
        style.spacing.interact_size = egui::vec2(32.0, 22.0);
        style.spacing.slider_width = 120.0;
        style.spacing.combo_width = 96.0;
        style
            .text_styles
            .insert(egui::TextStyle::Body, body());
        style
            .text_styles
            .insert(egui::TextStyle::Button, body());
        style
            .text_styles
            .insert(egui::TextStyle::Small, small());
        style
            .text_styles
            .insert(egui::TextStyle::Heading, FontId::proportional(16.0));
        style
            .text_styles
            .insert(egui::TextStyle::Monospace, FontId::monospace(12.0));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WCAG relative luminance of an sRGB colour.
    fn luminance(color: Color32) -> f32 {
        let channel = |value: u8| {
            let value = f32::from(value) / 255.0;
            if value <= 0.040_45 {
                value / 12.92
            } else {
                ((value + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * channel(color.r()) + 0.7152 * channel(color.g()) + 0.0722 * channel(color.b())
    }

    fn contrast(a: Color32, b: Color32) -> f32 {
        let (hi, lo) = if luminance(a) > luminance(b) {
            (luminance(a), luminance(b))
        } else {
            (luminance(b), luminance(a))
        };
        (hi + 0.05) / (lo + 0.05)
    }

    #[test]
    fn text_is_readable_on_every_surface() {
        for surface in [BG, PANEL, RAISED, HOVER, SLOT] {
            assert!(contrast(TEXT, surface) >= 7.0, "text on {surface:?}");
            assert!(contrast(DIM, surface) >= 4.5, "dim text on {surface:?}");
            assert!(contrast(ACCENT, surface) >= 4.5, "accent on {surface:?}");
        }
        // Labels sit on accent fills (a playing cell, a lit lamp) and on clip colours.
        assert!(contrast(ON_ACCENT, ACCENT) >= 7.0);
        for entry in crate::editor::model::catalog() {
            let color = clip_color(entry.id);
            assert!(
                contrast(ON_ACCENT, color) >= 4.5,
                "{}: block title on {color:?}",
                entry.id
            );
        }
    }

    #[test]
    fn a_clip_colour_is_stable_and_spread() {
        assert_eq!(clip_color("tunnel"), clip_color("tunnel"));
        let mut distinct: Vec<Color32> = crate::editor::model::catalog()
            .iter()
            .map(|entry| clip_color(entry.id))
            .collect();
        let total = distinct.len();
        distinct.sort_by_key(|color| color.to_array());
        distinct.dedup();
        // The catalog grows with every style pack: most clips must still look different.
        assert!(distinct.len() * 10 >= total * 9, "{} of {total} hues", distinct.len());
    }
}
