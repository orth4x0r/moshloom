//! The top bar: where the show is and what it is doing, readable from a step back.
//!
//! Left to right, in the order the questions come during a rehearsal: which view am I in, what
//! drives the show, is Live connected, how fast, where, which song, how loud does the analysis
//! think it is, did it just see a build or a drop, and is the frame inside its budget. The stage
//! preview and the stage button sit at the right end.
//!
//! What fits is decided by [`top_bar_plan`], a pure function of the width: a narrow window drops
//! the brand, the song, the frame times and the preview, in that order, and never overlaps.

use egui::{Align2, Pos2, Rect, Vec2};
use mosh_core::ShowMode;

use crate::budget::{CPU_MS_PER_FRAME, GPU_MS_PER_FRAME};
use crate::core::SourceKind;
use crate::editor::model::{Command, EditorModel, View, bar_len_at};
use crate::editor::theme;
use crate::editor::ui::widgets::{self, caption, clipped_text};
use crate::editor::ui::{EditorUi, UiAction};
use crate::preview::fit;

const BRAND: f32 = 104.0;
const TEMPO: f32 = 62.0;
const POSITION: f32 = 70.0;
const FRAME: f32 = 74.0;
const PREVIEW: f32 = 86.0;
const STAGE: f32 = 84.0;
const HELP: f32 = 26.0;

/// What the top bar shows at a given width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TopBarPlan {
    pub brand: bool,
    /// Short labels on the view switch, the mode switch and the lamps.
    pub compact: bool,
    /// The song name's width; 0 hides it.
    pub song: f32,
    pub frame: bool,
    pub preview: bool,
    /// The width everything shown adds up to, gaps included.
    pub used: f32,
}

impl TopBarPlan {
    fn view(self) -> f32 {
        if self.compact { 2.0 * 46.0 } else { 2.0 * 96.0 }
    }

    fn mode(self) -> f32 {
        if self.compact { 3.0 * 42.0 } else { 3.0 * 68.0 }
    }

    fn lamps(self) -> f32 {
        if self.compact { 3.0 * 26.0 } else { 3.0 * 52.0 }
    }

    fn source(self) -> f32 {
        if self.compact { 62.0 } else { 78.0 }
    }

    fn intensity(self) -> f32 {
        if self.compact { 84.0 } else { 104.0 }
    }

    /// The gap between groups.
    fn pad(self) -> f32 {
        if self.compact { 8.0 } else { 12.0 }
    }
}

/// Decide what fits in `width`: the essentials always (view, mode, source, tempo, position,
/// intensity, lamps, stage, help), then the preview, the frame times, the song and the brand as
/// room allows.
pub fn top_bar_plan(width: f32) -> TopBarPlan {
    let total = |plan: TopBarPlan| {
        let mut items = vec![
            plan.view(),
            plan.mode(),
            plan.source(),
            TEMPO,
            POSITION,
            plan.intensity(),
            plan.lamps(),
            STAGE,
            HELP,
        ];
        if plan.brand {
            items.push(BRAND);
        }
        if plan.song > 0.0 {
            items.push(plan.song);
        }
        if plan.frame {
            items.push(FRAME);
        }
        if plan.preview {
            items.push(PREVIEW);
        }
        items.iter().sum::<f32>() + plan.pad() * (items.len() as f32 + 1.0)
    };
    let mut plan = TopBarPlan {
        brand: false,
        compact: true,
        song: 0.0,
        frame: false,
        preview: false,
        used: 0.0,
    };
    // Each step is kept only if the bar still fits with it.
    let steps: [fn(&mut TopBarPlan); 5] = [
        |plan| plan.preview = true,
        |plan| plan.compact = false,
        |plan| plan.frame = true,
        |plan| plan.song = 96.0,
        |plan| plan.brand = true,
    ];
    for step in steps {
        let mut next = plan;
        step(&mut next);
        if total(next) <= width {
            plan = next;
        }
    }
    // Whatever is left over goes to the song name, which is the one thing that can be long.
    if plan.song > 0.0 {
        plan.song = (plan.song + (width - total(plan))).clamp(96.0, 220.0);
    }
    plan.used = total(plan);
    plan
}

/// Bar and beat as Live's transport shows them: `17.3` is the third beat of bar 17.
pub fn position_label(bar: i64, bar_phase: f32, beats_per_bar: f64) -> String {
    let beats = beats_per_bar.max(1.0);
    let beat = ((f64::from(bar_phase.clamp(0.0, 0.999_9)) * beats).floor() as i64 + 1).max(1);
    format!("{}.{}", bar + 1, beat)
}

pub fn draw(ui: &mut egui::Ui, state: &mut EditorUi, model: &EditorModel, out: &mut Vec<Command>) {
    let rect = ui.max_rect();
    let plan = top_bar_plan(rect.width());
    let status = *model.status();
    let project = model.project();
    let painter = ui.painter().clone();
    painter.hline(rect.x_range(), rect.bottom() - 0.5, theme::line());

    let top = rect.top() + 7.0;
    let control_top = rect.top() + 22.0;
    let control_h = 26.0;
    let pad = plan.pad();
    let mut x = rect.left() + pad;
    // A group: its caption, and the rectangle its control goes in.
    let group = |x: &mut f32, width: f32, title: &str| -> Rect {
        caption(&painter, Pos2::new(*x, top), title);
        let area = Rect::from_min_size(Pos2::new(*x, control_top), Vec2::new(width, control_h));
        *x += width + pad;
        area
    };

    if plan.brand {
        let area = Rect::from_min_size(Pos2::new(x, rect.top()), Vec2::new(BRAND, rect.height()));
        painter.text(
            Pos2::new(area.left(), rect.top() + 12.0),
            Align2::LEFT_TOP,
            "ilanaVisual",
            egui::FontId::proportional(17.0),
            theme::TEXT,
        );
        painter.text(
            Pos2::new(area.left() + 1.0, rect.top() + 34.0),
            Align2::LEFT_TOP,
            "ILANA AUDIO",
            egui::FontId::proportional(9.5),
            theme::FAINT,
        );
        x += BRAND + pad;
    }

    // View: the two main views, Tab toggles.
    let area = group(&mut x, plan.view(), "VIEW  (TAB)");
    let labels: [&str; 2] = if plan.compact {
        ["Arr", "Ses"]
    } else {
        ["Arrangement", "Session"]
    };
    let selected = match model.view() {
        View::Arrangement => 0,
        View::Session => 1,
    };
    if let Some(index) = widgets::segmented(
        ui,
        area,
        "view",
        &labels,
        Some(selected),
        &[
            "the timeline: lanes, blocks, song markers (Tab)",
            "the clip grid: layers x scenes (Tab)",
        ],
    ) {
        out.push(Command::SelectView(if index == 0 {
            View::Arrangement
        } else {
            View::Session
        }));
    }

    // Show mode: what drives the stage.
    let area = group(&mut x, plan.mode(), "SHOW MODE  (M)");
    let labels: [&str; 3] = if plan.compact {
        ["Arr", "Ses", "Auto"]
    } else {
        ["Follow", "Manual", "Autopilot"]
    };
    let modes = [ShowMode::Arrangement, ShowMode::Session, ShowMode::Autopilot];
    if let Some(index) = widgets::segmented(
        ui,
        area,
        "mode",
        &labels,
        modes.iter().position(|mode| *mode == project.mode),
        &[
            "Arrangement: the timeline follows Live's playhead",
            "Session: clips are launched by hand",
            "Autopilot: clips launch themselves on drops and level changes",
        ],
    ) {
        out.push(Command::SetMode(modes[index]));
    }

    // Source: is Live connected.
    let area = group(&mut x, plan.source(), "SOURCE");
    let (dot, label) = match status.source.0 {
        SourceKind::Plugin => (theme::ACCENT, "Live"),
        SourceKind::Loopback => (theme::WARN, if plan.compact { "loop" } else { "loopback" }),
        SourceKind::Silent => (theme::FAINT, "silent"),
    };
    painter.circle_filled(Pos2::new(area.left() + 5.0, area.center().y), 4.0, dot);
    painter.text(
        Pos2::new(area.left() + 16.0, area.center().y),
        Align2::LEFT_CENTER,
        label,
        theme::strong(),
        theme::TEXT,
    );
    ui.interact(area, ui.scope_id().with("source"), egui::Sense::hover())
        .on_hover_text(match status.source.0 {
            SourceKind::Plugin => "the ilanaVisual Tap plugin on Live's master: audio and transport",
            SourceKind::Loopback => {
                "no plugin: listening to the sound card, tempo and bars are estimated"
            }
            SourceKind::Silent => "no audio at all: the show keeps drawing",
        });

    let area = group(&mut x, TEMPO, "BPM");
    painter.text(
        area.left_center(),
        Align2::LEFT_CENTER,
        format!("{:.1}", status.tempo_bpm),
        theme::big(),
        theme::TEXT,
    );

    // Position, with the bar's progress under it.
    let area = group(&mut x, POSITION, "BAR . BEAT");
    let beats = bar_len_at(&project.arrangement.sig_map, status.ppq);
    painter.text(
        area.left_center(),
        Align2::LEFT_CENTER,
        position_label(status.bar, status.bar_phase, beats),
        theme::big(),
        if status.transport {
            theme::ACCENT
        } else {
            theme::TEXT
        },
    );
    let track = Rect::from_min_size(
        Pos2::new(area.left(), area.bottom() + 1.0),
        Vec2::new(POSITION - 8.0, 2.0),
    );
    painter.rect_filled(track, 1.0, theme::RAISED);
    painter.rect_filled(
        Rect::from_min_size(
            track.min,
            Vec2::new(track.width() * status.bar_phase.clamp(0.0, 1.0), 2.0),
        ),
        1.0,
        theme::ACCENT,
    );

    if plan.song > 0.0 {
        let area = group(&mut x, plan.song, "SONG");
        let name = status
            .song
            .and_then(|song| project.arrangement.songs.get(usize::from(song)))
            .map(|song| song.name.as_str());
        clipped_text(
            &painter,
            area,
            Align2::LEFT_CENTER,
            name.unwrap_or("-"),
            theme::strong(),
            if name.is_some() { theme::TEXT } else { theme::FAINT },
        );
    }

    // Intensity: the continuous meter, and the level 1-4 it was quantized to.
    let intensity = plan.intensity();
    let area = group(&mut x, intensity, "INTENSITY");
    let meter = Rect::from_min_size(
        Pos2::new(area.left(), area.top() + 4.0),
        Vec2::new(intensity - 4.0, 6.0),
    );
    painter.rect_filled(meter, 3.0, theme::RAISED);
    painter.rect_filled(
        Rect::from_min_size(
            meter.min,
            Vec2::new(meter.width() * status.intensity.clamp(0.0, 1.0), 6.0),
        ),
        3.0,
        theme::ACCENT,
    );
    let pip = (intensity - 4.0 - 3.0 * 3.0) / 4.0;
    for level in 1..=4u8 {
        let cell = Rect::from_min_size(
            Pos2::new(
                area.left() + (pip + 3.0) * f32::from(level - 1),
                area.top() + 14.0,
            ),
            Vec2::new(pip, 12.0),
        );
        let lit = level <= status.level;
        painter.rect_filled(cell, 2.0, if lit { theme::ACCENT } else { theme::RAISED });
        painter.text(
            cell.center(),
            Align2::CENTER_CENTER,
            level,
            egui::FontId::proportional(9.5),
            if lit { theme::ON_ACCENT } else { theme::FAINT },
        );
    }

    // Event lamps: lit by the event, fading after it.
    let area = group(&mut x, plan.lamps(), "EVENTS");
    let names: [&str; 3] = if plan.compact {
        ["B", "D", "K"]
    } else {
        ["BUILD", "DROP", "BREAK"]
    };
    let lamp = area.width() / 3.0;
    for (index, name) in names.iter().enumerate() {
        let cell = Rect::from_min_size(
            Pos2::new(area.left() + lamp * index as f32, area.top() + 3.0),
            Vec2::new(lamp - 3.0, 20.0),
        );
        let level = status.lamps[index].clamp(0.0, 1.0);
        painter.rect_filled(cell, theme::RADIUS, theme::RAISED);
        if level > 0.0 {
            painter.rect_filled(cell, theme::RADIUS, theme::alpha(theme::ACCENT, level));
        }
        painter.text(
            cell.center(),
            Align2::CENTER_CENTER,
            *name,
            egui::FontId::proportional(10.0),
            if level > 0.5 {
                theme::ON_ACCENT
            } else {
                theme::DIM
            },
        );
    }
    if status.locked {
        painter.text(
            Pos2::new(area.left(), rect.bottom() - 3.0),
            Align2::LEFT_BOTTOM,
            "LOCKED (L)",
            egui::FontId::proportional(9.0),
            theme::WARN,
        );
    }

    if plan.frame {
        let area = group(&mut x, FRAME, "FRAME  ms");
        let line = |y: f32, name: &str, value: f32, budget: f32| {
            painter.text(
                Pos2::new(area.left(), y),
                Align2::LEFT_TOP,
                name,
                theme::small(),
                theme::FAINT,
            );
            painter.text(
                Pos2::new(area.left() + 30.0, y),
                Align2::LEFT_TOP,
                format!("{value:.1}"),
                theme::small(),
                if value > budget { theme::WARN } else { theme::DIM },
            );
        };
        line(area.top(), "GPU", status.gpu_ms, GPU_MS_PER_FRAME);
        line(area.top() + 13.0, "CPU", status.cpu_ms, CPU_MS_PER_FRAME);
        ui.interact(area, ui.scope_id().with("frame"), egui::Sense::hover())
            .on_hover_text("the stage's frame: GPU under 6 ms, render thread under 2 ms");
    }

    // The right end: help, the stage button, the preview.
    let mut right = rect.right() - pad;
    let help = Rect::from_min_size(
        Pos2::new(right - HELP, control_top),
        Vec2::new(HELP, control_h),
    );
    if widgets::flat_button(ui, help, "help", "?", model.panels().help)
        .on_hover_text("shortcuts (?)")
        .clicked()
    {
        out.push(Command::SetHelp(!model.panels().help));
    }
    right = help.left() - pad;

    let stage = Rect::from_min_size(
        Pos2::new(right - STAGE, control_top),
        Vec2::new(STAGE, control_h),
    );
    caption(&painter, Pos2::new(stage.left(), top), "STAGE  (F11)");
    let label = if state.stage.fullscreen {
        "Window"
    } else {
        "Fullscreen"
    };
    if widgets::flat_button(ui, stage, "stage", label, state.stage.fullscreen)
        .on_hover_text(if state.stage.fullscreen {
            "put the stage back in a window (F11, or Esc on the stage)"
        } else if state.stage.shared_monitor {
            "stage fullscreen on this screen: Esc brings the controls back (F11)"
        } else {
            "stage fullscreen on its monitor (F11)"
        })
        .clicked()
    {
        state.actions.push(UiAction::ToggleStageFullscreen);
    }
    right = stage.left() - pad;

    if plan.preview {
        let slot = Rect::from_min_size(
            Pos2::new(right - PREVIEW, rect.top() + 4.0),
            Vec2::new(PREVIEW, rect.height() - 9.0),
        );
        let on = state.preview_enabled();
        let response = ui
            .interact(slot, ui.scope_id().with("preview"), egui::Sense::click())
            .on_hover_text(if on {
                "what the stage shows, a few times a second (click or P: off)"
            } else {
                "stage preview is off (click or P: on)"
            });
        painter.rect_filled(slot, 3.0, theme::SLOT);
        match state.preview.as_ref() {
            Some((texture, aspect)) if on => {
                let (dx, dy, width, height) = fit(*aspect, (slot.width(), slot.height()));
                let picture = Rect::from_min_size(
                    slot.min + Vec2::new(dx, dy),
                    Vec2::new(width, height),
                );
                painter.image(
                    texture.id(),
                    picture,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            }
            _ => {
                painter.text(
                    slot.center(),
                    Align2::CENTER_CENTER,
                    if on { "stage" } else { "preview (P)" },
                    theme::small(),
                    theme::FAINT,
                );
            }
        }
        painter.rect_stroke(
            slot,
            3.0,
            if response.hovered() {
                egui::Stroke::new(1.0, theme::FAINT)
            } else {
                theme::line()
            },
            egui::StrokeKind::Inside,
        );
        if response.clicked() {
            state.preview_on = Some(!on);
        }
        state.preview_shown = on;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_top_bar_fits_every_width_it_is_given() {
        for width in (800..=2560).step_by(20) {
            let plan = top_bar_plan(width as f32);
            assert!(
                plan.used <= width as f32,
                "{width}: {plan:?} needs {} px",
                plan.used
            );
        }
        // 800 wide: the essentials, with short labels.
        let narrow = top_bar_plan(800.0);
        assert!(narrow.compact && !narrow.brand && narrow.song == 0.0 && !narrow.frame);
        // 1280: full labels and the preview.
        let laptop = top_bar_plan(1280.0);
        assert!(!laptop.compact && laptop.preview && laptop.frame, "{laptop:?}");
        // 1920: everything, and the song name gets the spare room.
        let wide = top_bar_plan(1920.0);
        assert!(wide.brand && wide.preview && wide.frame && wide.song >= 200.0, "{wide:?}");
    }

    #[test]
    fn the_position_reads_like_lives_transport() {
        assert_eq!(position_label(0, 0.0, 4.0), "1.1");
        assert_eq!(position_label(16, 0.5, 4.0), "17.3");
        assert_eq!(position_label(16, 0.99, 4.0), "17.4");
        // The phase can touch 1.0 on the last frame of a bar: still beat 4, not beat 5.
        assert_eq!(position_label(16, 1.0, 4.0), "17.4");
        // 6/8 is three quarter-note beats to the bar.
        assert_eq!(position_label(3, 0.7, 3.0), "4.3");
    }
}
