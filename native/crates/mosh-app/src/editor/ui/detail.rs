//! The detail panel: the selected clip, left to right in the order a look is built.
//!
//! ```text
//! | CLIP              | PARAMETERS   level [1][2][3][4]  pulse | EFFECTS      | TRANSITION / FOLLOW |
//! | picture, name     | speed    ====o====   lo  hi            | chips        | kind, beats         |
//! | visualizer, seed  | density  ==o======   lo  hi            | amount/param | follow, after, react|
//! | source, preset    | ...                                    |              |                     |
//! ```
//!
//! A parameter is a position (the slider) inside a range (lo, hi) that depends on the intensity
//! level: the level tabs choose which level's ranges and pulse depth are shown, and follow the
//! live level until one is clicked. A slider of a clip that is on stage moves the stage at once;
//! the project itself still lands on the next bar.

use egui::{Align2, Pos2, Rect, Sense, Stroke, Vec2};
use mosh_core::{Clip, EffectKind, FollowAction, TransitionKind};

use crate::editor::model::{Command, EditorModel, MediaKind, catalog, catalog_entry, param_labels};
use crate::editor::theme;
use crate::editor::ui::widgets::{self, caption};
use crate::editor::ui::{EditorUi, clip_slot, thumb_params};

pub const HEIGHT: f32 = 268.0;
const CLIP_COLUMN: f32 = 204.0;
const EFFECT_COLUMN: f32 = 236.0;
const LAST_COLUMN: f32 = 204.0;
/// The parameters column never gets narrower than this; the panel scrolls sideways instead.
const PARAMS_MIN: f32 = 372.0;
/// The gap between columns.
const GAP: f32 = 12.0;

/// Which half of a clip the detail panel shows beside the clip column when the panel is too
/// narrow for all of it at once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DetailPage {
    /// The eight parameters and their ranges.
    #[default]
    Parameters,
    /// Effects, transition and follow action.
    Effects,
}

/// How the detail panel divides a given width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetailPlan {
    /// The panel is too narrow for its four columns: the clip column gets a switch, and the
    /// rest shows one page at a time. Nothing is cut off and nothing needs a sideways scroll.
    pub paged: bool,
    pub clip: f32,
    pub params: f32,
    pub effects: f32,
    pub last: f32,
}

impl DetailPlan {
    /// The width the columns that are shown together add up to, gaps included.
    pub fn used(&self, page: DetailPage) -> f32 {
        if !self.paged {
            return self.clip + self.params + self.effects + self.last + GAP * 3.0;
        }
        match page {
            DetailPage::Parameters => self.clip + GAP + self.params,
            DetailPage::Effects => self.clip + GAP + self.effects + GAP + self.last,
        }
    }
}

/// Divide `width` between the detail panel's columns.
///
/// Wide enough for all four (1052 points): clip, parameters (which take the spare room),
/// effects, transition. Narrower (the control window beside the stage on a laptop is 605): the
/// clip column and one page, with every column a little slimmer.
pub fn detail_plan(width: f32) -> DetailPlan {
    let all = CLIP_COLUMN + PARAMS_MIN + EFFECT_COLUMN + LAST_COLUMN + GAP * 3.0;
    if width >= all {
        return DetailPlan {
            paged: false,
            clip: CLIP_COLUMN,
            params: PARAMS_MIN + (width - all),
            effects: EFFECT_COLUMN,
            last: LAST_COLUMN,
        };
    }
    let clip = 180.0;
    let right = (width - clip - GAP).max(PARAMS_MIN);
    let last = 190.0;
    DetailPlan {
        paged: true,
        clip,
        params: right,
        effects: (right - GAP - last).max(176.0),
        last,
    }
}

/// The nine effects the renderer has a slot for, in `mosh_core::EffectKind` order.
const EFFECTS: [EffectKind; 9] = [
    EffectKind::Fold,
    EffectKind::Mirror,
    EffectKind::Streak,
    EffectKind::Crush,
    EffectKind::Zoom,
    EffectKind::Spin,
    EffectKind::Echo,
    EffectKind::Negative,
    EffectKind::Mosh,
];

const TRANSITIONS: [(TransitionKind, &str); 5] = [
    (TransitionKind::Cut, "Cut"),
    (TransitionKind::Crossfade, "Fade"),
    (TransitionKind::Datamosh, "Mosh"),
    (TransitionKind::Melt, "Melt"),
    (TransitionKind::Strobe, "Strobe"),
];

const FOLLOWS: [(FollowAction, &str, &str); 4] = [
    (FollowAction::None, "Stay", "keep playing until something else is launched"),
    (FollowAction::Next, "Next", "go to the next clip in this layer's column"),
    (FollowAction::Random, "Random", "go to a random clip in this layer's column"),
    (
        FollowAction::RandomProcedural,
        "New look",
        "stay on this clip and draw a new look inside its ranges",
    ),
];

pub fn draw(ui: &mut egui::Ui, state: &mut EditorUi, model: &EditorModel, out: &mut Vec<Command>) {
    let Some(clip) = model.selected_clip() else {
        let rect = ui.max_rect();
        ui.painter().text(
            rect.center(),
            Align2::CENTER_CENTER,
            "No clip selected. Click a block or a slot to edit it here, or drag a visualizer from the browser to make one.",
            theme::body(),
            theme::DIM,
        );
        return;
    };

    egui::ScrollArea::horizontal()
        .id_salt("detail_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let height = ui.available_height();
            let plan = detail_plan(ui.available_width());
            let page = state.detail_page;
            ui.horizontal_top(|ui| {
                ui.spacing_mut().item_spacing.x = GAP;
                column(ui, plan.clip, height, |ui| {
                    clip_column(ui, state, clip, plan, out);
                });
                if !plan.paged || page == DetailPage::Parameters {
                    column(ui, plan.params, height, |ui| {
                        params_column(ui, state, model, clip, out);
                    });
                }
                if !plan.paged || page == DetailPage::Effects {
                    column(ui, plan.effects, height, |ui| {
                        effects_column(ui, clip, plan.effects, out);
                    });
                    column(ui, plan.last, height, |ui| {
                        last_column(ui, clip, plan.last, out);
                    });
                }
            });
        });
}

/// One fixed-width column of the panel.
fn column(ui: &mut egui::Ui, width: f32, height: f32, add: impl FnOnce(&mut egui::Ui)) {
    ui.allocate_ui_with_layout(
        Vec2::new(width, height),
        egui::Layout::top_down(egui::Align::Min),
        |ui| {
            ui.set_width(width);
            ui.set_min_height(height);
            ui.spacing_mut().item_spacing.y = 5.0;
            add(ui);
        },
    );
}

fn heading(ui: &mut egui::Ui, text: &str) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 13.0), Sense::hover());
    caption(ui.painter(), rect.min, text);
}

/// Picture, name, visualizer, palette, seed, media source, save as preset.
fn clip_column(
    ui: &mut egui::Ui,
    state: &mut EditorUi,
    clip: &Clip,
    plan: DetailPlan,
    out: &mut Vec<Command>,
) {
    let column_width = plan.clip;
    let id = clip.id;
    heading(ui, "CLIP");
    ui.horizontal(|ui| {
        let (picture, _) = ui.allocate_exact_size(Vec2::new(96.0, 54.0), Sense::hover());
        ui.painter()
            .rect_filled(picture, 3.0, theme::clip_color(&clip.visualizer));
        if let Some(texture) = state.thumbs.get(
            ui.ctx(),
            clip_slot(id),
            &clip.visualizer,
            &thumb_params(clip),
        ) {
            ui.painter().image(
                texture,
                picture,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }
        ui.vertical(|ui| {
            // Seed: a number, and a die that rolls a new one.
            let mut seed = clip.params.seed;
            if ui
                .add(egui::DragValue::new(&mut seed).speed(1.0).prefix("seed "))
                .on_hover_text("the seed fixes the look; a preset carries it")
                .changed()
            {
                out.push(Command::SetSeed { clip: id, seed });
            }
            ui.horizontal(|ui| {
                let (dice, _) = ui.allocate_exact_size(Vec2::splat(24.0), Sense::hover());
                if widgets::icon_button(
                    ui,
                    dice,
                    "dice",
                    theme::RAISED,
                    theme::TEXT,
                    widgets::dice_icon,
                )
                .on_hover_text("roll a new seed")
                .clicked()
                {
                    out.push(Command::SetSeed {
                        clip: id,
                        seed: state.roll_seed() % 1_000_000,
                    });
                }
                let mut palette = u32::from(clip.params.palette) + 1;
                if ui
                    .add(
                        egui::DragValue::new(&mut palette)
                            .range(1..=8)
                            .speed(0.05)
                            .prefix("pal "),
                    )
                    .on_hover_text("colour palette, 1 to 8")
                    .changed()
                {
                    out.push(Command::SetPalette {
                        clip: id,
                        palette: (palette - 1) as u16,
                    });
                }
            });
        });
    });

    if let Some(name) = widgets::text_field(
        ui,
        &mut state.draft,
        widgets::field_key(("clip_name", id)),
        &clip.name,
        column_width - 12.0,
        "clip name",
        1,
    ) {
        out.push(Command::RenameClip { clip: id, name });
    }

    let current = catalog_entry(&clip.visualizer).map_or(clip.visualizer.as_str(), |entry| entry.name);
    egui::ComboBox::from_id_salt("visualizer")
        .width(column_width - 12.0)
        .height(320.0)
        .selected_text(current)
        .show_ui(ui, |ui| {
            let mut group = "";
            for entry in catalog() {
                if entry.group != group {
                    group = entry.group;
                    ui.add(egui::Label::new(
                        egui::RichText::new(group).font(theme::small()).color(theme::FAINT),
                    ));
                }
                if ui
                    .selectable_label(clip.visualizer == entry.id, entry.name)
                    .clicked()
                {
                    out.push(Command::SetVisualizer {
                        clip: id,
                        visualizer: entry.id.to_string(),
                    });
                }
            }
        });

    media_source(ui, state, clip, column_width - 12.0, out);

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = theme::GAP;
        ui.add(
            egui::TextEdit::singleline(&mut state.preset_name)
                .hint_text("preset name")
                .desired_width(column_width - 12.0 - 96.0),
        );
        if ui
            .button("Save preset")
            .on_hover_text("keep this look (visualizer, parameters, seed) in the browser")
            .clicked()
        {
            out.push(Command::SavePreset {
                name: state.preset_name.clone(),
            });
            state.preset_name.clear();
        }
    });

    // Too narrow for everything at once: this switch says which half is beside this column.
    if plan.paged {
        ui.add_space(2.0);
        heading(ui, "SHOWING");
        let (rect, _) =
            ui.allocate_exact_size(Vec2::new(column_width - 12.0, 24.0), Sense::hover());
        let pages = [DetailPage::Parameters, DetailPage::Effects];
        if let Some(index) = widgets::segmented(
            ui,
            rect,
            "detail_page",
            &["Parameters", "Effects"],
            pages.iter().position(|page| *page == state.detail_page),
            &[
                "the eight parameters and the range each moves in",
                "effects, the transition in and the follow action",
            ],
        ) {
            state.detail_page = pages[index];
        }
    }
}

/// The clip's media source, for the visualizers that have one: a video file, a camera, words.
fn media_source(
    ui: &mut egui::Ui,
    state: &mut EditorUi,
    clip: &Clip,
    width: f32,
    out: &mut Vec<Command>,
) {
    let id = clip.id;
    let kind = MediaKind::of(&clip.visualizer);
    let stored = clip.video.as_deref().unwrap_or("");
    let value = stored.strip_prefix(kind.prefix()).unwrap_or(stored);
    match kind {
        MediaKind::None => {}
        MediaKind::File => {
            if let Some(path) = widgets::text_field(
                ui,
                &mut state.draft,
                widgets::field_key(("video", id)),
                value,
                width,
                "path to a video file",
                1,
            ) {
                out.push(Command::SetVideo {
                    clip: id,
                    path: Some(path),
                });
            }
        }
        MediaKind::Camera => {
            let devices = camera_devices();
            let shown = if value.is_empty() { "default camera" } else { value };
            egui::ComboBox::from_id_salt("camera")
                .width(width)
                .selected_text(shown)
                .show_ui(ui, |ui| {
                    if ui.selectable_label(value.is_empty(), "default camera").clicked() {
                        out.push(Command::SetVideo {
                            clip: id,
                            path: Some("camera:".to_string()),
                        });
                    }
                    for device in devices {
                        if ui.selectable_label(value == device, device.as_str()).clicked() {
                            out.push(Command::SetVideo {
                                clip: id,
                                path: Some(format!("camera:{device}")),
                            });
                        }
                    }
                })
                .response
                .on_hover_text("which camera this clip shows");
        }
        MediaKind::Text => {
            if let Some(words) = widgets::text_field(
                ui,
                &mut state.draft,
                widgets::field_key(("words", id)),
                value,
                width,
                "the words (one line per line)",
                3,
            ) {
                out.push(Command::SetVideo {
                    clip: id,
                    path: Some(format!("text:{words}")),
                });
            }
        }
    }
}

/// The cameras Windows reports, asked once: enumerating devices is not a per-frame job.
fn camera_devices() -> &'static [String] {
    static DEVICES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    DEVICES.get_or_init(mosh_visuals::camera::CameraLayer::devices)
}

/// Eight named sliders, and the range each one moves in at the chosen intensity level.
fn params_column(
    ui: &mut egui::Ui,
    state: &mut EditorUi,
    model: &EditorModel,
    clip: &Clip,
    out: &mut Vec<Command>,
) {
    let id = clip.id;
    let live = model.status().level.clamp(1, 4);
    let level = state.level_tab.unwrap_or(live).clamp(1, 4);
    let slot = usize::from(level - 1);
    let width = ui.available_width();

    // Heading, then the level tabs and that level's pulse depth.
    heading(ui, "PARAMETERS   position, and the range it moves in at level");
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        let (tabs, _) = ui.allocate_exact_size(Vec2::new(4.0 * 30.0, 22.0), Sense::hover());
        if let Some(index) = widgets::segmented(
            ui,
            tabs,
            "level_tab",
            &["1", "2", "3", "4"],
            Some(slot),
            &[
                "the ranges the parameters move in at intensity level 1 (quiet)",
                "level 2",
                "level 3",
                "level 4 (the drop)",
            ],
        ) {
            let picked = index as u8 + 1;
            // Clicking the tab that is already shown hands the tabs back to the live level.
            state.level_tab = if state.level_tab == Some(picked) {
                None
            } else {
                Some(picked)
            };
        }
        // The live level, as a dot under its tab.
        let dot = Pos2::new(
            tabs.left() + 30.0 * (f32::from(live) - 0.5),
            tabs.bottom() + 2.0,
        );
        ui.painter().circle_filled(dot, 2.0, theme::ACCENT);
        let (mode, _) = ui.allocate_exact_size(Vec2::new(64.0, 22.0), Sense::hover());
        ui.painter().text(
            mode.left_center(),
            Align2::LEFT_CENTER,
            if state.level_tab.is_some() {
                "held"
            } else {
                "follows live"
            },
            theme::small(),
            theme::FAINT,
        );

        let mut depth = clip.pulse_depth[slot];
        ui.spacing_mut().slider_width = 64.0;
        if ui
            .add(
                egui::Slider::new(&mut depth, 0.0..=1.0)
                    .show_value(false)
                    .text("beat pulse"),
            )
            .on_hover_text("how hard the beat pushes the parameters at this level")
            .changed()
        {
            out.push(Command::SetPulseDepth {
                clip: id,
                level,
                depth,
            });
        }
    });
    ui.spacing_mut().item_spacing.y = 2.0;

    // Two columns of four when there is room, one of eight in a scroll area when not.
    let labels = param_labels(&clip.visualizer);
    let two = width >= 620.0;
    let row_width = if two { (width - 16.0) * 0.5 } else { width - 14.0 };
    let mut row = |ui: &mut egui::Ui, index: usize| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            let (name, _) = ui.allocate_exact_size(Vec2::new(94.0, 21.0), Sense::hover());
            widgets::clipped_text(
                ui.painter(),
                name,
                Align2::LEFT_CENTER,
                labels[index],
                theme::body(),
                theme::TEXT,
            );
            let range = clip.ranges[slot][index];
            ui.spacing_mut().slider_width = (row_width - 94.0 - 2.0 * 46.0 - 20.0).max(50.0);
            let mut value = clip.params.p[index];
            let slider = ui.add(egui::Slider::new(&mut value, 0.0..=1.0).show_value(false));
            // Where the value lands inside the level's range, under the slider.
            let track = slider.rect;
            let x = |t: f32| track.left() + 7.0 + (track.width() - 14.0) * t.clamp(0.0, 1.0);
            ui.painter().hline(
                x(range.lo)..=x(range.hi),
                track.bottom() - 1.0,
                Stroke::new(2.0, theme::ACCENT_DIM),
            );
            if slider.changed() {
                out.push(Command::SetClipParam {
                    clip: id,
                    index,
                    value,
                });
            }
            let (mut lo, mut hi) = (range.lo, range.hi);
            let a = ui
                .add(egui::DragValue::new(&mut lo).speed(0.005).range(0.0..=1.0).fixed_decimals(2))
                .on_hover_text("lowest value at this level");
            let b = ui
                .add(egui::DragValue::new(&mut hi).speed(0.005).range(0.0..=1.0).fixed_decimals(2))
                .on_hover_text("highest value at this level");
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
    };
    if two {
        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = 16.0;
            for half in 0..2 {
                ui.vertical(|ui| {
                    ui.set_width(row_width);
                    for index in 0..4 {
                        row(ui, half * 4 + index);
                    }
                });
            }
        });
    } else {
        egui::ScrollArea::vertical()
            .id_salt("params_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for index in 0..8 {
                    row(ui, index);
                }
            });
    }
}

/// Effect chips, and amount + parameter for the ones that are on.
fn effects_column(ui: &mut egui::Ui, clip: &Clip, width: f32, out: &mut Vec<Command>) {
    // Two sliders beside an effect's name, sharing what the name leaves.
    let slider = ((width - 58.0 - 26.0) * 0.5).clamp(40.0, 70.0);
    let id = clip.id;
    heading(ui, "EFFECTS   click to add or remove");
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = Vec2::new(4.0, 4.0);
        for kind in EFFECTS {
            let slot = clip.effects.iter().position(|slot| slot.kind == kind);
            if widgets::chip(ui, &format!("{kind:?}"), slot.is_some()).clicked() {
                out.push(match slot {
                    Some(index) => Command::RemoveEffect { clip: id, index },
                    None => Command::AddEffect { clip: id, kind },
                });
            }
        }
    });
    ui.add_space(2.0);
    egui::ScrollArea::vertical()
        .id_salt("effects_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            if clip.effects.is_empty() {
                ui.add(egui::Label::new(
                    egui::RichText::new("no effects on this clip")
                        .font(theme::small())
                        .color(theme::FAINT),
                ));
            }
            for (index, slot) in clip.effects.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    ui.spacing_mut().slider_width = slider;
                    let (name, _) = ui.allocate_exact_size(Vec2::new(58.0, 22.0), Sense::hover());
                    widgets::clipped_text(
                        ui.painter(),
                        name,
                        Align2::LEFT_CENTER,
                        &format!("{:?}", slot.kind),
                        theme::body(),
                        theme::TEXT,
                    );
                    let (mut amount, mut param) = (slot.amount, slot.param);
                    let a = ui
                        .add(egui::Slider::new(&mut amount, 0.0..=1.0).show_value(false))
                        .on_hover_text("amount");
                    let b = ui
                        .add(egui::Slider::new(&mut param, 0.0..=1.0).show_value(false))
                        .on_hover_text("the effect's own parameter");
                    if a.changed() || b.changed() {
                        out.push(Command::SetEffect {
                            clip: id,
                            index,
                            amount,
                            param,
                        });
                    }
                });
            }
        });
}

/// How the clip comes in, and what happens after it.
fn last_column(ui: &mut egui::Ui, clip: &Clip, width: f32, out: &mut Vec<Command>) {
    let id = clip.id;
    heading(ui, "TRANSITION IN");
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width - 6.0, 24.0), Sense::hover());
    let labels: Vec<&str> = TRANSITIONS.iter().map(|(_, name)| *name).collect();
    if let Some(index) = widgets::segmented(
        ui,
        rect,
        "transition",
        &labels,
        TRANSITIONS
            .iter()
            .position(|(kind, _)| *kind == clip.transition_in.kind),
        &[
            "cut straight in",
            "crossfade from the clip before",
            "datamosh from the clip before",
            "melt from the clip before",
            "strobe in (flashes are at least 400 ms apart)",
        ],
    ) {
        out.push(Command::SetTransition {
            clip: id,
            kind: TRANSITIONS[index].0,
            beats: clip.transition_in.beats,
        });
    }
    let mut beats = clip.transition_in.beats;
    if ui
        .add(
            egui::DragValue::new(&mut beats)
                .speed(0.25)
                .range(0.0..=64.0)
                .prefix("over ")
                .suffix(" beats"),
        )
        .changed()
    {
        out.push(Command::SetTransition {
            clip: id,
            kind: clip.transition_in.kind,
            beats,
        });
    }

    ui.add_space(6.0);
    heading(ui, "FOLLOW ACTION   Session and Autopilot");
    let current = FOLLOWS
        .iter()
        .find(|(action, _, _)| *action == clip.follow)
        .map_or("Stay", |(_, name, _)| *name);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        egui::ComboBox::from_id_salt("follow")
            .width(86.0)
            .selected_text(current)
            .show_ui(ui, |ui| {
                for (action, name, about) in FOLLOWS {
                    if ui
                        .selectable_label(clip.follow == action, name)
                        .on_hover_text(about)
                        .clicked()
                    {
                        out.push(Command::SetFollow {
                            clip: id,
                            action,
                            after_bars: clip.follow_after_bars,
                        });
                    }
                }
            });
        let mut bars = u32::from(clip.follow_after_bars);
        if ui
            .add(
                egui::DragValue::new(&mut bars)
                    .speed(0.1)
                    .range(1..=64)
                    .prefix("after ")
                    .suffix(" bars"),
            )
            .changed()
        {
            out.push(Command::SetFollow {
                clip: id,
                action: clip.follow,
                after_bars: bars as u16,
            });
        }
    });
    let mut react = clip.react_to_events;
    if ui
        .checkbox(&mut react, "also on a drop or a level up")
        .on_hover_text("fire the follow action on the next downbeat after a drop or a level up")
        .changed()
    {
        out.push(Command::SetReactToEvents { clip: id, on: react });
    }
}
