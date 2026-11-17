//! The control window's widgets: they read the model and return commands, and never change it.
//!
//! ```text
//! +--------------------------------------------------------------------------------------+
//! | top bar: view | show mode | source | tempo | bar.beat | song | intensity | events | preview
//! +-----------+--------------------------------------------------------------------------+
//! | browser   |  the main view: ARRANGEMENT (timeline) or SESSION (clip grid); Tab toggles |
//! | (B)       |                                                                          |
//! |           +--------------------------------------------------------------------------+
//! |           |  detail: the selected clip (Shift+Tab)                                   |
//! +-----------+--------------------------------------------------------------------------+
//! | status: panel toggles, the last message, launch quantize, controller                 |
//! +--------------------------------------------------------------------------------------+
//! ```
//!
//! Keeping the egui tree in its own module is what makes the control window's cost measurable
//! without a display (`tests/ui.rs` runs this against a headless `egui::Context`), and it is
//! also the rule of the editor: the layer here maps widgets to [`Command`]s; applying them, and
//! every decision about the running show, stays in [`super::model`] and the shell.
//!
//! Drafts (a path, a name being typed, a block being dragged) are widget state and live in
//! [`EditorUi`]: dragging a block is not an edit to the show until the mouse button is released.

mod arrangement;
mod browser;
mod detail;
mod session;
mod topbar;
pub mod widgets;

use std::collections::{HashMap, HashSet, VecDeque};

use egui::{Align2, Key, Rect, Sense};
use mosh_core::{ClipId, ClipParams, LaunchQuantize, ShowMode};

use crate::editor::model::{ClipSource, Command, EditorModel, View};
use crate::editor::theme;
use crate::preview::{Picture, ThumbRequest, has_thumbnail, thumb_key};

pub use arrangement::{SpanDrag, SpanZone, TimeAxis, drag_result, hit_span};
pub use topbar::{TopBarPlan, top_bar_plan};

/// How many thumbnails the cache keeps (96 x 54 RGBA each: 21 KB, so about 5 MB when full).
const THUMB_CACHE: usize = 240;
/// A look must hold still this long before its thumbnail is drawn, so dragging a slider does
/// not queue a picture per frame.
const THUMB_SETTLE_S: f64 = 0.25;
/// Thumbnails asked for and not yet delivered, at most.
const THUMB_PENDING: usize = 6;
/// The recent-projects list.
pub const RECENT_MAX: usize = 8;

/// What is being dragged from the browser or the grid.
#[derive(Clone, Debug, PartialEq)]
pub enum DragPayload {
    /// A visualizer or a preset from the browser: dropping it makes a clip.
    Source(ClipSource, String),
    /// A session slot: dropping it on another swaps the two.
    Cell { layer: usize, scene: usize },
}

/// Things the control window asks of the shell rather than of the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiAction {
    /// `F11`, or the Stage button: the stage window in and out of fullscreen.
    ToggleStageFullscreen,
}

/// What the shell tells the widgets about the stage window, for the Stage button's label.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageInfo {
    pub fullscreen: bool,
    /// The stage shares the control window's monitor.
    pub shared_monitor: bool,
}

/// Clip thumbnails: asked for by look, drawn once on the render thread, cached here.
#[derive(Default)]
pub struct Thumbs {
    textures: HashMap<u64, egui::TextureHandle>,
    order: VecDeque<u64>,
    /// Asked for and not delivered yet.
    pending: HashSet<u64>,
    /// Looks that have no picture (the render thread said so): never asked for again.
    blank: HashSet<u64>,
    outbox: Vec<ThumbRequest>,
    /// Per slot (a clip, a browser row): the look it wants, since when, and the look it shows.
    slots: HashMap<u64, (u64, f64, Option<u64>)>,
}

impl Thumbs {
    /// The texture to draw for `slot`, asking for a new one when its look has settled.
    pub fn get(
        &mut self,
        ctx: &egui::Context,
        slot: u64,
        visualizer: &str,
        params: &ClipParams,
    ) -> Option<egui::TextureId> {
        if !has_thumbnail(visualizer) {
            return None;
        }
        let now = ctx.input(|input| input.time);
        let key = thumb_key(visualizer, params);
        let entry = self.slots.entry(slot).or_insert((key, now, None));
        if entry.0 != key {
            entry.0 = key;
            entry.1 = now;
        }
        if self.textures.contains_key(&key) {
            entry.2 = Some(key);
        } else if !self.pending.contains(&key) && !self.blank.contains(&key) {
            if now - entry.1 < THUMB_SETTLE_S {
                ctx.request_repaint_after(std::time::Duration::from_secs_f64(THUMB_SETTLE_S));
            } else if self.pending.len() < THUMB_PENDING {
                self.pending.insert(key);
                self.outbox.push(ThumbRequest {
                    key,
                    visualizer: visualizer.to_string(),
                    params: *params,
                });
            }
        }
        let shown = entry.2?;
        self.textures.get(&shown).map(egui::TextureHandle::id)
    }

    /// The requests made since the last call, for the shell to send to the render thread.
    pub fn take_requests(&mut self) -> Vec<ThumbRequest> {
        std::mem::take(&mut self.outbox)
    }

    /// A thumbnail arrived (`None`: that look has no picture).
    pub fn deliver(&mut self, ctx: &egui::Context, key: u64, picture: Option<Picture>) {
        self.pending.remove(&key);
        let Some(picture) = picture else {
            self.blank.insert(key);
            return;
        };
        let image = egui::ColorImage::from_rgba_unmultiplied(
            [picture.width as usize, picture.height as usize],
            &picture.rgba,
        );
        let texture = ctx.load_texture(
            format!("thumb-{key:016x}"),
            image,
            egui::TextureOptions::LINEAR,
        );
        self.textures.insert(key, texture);
        self.order.push_back(key);
        while self.order.len() > THUMB_CACHE {
            if let Some(old) = self.order.pop_front() {
                self.textures.remove(&old);
            }
        }
    }

    pub fn cached(&self) -> usize {
        self.textures.len()
    }
}

/// Widget state that is not part of the project.
pub struct EditorUi {
    /// The `.als` to import.
    pub als_path: String,
    /// The name for a preset saved from the selected clip (`""` takes the clip's name).
    pub preset_name: String,
    /// Where the project is saved to, or read from.
    pub project_path: String,
    /// Projects opened or saved lately, newest first.
    pub recent: Vec<String>,
    /// The browser's filter text.
    pub filter: String,
    /// Which browser sections are open: visualizers, presets, project.
    pub sections: [bool; 3],
    /// The intensity level whose ranges the detail panel edits; `None` follows the live level.
    pub level_tab: Option<u8>,
    /// The arrangement's horizontal scale and scroll.
    pub px_per_beat: f32,
    pub scroll_beats: f64,
    /// Keep the playhead in view while Live plays.
    pub follow: bool,
    /// A block being dragged on the arrangement.
    pub drag: Option<SpanDrag>,
    /// A song marker being dragged: its index and where it is now, in beats.
    pub marker_drag: Option<(usize, f64)>,
    /// The field being typed in.
    pub draft: Option<widgets::TextDraft>,
    pub thumbs: Thumbs,
    /// The live preview of the stage and its aspect (width / height).
    pub preview: Option<(egui::TextureHandle, f32)>,
    /// The performer's choice for the preview (`P`, or a click on it). `None` is automatic: on
    /// when the stage is on another monitor, off when it is a window right beside the controls
    /// (where the stage itself is the preview, and the readback would be paid for nothing).
    pub preview_on: Option<bool>,
    /// The preview was drawn this frame, so the shell keeps the render thread sending it.
    pub preview_shown: bool,
    pub stage: StageInfo,
    /// Requests for the shell, drained after each frame.
    pub actions: Vec<UiAction>,
    /// The dice button's generator state.
    pub seed_state: u64,
}

impl Default for EditorUi {
    fn default() -> EditorUi {
        EditorUi {
            als_path: String::new(),
            preset_name: String::new(),
            project_path: "show.ron".to_string(),
            recent: Vec::new(),
            filter: String::new(),
            sections: [true, true, true],
            level_tab: None,
            px_per_beat: 14.0,
            scroll_beats: 0.0,
            follow: true,
            drag: None,
            marker_drag: None,
            draft: None,
            thumbs: Thumbs::default(),
            preview: None,
            preview_on: None,
            preview_shown: false,
            stage: StageInfo::default(),
            actions: Vec::new(),
            seed_state: 0x9e37_79b9_7f4a_7c15,
        }
    }
}

impl EditorUi {
    /// Show a new preview frame.
    pub fn set_preview(&mut self, ctx: &egui::Context, picture: Picture) {
        let aspect = picture.width as f32 / picture.height.max(1) as f32;
        let image = egui::ColorImage::from_rgba_unmultiplied(
            [picture.width as usize, picture.height as usize],
            &picture.rgba,
        );
        match self.preview.as_mut() {
            Some((texture, old)) => {
                texture.set(image, egui::TextureOptions::LINEAR);
                *old = aspect;
            }
            None => {
                let texture = ctx.load_texture("stage-preview", image, egui::TextureOptions::LINEAR);
                self.preview = Some((texture, aspect));
            }
        }
    }

    /// Whether the preview is switched on (by the performer, or automatically).
    pub fn preview_enabled(&self) -> bool {
        self.preview_on.unwrap_or(!self.stage.shared_monitor)
    }

    /// Whether the render thread should be sending preview frames: the preview is on *and* the
    /// top bar had room to draw it.
    pub fn wants_preview(&self) -> bool {
        self.preview_enabled() && self.preview_shown
    }

    /// A new seed for the dice button: a xorshift step, so every roll differs and no clock or
    /// OS randomness is involved (a roll in a test is the same roll every run).
    pub fn roll_seed(&mut self) -> u64 {
        let mut x = self.seed_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.seed_state = x;
        x
    }

    /// Put `path` at the top of the recent list.
    pub fn note_recent(&mut self, path: &str) {
        push_recent(&mut self.recent, path);
    }
}

/// Put `path` first in a recent-files list: no duplicates (case-insensitive, as Windows paths
/// are), at most [`RECENT_MAX`] entries.
pub fn push_recent(recent: &mut Vec<String>, path: &str) {
    let path = path.trim();
    if path.is_empty() {
        return;
    }
    recent.retain(|other| !other.eq_ignore_ascii_case(path));
    recent.insert(0, path.to_string());
    recent.truncate(RECENT_MAX);
}

/// The thumbnail slot of a clip (browser rows use their own ids, far above any clip id).
pub(crate) fn clip_slot(clip: ClipId) -> u64 {
    u64::from(clip)
}

/// The parameters a clip's thumbnail is drawn with: its look at level 3, a little after a beat.
pub(crate) fn thumb_params(clip: &mosh_core::Clip) -> ClipParams {
    ClipParams {
        p: mosh_show::mapping::resolve_params(clip, 3, 0.3, &clip.params.p),
        palette: clip.params.palette,
        seed: clip.params.seed,
    }
}

fn quantize_label(quantize: LaunchQuantize) -> &'static str {
    match quantize {
        LaunchQuantize::None => "now",
        LaunchQuantize::Beat => "beat",
        LaunchQuantize::Bar => "bar",
        LaunchQuantize::Phrase => "phrase",
    }
}

/// Draw the whole control window and return what the performer asked for.
///
/// One frame of widgets, one batch of commands: the caller applies them in order, so a frame
/// that emits several (a clip selected and its parameter dragged) cannot interleave.
pub fn draw(root: &mut egui::Ui, state: &mut EditorUi, model: &EditorModel) -> Vec<Command> {
    let mut out = Vec::new();
    state.preview_shown = false;

    shortcuts(root, state, model, &mut out);

    egui::Panel::top("top_bar")
        .exact_size(theme::TOP_BAR)
        .resizable(false)
        .frame(egui::Frame::new().fill(theme::BG))
        .show(root, |ui| topbar::draw(ui, state, model, &mut out));
    egui::Panel::bottom("status_bar")
        .exact_size(theme::ROW)
        .resizable(false)
        .frame(egui::Frame::new().fill(theme::BG))
        .show(root, |ui| status_bar(ui, model, &mut out));
    if model.panels().browser {
        egui::Panel::left("browser")
            .exact_size(browser::WIDTH)
            .resizable(false)
            .frame(egui::Frame::new().fill(theme::PANEL).inner_margin(theme::GAP * 2.0))
            .show(root, |ui| browser::draw(ui, state, model, &mut out));
    }
    if model.panels().detail {
        egui::Panel::bottom("detail")
            .exact_size(detail::HEIGHT)
            .resizable(false)
            .frame(egui::Frame::new().fill(theme::PANEL).inner_margin(theme::GAP * 2.0))
            .show(root, |ui| detail::draw(ui, state, model, &mut out));
    }
    egui::CentralPanel::default()
        .frame(egui::Frame::new().fill(theme::BG))
        .show(root, |ui| match model.view() {
            View::Arrangement => arrangement::draw(ui, state, model, &mut out),
            View::Session => session::draw(ui, state, model, &mut out),
        });

    drag_label(root);
    if model.panels().help {
        help(root, &mut out);
    }

    // Tab is a view switch here, not a walk through the widgets: give back the focus egui just
    // moved, unless it is in a text field (where Tab and every other key belong to the field).
    let ctx = root.ctx().clone();
    if !ctx.text_edit_focused() {
        ctx.memory_mut(|memory| {
            if let Some(id) = memory.focused() {
                memory.surrender_focus(id);
            }
        });
    }
    out
}

/// The keyboard: everything that is not typing into a field.
fn shortcuts(ui: &mut egui::Ui, state: &mut EditorUi, model: &EditorModel, out: &mut Vec<Command>) {
    let ctx = ui.ctx().clone();
    if ctx.text_edit_focused() {
        return;
    }
    let project = model.project();
    let layers = model.lane_count();
    ctx.input(|input| {
        let plain = !input.modifiers.command && !input.modifiers.alt;
        let shift = input.modifiers.shift;
        let pressed = |key: Key| input.key_pressed(key);
        let typed = |text: &str| {
            input
                .events
                .iter()
                .any(|event| matches!(event, egui::Event::Text(typed) if typed == text))
        };

        if pressed(Key::Tab) && plain {
            out.push(if shift {
                Command::SetDetail(!model.panels().detail)
            } else {
                Command::ToggleView
            });
        }
        if pressed(Key::B) && plain {
            out.push(Command::SetBrowser(!model.panels().browser));
        }
        if pressed(Key::F1) || typed("?") {
            out.push(Command::SetHelp(!model.panels().help));
        }
        if pressed(Key::Escape) && model.panels().help {
            out.push(Command::SetHelp(false));
        }
        if pressed(Key::F11) {
            state.actions.push(UiAction::ToggleStageFullscreen);
        }
        if pressed(Key::P) && plain {
            state.preview_on = Some(!state.preview_enabled());
        }
        if pressed(Key::F) && plain {
            state.follow = !state.follow;
        }
        if pressed(Key::M) && plain {
            out.push(Command::SetMode(match project.mode {
                ShowMode::Arrangement => ShowMode::Session,
                ShowMode::Session => ShowMode::Autopilot,
                ShowMode::Autopilot => ShowMode::Arrangement,
            }));
        }
        if pressed(Key::L) && plain {
            out.push(Command::Lock(!model.status().locked));
        }

        // 1-9 launch scenes, 0 stops every layer: the top row of the keyboard is the scene
        // column of the grid.
        if plain && !shift {
            const DIGITS: [Key; 9] = [
                Key::Num1,
                Key::Num2,
                Key::Num3,
                Key::Num4,
                Key::Num5,
                Key::Num6,
                Key::Num7,
                Key::Num8,
                Key::Num9,
            ];
            for (scene, key) in DIGITS.into_iter().enumerate() {
                if pressed(key) && scene < project.session.scenes.len() {
                    out.push(Command::LaunchScene(scene));
                }
            }
            if pressed(Key::Num0) {
                for layer in 0..layers {
                    out.push(Command::StopLayer(layer));
                }
            }
        }

        if input.modifiers.command {
            if pressed(Key::Z) {
                out.push(if shift { Command::Redo } else { Command::Undo });
            }
            if pressed(Key::Y) {
                out.push(Command::Redo);
            }
            if pressed(Key::S) {
                out.push(match model.path() {
                    Some(_) => Command::Save,
                    None => Command::SaveProject {
                        path: state.project_path.trim().to_string(),
                    },
                });
            }
            if pressed(Key::D)
                && let Some(span) = model.selected_span()
            {
                out.push(Command::DuplicateSpan(span));
            }
        }

        if (pressed(Key::Delete) || pressed(Key::Backspace)) && plain {
            match model.view() {
                View::Arrangement => {
                    if let Some(span) = model.selected_span() {
                        out.push(Command::RemoveSpan(span));
                    }
                }
                View::Session => {
                    if let Some((layer, scene)) = selected_cell(model) {
                        out.push(Command::ClearCell { layer, scene });
                    }
                }
            }
        }
        if pressed(Key::Enter)
            && plain
            && let Some((layer, scene)) = selected_cell(model)
        {
            out.push(Command::Launch { layer, scene });
        }
    });
}

/// The session slot the selection is in: the selected row's slot that holds the selected clip.
pub(crate) fn selected_cell(model: &EditorModel) -> Option<(usize, usize)> {
    let clip = model.selected()?;
    let scene = model.selected_scene()?;
    let row = model.project().session.scenes.get(scene)?;
    let layer = row.cells.iter().position(|cell| *cell == Some(clip))?;
    Some((layer, scene))
}

/// The strip at the bottom: panel toggles at the ends, the last message in the middle.
fn status_bar(ui: &mut egui::Ui, model: &EditorModel, out: &mut Vec<Command>) {
    let rect = ui.max_rect();
    ui.painter()
        .hline(rect.x_range(), rect.top(), theme::line());
    let panels = model.panels();
    let pad = theme::GAP * 2.0;
    let button = |x: f32, width: f32| {
        Rect::from_min_size(
            egui::pos2(x, rect.top() + 3.0),
            egui::vec2(width, rect.height() - 6.0),
        )
    };

    let browser = button(rect.left() + pad, 72.0);
    if widgets::flat_button(ui, browser, "toggle_browser", "Browser", panels.browser)
        .on_hover_text("visualizers, presets, the show file (B)")
        .clicked()
    {
        out.push(Command::SetBrowser(!panels.browser));
    }
    let detail = button(rect.right() - pad - 64.0, 64.0);
    if widgets::flat_button(ui, detail, "toggle_detail", "Detail", panels.detail)
        .on_hover_text("the selected clip (Shift+Tab)")
        .clicked()
    {
        out.push(Command::SetDetail(!panels.detail));
    }

    // Launch quantize: a segmented control, because it is changed mid-rehearsal.
    let quantize = model.project().session.quantize;
    let kinds = [
        LaunchQuantize::None,
        LaunchQuantize::Beat,
        LaunchQuantize::Bar,
        LaunchQuantize::Phrase,
    ];
    let labels: Vec<&str> = kinds.iter().map(|kind| quantize_label(*kind)).collect();
    let segment = 4.0 * 48.0;
    let quantize_rect = button(detail.left() - pad - segment, segment);
    let wide = rect.width() >= 760.0;
    let mut right_edge = detail.left() - pad;
    if wide {
        if let Some(index) = widgets::segmented(
            ui,
            quantize_rect,
            "quantize",
            &labels,
            kinds.iter().position(|kind| *kind == quantize),
            &[
                "launch at once",
                "launch on the next beat",
                "launch on the next bar",
                "launch on the next phrase (4 bars)",
            ],
        ) {
            out.push(Command::SetQuantize(kinds[index]));
        }
        ui.painter().text(
            egui::pos2(quantize_rect.left() - pad, rect.center().y),
            Align2::RIGHT_CENTER,
            "launch on",
            theme::small(),
            theme::FAINT,
        );
        right_edge = quantize_rect.left() - pad - 56.0;
        if model.status().pads {
            ui.painter().text(
                egui::pos2(right_edge - pad, rect.center().y),
                Align2::RIGHT_CENTER,
                "APC mini",
                theme::small(),
                theme::ACCENT,
            );
            right_edge -= 60.0;
        }
    }

    let hint = if model.modified() {
        "edits land on the next bar  -  Ctrl+S saves  -  ? for shortcuts"
    } else {
        "Tab switches view  -  edits land on the next bar  -  ? for shortcuts"
    };
    let message = Rect::from_min_max(
        egui::pos2(browser.right() + pad * 1.5, rect.top()),
        egui::pos2(right_edge - pad, rect.bottom()),
    );
    widgets::clipped_text(
        ui.painter(),
        message,
        Align2::LEFT_CENTER,
        model.message().unwrap_or(hint),
        theme::small(),
        if model.message().is_some() {
            theme::TEXT
        } else {
            theme::FAINT
        },
    );
}

/// What is being dragged, next to the pointer.
fn drag_label(ui: &mut egui::Ui) {
    let ctx = ui.ctx();
    let Some(payload) = egui::DragAndDrop::payload::<DragPayload>(ctx) else {
        return;
    };
    let Some(pointer) = ctx.pointer_latest_pos() else {
        return;
    };
    let label = match payload.as_ref() {
        DragPayload::Source(_, name) => name.clone(),
        DragPayload::Cell { layer, scene } => format!("slot {}.{}", layer + 1, scene + 1),
    };
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Tooltip,
        egui::Id::unique("drag_label"),
    ));
    let galley = painter.layout_no_wrap(label, theme::body(), theme::ON_ACCENT);
    let rect = Rect::from_min_size(
        pointer + egui::vec2(12.0, 8.0),
        galley.size() + egui::vec2(theme::GAP * 4.0, theme::GAP * 2.0),
    );
    painter.rect_filled(rect, theme::RADIUS, theme::ACCENT);
    painter.galley(
        rect.min + egui::vec2(theme::GAP * 2.0, theme::GAP),
        galley,
        theme::ON_ACCENT,
    );
}

/// The shortcuts, in the order a show is built: views, building, rehearsing, the stage.
pub const SHORTCUTS: [(&str, &[(&str, &str)]); 4] = [
    (
        "Views",
        &[
            ("Tab", "Arrangement / Session"),
            ("Shift+Tab", "clip detail panel"),
            ("B", "browser"),
            ("P", "stage preview"),
            ("? or F1", "this overlay"),
        ],
    ),
    (
        "Building",
        &[
            ("drag from browser", "make a clip in a slot or on a lane"),
            ("click", "select a clip, edit it in the detail panel"),
            ("drag a block", "move it (snaps to bars, Alt: free)"),
            ("drag a block's edge", "make it longer or shorter"),
            ("double-click a lane", "new block there"),
            ("double-click the ruler", "song marker"),
            ("Ctrl+D", "duplicate the block"),
            ("Delete", "remove the block / empty the slot"),
            ("Ctrl+Z / Ctrl+Y", "undo / redo"),
            ("Ctrl+S", "save the show"),
        ],
    ),
    (
        "Rehearsing",
        &[
            ("1 - 9", "launch a scene"),
            ("0", "stop every layer"),
            ("Enter", "launch the selected slot"),
            ("M", "show mode: follow Live, manual, Autopilot"),
            ("L", "lock automation (manual launches still work)"),
            ("F", "timeline follows the playhead"),
            ("Ctrl+wheel / wheel", "zoom / scroll the timeline"),
        ],
    ),
    (
        "Stage",
        &[
            ("F11", "stage fullscreen on / off"),
            ("Esc", "leave fullscreen, back to the controls"),
            ("close this window", "quit ilanaVisual"),
        ],
    ),
];

/// The `?` overlay: every shortcut, on top of everything, closed by a click or `Esc`.
fn help(ui: &mut egui::Ui, out: &mut Vec<Command>) {
    let ctx = ui.ctx().clone();
    let screen = ctx.content_rect();
    let layer = egui::LayerId::new(egui::Order::Foreground, egui::Id::unique("help"));
    let painter = ctx.layer_painter(layer);
    painter.rect_filled(screen, 0.0, theme::alpha(theme::BG, 0.82));

    let line = 19.0;
    let column = 410.0;
    let columns = if screen.width() >= column * 2.0 + 80.0 { 2 } else { 1 };
    let groups: Vec<f32> = SHORTCUTS
        .iter()
        .map(|(_, rows)| (rows.len() as f32 + 1.8) * line)
        .collect();
    let height = if columns == 2 {
        (groups[0] + groups[1]).max(groups[2] + groups[3])
    } else {
        groups.iter().sum()
    };
    let card = Rect::from_center_size(
        screen.center(),
        egui::vec2(column * columns as f32 + 48.0, height + 72.0)
            .min(screen.size() - egui::vec2(16.0, 16.0)),
    );
    painter.rect_filled(card, 8.0, theme::PANEL);
    painter.rect_stroke(card, 8.0, theme::line(), egui::StrokeKind::Inside);
    let painter = painter.with_clip_rect(card);
    painter.text(
        card.left_top() + egui::vec2(24.0, 18.0),
        Align2::LEFT_TOP,
        "ilanaVisual  -  shortcuts",
        theme::big(),
        theme::TEXT,
    );

    let mut x = card.left() + 24.0;
    let mut y = card.top() + 58.0;
    for (index, (title, rows)) in SHORTCUTS.iter().enumerate() {
        if columns == 2 && index == 2 {
            x += column;
            y = card.top() + 58.0;
        }
        painter.text(
            egui::pos2(x, y),
            Align2::LEFT_TOP,
            title.to_uppercase(),
            theme::small(),
            theme::ACCENT,
        );
        y += line * 1.2;
        for (keys, what) in rows.iter() {
            painter.text(egui::pos2(x, y), Align2::LEFT_TOP, *keys, theme::body(), theme::TEXT);
            painter.text(
                egui::pos2(x + 150.0, y),
                Align2::LEFT_TOP,
                *what,
                theme::body(),
                theme::DIM,
            );
            y += line;
        }
        y += line * 0.6;
    }

    // A click anywhere closes it; the overlay swallows the click so nothing under it reacts.
    let response = ui.interact(screen, egui::Id::unique("help_backdrop"), Sense::click());
    if response.clicked() {
        out.push(Command::SetHelp(false));
    }
}
