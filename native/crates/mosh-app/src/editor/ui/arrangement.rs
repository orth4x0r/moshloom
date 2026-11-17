//! The Arrangement view: the show as a timeline that follows Live's playhead.
//!
//! ```text
//! | - + Fit Follow | 1       5       9       13      17   <- bars (drag to scroll)
//! |                | [Intro]         [Song two]           <- song markers (double-click the ruler)
//! |----------------|------------------------------------
//! | 1 Visualizer   | [tunnel      ][fluid        ]        <- a lane per layer, blocks in bars
//! |   Normal 100%  |        |                             <- the playhead
//! | 2 Adjustment   | ////[ echo ]///////////////////
//! ```
//!
//! It keeps the web version's directness: coloured blocks with a picture of what they show,
//! click one and it is in the detail panel. Blocks are dragged to move, dragged by an edge to
//! resize, and snap to bar lines (hold Alt for free placement). Nothing is edited while the
//! mouse is down: the drag is a ghost, and the one command is sent on release.
//!
//! The geometry ([`TimeAxis`], [`hit_span`], [`drag_result`]) is pure and tested without a
//! window.

use egui::{Align2, CursorIcon, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};
use mosh_core::{ArrClip, LayerKind};

use crate::editor::model::{
    ClipSource, Command, EditorModel, MAX_LAYERS, SpanRef, bars_at, ppq_at_bar, snap_to_bar,
};
use crate::editor::theme;
use crate::editor::ui::session::layer_header;
use crate::editor::ui::widgets::{self, clipped_text};
use crate::editor::ui::{DragPayload, EditorUi, clip_slot, thumb_params};

const HEADER: f32 = 178.0;
const RULER: f32 = 22.0;
const MARKERS: f32 = 22.0;
const LANE: f32 = 62.0;
const TITLE: f32 = 16.0;
/// How close to a block's end the pointer grabs the edge instead of the block, in pixels.
const EDGE_PX: f32 = 7.0;
const MIN_ZOOM: f32 = 1.5;
const MAX_ZOOM: f32 = 96.0;

/// The horizontal mapping between beats and pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeAxis {
    /// The x of the timeline's left edge.
    pub left: f32,
    pub px_per_beat: f32,
    /// The beat at the left edge.
    pub scroll_beats: f64,
}

impl TimeAxis {
    pub fn x(&self, ppq: f64) -> f32 {
        self.left + ((ppq - self.scroll_beats) * f64::from(self.px_per_beat)) as f32
    }

    pub fn ppq(&self, x: f32) -> f64 {
        self.scroll_beats + f64::from(x - self.left) / f64::from(self.px_per_beat.max(1e-3))
    }

    /// Zoom by `factor` keeping the beat under `anchor_x` where it is (zooming at the pointer).
    pub fn zoomed(self, factor: f32, anchor_x: f32) -> TimeAxis {
        let anchor = self.ppq(anchor_x);
        let px_per_beat = (self.px_per_beat * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        let scroll_beats = anchor - f64::from(anchor_x - self.left) / f64::from(px_per_beat);
        TimeAxis {
            left: self.left,
            px_per_beat,
            scroll_beats: scroll_beats.max(0.0),
        }
    }
}

/// Which part of a block the pointer is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanZone {
    /// The left edge: dragging it moves the block's start.
    Start,
    /// The block itself: dragging it moves the block.
    Body,
    /// The right edge: dragging it moves the block's end.
    End,
}

/// The block of `lane` under `ppq` and the part of it, with edges `edge_beats` wide.
///
/// Where blocks overlap, the one that starts last is on top (the one the show plays). A block
/// too short to have a body between two edges is all body: it can still be moved.
pub fn hit_span(lane: &[ArrClip], ppq: f64, edge_beats: f64) -> Option<(usize, SpanZone)> {
    let (index, span) = lane
        .iter()
        .enumerate()
        .filter(|(_, span)| span.start_ppq <= ppq && ppq < span.end_ppq)
        .max_by(|a, b| a.1.start_ppq.total_cmp(&b.1.start_ppq))?;
    let zone = if span.end_ppq - span.start_ppq < edge_beats * 3.0 {
        SpanZone::Body
    } else if ppq - span.start_ppq < edge_beats {
        SpanZone::Start
    } else if span.end_ppq - ppq < edge_beats {
        SpanZone::End
    } else {
        SpanZone::Body
    };
    Some((index, zone))
}

/// A block being dragged: what was grabbed, and where the ghost is now.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpanDrag {
    pub span: SpanRef,
    pub zone: SpanZone,
    /// Where on the timeline the pointer was when the drag began.
    pub grab_ppq: f64,
    pub original: ArrClip,
    /// The lane the ghost is on.
    pub lane: usize,
    pub start_ppq: f64,
    pub end_ppq: f64,
}

/// Where a dragged block ends up for a pointer at `pointer_ppq`, before snapping:
/// `(start, end)` in beats. Moving keeps the length; an edge stops at the other edge.
pub fn drag_result(
    zone: SpanZone,
    original: &ArrClip,
    grab_ppq: f64,
    pointer_ppq: f64,
) -> (f64, f64) {
    let delta = pointer_ppq - grab_ppq;
    let len = original.end_ppq - original.start_ppq;
    match zone {
        SpanZone::Body => {
            let start = (original.start_ppq + delta).max(0.0);
            (start, start + len)
        }
        SpanZone::Start => (
            (original.start_ppq + delta).clamp(0.0, original.end_ppq),
            original.end_ppq,
        ),
        SpanZone::End => (
            original.start_ppq,
            (original.end_ppq + delta).max(original.start_ppq),
        ),
    }
}

/// The span the timeline covers: the last block, the last marker plus a phrase, at least 16 bars.
fn content_beats(arrangement: &mosh_core::Arrangement) -> f64 {
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
    clips.max(markers).max(64.0)
}

/// How many bars apart the ruler's numbers are, so they never run into each other.
fn label_step(px_per_bar: f32) -> i64 {
    [1, 2, 4, 8, 16, 32, 64, 128]
        .into_iter()
        .find(|step| *step as f32 * px_per_bar >= 52.0)
        .unwrap_or(256)
}

pub fn draw(ui: &mut egui::Ui, state: &mut EditorUi, model: &EditorModel, out: &mut Vec<Command>) {
    let project = model.project();
    let arrangement = &project.arrangement;
    let sig_map = &arrangement.sig_map;
    let status = *model.status();
    let area = ui.max_rect();
    let timeline_left = area.left() + HEADER;
    let timeline_width = (area.width() - HEADER).max(40.0);
    let content = content_beats(arrangement);
    let (alt, zoom, scroll_x, pointer) = ui.input(|input| {
        (
            input.modifiers.alt,
            input.zoom_delta(),
            input.smooth_scroll_delta.x,
            input.pointer.latest_pos(),
        )
    });
    let snap = !alt;

    // Zoom at the pointer, scroll sideways with the wheel, keep the playhead in view.
    let mut axis = TimeAxis {
        left: timeline_left,
        px_per_beat: state.px_per_beat.clamp(MIN_ZOOM, MAX_ZOOM),
        scroll_beats: state.scroll_beats.max(0.0),
    };
    let over_timeline = pointer.is_some_and(|pos| pos.x >= timeline_left && area.contains(pos));
    if over_timeline {
        if (zoom - 1.0).abs() > 1e-3 {
            axis = axis.zoomed(zoom, pointer.map_or(timeline_left, |pos| pos.x));
        }
        if scroll_x != 0.0 {
            axis.scroll_beats -= f64::from(scroll_x / axis.px_per_beat);
        }
    }
    let visible_beats = f64::from(timeline_width / axis.px_per_beat);
    if state.follow && status.transport && state.drag.is_none() {
        let x = axis.x(status.ppq);
        if x < timeline_left + timeline_width * 0.05 || x > timeline_left + timeline_width * 0.9 {
            axis.scroll_beats = status.ppq - visible_beats * 0.15;
        }
    }
    let max_scroll = (content + 16.0 - visible_beats * 0.5).max(0.0);
    axis.scroll_beats = axis.scroll_beats.clamp(0.0, max_scroll);
    state.px_per_beat = axis.px_per_beat;
    state.scroll_beats = axis.scroll_beats;

    // The corner: zoom, fit, follow.
    let (top, _) = ui.allocate_exact_size(Vec2::new(area.width(), RULER + MARKERS), Sense::hover());
    let corner = Rect::from_min_size(top.min, Vec2::new(HEADER, RULER + MARKERS));
    ui.painter().rect_filled(corner, 0.0, theme::PANEL);
    let tool = |x: f32, width: f32| {
        Rect::from_min_size(
            Pos2::new(corner.left() + x, corner.top() + 9.0),
            Vec2::new(width, 26.0),
        )
    };
    let centre = timeline_left + timeline_width * 0.5;
    if widgets::flat_button(ui, tool(6.0, 26.0), "zoom_out", "-", false)
        .on_hover_text("zoom out (Ctrl+wheel)")
        .clicked()
    {
        let next = axis.zoomed(1.0 / 1.4, centre);
        state.px_per_beat = next.px_per_beat;
        state.scroll_beats = next.scroll_beats;
    }
    if widgets::flat_button(ui, tool(36.0, 26.0), "zoom_in", "+", false)
        .on_hover_text("zoom in (Ctrl+wheel)")
        .clicked()
    {
        let next = axis.zoomed(1.4, centre);
        state.px_per_beat = next.px_per_beat;
        state.scroll_beats = next.scroll_beats;
    }
    if widgets::flat_button(ui, tool(66.0, 40.0), "zoom_fit", "Fit", false)
        .on_hover_text("show the whole arrangement")
        .clicked()
    {
        state.px_per_beat = (timeline_width / (content as f32 + 8.0)).clamp(MIN_ZOOM, MAX_ZOOM);
        state.scroll_beats = 0.0;
    }
    if widgets::flat_button(ui, tool(110.0, 60.0), "follow", "Follow", state.follow)
        .on_hover_text("keep the playhead in view while Live plays (F)")
        .clicked()
    {
        state.follow = !state.follow;
    }

    // The ruler: bar numbers. Drag it to scroll, double-click it for a song marker.
    let ruler = Rect::from_min_size(
        Pos2::new(timeline_left, top.top()),
        Vec2::new(timeline_width, RULER),
    );
    let marker_row = Rect::from_min_size(
        Pos2::new(timeline_left, ruler.bottom()),
        Vec2::new(timeline_width, MARKERS),
    );
    let ruler_response = ui
        .interact(ruler, ui.scope_id().with("ruler"), Sense::click_and_drag())
        .on_hover_text("drag to scroll, double-click to add a song marker");
    if ruler_response.dragged() {
        state.scroll_beats = (state.scroll_beats
            - f64::from(ruler_response.drag_delta().x / axis.px_per_beat))
        .clamp(0.0, max_scroll);
        state.follow = false;
    }
    if ruler_response.double_clicked()
        && let Some(pos) = ruler_response.interact_pointer_pos()
    {
        let ppq = axis.ppq(pos.x).max(0.0);
        out.push(Command::AddMarker {
            name: String::new(),
            ppq: if snap { snap_to_bar(sig_map, ppq) } else { ppq },
        });
    }
    let painter = ui.painter().with_clip_rect(Rect::from_min_max(
        Pos2::new(timeline_left, area.top()),
        area.max,
    ));
    painter.rect_filled(ruler.union(marker_row), 0.0, theme::PANEL);

    let px_per_bar = axis.px_per_beat * 4.0;
    let step = label_step(px_per_bar);
    let first_bar = bars_at(sig_map, axis.scroll_beats).floor() as i64;
    let last_bar = bars_at(sig_map, axis.scroll_beats + visible_beats).ceil() as i64 + 1;
    let bar_lines: Vec<(i64, f32)> = (first_bar..=last_bar)
        .map(|bar| (bar, axis.x(ppq_at_bar(sig_map, bar as f64))))
        .collect();
    for (bar, x) in &bar_lines {
        if bar % step == 0 {
            painter.vline(*x, ruler.y_range(), Stroke::new(1.0, theme::FAINT));
            painter.text(
                Pos2::new(x + 4.0, ruler.center().y),
                Align2::LEFT_CENTER,
                bar + 1,
                theme::small(),
                theme::DIM,
            );
        } else if px_per_bar >= 10.0 {
            painter.vline(
                *x,
                (ruler.bottom() - 5.0)..=ruler.bottom(),
                Stroke::new(1.0, theme::LINE),
            );
        }
    }

    // Song markers: flags on their own row.
    for (index, song) in arrangement.songs.iter().enumerate() {
        let at = match state.marker_drag {
            Some((dragged, ppq)) if dragged == index => ppq,
            _ => song.start_ppq,
        };
        let x = axis.x(at);
        let current = status.song == Some(index as u16);
        let galley = painter.layout_no_wrap(song.name.clone(), theme::small(), theme::TEXT);
        let flag = Rect::from_min_size(
            Pos2::new(x, marker_row.top() + 2.0),
            Vec2::new(galley.size().x + 12.0, MARKERS - 4.0),
        );
        if flag.right() < timeline_left || flag.left() > area.right() {
            continue;
        }
        let response = ui.interact(
            flag.intersect(marker_row),
            ui.scope_id().with(("marker", index)),
            Sense::click_and_drag(),
        );
        painter.rect_filled(
            flag,
            2.0,
            if current {
                theme::ACCENT
            } else if response.hovered() {
                theme::HOVER
            } else {
                theme::RAISED
            },
        );
        painter.galley(
            flag.min + Vec2::new(6.0, 3.0),
            galley,
            if current { theme::ON_ACCENT } else { theme::TEXT },
        );
        if response.dragged()
            && let Some(pos) = response.interact_pointer_pos()
        {
            let ppq = axis.ppq(pos.x).max(0.0);
            state.marker_drag = Some((index, if snap { snap_to_bar(sig_map, ppq) } else { ppq }));
        }
        if response.drag_stopped()
            && let Some((dragged, ppq)) = state.marker_drag.take()
            && dragged == index
        {
            out.push(Command::MoveMarker {
                index,
                ppq,
                snap: false,
            });
        }
        response
            .on_hover_text("a song starts here: drag to move, right-click to rename or remove")
            .context_menu(|ui| {
                ui.label("song name");
                if let Some(name) = widgets::text_field(
                    ui,
                    &mut state.draft,
                    widgets::field_key(("marker_name", index)),
                    &song.name,
                    150.0,
                    "name",
                    1,
                ) {
                    out.push(Command::RenameMarker { index, name });
                }
                if ui.button("Remove this marker").clicked() {
                    out.push(Command::RemoveMarker { index });
                    ui.close();
                }
            });
    }
    ui.painter()
        .hline(area.x_range(), top.bottom() - 0.5, theme::line());

    // The lanes.
    let lanes = model.lane_count();
    let layers = &project.session.layers;
    let lanes_top = top.bottom();
    let mut lanes_bottom = lanes_top;
    egui::ScrollArea::vertical()
        .id_salt("lanes_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let height = LANE * lanes as f32 + 40.0;
            let (rect, _) = ui.allocate_exact_size(Vec2::new(area.width(), height), Sense::hover());
            let clip_rect = ui.clip_rect();
            let timeline = Rect::from_min_max(
                Pos2::new(timeline_left, clip_rect.top().max(lanes_top)),
                Pos2::new(area.right(), clip_rect.bottom()),
            );
            let painter = ui.painter().with_clip_rect(timeline);
            let lane_at = |y: f32| -> Option<usize> {
                let index = ((y - rect.top()) / LANE).floor();
                (index >= 0.0 && (index as usize) < lanes).then_some(index as usize)
            };
            lanes_bottom = rect.top() + LANE * lanes as f32;

            for lane in 0..lanes {
                let y = rect.top() + LANE * lane as f32;
                let header = Rect::from_min_size(
                    Pos2::new(area.left(), y),
                    Vec2::new(HEADER - 1.0, LANE - 1.0),
                );
                let body = Rect::from_min_size(
                    Pos2::new(timeline_left, y),
                    Vec2::new(timeline_width, LANE - 1.0),
                );
                let kind = layers.get(lane).map_or(LayerKind::Visualizer, |layer| layer.kind);
                match layers.get(lane) {
                    Some(layer) => layer_header(ui, header, lane, layer, model, out),
                    None => {
                        ui.painter().rect_filled(header, 0.0, theme::PANEL);
                        ui.painter().text(
                            header.left_center() + Vec2::new(8.0, 0.0),
                            Align2::LEFT_CENTER,
                            format!("lane {} (no layer)", lane + 1),
                            theme::body(),
                            theme::FAINT,
                        );
                    }
                }

                painter.rect_filled(body, 0.0, theme::SLOT);
                widgets::kind_texture(&painter, body, kind);
                for (bar, x) in &bar_lines {
                    if bar % step == 0 {
                        painter.vline(*x, body.y_range(), Stroke::new(1.0, theme::LINE));
                    } else if px_per_bar >= 14.0 {
                        painter.vline(
                            *x,
                            body.y_range(),
                            Stroke::new(1.0, theme::alpha(theme::LINE, 0.45)),
                        );
                    }
                }
                painter.hline(body.x_range(), body.bottom() + 0.5, theme::line());

                let spans: &[ArrClip] = arrangement.lanes.get(lane).map_or(&[], Vec::as_slice);
                let response = ui.interact(
                    body.intersect(timeline),
                    ui.scope_id().with(("lane", lane)),
                    Sense::click_and_drag(),
                );
                let edge = f64::from(EDGE_PX / axis.px_per_beat);
                let hover = response
                    .hover_pos()
                    .and_then(|pos| hit_span(spans, axis.ppq(pos.x), edge));

                // The blocks.
                for (index, span) in spans.iter().enumerate() {
                    let block = Rect::from_min_max(
                        Pos2::new(axis.x(span.start_ppq), body.top() + 3.0),
                        Pos2::new(axis.x(span.end_ppq) - 1.0, body.bottom() - 3.0),
                    );
                    if block.right() < timeline.left() || block.left() > timeline.right() {
                        continue;
                    }
                    let dragged = state
                        .drag
                        .is_some_and(|drag| drag.span == SpanRef { lane, index });
                    let selected = model.selected_span() == Some(SpanRef { lane, index });
                    let live = status.playing.get(lane).copied().flatten() == Some(span.clip)
                        && span.start_ppq <= status.ppq
                        && status.ppq < span.end_ppq;
                    paint_block(
                        ui,
                        &painter,
                        state,
                        model,
                        block,
                        span,
                        if dragged { 0.35 } else { 1.0 },
                        selected,
                        live,
                    );
                }

                match hover {
                    Some((_, SpanZone::Start | SpanZone::End)) => {
                        ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
                    }
                    Some((_, SpanZone::Body)) if state.drag.is_none() => {
                        ui.ctx().set_cursor_icon(CursorIcon::Grab);
                    }
                    _ => {}
                }

                let pressed_at = response
                    .interact_pointer_pos()
                    .map(|pos| axis.ppq(pos.x).max(0.0));
                // A drag is noticed a few pixels after the press: it starts where the button
                // went down, or the block would jump by that much.
                let origin = ui
                    .input(|input| input.pointer.press_origin())
                    .map(|pos| axis.ppq(pos.x).max(0.0));
                if response.drag_started()
                    && let Some(ppq) = origin.or(pressed_at)
                    && let Some((index, zone)) = hit_span(spans, ppq, edge)
                {
                    let span = SpanRef { lane, index };
                    out.push(Command::SelectSpan(span));
                    state.drag = Some(SpanDrag {
                        span,
                        zone,
                        grab_ppq: ppq,
                        original: spans[index],
                        lane,
                        start_ppq: spans[index].start_ppq,
                        end_ppq: spans[index].end_ppq,
                    });
                }
                if response.dragged()
                    && let Some(drag) = state.drag.as_mut()
                    && drag.span.lane == lane
                    && let Some(pos) = response.interact_pointer_pos()
                {
                    let (start, end) =
                        drag_result(drag.zone, &drag.original, drag.grab_ppq, axis.ppq(pos.x));
                    let snapped = |ppq: f64| if snap { snap_to_bar(sig_map, ppq) } else { ppq };
                    match drag.zone {
                        SpanZone::Body => {
                            drag.start_ppq = snapped(start);
                            drag.end_ppq = drag.start_ppq + (end - start);
                            drag.lane = lane_at(pos.y).unwrap_or(drag.lane);
                        }
                        SpanZone::Start => drag.start_ppq = snapped(start).min(drag.end_ppq),
                        SpanZone::End => drag.end_ppq = snapped(end).max(drag.start_ppq),
                    }
                    ui.ctx().set_cursor_icon(match drag.zone {
                        SpanZone::Body => CursorIcon::Grabbing,
                        _ => CursorIcon::ResizeHorizontal,
                    });
                }
                if response.drag_stopped()
                    && let Some(drag) = state.drag.take()
                {
                    let moved = (drag.start_ppq - drag.original.start_ppq).abs() > 1e-6
                        || (drag.end_ppq - drag.original.end_ppq).abs() > 1e-6
                        || drag.lane != drag.span.lane;
                    if moved {
                        out.push(match drag.zone {
                            SpanZone::Body => Command::MoveSpan {
                                span: drag.span,
                                to_lane: drag.lane,
                                start_ppq: drag.start_ppq,
                                snap,
                            },
                            SpanZone::Start | SpanZone::End => Command::ResizeSpan {
                                span: drag.span,
                                start_ppq: drag.start_ppq,
                                end_ppq: drag.end_ppq,
                                snap,
                            },
                        });
                    }
                }
                if response.clicked()
                    && let Some(ppq) = pressed_at
                {
                    out.push(match hit_span(spans, ppq, edge) {
                        Some((index, _)) => Command::SelectSpan(SpanRef { lane, index }),
                        None => Command::SelectClip(None),
                    });
                }
                if response.double_clicked()
                    && let Some(ppq) = pressed_at
                    && hit_span(spans, ppq, edge).is_none()
                {
                    // A new block of whatever is being worked on: the selected clip's
                    // visualizer, or the first one in the catalog.
                    let visualizer = model
                        .selected_clip()
                        .map(|clip| clip.visualizer.clone())
                        .or_else(|| {
                            crate::editor::model::catalog()
                                .first()
                                .map(|entry| entry.id.to_string())
                        });
                    if let Some(visualizer) = visualizer {
                        out.push(Command::AddSpan {
                            lane,
                            source: ClipSource::Visualizer(visualizer),
                            start_ppq: if snap {
                                // The bar the pointer is in, not the nearest line.
                                ppq_at_bar(sig_map, bars_at(sig_map, ppq).floor())
                            } else {
                                ppq
                            },
                            len_ppq: None,
                            snap: false,
                        });
                    }
                }
                if response.secondary_clicked()
                    && let Some(ppq) = pressed_at
                    && let Some((index, _)) = hit_span(spans, ppq, edge)
                {
                    out.push(Command::SelectSpan(SpanRef { lane, index }));
                }
                response.context_menu(|ui| match model.selected_span() {
                    Some(span) if span.lane == lane => {
                        if ui.button("Duplicate   Ctrl+D").clicked() {
                            out.push(Command::DuplicateSpan(span));
                            ui.close();
                        }
                        if ui.button("Delete   Del").clicked() {
                            out.push(Command::RemoveSpan(span));
                            ui.close();
                        }
                    }
                    _ => {
                        ui.label("double-click a lane for a new block");
                    }
                });

                // Something from the browser over this lane: show where it would land.
                if let Some(payload) = response.dnd_hover_payload::<DragPayload>()
                    && let DragPayload::Source(..) = payload.as_ref()
                    && let Some(pos) = response.hover_pos()
                {
                    let ppq = axis.ppq(pos.x).max(0.0);
                    let start = if snap {
                        ppq_at_bar(sig_map, bars_at(sig_map, ppq).floor())
                    } else {
                        ppq
                    };
                    let end = ppq_at_bar(sig_map, bars_at(sig_map, start) + 4.0);
                    let ghost = Rect::from_min_max(
                        Pos2::new(axis.x(start), body.top() + 3.0),
                        Pos2::new(axis.x(end), body.bottom() - 3.0),
                    );
                    painter.rect_filled(ghost, theme::RADIUS, theme::alpha(theme::ACCENT, 0.25));
                    painter.rect_stroke(
                        ghost,
                        theme::RADIUS,
                        Stroke::new(1.5, theme::ACCENT),
                        StrokeKind::Inside,
                    );
                }
                if let Some(payload) = response.dnd_release_payload::<DragPayload>()
                    && let DragPayload::Source(source, _) = payload.as_ref()
                    && let Some(pos) = response.hover_pos().or(pointer)
                {
                    let ppq = axis.ppq(pos.x).max(0.0);
                    out.push(Command::AddSpan {
                        lane,
                        source: source.clone(),
                        start_ppq: if snap {
                            ppq_at_bar(sig_map, bars_at(sig_map, ppq).floor())
                        } else {
                            ppq
                        },
                        len_ppq: None,
                        snap: false,
                    });
                }
            }

            // The ghost of the block being dragged, on the lane it would land on.
            if let Some(drag) = state.drag {
                let y = rect.top() + LANE * drag.lane as f32;
                let ghost = Rect::from_min_max(
                    Pos2::new(axis.x(drag.start_ppq), y + 3.0),
                    Pos2::new(axis.x(drag.end_ppq) - 1.0, y + LANE - 4.0),
                );
                paint_block(
                    ui,
                    &painter,
                    state,
                    model,
                    ghost,
                    &ArrClip {
                        clip: drag.original.clip,
                        start_ppq: drag.start_ppq,
                        end_ppq: drag.end_ppq,
                    },
                    0.9,
                    true,
                    false,
                );
                let bars = bars_at(sig_map, drag.end_ppq) - bars_at(sig_map, drag.start_ppq);
                painter.text(
                    ghost.left_bottom() + Vec2::new(4.0, -3.0),
                    Align2::LEFT_BOTTOM,
                    format!(
                        "bar {:.2}  -  {:.2} bars",
                        bars_at(sig_map, drag.start_ppq) + 1.0,
                        bars
                    ),
                    theme::small(),
                    theme::TEXT,
                );
            }

            // Under the lanes: a new layer, and how to start.
            let below = rect.top() + LANE * lanes as f32 + 6.0;
            if layers.len() < MAX_LAYERS {
                let add = Rect::from_min_size(
                    Pos2::new(area.left() + 6.0, below),
                    Vec2::new(84.0, 24.0),
                );
                if widgets::flat_button(ui, add, "add_lane", "+ Layer", false)
                    .on_hover_text("add a layer on top")
                    .clicked()
                {
                    out.push(Command::AddLayer);
                }
            }
            if arrangement.lanes.iter().all(Vec::is_empty) {
                ui.painter().text(
                    Pos2::new(timeline_left + 12.0, below + 4.0),
                    Align2::LEFT_TOP,
                    "Drag a visualizer from the browser onto a lane, or double-click a lane.",
                    theme::body(),
                    theme::DIM,
                );
            }
        });

    // The playhead, over the ruler and every lane.
    let x = axis.x(status.ppq);
    if x >= timeline_left && x <= area.right() && (status.transport || status.ppq > 0.0) {
        let painter = ui.painter().with_clip_rect(Rect::from_min_max(
            Pos2::new(timeline_left, area.top()),
            area.max,
        ));
        let bottom = lanes_bottom.min(area.bottom());
        painter.vline(x, top.top()..=bottom, Stroke::new(1.5, theme::ACCENT));
        painter.add(egui::Shape::convex_polygon(
            vec![
                Pos2::new(x - 5.0, top.top()),
                Pos2::new(x + 5.0, top.top()),
                Pos2::new(x, top.top() + 7.0),
            ],
            theme::ACCENT,
            Stroke::NONE,
        ));
    }
}

/// One block: a title strip in the clip's colour, a body with a picture of its look.
#[allow(clippy::too_many_arguments)]
fn paint_block(
    ui: &egui::Ui,
    painter: &egui::Painter,
    state: &mut EditorUi,
    model: &EditorModel,
    block: Rect,
    span: &ArrClip,
    opacity: f32,
    selected: bool,
    live: bool,
) {
    let clip = model.clip(span.clip);
    let visualizer = clip.map_or("", |clip| clip.visualizer.as_str());
    let color = theme::clip_color(visualizer);
    let fade = |color: egui::Color32| theme::alpha(color, opacity);
    painter.rect_filled(block, theme::RADIUS, fade(theme::dimmed(color, 0.4)));
    let title = Rect::from_min_size(block.min, Vec2::new(block.width(), TITLE.min(block.height())));
    painter.rect_filled(title, theme::RADIUS, fade(color));

    let body = Rect::from_min_max(Pos2::new(block.left(), title.bottom()), block.max).shrink(2.0);
    if body.height() > 8.0
        && block.width() > 44.0
        && let Some(clip) = clip
        && let Some(texture) = state.thumbs.get(
            ui.ctx(),
            clip_slot(clip.id),
            &clip.visualizer,
            &thumb_params(clip),
        )
    {
        let picture = Rect::from_min_size(
            body.min,
            Vec2::new((body.height() * 16.0 / 9.0).min(body.width()), body.height()),
        );
        painter.with_clip_rect(block.intersect(painter.clip_rect())).image(
            texture,
            picture,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            fade(egui::Color32::WHITE),
        );
    }
    if block.width() > 18.0 {
        let name = clip.map_or_else(|| format!("clip {}", span.clip), |clip| clip.name.clone());
        clipped_text(
            painter,
            title.shrink2(Vec2::new(5.0, 0.0)),
            Align2::LEFT_CENTER,
            &name,
            theme::small(),
            theme::ON_ACCENT,
        );
    }
    if selected || live {
        painter.rect_stroke(
            block,
            theme::RADIUS,
            Stroke::new(
                if selected { 1.5 } else { 1.0 },
                if live { theme::ACCENT } else { theme::TEXT },
            ),
            StrokeKind::Inside,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(clip: u32, start_ppq: f64, end_ppq: f64) -> ArrClip {
        ArrClip {
            clip,
            start_ppq,
            end_ppq,
        }
    }

    #[test]
    fn the_axis_maps_beats_to_pixels_and_back() {
        let axis = TimeAxis {
            left: 200.0,
            px_per_beat: 10.0,
            scroll_beats: 8.0,
        };
        assert_eq!(axis.x(8.0), 200.0);
        assert_eq!(axis.x(12.0), 240.0);
        assert_eq!(axis.ppq(240.0), 12.0);
        assert_eq!(axis.ppq(axis.x(33.25)), 33.25);

        // Zooming keeps the beat under the pointer under the pointer.
        let zoomed = axis.zoomed(2.0, 240.0);
        assert_eq!(zoomed.px_per_beat, 20.0);
        assert!((zoomed.ppq(240.0) - 12.0).abs() < 1e-9);
        // The zoom has limits, and the timeline never scrolls before beat 0.
        assert_eq!(axis.zoomed(1000.0, 240.0).px_per_beat, MAX_ZOOM);
        let out = axis.zoomed(0.001, 900.0);
        assert_eq!(out.px_per_beat, MIN_ZOOM);
        assert_eq!(out.scroll_beats, 0.0);
    }

    #[test]
    fn a_block_is_grabbed_by_its_body_or_its_edges() {
        let lane = [span(1, 0.0, 16.0), span(2, 16.0, 32.0), span(3, 40.0, 41.0)];
        assert_eq!(hit_span(&lane, 8.0, 0.5), Some((0, SpanZone::Body)));
        assert_eq!(hit_span(&lane, 0.2, 0.5), Some((0, SpanZone::Start)));
        assert_eq!(hit_span(&lane, 15.8, 0.5), Some((0, SpanZone::End)));
        // The range is half-open: the join belongs to the block that starts there.
        assert_eq!(hit_span(&lane, 16.0, 0.5), Some((1, SpanZone::Start)));
        assert_eq!(hit_span(&lane, 36.0, 0.5), None);
        // A block shorter than three edges is all body, so it can still be moved.
        assert_eq!(hit_span(&lane, 40.1, 0.5), Some((2, SpanZone::Body)));
        // Overlapping blocks: the one that starts last is on top, as the show plays it.
        let stacked = [span(1, 0.0, 32.0), span(2, 8.0, 12.0)];
        assert_eq!(hit_span(&stacked, 10.0, 0.5), Some((1, SpanZone::Body)));
    }

    #[test]
    fn dragging_moves_or_resizes_and_stops_at_the_limits() {
        let block = span(1, 16.0, 32.0);
        // Body: the block moves by the pointer's travel and keeps its length.
        assert_eq!(drag_result(SpanZone::Body, &block, 20.0, 28.0), (24.0, 40.0));
        // It cannot be dragged before the start of the timeline.
        assert_eq!(drag_result(SpanZone::Body, &block, 20.0, -100.0), (0.0, 16.0));
        // Start edge: the end stays, and the start cannot pass it.
        assert_eq!(drag_result(SpanZone::Start, &block, 16.0, 20.0), (20.0, 32.0));
        assert_eq!(drag_result(SpanZone::Start, &block, 16.0, 99.0), (32.0, 32.0));
        assert_eq!(drag_result(SpanZone::Start, &block, 16.0, -5.0), (0.0, 32.0));
        // End edge: the start stays, and the end cannot pass it.
        assert_eq!(drag_result(SpanZone::End, &block, 32.0, 40.0), (16.0, 40.0));
        assert_eq!(drag_result(SpanZone::End, &block, 32.0, 2.0), (16.0, 16.0));
    }

    #[test]
    fn ruler_numbers_keep_their_distance() {
        assert_eq!(label_step(56.0), 1);
        assert_eq!(label_step(28.0), 2);
        assert_eq!(label_step(6.0), 16);
        assert_eq!(label_step(0.1), 256);
    }
}
