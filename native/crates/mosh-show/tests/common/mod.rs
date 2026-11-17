//! Fixtures shared by the mosh-show integration tests.
//!
//! One layer is the common case: the grid helpers take one `&[Option<ClipId>]` per scene.

#![allow(dead_code)]

use mosh_core::{
    AnalysisFrame, ArrClip, Arrangement, BlendMode, Clip, ClipId, ClipParams, EventFlags,
    FollowAction, Layer, LayerKind, LaunchQuantize, ParamRange, Preset, Project, SceneRow,
    SessionGrid, ShowMode, TransportSource,
};

/// A 4/4 frame at `ppq` (bars are 4 beats), the way mosh-analysis reports one.
pub fn frame(ppq: f64, level: u8, events: EventFlags) -> AnalysisFrame {
    let bar = (ppq / 4.0).floor();
    AnalysisFrame {
        time_s: ppq * 0.5,
        ppq,
        beat_phase: ppq.rem_euclid(1.0) as f32,
        bar_phase: ((ppq - bar * 4.0) / 4.0) as f32,
        beat_index: ppq.floor() as i64,
        bar_index: bar as i64,
        tempo_bpm: 120.0,
        playing: true,
        source: TransportSource::Plugin,
        bands: [0.5, 0.5, 0.5],
        mel: [0.0; 24],
        onset: 0.0,
        loudness_db: -12.0,
        loudness: 0.5,
        brightness: 0.5,
        intensity: level as f32 / 4.0,
        level,
        events,
    }
}

/// A plain clip: no follow action, authored params at 0.5, full 0..1 ranges, no pulse.
pub fn clip(id: ClipId) -> Clip {
    Clip {
        id,
        name: format!("clip{id}"),
        visualizer: "tunnel".to_string(),
        params: ClipParams { p: [0.5; 8], palette: 0, seed: 0 },
        ranges: [[ParamRange::default(); 8]; 4],
        pulse_depth: [0.0; 4],
        effects: Vec::new(),
        transition_in: mosh_core::Transition::default(),
        follow: FollowAction::None,
        follow_after_bars: 1,
        react_to_events: false,
        video: None,
    }
}

/// A clip that follows itself with `follow` every bar.
pub fn following(id: ClipId, follow: FollowAction) -> Clip {
    let mut c = clip(id);
    c.follow = follow;
    c.follow_after_bars = 1;
    c
}

pub fn layer() -> Layer {
    Layer { kind: LayerKind::Visualizer, blend: BlendMode::Normal, opacity: 1.0, internal_scale: 0.5 }
}

/// A session grid with one column per entry of `rows`, one row per scene.
pub fn grid(rows: &[&[Option<ClipId>]], quantize: LaunchQuantize) -> SessionGrid {
    let columns = rows.first().map(|r| r.len()).unwrap_or(0);
    SessionGrid {
        layers: (0..columns).map(|_| layer()).collect(),
        scenes: rows
            .iter()
            .enumerate()
            .map(|(i, cells)| SceneRow { name: format!("scene{i}"), cells: cells.to_vec() })
            .collect(),
        quantize,
    }
}

pub fn project(mode: ShowMode, clips: Vec<Clip>, session: SessionGrid) -> Project {
    Project {
        schema: mosh_core::PROJECT_SCHEMA,
        mode,
        clips,
        presets: Vec::new(),
        session,
        arrangement: Arrangement::default(),
    }
}

/// A project whose arrangement holds `clips` in lane 0: `(id, start_ppq, end_ppq)`.
pub fn with_lane0(mut p: Project, clips: &[(ClipId, f64, f64)]) -> Project {
    p.arrangement.lanes = vec![
        clips
            .iter()
            .map(|(clip, start, end)| ArrClip { clip: *clip, start_ppq: *start, end_ppq: *end })
            .collect(),
    ];
    p
}

pub fn song(name: &str, start_ppq: f64) -> mosh_core::SongMarker {
    mosh_core::SongMarker { name: name.to_string(), start_ppq }
}

pub fn preset(name: &str, visualizer: &str, p: [f32; 8], seed: u64) -> Preset {
    Preset {
        name: name.to_string(),
        visualizer: visualizer.to_string(),
        params: ClipParams { p, palette: 3, seed },
    }
}

/// The clip id of every layer, in layer order.
pub fn ids(plan: &mosh_core::RenderPlan) -> Vec<Option<ClipId>> {
    plan.layers.iter().map(|l| l.clip).collect()
}
