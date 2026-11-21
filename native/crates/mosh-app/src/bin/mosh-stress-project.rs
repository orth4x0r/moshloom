//! mosh-stress-project: writes the heaviest project the budget covers (3 visualizer layers with
//! datamosh, effects and a fluid layer, plus a video overlay), for measuring a whole frame.
//!
//!     mosh-stress-project <out.ron> <overlay video path>

use mosh_core::{
    BlendMode, Clip, ClipParams, EffectKind, EffectSlot, FollowAction, Layer, LayerKind,
    LaunchQuantize, ParamRange, Project, SceneRow, SessionGrid, ShowMode, Transition,
    TransitionKind,
};

fn clip(id: u32, visualizer: &str, effects: Vec<EffectSlot>, video: Option<String>) -> Clip {
    Clip {
        id,
        name: format!("{visualizer} (stress)"),
        visualizer: visualizer.to_string(),
        params: ClipParams { p: [0.5; 8], palette: 0, seed: 7 },
        ranges: [[ParamRange::default(); 8]; 4],
        pulse_depth: [0.5; 4],
        effects,
        transition_in: Transition { kind: TransitionKind::Crossfade, beats: 2.0 },
        follow: FollowAction::None,
        follow_after_bars: 8,
        react_to_events: false,
        video,
    }
}

fn slot(kind: EffectKind, amount: f32) -> EffectSlot {
    EffectSlot {
        kind,
        amount,
        param: if kind == EffectKind::Mosh { 0.0 } else { 0.5 },
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(out), Some(video)) = (args.first(), args.get(1)) else {
        eprintln!("usage: mosh-stress-project <out.ron> <overlay video path>");
        std::process::exit(2);
    };
    let layer = |kind, blend, opacity| Layer { kind, blend, opacity, internal_scale: 0.5 };
    let layers = vec![
        layer(LayerKind::Visualizer, BlendMode::Normal, 1.0),
        layer(LayerKind::Visualizer, BlendMode::Add, 0.7),
        layer(LayerKind::Visualizer, BlendMode::Screen, 0.6),
        layer(LayerKind::Overlay, BlendMode::LumaKey, 0.8),
    ];
    let clips = vec![
        clip(1, "tunnel", vec![slot(EffectKind::Mosh, 0.8), slot(EffectKind::Zoom, 0.5), slot(EffectKind::Echo, 0.5)], None),
        clip(2, "fluid", vec![slot(EffectKind::Mirror, 0.6)], None),
        clip(3, "numogram", vec![slot(EffectKind::Mosh, 0.6), slot(EffectKind::Streak, 0.4)], None),
        clip(4, "overlay", Vec::new(), Some(video.clone())),
    ];
    let project = Project {
        mode: ShowMode::Autopilot,
        session: SessionGrid {
            layers,
            scenes: vec![SceneRow { name: "stress".into(), cells: vec![Some(1), Some(2), Some(3), Some(4)] }],
            quantize: LaunchQuantize::Bar,
        },
        clips,
        ..Project::default()
    };
    if let Err(error) = mosh_show::io::save(&project, out) {
        eprintln!("mosh-stress-project: {error:?}");
        std::process::exit(1);
    }
    eprintln!("wrote {out}");
}
