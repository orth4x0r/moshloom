//! The facade runs end to end on the stubs: an empty layer and a scene layer draw without error.

use mosh_core::{AnalysisFrame, BlendMode, ClipParams, LayerState};
use mosh_render::{FrameRecorder, Gpu, GpuPreference};
use mosh_visuals::{LayerSlot, Stage, Visuals};

fn state(visualizer: Option<&str>) -> LayerState {
    LayerState {
        layer: 0,
        clip: visualizer.map(|_| 1),
        visualizer: visualizer.map(str::to_owned),
        params: ClipParams::default(),
        effects: Vec::new(),
        transition: None,
        blend: BlendMode::Normal,
        opacity: 1.0,
        seed: 0,
    }
}

#[test]
fn facade_draws_on_stubs() {
    let gpu = Gpu::new(GpuPreference::Integrated).expect("a Vulkan device");
    let mut visuals = Visuals::new(&gpu, 64, 36).expect("visuals");
    let mut slot = LayerSlot::new(&gpu, 0, 64, 36);
    let frame = AnalysisFrame::default();
    {
        let mut rec = FrameRecorder::new(&gpu);
        visuals.draw_layer(&mut rec, &mut slot, &state(None), &frame, 0.0).unwrap();
        visuals.draw_layer(&mut rec, &mut slot, &state(Some("tunnel")), &frame, 0.0).unwrap();
        let ticket = rec.submit();
        gpu.wait_frame(ticket);
    }
    assert_eq!(slot.output_stage, Stage::Scene);
    assert!(gpu.validation_messages().is_empty(), "{:?}", gpu.validation_messages());
}
