//! The facade the app calls once per layer per frame. Owner: V-scenes (Task 2.1) may change the
//! bodies; the public signatures are frozen by the coordinator.

use mosh_core::{AnalysisFrame, EffectKind, LayerState};
use mosh_render::{FlashLimiter, FrameRecorder, Gpu, LayerTarget, RenderError};

use crate::effects::EffectChain;
use crate::fluid::FluidVisualizer;
use crate::mosh::MoshPass;
use crate::overlay::OverlayLayer;
use crate::scenes::Scenes;
use crate::transitions::TransitionPass;

/// Which target holds a layer's current picture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Scene,
    Mosh,
    Effects,
    Transition,
}

/// Per-layer GPU state: one scene target and three post targets at the layer's internal size.
pub struct LayerSlot {
    pub layer: usize,
    pub scene: LayerTarget,
    /// [mosh, effects, transition]
    pub post: [LayerTarget; 3],
    /// Where this frame's picture ended up.
    pub output_stage: Stage,
    /// Where last frame's picture ended up (a transition's "from" picture).
    pub last_output_stage: Stage,
    /// The clip drawn last frame, to notice a clip change.
    pub last_clip: Option<u32>,
}

impl LayerSlot {
    pub fn new(gpu: &Gpu, layer: usize, width: u32, height: u32) -> LayerSlot {
        LayerSlot {
            layer,
            scene: LayerTarget::new(gpu, width, height),
            post: [
                LayerTarget::new(gpu, width, height),
                LayerTarget::new(gpu, width, height),
                LayerTarget::new(gpu, width, height),
            ],
            output_stage: Stage::Scene,
            last_output_stage: Stage::Scene,
            last_clip: None,
        }
    }

    pub fn target(&self, stage: Stage) -> &LayerTarget {
        match stage {
            Stage::Scene => &self.scene,
            Stage::Mosh => &self.post[0],
            Stage::Effects => &self.post[1],
            Stage::Transition => &self.post[2],
        }
    }

    /// The picture to composite this frame.
    pub fn output(&self) -> &LayerTarget {
        self.target(self.output_stage)
    }
}

/// Everything that draws layers. One per app, shared by all layers.
pub struct Visuals {
    pub scenes: Scenes,
    pub fluid: FluidVisualizer,
    pub overlay: OverlayLayer,
    pub mosh: MoshPass,
    pub effects: EffectChain,
    pub transitions: TransitionPass,
    pub flash: FlashLimiter,
}

impl Visuals {
    /// `width`/`height`: the layers' internal size (output size times `Layer::internal_scale`).
    pub fn new(gpu: &Gpu, width: u32, height: u32) -> Result<Visuals, RenderError> {
        Ok(Visuals {
            scenes: Scenes::new(gpu)?,
            fluid: FluidVisualizer::new(gpu, width, height)?,
            overlay: OverlayLayer::new(gpu)?,
            mosh: MoshPass::new(gpu, width, height)?,
            effects: EffectChain::new(gpu)?,
            transitions: TransitionPass::new(gpu)?,
            flash: FlashLimiter::new(),
        })
    }

    /// Draw one layer for this frame. `t` is seconds since the app started (for animation that
    /// does not follow the beat). Video for overlay clips is supplied through `self.overlay`.
    pub fn draw_layer(
        &mut self,
        rec: &mut FrameRecorder<'_>,
        slot: &mut LayerSlot,
        state: &LayerState,
        frame: &AnalysisFrame,
        t: f32,
    ) -> Result<(), RenderError> {
        slot.last_output_stage = slot.output_stage;
        slot.output_stage = Stage::Scene;
        let Some(visualizer) = state.visualizer.as_deref() else {
            slot.last_clip = None;
            return Ok(());
        };

        // A clip change hands the transition its "from" picture (last frame's output) and starts
        // the datamosh over from the new clip (an I-frame), as the web app does.
        if state.clip != slot.last_clip {
            self.mosh.invalidate(slot.layer);
            if state.transition.is_some() {
                let from = slot.target(slot.last_output_stage);
                self.transitions.begin(rec, slot.layer, from)?;
            }
        }
        slot.last_clip = state.clip;

        if visualizer == "fluid" {
            self.fluid.draw(rec, &slot.scene, frame, &state.params, t)?;
        } else if visualizer == "overlay" {
            self.overlay.draw(rec, slot.layer, &slot.scene)?;
        } else {
            self.scenes.draw(rec, &slot.scene, visualizer, frame, &state.params, t)?;
        }

        let mosh = state
            .effects
            .iter()
            .filter(|e| e.kind == EffectKind::Mosh)
            .map(|e| e.amount)
            .fold(0.0f32, f32::max);
        if mosh > 0.0
            && self.mosh.apply(
                rec,
                slot.layer,
                slot.target(slot.output_stage),
                &slot.post[0],
                mosh,
                frame,
                self.fluid.velocity(),
            )?
        {
            slot.output_stage = Stage::Mosh;
        }

        if self.effects.apply(
            rec,
            slot.target(slot.output_stage),
            &slot.post[1],
            &state.effects,
            frame,
            t,
        )? {
            slot.output_stage = Stage::Effects;
        }

        if let Some(transition) = &state.transition
            && self.transitions.apply(
                rec,
                slot.layer,
                slot.target(slot.output_stage),
                &slot.post[2],
                transition,
                frame,
                &mut self.flash,
            )?
        {
            slot.output_stage = Stage::Transition;
        }
        Ok(())
    }
}
