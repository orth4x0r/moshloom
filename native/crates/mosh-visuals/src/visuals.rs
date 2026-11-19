//! The facade the app calls once per layer per frame. Owner: V-scenes (Task 2.1) may change the
//! bodies; the public signatures are frozen by the coordinator.

use mosh_core::{AnalysisFrame, EffectKind, LayerState, TransitionKind};
use mosh_render::{FlashLimiter, FrameRecorder, Gpu, LayerTarget, RenderError};

use crate::camera::CameraLayer;
use crate::effects::EffectChain;
use crate::fluid::FluidVisualizer;
use crate::mosh::MoshPass;
use crate::overlay::OverlayLayer;
use crate::particles::Particles;
use crate::scenes::Scenes;
use crate::text::TextLayer;
use crate::transitions::TransitionPass;

/// The mosh strength a Datamosh or Melt clip change runs at, even when the clip has no mosh slot of
/// its own. The web app derives no amount from a transition - it uses the clip's `mosh` - so this
/// is a native decision, and it is the same for both: `decide()` treats them identically
/// (`tr != T_XFADE` sets `P.skipres=1`), and `skipres` only means anything with the pass running.
/// 0.9 is above the coarse threshold, as a real mosh entry usually is.
const CARRY_AMOUNT: f32 = 0.9;

/// Which target holds a layer's current picture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Scene,
    Mosh,
    Effects,
    Transition,
}

/// Per-layer GPU state: one scene target and three post targets at the layer's internal size.
///
/// The scene target holds a ping-pong pair so [`MoshPass`] can read last frame's source picture
/// directly for its residual (`src - prev_src`) without a per-frame full-size copy pass. The mosh
/// and effects targets also keep a ping-pong pair for feedback and echo history; only the
/// transition output target holds a single image.
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
                LayerTarget::new_single(gpu, width, height),
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
    pub camera: CameraLayer,
    pub text: TextLayer,
    pub particles: Particles,
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
            camera: CameraLayer::new(gpu)?,
            text: TextLayer::new(gpu)?,
            particles: Particles::new(gpu, width, height)?,
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

        // A clip change hands the transition its "from" picture, and decides what happens to the
        // mosh buffer. `decide()` in engine.c, on the frame a clip is entered:
        //
        //   if (lastK<0 || tr==T_CUT) iframe=1;
        //   else if (tr==T_STROBE) { iframe=1; flash=1; }
        //   else if (tr!=T_XFADE) P.skipres=1;
        //
        // So: Cut and Strobe (and the first clip, which has no transition) take an I-frame;
        // Crossfade is an ordinary P-frame with the residual on; Datamosh and Melt keep the
        // buffer and skip the residual (`MoshPass::carry_over`), which is what lets the new
        // clip's motion push the old picture. `state.transition` is `None` when the clip enters
        // with no transition, which is the web's default `T_CUT`.
        let entry = state
            .transition
            .map_or(TransitionKind::Cut, |transition| transition.kind);
        // `tr != T_XFADE && tr != T_CUT/STROBE`: the two kinds whose picture is the carried
        // buffer, so the mosh below runs for as long as they last (see `CARRY_AMOUNT`).
        let carry_entry = matches!(entry, TransitionKind::Datamosh | TransitionKind::Melt);
        if state.clip != slot.last_clip {
            match entry {
                TransitionKind::Crossfade => {}
                TransitionKind::Datamosh | TransitionKind::Melt => {
                    self.mosh.carry_over(slot.layer);
                }
                TransitionKind::Cut | TransitionKind::Strobe => {
                    self.mosh.invalidate(slot.layer);
                }
            }
            // Only a blending transition needs the saved "from" picture: Cut copies `src`, and
            // Datamosh's `apply` draws nothing at all (the carried mosh buffer is the effect), so
            // neither would ever read it.
            if entry != TransitionKind::Cut && entry != TransitionKind::Datamosh {
                let from = slot.target(slot.last_output_stage);
                self.transitions.begin(rec, slot.layer, from)?;
            }
        }
        slot.last_clip = state.clip;

        if visualizer == "fluid" {
            self.fluid.draw(rec, &slot.scene, frame, &state.params, t)?;
        } else if visualizer == "overlay" {
            self.overlay.draw(rec, slot.layer, &slot.scene)?;
        } else if visualizer == "camera" {
            self.camera
                .draw(rec, slot.layer, &slot.scene, &state.params)?;
        } else if visualizer == "text" {
            self.text
                .draw(rec, slot.layer, &slot.scene, frame, &state.params, t)?;
        } else if visualizer == "particles" {
            self.particles
                .draw(rec, slot.layer, &slot.scene, frame, &state.params, t)?;
        } else {
            self.scenes
                .draw(rec, &slot.scene, visualizer, frame, &state.params, t)?;
        }

        let mut mosh = state
            .effects
            .iter()
            .filter(|e| e.kind == EffectKind::Mosh)
            .map(|e| e.amount)
            .fold(0.0f32, f32::max);
        if carry_entry {
            // The clip change *is* a mosh: keep the buffer running at mosh strength for the whole
            // transition, even when the new clip asks for no mosh of its own. The web app derives
            // no amount from a transition (it uses the clip's `mosh`), so this is a native
            // decision: `P.skipres` only means anything with the pass running, and a Datamosh (or
            // Melt) transition on a clip with no mosh slot would otherwise be an I-frame.
            mosh = mosh.max(CARRY_AMOUNT);
        }
        if mosh > 0.0 {
            if self.mosh.apply(
                rec,
                slot.layer,
                slot.target(slot.output_stage),
                &slot.post[0],
                mosh,
                frame,
                self.fluid.velocity(),
            )? {
                slot.output_stage = Stage::Mosh;
            }
        } else {
            // No mosh slot on a layer that moshed last frame is an I-frame, not a gap: the web app
            // does `if (m <= 0.001) iframe = 1`, so the buffer is forgotten here instead of the
            // next mosh frame carrying a picture the show has already left behind.
            self.mosh.invalidate(slot.layer);
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
