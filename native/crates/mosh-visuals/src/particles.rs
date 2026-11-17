//! Particle layer: a GPU particle field driven by the music (kicks burst, bands steer, intensity
//! sets density). Owner: L-particles. Stub: draws black. Signatures are frozen.

use mosh_core::{AnalysisFrame, ClipParams};
use mosh_render::{FrameRecorder, Gpu, LayerTarget, RenderError};

/// What `p0..p7` mean for `particles` (shown in the clip editor).
pub const PARAMS: [&str; 8] = [
    "count", "size", "speed", "burst", "swirl", "trail", "colour", "gravity",
];

pub struct Particles {}

impl Particles {
    /// `width`/`height`: the layers' internal size.
    pub fn new(_gpu: &Gpu, _width: u32, _height: u32) -> Result<Particles, RenderError> {
        Ok(Particles {})
    }

    /// Step `layer`'s particles once and draw them into `dst`.
    pub fn draw(
        &mut self,
        _rec: &mut FrameRecorder<'_>,
        _layer: usize,
        _dst: &LayerTarget,
        _frame: &AnalysisFrame,
        _params: &ClipParams,
        _t: f32,
    ) -> Result<(), RenderError> {
        Ok(())
    }
}
