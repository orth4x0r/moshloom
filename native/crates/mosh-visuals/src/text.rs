//! Text layer: words (song titles, lyrics, phrases) drawn in glitchy Y2K / cyber type that reacts
//! to the music. Owner: L-text. Stub: draws black. Signatures are frozen.

use mosh_core::{AnalysisFrame, ClipParams};
use mosh_render::{FrameRecorder, Gpu, LayerTarget, RenderError};

/// What `p0..p7` mean for `text` (shown in the clip editor).
pub const PARAMS: [&str; 8] = [
    "size", "font", "spacing", "jitter", "glitch", "outline", "beat pop", "scroll",
];

pub struct TextLayer {}

impl TextLayer {
    pub fn new(_gpu: &Gpu) -> Result<TextLayer, RenderError> {
        Ok(TextLayer {})
    }

    /// The words `layer` shows (`\n` separates lines). Called by the app when a text clip starts
    /// or its text changes; laying out the glyphs happens here, not per frame.
    pub fn set_text(&mut self, _gpu: &Gpu, _layer: usize, _text: &str) {}

    /// Draw `layer`'s text into `dst`.
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
