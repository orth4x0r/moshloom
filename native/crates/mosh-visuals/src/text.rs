//! Text layer: words (song titles, lyrics, phrases) drawn in contemporary display type that reacts
//! to the music. Owner: L-text.
//!
//! # What happens when
//!
//! `set_text` is the whole of the work: it lays the words out (wrapped, each line centred, spacing
//! and kerning from the font's advances and GPOS pairs), rasterises the block once, turns it into a
//! continuous signed distance field and hands it to a `VideoTexture`. `draw` records one
//! full-screen pass and nothing else: the layout and the glyphs are never touched per frame.
//!
//! # One face, baked when the look needs it
//!
//! `p1` ("font") picks one of three looks, and each look has its own display face:
//!
//! | `p1` | look | face | treatment |
//! | --- | --- | --- | --- |
//! | 0 | Chrome | Unbounded | liquid chrome: environment reflection, iridescent rim, soft specular |
//! | 1 | Sigil | Grenze Gotisch | blackletter polished silver, thorn barbs, blood-red glow |
//! | 2 | Clean | Space Grotesk | crisp minimal, soft shadow, thin cool rim |
//!
//! Only the face the layer last *drew* is baked. A text change costs one face's bake, not three;
//! the look that is not in use is baked on the next `set_text` after `draw` asks for it, which is
//! the frame the editor changes `p1`. The app calls `set_text` every frame with the same words, so
//! that bake lands on the following frame without a per-frame cost.
//!
//! Each face's sheet lives in a texture of [`sheet::capacity`] pixels — the largest block the wrap
//! width and line cap allow — so a text change never reallocates it. Growing a texture means
//! dropping the old image, and dropping it waits for the device; the app calls `set_text` on the
//! render thread.
//!
//! # Parameters
//!
//! | param | meaning |
//! | --- | --- |
//! | `p0` size | the block's height as a fraction of the layer, 0.20..0.88 |
//! | `p1` font | three looks: chrome, sigil, clean |
//! | `p2` spacing | widens the block about its centre, 0.92x..1.47x |
//! | `p3` jitter | a 24 Hz wobble of the whole block (never per glyph: that would tear it) |
//! | `p4` glitch | slice displacement, RGB split, square cut-outs; intensity adds to it |
//! | `p5` outline | the rim outside the glyphs |
//! | `p6` beat pop | the word breathes and its fill lifts on the beat with the low band |
//! | `p7` scroll | rolls the block from below the frame to above it |
//!
//! Nothing here flashes the whole frame: every bright pixel is inside the glyphs, so the 400 ms
//! flash rule has nothing to limit.
//!
//! # Row order
//!
//! A layer target is V-mirrored relative to the stage: the compositor samples a layer with
//! `fullscreen.vert`'s uv (0 at the bottom of the output), so one flip happens between the layer
//! and the picture an audience sees, and there is no second flip at present. The pass therefore
//! draws in audience coordinates (`screen` in `text/text.wgsl`) and a layer read-back is upside
//! down; `tests/text.rs::the_word_is_upright_on_the_stage` pins it through the compositor.

mod layout;
mod sheet;

use mosh_core::{AnalysisFrame, ClipParams};
use mosh_render::upload::VideoTexture;
use mosh_render::{FrameRecorder, Gpu, LayerTarget, Pipeline, RenderError};

use layout::Face;
use sheet::Sheet;

/// What `p0..p7` mean for `text` (shown in the clip editor).
pub const PARAMS: [&str; 8] = [
    "size", "font", "spacing", "jitter", "glitch", "outline", "beat pop", "scroll",
];

/// Bytes of the pass's uniform block. Keep in sync with `text/text.wgsl` (a unit test reflects the
/// shader and compares).
const UNIFORM_BYTES: usize = 96;

/// A show has far fewer layers than this; the cap only stops a bogus index from growing the
/// per-layer vector without bound.
const MAX_LAYERS: usize = 32;

/// How many looks `p1` has, one per face.
const LOOKS: usize = 3;

/// The pass, compiled once in [`TextLayer::new`] with the same naga options as every other shader.
const SHADER: &str = include_str!("text/text.wgsl");

/// The three display faces, one per look. Each is under the SIL Open Font License 1.1 and its
/// licence file sits next to it (`native/assets/fonts/README.md`). Variable fonts: the default
/// (regular) instance is what `fontdue` rasterises.
const FACE_CHROME: &[u8] = include_bytes!("../../../assets/fonts/unbounded/Unbounded[wght].ttf");
const FACE_SIGIL: &[u8] =
    include_bytes!("../../../assets/fonts/grenzegotisch/GrenzeGotisch[wght].ttf");
const FACE_CLEAN: &[u8] = include_bytes!("../../../assets/fonts/spacegrotesk/SpaceGrotesk[wght].ttf");

/// `p1` -> 0 chrome, 1 sigil, 2 clean.
fn look_index(font: f32) -> usize {
    ((font.clamp(0.0, 0.999) * LOOKS as f32) as usize).min(LOOKS - 1)
}

/// One baked face on the GPU, plus the numbers the shader needs to map into it.
struct TextSheet {
    texture: Option<VideoTexture>,
    width: u32,
    height: u32,
    inset_min: [f32; 2],
    inset_max: [f32; 2],
    /// The BGRA bytes of the last bake. Kept so a text change reuses the allocation instead of
    /// creating a few megabytes on the render thread.
    pixels: Vec<u8>,
    /// Rows of `pixels` the last bake touched, so the next one clears only those.
    written_rows: u32,
}

/// What a layer shows: the words it was given, and one sheet per look.
struct LayerText {
    text: String,
    /// One slot per look. Only the look in use is baked; a slot's texture is kept across text
    /// changes (the sheets are all [`sheet::capacity`] sized), because dropping one would wait for
    /// the device — and the app calls `set_text` on the render thread.
    sheets: [Option<TextSheet>; LOOKS],
    /// Which looks hold the words in `text` (false = stale, re-bake before drawing).
    fresh: [bool; LOOKS],
}

pub struct TextLayer {
    pipeline: Pipeline,
    faces: [Option<Face>; LOOKS],
    /// The largest sheet each face can need, so a texture is allocated once per look.
    capacities: [(u32, u32); LOOKS],
    /// The look `draw` last wanted; `set_text` bakes this one.
    look: usize,
    layers: Vec<Option<LayerText>>,
}

impl TextLayer {
    pub fn new(gpu: &Gpu) -> Result<TextLayer, RenderError> {
        let words = mosh_shaders::compile_wgsl(SHADER)
            .map_err(|error| RenderError::Stage(format!("text shader: {error}")))?;
        let pipeline =
            Pipeline::fullscreen(gpu, &words, UNIFORM_BYTES, 1)?.with_name("text.glyphs");
        let faces = [
            Face::new(FACE_CHROME, true),
            Face::new(FACE_SIGIL, true),
            Face::new(FACE_CLEAN, false),
        ];
        let capacities = std::array::from_fn(|index| {
            faces[index]
                .as_ref()
                .map(sheet::capacity)
                .unwrap_or((sheet::MAX_SHEET_WIDTH, sheet::MAX_SHEET_HEIGHT))
        });
        Ok(TextLayer {
            pipeline,
            faces,
            capacities,
            look: 0,
            layers: Vec::new(),
        })
    }

    /// The words `layer` shows (`\n` separates lines). Called by the app when a text clip starts
    /// or its text changes; laying out the glyphs happens here, not per frame.
    ///
    /// Only the face of the look last drawn is baked; a look this layer has not drawn yet is baked
    /// on the next call after `draw` asks for it. Repeating the same text is free. Empty or
    /// whitespace-only text drops the sheets, and the layer then draws transparent black.
    pub fn set_text(&mut self, gpu: &Gpu, layer: usize, text: &str) {
        if layer >= MAX_LAYERS {
            return;
        }
        if self.layers.len() <= layer {
            self.layers.resize_with(layer + 1, || None);
        }
        let entry = self.layers[layer].get_or_insert_with(|| LayerText {
            text: String::new(),
            sheets: [None, None, None],
            fresh: [false; LOOKS],
        });
        if entry.text != text {
            entry.text.clear();
            entry.text.push_str(text);
            entry.fresh = [false; LOOKS];
        }
        let look = self.look;
        TextLayer::bake(
            self.faces[look].as_ref(),
            self.capacities[look],
            gpu,
            entry,
            look,
            text,
        );
    }

    /// Bakes the look's face with `text` into the layer's sheet for that look, reusing the texture
    /// it already holds. Empty or whitespace-only text drops the sheet's freshness, and the layer
    /// then draws transparent black.
    fn bake(
        face: Option<&Face>,
        capacity: (u32, u32),
        gpu: &Gpu,
        entry: &mut LayerText,
        look: usize,
        text: &str,
    ) {
        if entry.fresh[look] {
            return;
        }
        if text.trim().is_empty() {
            entry.fresh[look] = false;
            return;
        }
        let Some(face) = face else {
            entry.fresh[look] = false;
            return;
        };
        let (width, height) = capacity;
        let held = entry.sheets[look].get_or_insert_with(|| TextSheet {
            texture: None,
            width,
            height,
            inset_min: [0.0; 2],
            inset_max: [1.0; 2],
            pixels: Vec::new(),
            written_rows: 0,
        });
        if (held.width, held.height) != (width, height) {
            // A different capacity: the texture has to be replaced (dropping it waits for the
            // device, which is why the capacity is a constant per face).
            held.texture = None;
            held.width = width;
            held.height = height;
            held.pixels = Vec::new();
        }
        let Some(baked) =
            Sheet::bake_into(face, text, width, height, &mut held.pixels, held.written_rows)
        else {
            entry.fresh[look] = false;
            return;
        };
        held.written_rows = baked.mask_h;
        let (inset_min, inset_max) = baked.inset_in(width, height);
        let uploaded = match held.texture.as_mut() {
            Some(texture) => texture.upload(gpu, width, height, &held.pixels),
            None => match VideoTexture::new(gpu, width, height) {
                Ok(mut texture) => match texture.upload(gpu, width, height, &held.pixels) {
                    Ok(()) => {
                        held.texture = Some(texture);
                        Ok(())
                    }
                    Err(error) => Err(error),
                },
                Err(error) => Err(error),
            },
        };
        held.inset_min = inset_min;
        held.inset_max = inset_max;
        entry.fresh[look] = uploaded.is_ok();
        if baked.dropped > 0 {
            // Logged once per text change, not per frame: the app repeats the same text every
            // frame, and that path returns before here.
            eprintln!(
                "text: dropped the last {} of {} characters (cap: {} lines, {} characters)",
                baked.dropped,
                text.chars().count(),
                layout::MAX_LINES,
                layout::MAX_CHARS
            );
        }
    }

    /// Draw `layer`'s text into `dst`.
    pub fn draw(
        &mut self,
        rec: &mut FrameRecorder<'_>,
        layer: usize,
        dst: &LayerTarget,
        frame: &AnalysisFrame,
        params: &ClipParams,
        t: f32,
    ) -> Result<(), RenderError> {
        let look = look_index(params.p[1]);
        self.look = look;
        // A look this layer has not drawn yet is baked here, once: the frame that changes `p1`
        // pays for its face, and every later frame reads the sheet. `FrameRecorder` carries the
        // device, which is what makes a bake possible from the draw path at all.
        let stale = self
            .layers
            .get(layer)
            .and_then(Option::as_ref)
            .is_some_and(|entry| !entry.fresh[look] && !entry.text.trim().is_empty());
        if stale {
            let text = self.layers[layer]
                .as_ref()
                .map(|entry| entry.text.clone())
                .unwrap_or_default();
            if let Some(entry) = self.layers[layer].as_mut() {
                TextLayer::bake(
                    self.faces[look].as_ref(),
                    self.capacities[look],
                    rec.gpu(),
                    entry,
                    look,
                    &text,
                );
            }
        }
        let pipeline = &self.pipeline;
        let Some(entry) = self.layers.get_mut(layer).and_then(Option::as_mut) else {
            return rec.clear(dst);
        };
        // The look this frame wants, else whatever face is fresh (one frame of the old look beats
        // a frame of black while a bake fails).
        let chosen = (0..LOOKS)
            .map(|offset| (look + offset) % LOOKS)
            .find(|index| entry.fresh[*index] && entry.sheets[*index].is_some());
        let Some(index) = chosen else {
            return rec.clear(dst);
        };
        let Some(sheet) = entry.sheets[index].as_mut() else {
            return rec.clear(dst);
        };
        let Some(texture) = sheet.texture.as_mut() else {
            return rec.clear(dst);
        };
        let uniforms = uniforms(
            dst,
            sheet.width,
            sheet.height,
            sheet.inset_min,
            sheet.inset_max,
            params,
            frame,
            t,
        );
        rec.upload_video(texture)?;
        rec.draw_video(pipeline, dst, &uniforms, texture)
    }
}

/// The pass's uniform block. Field order is `text/text.wgsl`'s `U`.
#[allow(clippy::too_many_arguments)]
fn uniforms(
    dst: &LayerTarget,
    width: u32,
    height: u32,
    inset_min: [f32; 2],
    inset_max: [f32; 2],
    params: &ClipParams,
    frame: &AnalysisFrame,
    t: f32,
) -> [u8; UNIFORM_BYTES] {
    let mut block = [0u8; UNIFORM_BYTES];
    let mut at = 0;
    let mut push = |values: &[f32]| {
        for value in values {
            block[at..at + 4].copy_from_slice(&value.to_le_bytes());
            at += 4;
        }
    };
    push(&[dst.width() as f32, dst.height() as f32]);
    push(&[width as f32, height as f32]);
    push(&inset_min);
    push(&inset_max);
    push(&params.p[0..4]);
    push(&params.p[4..8]);
    push(&[
        frame.bands[0],
        frame.bands[1],
        frame.bands[2],
        frame.intensity,
    ]);
    push(&[t, frame.beat_phase, frame.loudness, 0.0]);
    block
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_three_faces_load() {
        assert!(
            Face::new(FACE_CHROME, true).is_some(),
            "Unbounded must parse; it is the chrome look"
        );
        assert!(
            Face::new(FACE_SIGIL, true).is_some(),
            "Grenze Gotisch must parse; it is the sigil look"
        );
        assert!(
            Face::new(FACE_CLEAN, false).is_some(),
            "Space Grotesk must parse; it is the clean look"
        );
    }

    #[test]
    fn the_looks_are_three_and_the_ids_are_stable() {
        assert_eq!(look_index(0.0), 0);
        assert_eq!(look_index(0.33), 0);
        assert_eq!(look_index(0.34), 1);
        assert_eq!(look_index(0.66), 1);
        assert_eq!(look_index(0.67), 2);
        assert_eq!(look_index(1.0), 2);
        assert_eq!(look_index(-3.0), 0);
    }

    #[test]
    fn the_shader_compiles() {
        mosh_shaders::compile_wgsl(SHADER).expect("text.wgsl compiles");
    }

    #[test]
    fn the_uniform_block_matches_the_shader() {
        let module = naga::front::wgsl::parse_str(SHADER).expect("text.wgsl parses");
        let mut declared = 0u32;
        for (_, variable) in module.global_variables.iter() {
            if variable.space != naga::AddressSpace::Uniform {
                continue;
            }
            if let naga::TypeInner::Struct { span, .. } = &module.types[variable.ty].inner {
                declared = declared.max(*span);
            }
        }
        assert_eq!(
            declared as usize, UNIFORM_BYTES,
            "the uniform block text.rs writes must be the struct text.wgsl reads"
        );
    }
}
