//! Text layer: words (song titles, lyrics, phrases) drawn in glitchy Y2K / cyber type that reacts
//! to the music. Owner: L-text.
//!
//! # What happens when
//!
//! `set_text` is the whole of the work: it lays the words out (lines from `\n`, each line centred,
//! kerning from the font), rasterises every glyph's signed distance field, packs the block into one
//! sheet image and hands it to a `VideoTexture`. `draw` records one full-screen pass and nothing
//! else: the layout and the glyphs are never touched per frame.
//!
//! # Two faces, three looks
//!
//! `p1` ("font") picks one of three looks, and a look owns a face: chrome (0) and clean (2) use
//! Ubuntu Light, thorny (1) uses Hack Regular. `set_text` bakes both faces once; `draw` binds the
//! sheet the current look needs. Styling on top of the face — SDF dilation for the bubble, the
//! bevel shading, the noise-warped thorny edge — lives in `text/text.wgsl`.
//!
//! # Parameters
//!
//! | param | meaning |
//! | --- | --- |
//! | `p0` size | the block's height as a fraction of the layer, 0.20..0.88 |
//! | `p1` font | three looks: chrome, thorny, clean |
//! | `p2` spacing | widens the block about its centre, 0.92x..1.47x |
//! | `p3` jitter | per-glyph offset and rotation, on a 24 Hz clock |
//! | `p4` glitch | slice displacement, RGB split, square cut-outs; intensity adds to it |
//! | `p5` outline | the dark rim outside the glyphs |
//! | `p6` beat pop | the word breathes and each letter pops on the beat with the low band |
//! | `p7` scroll | rolls the block from below the frame to above it |
//!
//! Nothing here flashes the whole frame: every bright pixel is inside the glyphs, so the 400 ms
//! flash rule has nothing to limit.

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

/// How many looks `p1` has.
const LOOKS: usize = 3;

/// The pass, compiled once in [`TextLayer::new`] with the same naga options as every other shader.
const SHADER: &str = include_str!("text/text.wgsl");

/// The two faces. Both are freely redistributable and their licence files sit next to them
/// (`native/assets/fonts/README.md`).
const FACE_DISPLAY: &[u8] = include_bytes!("../../../assets/fonts/Ubuntu-Light.ttf");
const FACE_MONO: &[u8] = include_bytes!("../../../assets/fonts/Hack-Regular.ttf");

/// Which face a look draws with.
fn face_of_look(look: usize) -> usize {
    if look == 1 { 1 } else { 0 }
}

/// `p1` -> 0 chrome, 1 thorny, 2 clean.
fn look_index(font: f32) -> usize {
    ((font.clamp(0.0, 0.999) * LOOKS as f32) as usize).min(LOOKS - 1)
}

/// One baked face on the GPU, plus the numbers the shader needs to map into it.
struct TextSheet {
    texture: VideoTexture,
    width: u32,
    height: u32,
    inset_min: [f32; 2],
    inset_max: [f32; 2],
}

/// What a layer shows: the words it was given, and their sheets per face.
struct LayerText {
    text: String,
    sheets: [Option<TextSheet>; 2],
}

pub struct TextLayer {
    pipeline: Pipeline,
    faces: [Option<Face>; 2],
    layers: Vec<Option<LayerText>>,
}

impl TextLayer {
    pub fn new(gpu: &Gpu) -> Result<TextLayer, RenderError> {
        let words = mosh_shaders::compile_wgsl(SHADER)
            .map_err(|error| RenderError::Stage(format!("text shader: {error}")))?;
        let pipeline =
            Pipeline::fullscreen(gpu, &words, UNIFORM_BYTES, 1)?.with_name("text.glyphs");
        Ok(TextLayer {
            pipeline,
            faces: [Face::new(FACE_DISPLAY), Face::new(FACE_MONO)],
            layers: Vec::new(),
        })
    }

    /// The words `layer` shows (`\n` separates lines). Called by the app when a text clip starts
    /// or its text changes; laying out the glyphs happens here, not per frame.
    ///
    /// Repeating the same text is free. Empty or whitespace-only text drops the sheets, and the
    /// layer then draws transparent black.
    pub fn set_text(&mut self, gpu: &Gpu, layer: usize, text: &str) {
        if layer >= MAX_LAYERS {
            return;
        }
        if self.layers.len() <= layer {
            self.layers.resize_with(layer + 1, || None);
        }
        let entry = self.layers[layer].get_or_insert_with(|| LayerText {
            text: String::new(),
            sheets: [None, None],
        });
        if entry.text == text {
            return;
        }
        entry.text.clear();
        entry.text.push_str(text);

        let blank = text.trim().is_empty();
        for (face, slot) in self.faces.iter().zip(entry.sheets.iter_mut()) {
            if blank {
                *slot = None;
                continue;
            }
            let Some(face) = face else {
                *slot = None;
                continue;
            };
            let Some(baked) = Sheet::bake(face, text) else {
                *slot = None;
                continue;
            };
            let (inset_min, inset_max) = baked.inset();
            let same_size = slot
                .as_ref()
                .is_some_and(|held| (held.width, held.height) == (baked.width, baked.height));
            if same_size {
                // The same words at another size would reallocate; the same size is a memcpy.
                let Some(held) = slot.as_mut() else {
                    continue;
                };
                if held
                    .texture
                    .upload(gpu, baked.width, baked.height, &baked.pixels)
                    .is_err()
                {
                    *slot = None;
                    continue;
                }
                held.inset_min = inset_min;
                held.inset_max = inset_max;
                continue;
            }
            let created =
                VideoTexture::new(gpu, baked.width, baked.height).and_then(|mut texture| {
                    texture.upload(gpu, baked.width, baked.height, &baked.pixels)?;
                    Ok(TextSheet {
                        texture,
                        width: baked.width,
                        height: baked.height,
                        inset_min,
                        inset_max,
                    })
                });
            *slot = created.ok();
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
        let preferred = face_of_look(look_index(params.p[1]));
        let other = 1 - preferred;
        // Two fields, two borrows: the pipeline is borrowed immutably, the layer's sheet mutably.
        let pipeline = &self.pipeline;
        let Some(entry) = self.layers.get_mut(layer).and_then(Option::as_mut) else {
            return rec.clear(dst);
        };
        // A look falls back to the other face when its own sheet failed to bake.
        let Some(sheet) = (if entry.sheets[preferred].is_some() {
            entry.sheets[preferred].as_mut()
        } else {
            entry.sheets[other].as_mut()
        }) else {
            return rec.clear(dst);
        };
        let uniforms = uniforms(dst, sheet, params, frame, t);
        rec.upload_video(&mut sheet.texture)?;
        rec.draw_video(pipeline, dst, &uniforms, &sheet.texture)
    }
}

/// The pass's uniform block. Field order is `text/text.wgsl`'s `U`.
fn uniforms(
    dst: &LayerTarget,
    sheet: &TextSheet,
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
    push(&[sheet.width as f32, sheet.height as f32]);
    push(&sheet.inset_min);
    push(&sheet.inset_max);
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
    fn both_faces_load() {
        assert!(
            Face::new(FACE_DISPLAY).is_some(),
            "the display face (Ubuntu Light) must parse; it backs the chrome and clean looks"
        );
        assert!(
            Face::new(FACE_MONO).is_some(),
            "the mono face (Hack) must parse; it is the thorny look"
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
        assert_eq!(face_of_look(0), 0);
        assert_eq!(face_of_look(1), 1);
        assert_eq!(face_of_look(2), 0);
    }

    #[test]
    fn the_shader_compiles() {
        // `TextLayer::new` compiles it at startup; this is the same compile without a GPU.
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
