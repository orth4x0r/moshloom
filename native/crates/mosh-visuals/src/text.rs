//! Text layer: words (song titles, lyrics, phrases) drawn in contemporary display type that reacts
//! to the music. Owner: L-text.
//!
//! # What happens when
//!
//! `set_text` is the whole of the work: it lays the words out (lines from `\n`, each line centred,
//! spacing and kerning from the font's advances), rasterises the block once, turns it into one
//! continuous signed distance field and hands it to a `VideoTexture` per face. `draw` records one
//! full-screen pass and nothing else: the layout and the glyphs are never touched per frame.
//!
//! # Three faces, three looks
//!
//! `p1` ("font") picks one of three looks, and each look has its own display face from
//! `native/assets/fonts/`:
//!
//! | `p1` | look | face | treatment |
//! | --- | --- | --- | --- |
//! | 0 | Chrome | Unbounded | liquid chrome: environment reflection, iridescent rim, soft specular |
//! | 1 | Sigil | Grenze Gotisch | blackletter polished silver, barbed accents, blood-red glow |
//! | 2 | Clean | Space Grotesk | crisp minimal, soft shadow, thin cool rim |
//!
//! `set_text` bakes one sheet per face; `draw` binds the sheet the current look needs. Styling on
//! top of the face — the bevel from the baked surface normals, the reflections, the barbs, the
//! shadow — lives in `text/text.wgsl`. Nothing in the styling is retro: no scanlines, no CRT or
//! VHS treatment, no pixel type, no acid-green outline.
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
    texture: VideoTexture,
    width: u32,
    height: u32,
    inset_min: [f32; 2],
    inset_max: [f32; 2],
    /// A zeroed buffer the size of the texture, allocated only when a smaller text has to be
    /// padded into a texture that is already allocated.
    scratch: Vec<u8>,
}

/// What a layer shows: the words it was given, and their sheet for each look.
struct LayerText {
    text: String,
    sheets: [Option<TextSheet>; LOOKS],
}

pub struct TextLayer {
    pipeline: Pipeline,
    faces: [Option<Face>; LOOKS],
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
            faces: [
                Face::new(FACE_CHROME),
                Face::new(FACE_SIGIL),
                Face::new(FACE_CLEAN),
            ],
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
            sheets: [None, None, None],
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
            // A texture that already fits the new text is reused, not reallocated: a lyric
            // changing line would otherwise allocate a VkImage and two staging buffers on the
            // render thread, which was most of this call's cost. A larger text grows it.
            let fits = slot
                .as_ref()
                .is_some_and(|held| held.width >= baked.width && held.height >= baked.height);
            if !fits {
                let created =
                    VideoTexture::new(gpu, baked.width, baked.height).map(|texture| TextSheet {
                        texture,
                        width: baked.width,
                        height: baked.height,
                        inset_min: [0.0; 2],
                        inset_max: [1.0; 2],
                        scratch: Vec::new(),
                    });
                *slot = created.ok();
                if slot.is_none() {
                    continue;
                }
            }
            let Some(held) = slot.as_mut() else {
                continue;
            };
            let (inset_min, inset_max) = baked.inset_in(held.width, held.height);
            let uploaded = if (baked.width, baked.height) == (held.width, held.height) {
                held.texture
                    .upload(gpu, held.width, held.height, &baked.pixels)
            } else {
                // The bake is smaller than the texture: pad it into the texture's top-left, with
                // the rest transparent, and point the shader's inset at the smaller block.
                if held.scratch.len() != (held.width * held.height * 4) as usize {
                    held.scratch = vec![0u8; (held.width * held.height * 4) as usize];
                }
                held.scratch.fill(0);
                for row in 0..baked.height {
                    let source = (row * baked.width * 4) as usize;
                    let target = (row * held.width * 4) as usize;
                    let count = (baked.width * 4) as usize;
                    held.scratch[target..target + count]
                        .copy_from_slice(&baked.pixels[source..source + count]);
                }
                held.texture
                    .upload(gpu, held.width, held.height, &held.scratch)
            };
            if uploaded.is_err() {
                *slot = None;
                continue;
            }
            held.inset_min = inset_min;
            held.inset_max = inset_max;
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
        // The pass writes partial alpha (glow, ring) and the layer blend reads the destination,
        // so the target must be black first: otherwise the word composited two frames ago tints
        // the glow, and a freshly allocated target would make the same frame two different
        // pictures. A clear is one `vkCmdClearColorImage`, not a shaded pass.
        rec.clear(dst)?;
        // A look falls back to another face when its own sheet failed to bake.
        let pipeline = &self.pipeline;
        let Some(entry) = self.layers.get_mut(layer).and_then(Option::as_mut) else {
            return Ok(());
        };
        let chosen = (0..LOOKS)
            .map(|offset| (look + offset) % LOOKS)
            .find(|index| entry.sheets[*index].is_some());
        let Some(index) = chosen else {
            return Ok(());
        };
        let Some(sheet) = entry.sheets[index].as_mut() else {
            return Ok(());
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
    fn all_three_faces_load() {
        // One display face per look, each under its own licence file.
        assert!(
            Face::new(FACE_CHROME).is_some(),
            "Unbounded must parse; it is the chrome look"
        );
        assert!(
            Face::new(FACE_SIGIL).is_some(),
            "Grenze Gotisch must parse; it is the sigil look"
        );
        assert!(
            Face::new(FACE_CLEAN).is_some(),
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
