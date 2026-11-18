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
//! | 1 | Sigil | Grenze Gotisch | blackletter polished silver, thorns grown out of its contour, a thin blood-red glow |
//! | 2 | Clean | Space Grotesk | crisp minimal, soft shadow, thin cool rim |
//!
//! The sigil look raises the words to uppercase before it bakes them: Grenze Gotisch's lowercase is
//! a polite book serif with none of the look's character, and the look is a sigil, not a caption.
//! Its thorns are geometry grown from the block's outer contour (`text/thorn.rs`), uploaded with
//! the pass's uniforms: sharp tapered spikes where the silhouette turns hardest, sparse, each one
//! its own sweep and bend, and never one that touches the ink.
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
mod thorn;

use mosh_core::{AnalysisFrame, ClipParams};
use mosh_render::upload::VideoTexture;
use mosh_render::{FrameRecorder, Gpu, LayerTarget, Pipeline, RenderError};

use layout::Face;
use sheet::Sheet;
use thorn::Thorn;

/// What `p0..p7` mean for `text` (shown in the clip editor).
pub const PARAMS: [&str; 8] = [
    "size", "font", "spacing", "jitter", "glitch", "outline", "beat pop", "scroll",
];

/// Bytes of the pass's uniform block. Keep in sync with `text/text.wgsl` (a unit test reflects the
/// shader and compares every member's offset): 96 bytes of scalar block, then [`thorn::MAX_THORNS`]
/// thorns at 32 bytes.
const UNIFORM_BYTES: usize = 96 + thorn::MAX_THORNS * THORN_BYTES;

/// The uniform block's scalar member offsets, in bytes. `text/text.wgsl`'s `TextUniforms` is the
/// other half of this contract: `the_uniform_block_members_match_the_shader` reflects the shader
/// and compares each of these, so a reordered or resized WGSL member cannot pass silently.
const DST_AT: usize = 0;
const SHEET_AT: usize = 8;
const INSET_MIN_AT: usize = 16;
const INSET_MAX_AT: usize = 24;
const PARAMS_AT: usize = 32;
const TAPS_AT: usize = 48;
const AUDIO_AT: usize = 64;
const CLOCK_AT: usize = 80;
const THORN_COUNT_AT: usize = 92;
const THORNS_AT: usize = 96;

/// Bytes of one thorn in the uniform block: its `Thorn` struct — `base`, `bend` and `tip` as a
/// `vec2` each, then the base half-width and a padding float.
const THORN_BYTES: usize = 32;

/// The `Thorn` struct's member offsets, in bytes, relative to the thorn.
const THORN_BASE_AT: usize = 0;
const THORN_BEND_AT: usize = 8;
const THORN_TIP_AT: usize = 16;
const THORN_WIDTH_AT: usize = 24;

/// A show has far fewer layers than this; the cap only stops a bogus index from growing the
/// per-layer vector without bound.
const MAX_LAYERS: usize = 32;

/// How many looks `p1` has, one per face.
const LOOKS: usize = 3;

/// The sigil look's index: the one that forces uppercase and grows thorns.
const SIGIL: usize = 1;

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
    /// The sigil look's thorns, packed for the shader ([`ThornWords`]).
    thorns: ThornWords,
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
            Face::new(FACE_SIGIL, true).map(|face| face.with_thorns(true)),
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
    /// whitespace-only text leaves every look's sheet stale — none is dropped — and the layer then
    /// draws transparent black, because a stale look is never chosen without fresh words.
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
    /// it already holds. Empty or whitespace-only text leaves the sheet stale, and the layer then
    /// draws transparent black.
    ///
    /// The sigil look shouts: Grenze Gotisch's lowercase is a polite book serif with none of the
    /// look's character, so that sheet alone is baked from the raised words. The other two looks
    /// draw the text exactly as typed.
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
        let raised;
        let drawn = if look == SIGIL {
            raised = text.to_uppercase();
            raised.as_str()
        } else {
            text
        };
        let (width, height) = capacity;
        let held = entry.sheets[look].get_or_insert_with(|| TextSheet {
            texture: None,
            width,
            height,
            inset_min: [0.0; 2],
            inset_max: [1.0; 2],
            thorns: ThornWords::default(),
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
            Sheet::bake_into(face, drawn, width, height, &mut held.pixels, held.written_rows)
        else {
            entry.fresh[look] = false;
            return;
        };
        held.written_rows = baked.mask_h;
        let (inset_min, inset_max) = baked.inset_in(width, height);
        let thorns = ThornWords::new(&baked.thorns, baked.block_h);
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
        held.thorns = thorns;
        entry.fresh[look] = uploaded.is_ok();
        if baked.dropped > 0 {
            // Logged once per text change, not per frame: the app repeats the same text every
            // frame, and that path returns before here.
            eprintln!(
                "text: dropped the last {} of {} characters (cap: {} lines, {} characters)",
                baked.dropped,
                drawn.chars().count(),
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
            (dst.width(), dst.height()),
            sheet.width,
            sheet.height,
            sheet.inset_min,
            sheet.inset_max,
            &sheet.thorns,
            params,
            frame,
            t,
        );
        rec.upload_video(texture)?;
        rec.draw_video(pipeline, dst, &uniforms, texture)
    }
}

/// The pass's uniform block. Field order and offsets are `text/text.wgsl`'s `TextUniforms`; the
/// `*_AT` constants above are that layout, and the reflection test compares them with the shader.
#[allow(clippy::too_many_arguments)]
fn uniforms(
    dst: (u32, u32),
    width: u32,
    height: u32,
    inset_min: [f32; 2],
    inset_max: [f32; 2],
    thorns: &ThornWords,
    params: &ClipParams,
    frame: &AnalysisFrame,
    t: f32,
) -> [u8; UNIFORM_BYTES] {
    let mut block = [0u8; UNIFORM_BYTES];
    // Every write names its member's offset: the layout is the `*_AT` constants, not a running
    // cursor that can silently drift.
    let mut write = |offset: usize, values: &[f32]| {
        debug_assert!(offset + values.len() * 4 <= UNIFORM_BYTES, "member at {offset}");
        for (slot, value) in values.iter().enumerate() {
            let field = offset + slot * 4;
            block[field..field + 4].copy_from_slice(&value.to_le_bytes());
        }
    };
    write(DST_AT, &[dst.0 as f32, dst.1 as f32]);
    write(SHEET_AT, &[width as f32, height as f32]);
    write(INSET_MIN_AT, &inset_min);
    write(INSET_MAX_AT, &inset_max);
    write(PARAMS_AT, &params.p[0..4]);
    write(TAPS_AT, &params.p[4..8]);
    write(
        AUDIO_AT,
        &[
            frame.bands[0],
            frame.bands[1],
            frame.bands[2],
            frame.intensity,
        ],
    );
    // `thorn_count` is the number of *thorns*, not of the words below: the shader indexes
    // `U.thorns[i]` once per thorn, so publishing the word count would read the array's tail (and
    // loop over it) for every Sigil pixel.
    write(
        CLOCK_AT,
        &[t, frame.beat_phase, frame.loudness],
    );
    write(THORN_COUNT_AT, &[thorns.count as f32]);
    for (index, thorn) in thorns.words.iter().take(thorn::MAX_THORNS).enumerate() {
        let start = THORNS_AT + index * THORN_BYTES;
        write(start + THORN_BASE_AT, &thorn[0][0..2]);
        write(start + THORN_BEND_AT, &thorn[0][2..4]);
        write(start + THORN_TIP_AT, &thorn[1][0..2]);
        write(start + THORN_WIDTH_AT, &thorn[1][2..3]);
    }
    block
}

/// The sigil look's thorns as the shader reads them: how many there are, and one entry per thorn
/// holding its `Thorn` struct's two `vec4` words, in the block's height units.
#[derive(Default)]
struct ThornWords {
    /// The number of thorns — what `thorn_count` carries. One per `U.thorns[i]`, never the number
    /// of `vec4` words (there are two of those per thorn).
    count: usize,
    /// One entry per thorn: `(base.xy, bend.xy)` and `(tip.xy, half-width, unused)`. Every number
    /// is divided by the block's height, because `block_uv` scaled by the block's aspect is that
    /// space; the shader reads them directly and a thorn keeps its weight relative to the block at
    /// every `size`.
    words: Vec<[[f32; 4]; 2]>,
}

impl ThornWords {
    /// Packs `thorns` (in block pixels) for the shader.
    fn new(thorns: &[Thorn], block_h: u32) -> ThornWords {
        let height = block_h.max(1) as f32;
        let words: Vec<[[f32; 4]; 2]> = thorns
            .iter()
            .take(thorn::MAX_THORNS)
            .map(|thorn| {
                [
                    [
                        thorn.base[0] / height,
                        thorn.base[1] / height,
                        thorn.bend[0] / height,
                        thorn.bend[1] / height,
                    ],
                    [
                        thorn.tip[0] / height,
                        thorn.tip[1] / height,
                        thorn.width / height,
                        0.0,
                    ],
                ]
            })
            .collect();
        ThornWords {
            count: words.len(),
            words,
        }
    }
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

    /// The thorn's taper is written twice — in `text/thorn.rs` (which the tests rasterise) and in
    /// `text/text.wgsl` (which the GPU draws) — so the two are compared instead of hoped for. A
    /// text parse is enough: each is a literal on one `const NAME: f32 = ...;` line.
    #[test]
    fn the_thorn_geometry_constants_match_the_shader() {
        for (name, expected) in [
            ("BEND_WIDTH", thorn::BEND_WIDTH),
            ("TIP_WIDTH", thorn::TIP_WIDTH),
        ] {
            assert_eq!(
                wgsl_const(SHADER, name),
                expected,
                "`{name}` in text.wgsl and `text/thorn.rs` must be the same number"
            );
        }
    }

    /// The float a `const NAME: f32 = <literal>;` line of the shader declares.
    fn wgsl_const(source: &str, name: &str) -> f32 {
        let needle = format!("const {name}: f32 = ");
        let start = source
            .find(&needle)
            .unwrap_or_else(|| panic!("text.wgsl declares no `{name}`"))
            + needle.len();
        let end = start + source[start..].find(';').expect("a ; after the literal");
        source[start..end]
            .trim()
            .parse()
            .unwrap_or_else(|error| panic!("`{name}`'s literal: {error}"))
    }

    /// The uniform block's layout, member by member: offsets, the thorn array's stride and the
    /// offsets inside `Thorn`. Swapping two members of `Thorn` in the shader (say `bend` and `tip`)
    /// keeps the total size, so only this test can catch it.
    #[test]
    fn the_uniform_block_members_match_the_shader() {
        let module = naga::front::wgsl::parse_str(SHADER).expect("text.wgsl parses");
        let uniform_type = module
            .global_variables
            .iter()
            .find(|(_, variable)| variable.space == naga::AddressSpace::Uniform)
            .map(|(_, variable)| variable.ty)
            .expect("text.wgsl declares a uniform block");
        let naga::TypeInner::Struct { members, span } = &module.types[uniform_type].inner else {
            panic!("the uniform block is a struct");
        };
        assert_eq!(
            *span as usize, UNIFORM_BYTES,
            "the uniform block text.rs writes must be the struct text.wgsl reads"
        );
        let offset = |name: &str| {
            members
                .iter()
                .find(|member| member.name.as_deref() == Some(name))
                .map(|member| member.offset as usize)
                .unwrap_or_else(|| panic!("text.wgsl's uniform block has no `{name}` member"))
        };
        for (name, expected) in [
            ("dst", DST_AT),
            ("sheet", SHEET_AT),
            ("inset_min", INSET_MIN_AT),
            ("inset_max", INSET_MAX_AT),
            ("p0", PARAMS_AT),
            ("p1", TAPS_AT),
            ("audio", AUDIO_AT),
            ("time", CLOCK_AT),
            ("beat", CLOCK_AT + 4),
            ("loud", CLOCK_AT + 8),
            ("thorn_count", THORN_COUNT_AT),
            ("thorns", THORNS_AT),
        ] {
            assert_eq!(offset(name), expected, "`{name}` sits at the wrong byte");
        }
        // The thorn array: one 32-byte `Thorn` each, and the struct's own member offsets, which is
        // where text.rs writes `base`, `bend`, `tip` and the half-width.
        let thorns = members
            .iter()
            .find(|member| member.name.as_deref() == Some("thorns"))
            .expect("the thorns member");
        let naga::TypeInner::Array { base, stride, .. } = &module.types[thorns.ty].inner else {
            panic!("`thorns` is an array");
        };
        assert_eq!(
            *stride as usize, THORN_BYTES,
            "one thorn's stride in the uniform block"
        );
        let naga::TypeInner::Struct {
            members: thorn_members,
            ..
        } = &module.types[*base].inner
        else {
            panic!("a thorn is a struct");
        };
        let thorn_offset = |name: &str| {
            thorn_members
                .iter()
                .find(|member| member.name.as_deref() == Some(name))
                .map(|member| member.offset as usize)
                .unwrap_or_else(|| panic!("the WGSL `Thorn` has no `{name}` member"))
        };
        assert_eq!(thorn_offset("base"), THORN_BASE_AT);
        assert_eq!(thorn_offset("bend"), THORN_BEND_AT);
        assert_eq!(thorn_offset("tip"), THORN_TIP_AT);
        assert_eq!(thorn_offset("width"), THORN_WIDTH_AT);
    }

    /// The count the shader loops over is the number of thorns, not the number of `vec4` words it
    /// takes to describe them: publishing the word count made the Sigil pass read past the end of
    /// the array and loop twice per thorn.
    #[test]
    fn the_thorn_count_in_the_uniform_block_is_the_thorn_count() {
        let thorns: Vec<Thorn> = (0..7)
            .map(|index| Thorn {
                base: [index as f32, 20.0],
                bend: [index as f32 + 1.0, 10.0],
                tip: [index as f32 + 2.0, 0.0],
                width: 2.0,
            })
            .collect();
        let words = ThornWords::new(&thorns, 100);
        assert_eq!(words.count, thorns.len(), "one uniform entry per thorn");
        assert_eq!(words.words.len(), thorns.len(), "two vec4 per thorn, one entry");
        let block = uniforms(
            (960, 540),
            512,
            256,
            [0.0; 2],
            [1.0; 2],
            &words,
            &ClipParams {
                p: [0.5; 8],
                palette: 0,
                seed: 0,
            },
            &AnalysisFrame::default(),
            1.0,
        );
        let at = |offset: usize| {
            f32::from_le_bytes(block[offset..offset + 4].try_into().expect("four bytes"))
        };
        let count = at(THORN_COUNT_AT);
        assert_eq!(
            count,
            thorns.len() as f32,
            "the shader's `thorn_count` is the number of thorns, not of their words"
        );
        assert!(
            count as usize <= thorn::MAX_THORNS,
            "{count} thorns do not fit the shader's array of {}",
            thorn::MAX_THORNS
        );
        // And the entry the shader indexes per thorn is the one the bake grew, at its own slot.
        let last = THORNS_AT + (thorns.len() - 1) * THORN_BYTES;
        assert_eq!(at(last), 6.0 / 100.0, "thorn 6 starts where thorn 6 is");
        assert_eq!(at(last + 8), 7.0 / 100.0, "its bend follows its base");
        assert_eq!(at(last + 16), 8.0 / 100.0, "its tip follows its bend");
        assert_eq!(at(last + THORN_WIDTH_AT), 2.0 / 100.0, "its half-width");
    }
}
