//! The 16 web scenes plus custom WGSL scenes. Owner: V-scenes (Task 2.1).
//!
//! # What a scene is here
//!
//! A scene is the web app's fragment body, verbatim, on top of a native prelude
//! (`scenes/prelude.wgsl`) that binds it to the descriptor layout `mosh-render` builds. The body
//! is read from `mosh-shaders/wgsl/` with `include_str!`, so a re-extraction of `gpu.js`
//! (`node scripts/extract_wgsl.mjs`) reaches the native app without a second copy of the shader
//! text; only the prelude is native, and its header lists the four things it changes and why
//! (binding numbers, `fract()` in `lut()`, a generated `ca()`, and no vertex stage: the renderer
//! supplies one). The modules are compiled once, in [`Scenes::new`], so a shader that does not
//! compile fails at construction and never mid-set: the frozen signature returns `Result`, but
//! nothing in it can fail any more, so a bad module panics with its name and naga's message, the
//! way `LayerTarget::new` panics when a target cannot be allocated.
//!
//! Scene 15 (`swarm`) is the one scene whose body cannot be reused: it draws instanced quads with
//! additive blending and `mosh-render` records one 3-vertex draw per pass, so `scenes/swarm.wgsl`
//! accumulates the streaks in the fragment shader, reading a per-frame particle table
//! (`scenes/swarm_pos.wgsl`) that stands in for the instances.
//!
//! Scene 02 (`numogram`) and scene 15 are also the two scenes whose extracted body cannot afford
//! the frame budget (3.4-9.5 and 1.6-4.5 ms at 960x540, where every other scene is 0.1-0.3 ms), so
//! both are drawn by a hand-written replacement of the same maths: `scenes/numogram.wgsl`, and
//! `scenes/swarm_fast.wgsl` + `scenes/swarm_pos_fast.wgsl` + `scenes/swarm_buckets.wgsl`. The
//! extracted and first-generation files they replace are still compiled by this file's tests,
//! which report the pixel difference and the GPU time of both.
//!
//! # The textures a scene binds
//!
//! | input | binding | what it is |
//! | --- | --- | --- |
//! | 0 | 1 | `palT`, 1024x8 palette LUT, generated once from engine.c's `KEYS`/`KEYN` |
//! | 1 | 3 | `caT`, 1x1 placeholder: `ca()` is generated in the prelude (no texture upload) |
//! | 2 | 5 | `melT`, 24x128 mel history, one column shifted per frame |
//! | 3 | 7 | `rdT`, 320x180 reaction-diffusion state, 12 steps per frame while Reaction is up |
//! | 4 | 9 | `mediaT`, 1x1 (the image/video pass is not a visualizer; scene 16 is not here) |
//! | 5 | 11 | the swarm's particle table, 176x3 — only the swarm pipeline has it |
//! | 6 | 13 | the swarm's tile buckets, 144x25 — only the swarm pipeline has it |
//!
//! # Uniforms
//!
//! [`LayerUniforms`] is the web `SU` plus `p0..p7`; a scene reads the 208-byte web prefix. The
//! mapping from a clip is engine.c's plan (`o[0..9]` in `engine.c`): `T = t * p0 + variant * 7.3`,
//! `hue = p2 + variant * 0.137`, `vari = p3`, `dens = p1`, and a palette of 0 means the scene's
//! `DEFPAL` entry. `p4..p7` are the layer's post amounts and are left in `p` for the effects lane.

use mosh_core::{AnalysisFrame, ClipParams};
use mosh_render::{FrameRecorder, Gpu, LayerTarget, Pipeline, RenderError};
use mosh_shaders::{
    LAYER_UNIFORM_BYTES, LayerUniforms, VISUALIZERS, WEB_UNIFORM_BYTES, layer_uniforms,
    visualizer_shader,
};

/// The native prelude every scene body is compiled on top of.
const PRELUDE: &str = include_str!("scenes/prelude.wgsl");
/// Scene 15, written for this renderer (see the module docs).
const SWARM: &str = include_str!("scenes/swarm.wgsl");
/// Scene 15's per-frame particle table, first generation: nothing draws it any more, and the perf
/// test still compiles it to compare the two generations.
#[cfg(test)]
const SWARM_POS: &str = include_str!("scenes/swarm_pos.wgsl");
/// Scene 15's replacement body: the same streaks, precomputed per particle (see the file).
const SWARM_FAST: &str = include_str!("scenes/swarm_fast.wgsl");
/// Scene 15's replacement particle table: per-particle box, ends, width and colour.
const SWARM_POS_FAST: &str = include_str!("scenes/swarm_pos_fast.wgsl");
/// Scene 15's tile buckets: which particles can reach which part of the layer.
const SWARM_BUCKETS: &str = include_str!("scenes/swarm_buckets.wgsl");
/// Scene 02's replacement body: the same zode glyph, evaluated per ring (see the file).
const NUMOGRAM: &str = include_str!("scenes/numogram.wgsl");
/// `palT`, generated once.
const PALETTE: &str = include_str!("scenes/palette.wgsl");
/// `melT`, the mel history shift.
const MEL: &str = include_str!("scenes/mel.wgsl");
/// The reaction-diffusion step, from `mosh-shaders` (scene 11's state).
const RD_SIM: &str = include_str!("../../mosh-shaders/wgsl/rd_sim.wgsl");
/// What an unknown visualizer id draws.
const BLACK: &str = include_str!("scenes/black.wgsl");

/// Textures every scene binds, in `mosh-render`'s input order (see the module docs).
const INPUTS: u32 = 5;

/// The palette a scene falls back to when the clip's palette is 0, from engine.c `DEFPAL`.
const DEFPAL: [u16; 16] = [1, 3, 2, 4, 1, 2, 6, 3, 3, 1, 7, 4, 2, 2, 1, 6];

const PAL_W: u32 = 1024;
const PAL_H: u32 = 8;
const MEL_W: u32 = 24;
const MEL_H: u32 = 128;
const RD_W: u32 = 320;
const RD_H: u32 = 180;
/// gpu.js steps the reaction-diffusion pair 12 times per frame while a Reaction layer is up.
const RD_STEPS: usize = 12;
/// Particles in the swarm's position table (`SWARM_MAX` in `scenes/swarm_fast.wgsl`).
const SWARM_PARTICLES: u32 = 176;
/// The swarm's tile grid and list length (`TILES_X`, `TILES_Y`, `CAPACITY` in
/// `scenes/swarm_buckets.wgsl` and `scenes/swarm_fast.wgsl`).
const SWARM_TILES_X: u32 = 16;
const SWARM_TILES_Y: u32 = 9;
const SWARM_BUCKET_CAPACITY: u32 = 24;

/// The 16 scenes plus the custom scenes registered through [`Scenes::set_custom`].
pub struct Scenes {
    /// One pipeline per visualizer, in `VISUALIZERS` order (`fluid` excluded).
    scene: Vec<(&'static str, Pipeline)>,
    /// `custom:<name>` pipelines, in registration order.
    custom: Vec<(String, Pipeline)>,
    black: Pipeline,
    palette: Pipeline,
    mel_shift: Pipeline,
    rd_step: Pipeline,
    swarm_positions: Pipeline,
    swarm_buckets: Pipeline,
    pal: LayerTarget,
    ca: LayerTarget,
    mel: LayerTarget,
    rd: LayerTarget,
    media: LayerTarget,
    swarm: LayerTarget,
    swarm_tiles: LayerTarget,
    /// `palT` is generated by the first frame that draws a scene.
    palette_ready: bool,
    /// The `t` the mel history last shifted at (one shift per frame, not per layer).
    mel_key: Option<u32>,
    /// The `t` the reaction-diffusion pair last stepped at.
    rd_key: Option<u32>,
    rd_seeded: bool,
}

impl Scenes {
    pub fn new(gpu: &Gpu) -> Result<Scenes, RenderError> {
        // Every scene module is compiled here, once: a shader that does not compile is a defect
        // in this crate, and it fails at construction instead of on stage mid-set.
        let mut scene = Vec::with_capacity(16);
        for info in VISUALIZERS {
            if info.id == "fluid" {
                continue;
            }
            let source = scene_source(info.id);
            let words = compile(&source, info.id);
            // The swarm is the one scene with more inputs than the five every scene binds: its
            // particle table and its tile buckets.
            let textures = if info.id == "swarm" {
                INPUTS + 2
            } else {
                INPUTS
            };
            let pipeline = Pipeline::fullscreen(gpu, &words, LAYER_UNIFORM_BYTES, textures)
                .unwrap_or_else(|error| {
                    panic!(
                        "scene {} pipeline: {error}; validation: {:#?}",
                        info.id,
                        gpu.validation_messages()
                    )
                })
                .with_name(info.id);
            scene.push((info.id, pipeline));
        }
        // Style packs (`styles`): native scenes on the scene prelude plus `P(i)`.
        let style_prelude = crate::styles::prelude();
        for style in crate::styles::all() {
            let words = compile(
                &format!(
                    "{style_prelude}
{}",
                    style.body
                ),
                style.id,
            );
            let pipeline = Pipeline::fullscreen(gpu, &words, LAYER_UNIFORM_BYTES, INPUTS)
                .unwrap_or_else(|error| panic!("style scene {} pipeline: {error}", style.id))
                .with_name(style.id);
            scene.push((style.id, pipeline));
        }
        Ok(Scenes {
            scene,
            custom: Vec::new(),
            black: build(gpu, BLACK, "scene:black", 0, 0),
            palette: build(gpu, PALETTE, "scene:palette", 0, 0),
            mel_shift: build(gpu, MEL, "scene:mel", LAYER_UNIFORM_BYTES, 1),
            // `rd_sim` comes from `mosh-shaders`, whose modules name the fragment entry point
            // `fs`; `mosh-render` looks for `fs_main`, so the committed text is renamed here.
            rd_step: build(
                gpu,
                &rename_fragment(RD_SIM),
                "scene:reaction-diffusion",
                LAYER_UNIFORM_BYTES,
                1,
            ),
            swarm_positions: build(
                gpu,
                SWARM_POS_FAST,
                "scene:swarm-positions",
                LAYER_UNIFORM_BYTES,
                1,
            ),
            swarm_buckets: build(
                gpu,
                SWARM_BUCKETS,
                "scene:swarm-buckets",
                LAYER_UNIFORM_BYTES,
                1,
            ),
            pal: LayerTarget::new(gpu, PAL_W, PAL_H),
            ca: LayerTarget::new(gpu, 1, 1),
            mel: LayerTarget::new(gpu, MEL_W, MEL_H),
            rd: LayerTarget::new(gpu, RD_W, RD_H),
            media: LayerTarget::new(gpu, 1, 1),
            // `swarm_pos_fast.wgsl` writes three rows per particle: box, ends, width + colour.
            swarm: LayerTarget::new(gpu, SWARM_PARTICLES, 3),
            // `swarm_buckets.wgsl` writes one tile per column and 1 + capacity rows.
            swarm_tiles: LayerTarget::new(
                gpu,
                SWARM_TILES_X * SWARM_TILES_Y,
                1 + SWARM_BUCKET_CAPACITY,
            ),
            palette_ready: false,
            mel_key: None,
            rd_key: None,
            rd_seeded: false,
        })
    }

    /// Draw scene `visualizer` ("tunnel", ..., or "custom:<name>") into `dst`.
    pub fn draw(
        &mut self,
        rec: &mut FrameRecorder<'_>,
        dst: &LayerTarget,
        visualizer: &str,
        frame: &AnalysisFrame,
        params: &ClipParams,
        t: f32,
    ) -> Result<(), RenderError> {
        let scene = self.scene.iter().position(|(id, _)| *id == visualizer);
        let custom = match scene {
            Some(_) => None,
            None => self.custom.iter().position(|(id, _)| id == visualizer),
        };
        self.update_tables(rec, frame, params, t, scene)?;

        let res = [dst.width() as f32, dst.height() as f32];
        let uniforms = scene_uniforms(frame, params, palette_for(params, scene), t, res);
        match (scene, custom) {
            (Some(index), _) => {
                let pipeline = &self.scene[index].1;
                if self.scene[index].0 == "swarm" {
                    // Three of this scene's own passes: the particle table for this layer, the
                    // tile buckets that tell a pixel which particles to look at, then the streaks.
                    rec.draw(
                        &self.swarm_positions,
                        &self.swarm,
                        uniforms.as_bytes(),
                        &[&self.pal],
                    )?;
                    rec.draw(
                        &self.swarm_buckets,
                        &self.swarm_tiles,
                        uniforms.as_bytes(),
                        &[&self.swarm],
                    )?;
                    let inputs = [
                        &self.pal,
                        &self.ca,
                        &self.mel,
                        &self.rd,
                        &self.media,
                        &self.swarm,
                        &self.swarm_tiles,
                    ];
                    rec.draw(pipeline, dst, uniforms.as_bytes(), &inputs)
                } else {
                    rec.draw(pipeline, dst, uniforms.as_bytes(), &self.inputs())
                }
            }
            (None, Some(index)) => rec.draw(
                &self.custom[index].1,
                dst,
                uniforms.as_bytes(),
                &self.inputs(),
            ),
            // An id that is not a visualizer is a black layer, never a panic and never the
            // previous frame's picture.
            (None, None) => rec.draw(&self.black, dst, &[], &[]),
        }
    }

    /// Compile and register a custom scene (`custom:<name>`) from WGSL written against
    /// `mosh_shaders::web_prelude()`. On error the previous version (if any) stays active.
    pub fn set_custom(&mut self, gpu: &Gpu, name: &str, wgsl: &str) -> Result<(), String> {
        // `mosh_shaders::compile_custom` cannot be used directly: `mosh-render` names the
        // fragment entry point `fs_main` and the web prelude names it `fs`, so both are renamed
        // here and the error lines are rebased into `wgsl` exactly the way `compile_custom` does.
        let prelude = mosh_shaders::web_prelude().replace("fn vs(", "fn vs_main(");
        let body = wgsl.replace("fn fs(", "fn fs_main(");
        let source = format!("{prelude}{body}");
        let words = mosh_shaders::compile_wgsl(&source).map_err(|error| {
            let prelude_lines = prelude.matches('\n').count() as u32;
            let mut rebased = error;
            rebased.line = rebased
                .line
                .map(|line| line.saturating_sub(prelude_lines).max(1));
            rebased.to_string()
        })?;
        let words = remap_web_bindings(&words);
        let pipeline = Pipeline::fullscreen(gpu, &words, LAYER_UNIFORM_BYTES, INPUTS)
            .map_err(|error| error.to_string())?
            .with_name(format!("custom:{name}"));
        let id = format!("custom:{name}");
        match self.custom.iter_mut().find(|(existing, _)| *existing == id) {
            Some(slot) => slot.1 = pipeline,
            None => self.custom.push((id, pipeline)),
        }
        Ok(())
    }

    /// The five textures every scene binds, in `mosh-render`'s input order.
    fn inputs(&self) -> [&LayerTarget; INPUTS as usize] {
        [&self.pal, &self.ca, &self.mel, &self.rd, &self.media]
    }

    /// Generates the palette once, and advances the mel history and the reaction-diffusion state
    /// once per frame (they are analysis-driven, not layer-driven).
    fn update_tables(
        &mut self,
        rec: &mut FrameRecorder<'_>,
        frame: &AnalysisFrame,
        params: &ClipParams,
        t: f32,
        scene: Option<usize>,
    ) -> Result<(), RenderError> {
        let reaction = scene.is_some_and(|index| self.scene[index].0 == "reaction");
        if !self.palette_ready {
            rec.draw(&self.palette, &self.pal, &[], &[])?;
            self.palette_ready = true;
        }
        let key = t.to_bits();
        if self.mel_key != Some(key) {
            let uniforms = scene_uniforms(frame, params, 1, t, [MEL_W as f32, MEL_H as f32]);
            let bytes = &uniforms.as_bytes()[..WEB_UNIFORM_BYTES];
            rec.draw(&self.mel_shift, &self.mel, bytes, &[&self.mel])?;
            self.mel_key = Some(key);
        }
        if reaction && self.rd_key != Some(key) {
            for _ in 0..RD_STEPS {
                let mut uniforms = scene_uniforms(frame, params, 1, t, [RD_W as f32, RD_H as f32]);
                if !self.rd_seeded {
                    uniforms.aux2 = 1.0;
                }
                // gpu.js: a kick (pulse > 0.7, bass > 0.3) seeds the pattern at a hashed point.
                if uniforms.pulse > 0.7 && uniforms.bass > 0.3 {
                    let beat = uniforms.beat.floor();
                    uniforms.aux0 = 0.1 + 0.8 * hash01(beat * 12.9898);
                    uniforms.aux1 = 0.1 + 0.8 * hash01(beat * 78.233);
                }
                let bytes = &uniforms.as_bytes()[..WEB_UNIFORM_BYTES];
                rec.draw(&self.rd_step, &self.rd, bytes, &[&self.rd])?;
                self.rd_seeded = true;
            }
            self.rd_key = Some(key);
        }
        Ok(())
    }
}

/// A scene's source: the native prelude plus a fragment body.
///
/// Two scenes are drawn by a hand-written body (see the module docs): scene 02 keeps its extracted
/// geometry but arranges it per ring, and scene 15 accumulates its streaks from a precomputed
/// particle table. Every other body comes from `mosh-shaders/wgsl/` (`include_str!`, so a
/// re-extraction is picked up by the next build). `mosh-shaders` exposes the SPIR-V and the module
/// names but not the WGSL text, and the native port needs the text: `lut()` has to wrap with
/// `fract()` and the entry points are named differently. Everything after the shared prelude's
/// vertex stage is kept verbatim (scene 02, for one, declares its own consts and helpers there).
fn scene_source(id: &str) -> String {
    match id {
        "numogram" => format!("{PRELUDE}\n{NUMOGRAM}"),
        "swarm" => format!("{PRELUDE}\n{SWARM_FAST}"),
        _ => extracted_source(id),
    }
}

/// The extracted web body of a scene, on the native prelude: what `scene_source` returns for the
/// fourteen scenes that keep their web shader, and what the perf test compares the two
/// hand-written scenes against.
fn extracted_source(id: &str) -> String {
    if id == "swarm" {
        return format!("{PRELUDE}\n{SWARM}");
    }
    let module = visualizer_shader(id).unwrap_or_else(|| panic!("no extracted module for {id}"));
    let text = scene_module(module).unwrap_or_else(|| panic!("no source for module {module}"));
    let body = scene_body(text)
        .unwrap_or_else(|| panic!("{module}: no `@vertex fn vs` line to split the prelude on"));
    let fragment = rename_fragment(&body);
    assert_ne!(
        fragment, body,
        "{module}: no `@fragment fn fs` entry point to rename"
    );
    format!("{PRELUDE}\n{fragment}")
}

/// The scene's own text: everything after the web prelude's `@vertex fn vs(...)` line.
fn scene_body(text: &str) -> Option<String> {
    let mut body = String::new();
    let mut after_prelude = false;
    for line in text.lines() {
        if !after_prelude {
            after_prelude = line.trim_start().starts_with("@vertex fn vs(");
            continue;
        }
        body.push_str(line);
        body.push('\n');
    }
    after_prelude.then_some(body)
}

/// Renames the fragment entry point: every module in `mosh-shaders` (extracted from `gpu.js`)
/// calls it `fs`, and `mosh-render`'s pipeline looks for `fs_main`. Same text, one word.
fn rename_fragment(text: &str) -> String {
    text.replace("@fragment fn fs(", "@fragment fn fs_main(")
}

/// The committed WGSL text of the scene modules, by module name (`visualizer_shader`).
macro_rules! scene_modules {
    ($($module:literal),* $(,)?) => {
        fn scene_module(name: &str) -> Option<&'static str> {
            match name {
                $($module => Some(include_str!(concat!("../../mosh-shaders/wgsl/", $module, ".wgsl"))),)*
                _ => None,
            }
        }
    };
}

scene_modules!(
    "scene00_tunnel",
    "scene01_loom",
    "scene02_numogram",
    "scene03_flesh",
    "scene04_rotor",
    "scene05_moire",
    "scene06_spectrum",
    "scene07_cells",
    "scene08_punch_cards",
    "scene09_vortex",
    "scene10_kaleido",
    "scene11_reaction",
    "scene12_grid",
    "scene13_glyph_rain",
    "scene14_scan",
    "scene15_swarm",
);

fn compile(source: &str, name: &str) -> Vec<u32> {
    mosh_shaders::compile_wgsl(source)
        .unwrap_or_else(|error| panic!("native scene {name} does not compile: {error}"))
}

fn build(gpu: &Gpu, source: &str, name: &str, uniforms: usize, textures: u32) -> Pipeline {
    let words = compile(source, name);
    Pipeline::fullscreen(gpu, &words, uniforms, textures)
        .unwrap_or_else(|error| {
            panic!(
                "{name} pipeline: {error}; validation: {:#?}",
                gpu.validation_messages()
            )
        })
        .with_name(name)
}

/// The palette a clip asks for, with engine.c's default when it leaves the field at 0.
fn palette_for(params: &ClipParams, scene: Option<usize>) -> u16 {
    match params.palette {
        1..=7 => params.palette,
        _ => scene.map_or(1, |index| DEFPAL[index]),
    }
}

/// The uniform block for one scene draw: engine.c's plan mapping (`o[0..9]`) on top of
/// [`layer_uniforms`].
fn scene_uniforms(
    frame: &AnalysisFrame,
    params: &ClipParams,
    palette: u16,
    t: f32,
    res: [f32; 2],
) -> LayerUniforms {
    // engine.c: `int v = (int)b->var`.
    let variant = params.p[3] as i32;
    let mut adjusted = *params;
    adjusted.palette = palette;
    // o[3] = fHue[f] + v * 0.137 + hueo (the native port has no analysis hue drift yet).
    adjusted.p[2] = params.p[2] + variant as f32 * 0.137;
    adjusted.p[3] = variant as f32;
    // o[2] = fTT[f] * speed + v * 7.3: `p0` is the scene's clock rate, `t` is the app's.
    let clock = t * params.p[0].max(0.0) + variant as f32 * 7.3;
    let mut uniforms = layer_uniforms(frame, &adjusted, clock, res);
    // gpu.js passes the wall frame counter, not the scene clock.
    uniforms.frame = t * 60.0;
    uniforms
}

/// `fract(sin(x) * 43758.5453)` into 0..1, as gpu.js writes the kick seed. The literal is the web
/// app's, digits and all, even though f32 rounds it.
#[allow(clippy::excessive_precision)]
fn hash01(x: f32) -> f32 {
    (x.sin() * 43758.5453).rem_euclid(1.0)
}

/// Rewrites a web custom shader's binding numbers into `mosh-render`'s layout, once, at compile
/// time: the web bind group is sampler at 1 and textures at 2..6, the renderer's is uniform at 0
/// and (image, sampler) pairs at (1,2), (3,4), ... So 1 -> 2 (the one immutable sampler), 2 -> 1
/// (palT), 4 -> 5 (melT), 5 -> 7 (rdT), 6 -> 9 (mediaT); caT stays at 3, which is input 1.
fn remap_web_bindings(words: &[u32]) -> Vec<u32> {
    const OP_DECORATE: u32 = 71;
    const DECORATION_BINDING: u32 = 33;
    let mut out = words.to_vec();
    let mut index = 5; // past the SPIR-V header
    while index + 3 < out.len() {
        let instruction = out[index];
        let opcode = instruction & 0xffff;
        let count = (instruction >> 16) as usize;
        if count == 0 {
            break;
        }
        if opcode == OP_DECORATE && out[index + 2] == DECORATION_BINDING {
            out[index + 3] = match out[index + 3] {
                1 => 2,
                2 => 1,
                4 => 5,
                5 => 7,
                6 => 9,
                other => other,
            };
        }
        index += count;
    }
    out
}

#[cfg(test)]
mod perf;

#[cfg(test)]
mod tests {
    use super::{DEFPAL, remap_web_bindings, scene_source};
    use mosh_shaders::VISUALIZERS;

    /// Every scene source assembles: the prelude plus a body, or the hand-written swarm.
    #[test]
    fn every_scene_has_a_source() {
        let scenes = VISUALIZERS.iter().filter(|info| info.id != "fluid");
        assert_eq!(scenes.clone().count(), 16);
        for info in scenes {
            let source = scene_source(info.id);
            assert!(
                source.contains("@fragment fn fs_main("),
                "{} has no fragment entry point",
                info.id
            );
        }
    }

    #[test]
    fn default_palettes_are_in_range() {
        assert_eq!(DEFPAL.len(), 16);
        assert!(DEFPAL.iter().all(|palette| (1..=7).contains(palette)));
    }

    /// The remap is a permutation on the bindings a web shader uses, and leaves everything else.
    #[test]
    fn binding_remap_moves_the_web_layout_onto_the_renderer_layout() {
        let decorate = |target: u32, decoration: u32, literal: u32| {
            [(4 << 16) | 71, target, decoration, literal]
        };
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 0, 0];
        for binding in [0, 1, 2, 4, 6] {
            words.extend_from_slice(&decorate(binding, 33, binding));
        }
        // OpDecorate %v9 Location 3: not a binding, untouched.
        words.extend_from_slice(&decorate(9, 30, 3));
        words.extend_from_slice(&[0x0001_0011]);

        let out = remap_web_bindings(&words);
        let mut bindings = Vec::new();
        let mut index = 5;
        while index + 3 < out.len() {
            let count = (out[index] >> 16) as usize;
            if count == 0 {
                break;
            }
            if out[index] & 0xffff == 71 && out[index + 2] == 33 {
                bindings.push(out[index + 3]);
            }
            index += count;
        }
        assert_eq!(bindings, vec![0, 2, 1, 5, 9]);
        assert_eq!(
            out[out.len() - 5 + 3],
            3,
            "a Location decoration is left alone"
        );
    }
}
