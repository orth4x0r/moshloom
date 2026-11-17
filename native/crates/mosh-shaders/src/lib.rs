//! mosh-shaders: the WGSL sources, their SPIR-V, the per-layer uniform ABI and the visualizer list.
//!
//! Task 1.12 of `docs/superpowers/plans/2026-11-17-native-port.md`. The shaders are not rewritten:
//! `scripts/extract_wgsl.mjs` reads `gpu.js` (read-only) and writes one complete WGSL module per
//! pass into `wgsl/`, `build.rs` compiles every one of them to SPIR-V with naga, and the result is
//! committed so `spirv()` is a pure lookup at run time.
//!
//! The extracted files are committed on purpose, so a change to a web shader shows up as a reviewable
//! `.wgsl` diff here. After editing `gpu.js`, run `node scripts/extract_wgsl.mjs`; `node
//! scripts/extract_wgsl.mjs --check` reports a stale tree without writing anything.
//!
//! # What is in `wgsl/`
//!
//! 34 modules, all compiled by the build step (a broken one fails the build, not a show):
//!
//! * `scene00_tunnel` … `scene15_swarm` — the 16 web scenes, in `WG_SCENES` order.
//! * `scene16_media` — the image/video pass. Not a visualizer: it samples `mediaT`.
//! * `custom_template` — the web app's custom-scene template as a complete module.
//! * `web_prelude` — `WG_HEAD + WG_VS`: the web uniform struct, the bindings, the helpers, the
//!   full-screen vertex stage. [`web_prelude()`] returns exactly this text.
//! * `layer_prelude` — the same prelude with the native struct `LU` (`SU` + `p0..p7`).
//! * `rd_sim` — the reaction-diffusion step (scene 11's state).
//! * `fx_mix`, `fx_copy`, `fx_pre`, `motion_estimate`, `fx_feedback`, `fx_output` — the post chain
//!   (transition mix, copy, effects pre-pass, block-matching motion, mosh feedback, output grade).
//! * `fluid_advect`, `fluid_force`, `fluid_curl`, `fluid_vorticity`, `fluid_divergence`,
//!   `fluid_jacobi`, `fluid_gradient` — the seven stable-fluids passes on a 256x144 grid.
//!
//! # Bind group 0, as the renderer must build it
//!
//! Scene modules (`scene*`, `custom_template`, `web_prelude`, `layer_prelude`):
//!
//! | binding | resource |
//! | --- | --- |
//! | 0 | uniform buffer, [`LayerUniforms`] (240 B). `custom_template`/`web_prelude`/`scene*` declare 208 B and read that prefix; `layer_prelude` declares 240 B |
//! | 1 | filtering sampler (`lin`): linear, `addressModeU/V = repeat` |
//! | 2 | `palT`: 1024x8 `rgba8unorm`, 8 palette rows of 1024 entries; `lut(v)` samples one row |
//! | 3 | `caT`: 512x512 `rgba8unorm`, three cell-automaton planes in rgb, `r` = the rule plane |
//! | 4 | `melT`: 24x128 `r8unorm`, 24 mel bands x 128 frames of history (newest row 0) |
//! | 5 | `rdT`: 320x180 `rgba16float`, reaction-diffusion state (ping-pong pair) |
//! | 6 | `mediaT`: the clip's image or video frame, `rgba8unorm`; a 1x1 opaque texture when unused |
//!
//! Other passes:
//!
//! | module | bindings |
//! | --- | --- |
//! | `rd_sim` | 0 uniform `SU`, 1 `rdT` |
//! | `fx_mix` | 0 uniform `FU`, 1 `aT`, 2 `bT` |
//! | `fx_copy` | 0 `aT` (no uniform) |
//! | `fx_pre` | 0 uniform `FU`, 1 sampler, 2 `inT` |
//! | `motion_estimate` | 0 uniform `FU`, 1 sampler, 2 `curT`, 3 `prvT` |
//! | `fx_feedback` | 0 uniform `FU`, 1 sampler, 2 `srcT`, 3 `prvT`, 4 `bufT`, 5 `mvT`, 6 `velT` |
//! | `fx_output` | 0 uniform `FU`, 1 sampler, 2 `bufT` |
//! | `fluid_*` | 0 uniform `FUB` (texel, dt, visc, imp, beat, bass, pulse, gain, vort), 1 sampler, 2 `tA`, 3 `tB`, 4 `tC`, 5 `tD` |
//!
//! [`bindings`] returns the same data, reflected from the compiled modules, so a mismatch between
//! this table and a shader is a test failure (`tests/compile.rs::scene_bindings_match_the_web_bind_group`).
//! Target formats, from `gpu.js`: scenes and the post chain render `rgba16float` (scene 15 is
//! additive), `fx_output` renders the swapchain format, `fx_copy` renders `rgba8unorm`
//! (thumbnails), the fluid passes render `rgba16float` at 256x144.
//!
//! # What this crate does not do
//!
//! It does not render anything. SPIR-V validity here means "naga accepted it"; that it executes,
//! paints, or matches a web frame is Task 2.1's and Task 1.7's problem, and no pixel of this lane
//! has been on a GPU.

#![forbid(unsafe_code)]

mod abi;
mod runtime;

pub use abi::{
    LAYER_UNIFORM_BYTES, LayerUniforms, WEB_UNIFORM_BYTES, layer_prelude, layer_uniforms, web_prelude,
};
pub use runtime::{ShaderError, compile_custom, compile_wgsl};

mod generated {
    include!(concat!(env!("OUT_DIR"), "/spirv.rs"));
}

/// One compiled module: its name (the file stem in `wgsl/`), its SPIR-V and its bindings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Module {
    pub name: &'static str,
    pub spirv: &'static [u32],
    pub bindings: &'static [Binding],
    /// Bytes of the module's uniform block, when it has one. `None` for `fx_copy`.
    pub uniform_bytes: Option<u32>,
}

/// One resource in bind group 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    pub group: u32,
    pub binding: u32,
    pub kind: BindingKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingKind {
    Uniform,
    Sampler,
    Texture,
}

/// Every compiled module, sorted by name.
pub fn modules() -> &'static [Module] {
    generated::MODULES
}

/// SPIR-V for a module, by name (the file stem in `wgsl/`).
pub fn spirv(name: &str) -> Option<&'static [u32]> {
    modules().iter().find(|m| m.name == name).map(|m| m.spirv)
}

/// Bind group 0 for a module, reflected from the compiled shader.
pub fn bindings(name: &str) -> Option<&'static [Binding]> {
    modules().iter().find(|m| m.name == name).map(|m| m.bindings)
}

/// Bytes of a module's uniform block, when it declares one.
pub fn uniform_bytes(name: &str) -> Option<u32> {
    modules().iter().find(|m| m.name == name).and_then(|m| m.uniform_bytes)
}

/// Module names, sorted.
pub fn shader_names() -> impl Iterator<Item = &'static str> {
    modules().iter().map(|m| m.name)
}

/// A visualizer the app can put on a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisualizerInfo {
    /// Stable id, used in `Clip::visualizer` (`"tunnel"`, `"fluid"`, ...).
    pub id: &'static str,
    /// Display name.
    pub name: &'static str,
    /// What `p0..p7` mean for this visualizer. Every slot is used by at least one layer the
    /// renderer feeds it (`p0` speed, `p1` density, `p2` hue, `p3` variant are the shared four;
    /// `p4..p7` are the layer's post amounts, which the web app also keeps per block).
    pub params: [&'static str; 8],
}

/// The 16 web scenes in `WG_SCENES` order, then `fluid`.
pub const VISUALIZERS: &[VisualizerInfo] = &[
    VisualizerInfo {
        id: "tunnel",
        name: "Tunnel",
        params: ["speed", "density (rings + facets)", "hue", "variant", "mosh", "glitch", "bloom", "rgb split"],
    },
    VisualizerInfo {
        id: "loom",
        name: "Loom",
        params: ["speed", "density (zoom)", "hue", "variant (cell rule)", "mosh", "glitch", "bloom", "rgb split"],
    },
    VisualizerInfo {
        id: "numogram",
        name: "Numogram",
        params: ["speed", "density (ring size)", "hue", "variant (twist)", "mosh", "glitch", "bloom", "rgb split"],
    },
    VisualizerInfo {
        id: "flesh",
        name: "Flesh",
        params: ["speed", "density (blob softness)", "hue", "variant", "mosh", "glitch", "bloom", "rgb split"],
    },
    VisualizerInfo {
        id: "rotor",
        name: "Rotor",
        params: ["speed", "density (stripe frequency)", "hue", "variant", "mosh", "glitch", "bloom", "rgb split"],
    },
    VisualizerInfo {
        id: "moire",
        name: "Moiré",
        params: ["speed", "density (line frequency)", "hue", "variant", "mosh", "glitch", "bloom", "rgb split"],
    },
    VisualizerInfo {
        id: "spectrum",
        name: "Spectrum",
        params: ["speed", "density (radial scale)", "hue", "variant (radial / bars)", "mosh", "glitch", "bloom", "rgb split"],
    },
    VisualizerInfo {
        id: "cells",
        name: "Cells",
        params: ["speed", "density (zoom)", "hue", "variant", "mosh", "glitch", "bloom", "rgb split"],
    },
    VisualizerInfo {
        id: "punch_cards",
        name: "Punch cards",
        params: ["speed", "density (columns)", "hue", "variant (punch rule)", "mosh", "glitch", "bloom", "rgb split"],
    },
    VisualizerInfo {
        id: "vortex",
        name: "Vortex",
        params: ["speed", "density (arms)", "hue", "variant", "mosh", "glitch", "bloom", "rgb split"],
    },
    VisualizerInfo {
        id: "kaleido",
        name: "Kaleido",
        params: ["speed", "density (segments)", "hue", "variant", "mosh", "glitch", "bloom", "rgb split"],
    },
    VisualizerInfo {
        id: "reaction",
        name: "Reaction",
        params: ["speed", "density (feed rate)", "hue", "variant (seed)", "mosh", "glitch", "bloom", "rgb split"],
    },
    VisualizerInfo {
        id: "grid",
        name: "Grid",
        params: ["speed", "density (line count)", "hue", "variant", "mosh", "glitch", "bloom", "rgb split"],
    },
    VisualizerInfo {
        id: "glyph_rain",
        name: "Glyph rain",
        params: ["speed", "density (columns)", "hue", "variant", "mosh", "glitch", "bloom", "rgb split"],
    },
    VisualizerInfo {
        id: "scan",
        name: "Scan",
        params: ["speed", "density (block count)", "hue", "variant", "mosh", "glitch", "bloom", "rgb split"],
    },
    VisualizerInfo {
        id: "swarm",
        name: "Swarm",
        params: ["speed", "density (streak count)", "hue", "variant", "mosh", "glitch", "bloom", "rgb split"],
    },
    VisualizerInfo {
        id: "fluid",
        name: "Fluid",
        params: ["viscosity", "dye amount", "force scale", "swirl", "colour cycle", "vorticity", "jacobi iterations", "dye fade"],
    },
];

/// The fragment module for a visualizer: one per scene.
///
/// `fluid` is seven passes with their own ping-pong targets (see the module list above), so it has
/// no single fragment module and returns `None`; the same is true of `custom:<name>`, which the
/// renderer compiles at run time with [`compile_custom`] or a [`layer_prelude`]-based compile.
pub fn visualizer_shader(id: &str) -> Option<&'static str> {
    /// `VISUALIZERS` order is `WG_SCENES` order.
    const SCENE_MODULES: [&str; 16] = [
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
    ];
    let index = VISUALIZERS.iter().position(|v| v.id == id)?;
    SCENE_MODULES.get(index).copied()
}

/// Look a visualizer up by id.
pub fn visualizer(id: &str) -> Option<&'static VisualizerInfo> {
    VISUALIZERS.iter().find(|v| v.id == id)
}
