//! Style packs: native scenes beyond the web app's 16, registered here and drawn by `Scenes` like
//! any other scene. Each pack is one module with a `SCENES` list; a lane owns one pack.
//!
//! | pack | module | lane |
//! | --- | --- | --- |
//! | Sigil + Smear | `cyber` | S-cyber |
//! | Chrome + Glass + Aura + Fold | `aero` | S-aero |
//! | Hyper + Wave | `novel` | S-novel |
//!
//! Twenty-two abstract and mathematical scenes in eight packs. The palette is contemporary abstract
//! digital art — chrome sigil linework, liquid metal, iridescent foil, gradient meshes, glass
//! dispersion, silk, topographic contours, sorted streaks, plus non-Euclidean Poincaré / Kleinian
//! geometry, 4D Hopf fibrations, spectral Chladni cymatics, aperiodic quasicrystals, and analytical
//! curl tensor streamlines — and there is deliberately nothing retro-literal in here: no synthwave
//! grid or sun, no CRT, no scanlines, no VHS, no glyph rain, no pixel font, no hills, globes or skies.
//! The look is carried by `paint.wgsl` (shared chrome, iridescence, grain and the highlight shoulder)
//! and by the per-scene bodies; `tests/styles_cyber.rs`, `tests/styles_aero.rs`, and
//! `tests/styles_novel.rs` hold all packs to lit, beat-reactive, unclipped pixels and to the 0.5 ms
//! budget.
//!
//! A style scene's `body` is WGSL written on top of [`prelude`]: the native scene prelude
//! (`scenes/prelude.wgsl`: `U`, `lut`, `mel`, `melh`, `XY`, `h1`, `h2`, the five inputs palT,
//! caT, melT, rdT, mediaT) with the clip's eight parameters added to `U` as `U.p[0..2]` and read
//! through `P(i)` (0..1 each). The body declares `@fragment fn fs_main(@builtin(position) pos:
//! vec4f) -> @location(0) vec4f`. Scene ids must be unique across every pack and must not clash
//! with `mosh_shaders::VISUALIZERS` or the layer visualizers (`fluid`, `overlay`, `camera`,
//! `text`, `particles`); `tests/styles.rs` checks this and compiles every body.

pub mod aero;
pub mod cyber;
pub mod novel;

/// One native scene.
#[derive(Clone, Copy, Debug)]
pub struct StyleScene {
    /// Stable id used in `Clip::visualizer`.
    pub id: &'static str,
    /// Display name.
    pub name: &'static str,
    /// Pack name shown in the browser ("Sigil", "Smear", "Chrome", "Glass", "Aura", "Fold", "Hyper", "Wave").
    pub pack: &'static str,
    /// What `p0..p7` mean for this scene (shown in the clip editor).
    pub params: [&'static str; 8],
    /// WGSL fragment body on top of [`prelude`].
    pub body: &'static str,
}

/// Every style scene, pack by pack.
pub fn all() -> impl Iterator<Item = &'static StyleScene> {
    cyber::SCENES
        .iter()
        .chain(aero::SCENES.iter())
        .chain(novel::SCENES.iter())
}

/// The prelude a style body is compiled on: the scene prelude with `p0..p7` readable.
pub fn prelude() -> String {
    let base = include_str!("../scenes/prelude.wgsl");
    let field = "mel: array<vec4f,6> };";
    assert!(
        base.contains(field),
        "scenes/prelude.wgsl changed its SU struct; update styles::prelude"
    );
    let with_p = base.replacen(field, "mel: array<vec4f,6>, p: array<vec4f,2> };", 1);
    format!(
        "{with_p}\n// The clip's eight parameters, 0..1 (mosh_shaders::LayerUniforms after the web prefix).\nfn P(i: i32) -> f32 {{ return U.p[i / 4][i % 4]; }}\n"
    )
}
