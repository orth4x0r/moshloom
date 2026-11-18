//! Chrome + Glass + Aura + Fold scenes. Owner: S-aero. Add scenes to `SCENES`; bodies live in
//! `styles/aero/*.wgsl` and are included with `include_str!`.
//!
//! Four packs, ten scenes, all of them abstract: no horizons, no grids, no globes, no hills, no sky.
//!
//! * **Chrome** is liquid metal: a droplet field lit by the reflection of a studio (`y2k-chrome`) and a
//!   field of iridescent films with bokeh behind them (`y2k-bubble-bloom`).
//! * **Glass** is light through a body: water caustics (`aero-caustic-pool`), crossing glass blades
//!   (`aero-glass-sheen`), a glossy orb (`aero-aqua-orb`) and a molten lens with real chromatic
//!   dispersion (`glass-refract`).
//! * **Aura** is atmosphere: a grainy gradient mesh (`aura-mesh`) and the topographic iso-lines of a
//!   slow noise (`contour-field`).
//! * **Fold** is material: crossing creases in holographic foil (`foil-fold`) and ribbons of satin light
//!   (`silk-ribbon`).
//!
//! Every body is compiled on the style prelude (`styles::prelude()`) plus `styles/paint.wgsl`, which is
//! prepended here and by `styles/cyber.rs`: the cosine palette, the rational glows, the film grain, the
//! iridescent ramp, the chrome studio, the dark plate and the highlight shoulder every scene shares.
//!
//! Every body follows the same four rules:
//!
//! * `p0` is the clock (the scene machinery multiplies `t` by it, see `scenes::scene_uniforms`), so
//!   `p0 = 0` freezes the motion and `p0 = 1` runs it;
//! * `p1..p7` are this scene's own controls, documented in `params` below and read through `P(i)`; every
//!   one of them moves pixels (`tests/styles_aero.rs` prints the counts);
//! * the audio reaches it as `U.pulse` (1 on the beat, decaying), `U.bass`/`U.mid`/`U.high` and
//!   `mel(i)`; high energy moves more, glows harder and swells the shapes, and `pulse` only ever lights
//!   part of the frame, never the whole frame, because flashes belong to the Strobe transition;
//! * cost: analytic 2D shapes, at most five loops, and the transcendental budget `paint.wgsl` documents,
//!   because the pack's budget is under 0.5 ms per scene at 960x540 (`tests/styles_aero.rs` prints the
//!   number and asserts it in release);
//! * no clipped white: every bright body ends in `shoulder()`, so a specular rolls off instead of
//!   flattening a patch of the frame to 255 (`tests/styles_aero.rs` measures the clipped share).

use super::StyleScene;

/// A scene body: the packs' shared paint, then this scene's own WGSL.
macro_rules! style_body {
    ($file:literal) => {
        concat!(include_str!("paint.wgsl"), "\n", include_str!(concat!("aero/", $file)))
    };
}

pub const SCENES: &[StyleScene] = &[
    StyleScene {
        id: "y2k-chrome",
        name: "Liquid Mercury",
        pack: "Chrome",
        params: [
            "Speed",
            "Droplet size",
            "Hue",
            "Chrome",
            "Iridescence",
            "Flow",
            "Sparkle",
            "Gloss",
        ],
        body: style_body!("y2k_chrome.wgsl"),
    },
    StyleScene {
        id: "y2k-bubble-bloom",
        name: "Bubble Bloom",
        pack: "Chrome",
        params: [
            "Speed",
            "Bubble size",
            "Hue",
            "Spread",
            "Gloss",
            "Iridescence",
            "Squash",
            "Sparkle",
        ],
        body: style_body!("y2k_bubble_bloom.wgsl"),
    },
    StyleScene {
        id: "aero-aqua-orb",
        name: "Aqua Orb",
        pack: "Glass",
        params: [
            "Speed",
            "Orb size",
            "Hue",
            "Haze",
            "Gloss",
            "Rim light",
            "Beads",
            "Reflection",
        ],
        body: style_body!("aero_aqua_orb.wgsl"),
    },
    StyleScene {
        id: "aero-caustic-pool",
        name: "Caustic Pool",
        pack: "Glass",
        params: [
            "Speed",
            "Caustic scale",
            "Hue",
            "Droplets",
            "Glass",
            "Depth",
            "Glow",
            "Sparkle",
        ],
        body: style_body!("aero_caustic_pool.wgsl"),
    },
    StyleScene {
        id: "aero-glass-sheen",
        name: "Shatter Blades",
        pack: "Glass",
        params: [
            "Speed",
            "Blade drift",
            "Hue",
            "Frost",
            "Sheen",
            "Bokeh",
            "Droplets",
            "Rim",
        ],
        body: style_body!("aero_glass_sheen.wgsl"),
    },
    StyleScene {
        id: "glass-refract",
        name: "Liquid Lens",
        pack: "Glass",
        params: [
            "Speed",
            "Lens size",
            "Hue",
            "Dispersion",
            "Refraction",
            "Frost",
            "Glow",
            "Grain",
        ],
        body: style_body!("glass_refract.wgsl"),
    },
    StyleScene {
        id: "aura-mesh",
        name: "Gradient Mesh",
        pack: "Aura",
        params: [
            "Speed",
            "Blob scale",
            "Hue",
            "Spread",
            "Bloom",
            "Drift",
            "Grain",
            "Contrast",
        ],
        body: style_body!("aura_mesh.wgsl"),
    },
    StyleScene {
        id: "contour-field",
        name: "Contour Field",
        pack: "Aura",
        params: [
            "Speed",
            "Contour density",
            "Hue",
            "Warp",
            "Line width",
            "Depth",
            "Bloom",
            "Grain",
        ],
        body: style_body!("contour_field.wgsl"),
    },
    StyleScene {
        id: "foil-fold",
        name: "Holo Foil",
        pack: "Fold",
        params: [
            "Speed",
            "Fold scale",
            "Hue",
            "Fold depth",
            "Holo shift",
            "Sheen",
            "Crease",
            "Grain",
        ],
        body: style_body!("foil_fold.wgsl"),
    },
    StyleScene {
        id: "silk-ribbon",
        name: "Silk Ribbons",
        pack: "Fold",
        params: [
            "Speed",
            "Ribbons",
            "Hue",
            "Width",
            "Flow",
            "Sheen",
            "Weave",
            "Grain",
        ],
        body: style_body!("silk_ribbon.wgsl"),
    },
];
