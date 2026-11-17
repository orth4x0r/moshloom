//! Y2K + Frutiger Aero scenes. Owner: S-aero. Add scenes to `SCENES`; bodies may live in
//! `styles/aero/*.wgsl` and be included with `include_str!`.
//!
//! Two packs, eight scenes. Y2K is the early-2000s screensaver: liquid chrome, holographic wire
//! globes, glossy bubbles, a wireframe horizon. Frutiger Aero is the glossy nature-tech of the same
//! years: aqua orbs with white highlights, sky and clouds over green hills, water caustics, frosted
//! glass. Both packs share the style prelude (`styles::prelude()`), so `U`, `lut`, `P(i)`, `XY`,
//! `h1`/`h2` and `fin` are what a body writes against, plus `aero/common.wgsl`, which is prepended
//! to every body and holds the palette, the cheap falloffs and the glitter.
//!
//! Every body follows the same four rules:
//!
//! * `p0` is the clock (the scene machinery multiplies `t` by it, see `scenes::scene_uniforms`), so
//!   `p0 = 0` freezes the motion and `p0 = 1` runs it;
//! * `p1..p7` are this scene's own controls, documented in `params` below and read through `P(i)`;
//! * the audio reaches it as `U.pulse` (1 on the beat, decaying), `U.bass`/`U.mid`/`U.high` and
//!   `mel(i)`; high energy moves more, glows harder and swells the shapes, and `pulse` only ever
//!   lights part of the frame, never the whole frame, because flashes belong to the Strobe
//!   transition;
//! * cost: analytic 2D shapes, at most five loops, and the transcendental budget `common.wgsl`
//!   documents, because the pack's budget is under 0.5 ms per scene at 960x540
//!   (`tests/styles_aero.rs` prints the number and asserts it in release).

use super::StyleScene;

/// A scene body: the pack's shared helpers, then this scene's own WGSL.
macro_rules! aero_body {
    ($file:literal) => {
        concat!(include_str!("aero/common.wgsl"), "\n", include_str!(concat!("aero/", $file)))
    };
}

pub const SCENES: &[StyleScene] = &[
    StyleScene {
        id: "y2k-chrome",
        name: "Liquid Chrome",
        pack: "Y2K",
        params: [
            "Speed",
            "Blob size",
            "Hue",
            "Chrome",
            "Iridescence",
            "Flow",
            "Sparkle",
            "Gloss",
        ],
        body: aero_body!("y2k_chrome.wgsl"),
    },
    StyleScene {
        id: "y2k-holo-globe",
        name: "Holo Globe",
        pack: "Y2K",
        params: [
            "Speed",
            "Globe size",
            "Hue",
            "Grid density",
            "Rim glow",
            "Sparkle",
            "Orbit tilt",
            "Lens flare",
        ],
        body: aero_body!("y2k_holo_globe.wgsl"),
    },
    StyleScene {
        id: "y2k-bubble-bloom",
        name: "Bubble Bloom",
        pack: "Y2K",
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
        body: aero_body!("y2k_bubble_bloom.wgsl"),
    },
    StyleScene {
        id: "y2k-cyber-horizon",
        name: "Cyber Horizon",
        pack: "Y2K",
        params: [
            "Speed",
            "Grid density",
            "Hue",
            "Sun size",
            "Scanlines",
            "Horizon glow",
            "Globe",
            "Sparkle",
        ],
        body: aero_body!("y2k_cyber_horizon.wgsl"),
    },
    StyleScene {
        id: "aero-aqua-orb",
        name: "Aqua Orb",
        pack: "Frutiger Aero",
        params: [
            "Speed",
            "Orb size",
            "Hue",
            "Clouds",
            "Gloss",
            "Rim light",
            "Bubbles",
            "Floor sheen",
        ],
        body: aero_body!("aero_aqua_orb.wgsl"),
    },
    StyleScene {
        id: "aero-aqua-hills",
        name: "Aqua Hills",
        pack: "Frutiger Aero",
        params: [
            "Speed",
            "Hill height",
            "Hue",
            "Clouds",
            "Grass gloss",
            "Sun glow",
            "Waterline",
            "Sparkle",
        ],
        body: aero_body!("aero_aqua_hills.wgsl"),
    },
    StyleScene {
        id: "aero-caustic-pool",
        name: "Caustic Pool",
        pack: "Frutiger Aero",
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
        body: aero_body!("aero_caustic_pool.wgsl"),
    },
    StyleScene {
        id: "aero-glass-sheen",
        name: "Glass Sheen",
        pack: "Frutiger Aero",
        params: [
            "Speed",
            "Panel drift",
            "Hue",
            "Frost",
            "Sheen",
            "Bokeh",
            "Droplets",
            "Rim",
        ],
        body: aero_body!("aero_glass_sheen.wgsl"),
    },
];
