//! Cybersigilism + glitch scenes. Owner: S-cyber. Add scenes to `SCENES`; bodies live in
//! `styles/cyber/*.wgsl` and are included with `include_str!`.
//!
//! # The two packs
//!
//! **Cybersigil** is sharp, thorny, mirrored linework: sigils built from distance fields, lit as
//! chrome (`sigil_thorn`, `sigil_halo`, `sigil_rune`, `sigil_fang`). Cold silver and ice blue with
//! blood-red accents on a near-black plate; tattoo-flash energy rather than neon glow.
//!
//! **Glitch** is signal failure: pixel sorting (`glitch_sort`), horizontal tearing with corrupt
//! macroblocks and RGB splits (`glitch_tear`), a curved broken CRT (`glitch_crt`) and an ASCII
//! glyph raster with databent colour quantisation (`glitch_glyph`).
//!
//! # What every scene in this pack does
//!
//! * reads `P(0)` as the clock (the app maps it into `U.T`), `U.dens`/`U.vari` from `p1`/`p3`, and
//!   the mel bands (`mel`, `melh`) and `U.bass`/`U.mid`/`U.high` for its shape;
//! * gets harder at high energy: energy (bands + pulse) widens thorns, lengthens runs, deepens
//!   tears, and lifts contrast;
//! * answers `U.pulse` on the beat with a ring, a flare, a displacement or a channel split;
//! * never flashes the whole frame on its own: global brightness moves at the beat's rate or
//!   slower, so nothing here strobes inside the 400 ms safety limit (the Strobe layer is the one
//!   place a fast flash is allowed).
//!
//! # `p0..p7` and the effects lane
//!
//! `scenes.rs` documents `p4..p7` as "the layer's post amounts and left in `p` for the effects
//! lane". This pack reads all eight because the style pack's contract is the whole clip's
//! parameter block, and each scene names what it does with `p4..p7` in `params`. If the app starts
//! writing the effect amounts into `p4..p7`, these four controls follow them; the scenes are built
//! so every value looks intentional (reported to the coordinator in the PR).

use super::StyleScene;

/// The pack a scene belongs to, as `catalog()` shows it.
const CYBERSIGIL: &str = "Cybersigil";
const GLITCH: &str = "Glitch";

pub const SCENES: &[StyleScene] = &[
    StyleScene {
        id: "sigil_thorn",
        name: "Thorn Sigil",
        pack: CYBERSIGIL,
        params: [
            "Spin", "Arms", "Hue", "Sway", "Spikes", "Bloom", "Frame", "Grain",
        ],
        body: include_str!("cyber/sigil_thorn.wgsl"),
    },
    StyleScene {
        id: "sigil_halo",
        name: "Barbed Halo",
        pack: CYBERSIGIL,
        params: [
            "Spin", "Rings", "Hue", "Twist", "Thorns", "Core", "Spacing", "Grain",
        ],
        body: include_str!("cyber/sigil_halo.wgsl"),
    },
    StyleScene {
        id: "sigil_glyph",
        name: "Sigil Glyph",
        pack: CYBERSIGIL,
        params: [
            "Rewrite", "Density", "Hue", "Curl", "Hook length", "Flash", "Ring", "Grain",
        ],
        body: include_str!("cyber/sigil_glyph.wgsl"),
    },
    StyleScene {
        id: "sigil_fang",
        name: "Chrome Fangs",
        pack: CYBERSIGIL,
        params: [
            "Lunge", "Columns", "Hue", "Shear", "Fang length", "Backlight", "Depth", "Grain",
        ],
        body: include_str!("cyber/sigil_fang.wgsl"),
    },
    StyleScene {
        id: "glitch_sort",
        name: "Pixel Sort",
        pack: GLITCH,
        params: [
            "Seed drift", "Run length", "Hue", "Threshold", "Smear", "RGB split", "Blocks", "Scanlines",
        ],
        body: include_str!("cyber/glitch_sort.wgsl"),
    },
    StyleScene {
        id: "glitch_tear",
        name: "Scan Tear",
        pack: GLITCH,
        params: [
            "Sweep", "Band height", "Hue", "Tear", "Macroblocks", "RGB split", "Contrast", "Scanlines",
        ],
        body: include_str!("cyber/glitch_tear.wgsl"),
    },
    StyleScene {
        id: "glitch_crt",
        name: "Broken CRT",
        pack: GLITCH,
        params: [
            "Roll", "Curvature", "Hue", "Tracking", "Grille", "Aberration", "Brightness", "Noise",
        ],
        body: include_str!("cyber/glitch_crt.wgsl"),
    },
    StyleScene {
        id: "glitch_glyph",
        name: "Glyph Raster",
        pack: GLITCH,
        params: [
            "Churn", "Cell size", "Hue", "Churn rate", "Row shift", "Quantise", "Cursor", "Scanlines",
        ],
        body: include_str!("cyber/glitch_glyph.wgsl"),
    },
];
