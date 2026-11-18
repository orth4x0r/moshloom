//! Sigil + Smear scenes. Owner: S-cyber. Add scenes to `SCENES`; bodies live in `styles/cyber/*.wgsl`
//! and are included with `include_str!`.
//!
//! # The two packs
//!
//! **Sigil** is thorny, mirrored linework cast in metal: sigils built from distance fields and lit as
//! polished chrome rods (`sigil_thorn`, `sigil_halo`, `sigil_glyph`, `sigil_fang`). Cold silver, a
//! blood-red rim and an iridescent turn, on a grainy plum plate with an aura pool — tattoo-flash energy
//! with weight, not neon wire.
//!
//! **Smear** is the abstract side of signal failure: pixel sorting as painting (`glitch_sort`: long
//! luminous sorted streaks over a melt) and datamosh as collage (`glitch_tear`: soft band warps and
//! stretched trails over woven ribbons). No scanlines, no cathode-ray curve, no glyph raster, no block
//! corruption: those read as a broken screen, and a broken screen is not the brief.
//!
//! # What every scene in this pack does
//!
//! * reads `P(0)` as the clock (the app maps it into `U.T`), `U.dens`/`U.vari` from `p1`/`p3`, and the
//!   mel bands (`mel`) and `U.bass`/`U.mid`/`U.high` for its shape;
//! * gets harder at high energy: energy widens strokes, lengthens runs, deepens warps, lifts contrast;
//! * answers `U.pulse` on the beat with a rim flare, a wider stroke, a longer drag or a shock ring;
//! * never flashes the whole frame on its own: global brightness moves at the beat's rate or slower,
//!   so nothing here strobes inside the 400 ms safety limit (the Strobe layer is the one place a fast
//!   flash is allowed);
//! * keeps its highlights under a shoulder (see `shoulder` in `styles/paint.wgsl`), so a specular is
//!   hot without a clipped white patch: `tests/styles_cyber.rs` measures the clipped share.
//!
//! # `p0..p7` and the effects lane
//!
//! `scenes.rs` documents `p4..p7` as "the layer's post amounts and left in `p` for the effects lane".
//! This pack reads all eight because the style pack's contract is the whole clip's parameter block,
//! and each scene names what it does with `p4..p7` in `params`. If the app starts writing the effect
//! amounts into `p4..p7`, these four controls follow them; the scenes are built so every value looks
//! intentional (reported to the coordinator in the PR).

use super::StyleScene;

/// A scene body: the packs' shared paint, then this scene's own WGSL.
macro_rules! style_body {
    ($file:literal) => {
        concat!(include_str!("paint.wgsl"), "\n", include_str!(concat!("cyber/", $file)))
    };
}

/// The pack a scene belongs to, as `catalog()` shows it.
const SIGIL: &str = "Sigil";
const SMEAR: &str = "Smear";

pub const SCENES: &[StyleScene] = &[
    StyleScene {
        id: "sigil_thorn",
        name: "Thorn Crest",
        pack: SIGIL,
        params: [
            "Spin",
            "Arms",
            "Hue",
            "Sway",
            "Thorn length",
            "Bloom",
            "Gate",
            "Grain",
        ],
        body: style_body!("sigil_thorn.wgsl"),
    },
    StyleScene {
        id: "sigil_halo",
        name: "Barbed Halo",
        pack: SIGIL,
        params: [
            "Spin",
            "Rings",
            "Hue",
            "Twist",
            "Thorn length",
            "Core glow",
            "Spacing",
            "Grain",
        ],
        body: style_body!("sigil_halo.wgsl"),
    },
    StyleScene {
        id: "sigil_glyph",
        name: "Chrome Rune",
        pack: SIGIL,
        params: [
            "Spin",
            "Density",
            "Hue",
            "Curl",
            "Stroke weight",
            "Flash",
            "Ring",
            "Grain",
        ],
        body: style_body!("sigil_glyph.wgsl"),
    },
    StyleScene {
        id: "sigil_fang",
        name: "Chrome Fangs",
        pack: SIGIL,
        params: [
            "Lunge",
            "Columns",
            "Hue",
            "Shear",
            "Fang length",
            "Backlight",
            "Depth",
            "Grain",
        ],
        body: style_body!("sigil_fang.wgsl"),
    },
    StyleScene {
        id: "glitch_sort",
        name: "Sorted Streaks",
        pack: SMEAR,
        params: [
            "Speed",
            "Run length",
            "Hue",
            "Threshold",
            "Drag",
            "Dispersion",
            "Melt",
            "Grain",
        ],
        body: style_body!("glitch_sort.wgsl"),
    },
    StyleScene {
        id: "glitch_tear",
        name: "Melt Drag",
        pack: SMEAR,
        params: [
            "Speed",
            "Band height",
            "Hue",
            "Warp",
            "Drag",
            "Bloom",
            "Melt",
            "Grain",
        ],
        body: style_body!("glitch_tear.wgsl"),
    },
];
