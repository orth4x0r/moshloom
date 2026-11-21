//! Hyper + Wave scenes: six novel procedural visual algorithms (non-Euclidean geometry, 4D
//! topology, Kleinian limit sets, spectral Helmholtz cymatics, aperiodic quasicrystal phason
//! interference, and divergence-free curl tensor streamlines).
//!
//! # The two packs
//!
//! * **Hyper** (Non-Euclidean & Topological Geometry):
//!   - `hyper_poincare` (**Poincaré Tessellation**): exact Möbius isometries and orthogonal
//!     circle-reflection folding in the Poincaré hyperbolic disk $\mathbb{D}^2$, with
//!     conformal-Jacobian screen-constant geodesic anti-aliasing.
//!   - `hyper_hopf` (**Hopf Fibration**): 4D inverse stereographic projection $S^3 \to \mathbb{R}^3$
//!     with isoclinic Clifford rotation and linked Villarceau fiber ribbons.
//!   - `hyper_klein` (**Kleinian Apollyon**): Apollonian / Schottky Kleinian limit-set folding
//!     with Maskit parabolic shear, conformal Lyapunov shading, and geodesic orbit traps.
//! * **Wave** (Spectral Physics & Field Harmonics):
//!   - `wave_chladni` (**Chladni Cymatics**): 4-mode Helmholtz standing-wave eigenfunctions driven
//!     directly by live `mel()` spectral bands, with analytic gradient-normalized nodal caustics.
//!   - `wave_quasicrystal` (**Quasicrystal Lattice**): de Bruijn 5/7/9-fold aperiodic plane-wave
//!     interference with live `mel()`-driven phason flips, exact 3D facet relief, and thin-film
//!     iridescence.
//!   - `wave_tensor` (**Tensor Streamlines**): divergence-free 2D curl dipole + shear flow field
//!     with symmetric analytical Line Integral Convolution (LIC) streamline advection.

use super::StyleScene;

/// A scene body: the packs' shared paint, then this scene's own WGSL.
macro_rules! style_body {
    ($file:literal) => {
        concat!(
            include_str!("paint.wgsl"),
            "\n",
            include_str!(concat!("novel/", $file))
        )
    };
}

const HYPER: &str = "Hyper";
const WAVE: &str = "Wave";

pub const SCENES: &[StyleScene] = &[
    StyleScene {
        id: "hyper_poincare",
        name: "Poincare Tessellation",
        pack: HYPER,
        params: [
            "Speed",
            "Symmetry",
            "Hue",
            "Curvature",
            "Drift",
            "Edge glow",
            "Contrast",
            "Grain",
        ],
        body: style_body!("hyper_poincare.wgsl"),
    },
    StyleScene {
        id: "hyper_hopf",
        name: "Hopf Fibration",
        pack: HYPER,
        params: [
            "Speed",
            "Fibers",
            "Hue",
            "4D tilt",
            "Twist",
            "Depth",
            "Sheen",
            "Grain",
        ],
        body: style_body!("hyper_hopf.wgsl"),
    },
    StyleScene {
        id: "hyper_klein",
        name: "Kleinian Apollyon",
        pack: HYPER,
        params: [
            "Speed",
            "Scale",
            "Hue",
            "Cusp",
            "Shear",
            "Filigree",
            "Glow",
            "Grain",
        ],
        body: style_body!("hyper_klein.wgsl"),
    },
    StyleScene {
        id: "wave_chladni",
        name: "Chladni Cymatics",
        pack: WAVE,
        params: [
            "Speed",
            "Mode order",
            "Hue",
            "Symmetry",
            "Crispness",
            "Phase glow",
            "Plate warp",
            "Grain",
        ],
        body: style_body!("wave_chladni.wgsl"),
    },
    StyleScene {
        id: "wave_quasicrystal",
        name: "Quasicrystal Lattice",
        pack: WAVE,
        params: [
            "Speed",
            "Scale",
            "Hue",
            "Fold order",
            "Phason",
            "Relief",
            "Iridescence",
            "Grain",
        ],
        body: style_body!("wave_quasicrystal.wgsl"),
    },
    StyleScene {
        id: "wave_tensor",
        name: "Tensor Streamlines",
        pack: WAVE,
        params: [
            "Speed",
            "Density",
            "Hue",
            "Turbulence",
            "Vortex pull",
            "Streak",
            "Sheen",
            "Grain",
        ],
        body: style_body!("wave_tensor.wgsl"),
    },
];
