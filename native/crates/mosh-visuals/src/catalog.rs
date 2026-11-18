//! Everything a clip can draw, for the UI's browser and the clip editor: the web scenes, the layer
//! visualizers and the style packs, with the names of their eight parameters.

use crate::styles;
use mosh_shaders::VISUALIZERS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisualizerEntry {
    pub id: &'static str,
    pub name: &'static str,
    /// Browser group: "Classic" (the web app's scenes), "Layers" (fluid, particles, camera, text,
    /// overlay video), a style pack's name, or [`LEGACY`].
    pub group: &'static str,
    pub params: [&'static str; 8],
}

const NONE: [&str; 8] = ["", "", "", "", "", "", "", ""];

/// The layer visualizers that are not scenes. `fluid` takes its names from `VISUALIZERS`.
const LAYER_VISUALIZERS: [(&str, &str, [&str; 8]); 4] = [
    ("particles", "Particles", NONE),
    ("camera", "Camera", NONE),
    ("text", "Text", NONE),
    ("overlay", "Video", NONE),
];

/// The browser group for web scenes whose look is retro or literal (a synthwave grid, terminal
/// glyphs, a scanline sweep, punch cards). They still draw, so a project that uses them keeps
/// working, but they sit at the end of the browser, folded, out of the way of the abstract looks.
pub const LEGACY: &str = "Legacy";

/// The web scenes in [`LEGACY`].
const LEGACY_SCENES: [&str; 4] = ["grid", "glyph_rain", "scan", "punch_cards"];

/// Every visualizer id a clip may use, in browser order.
pub fn catalog() -> Vec<VisualizerEntry> {
    let mut out = Vec::new();
    let mut legacy = Vec::new();
    for info in VISUALIZERS {
        let group = if info.id == "fluid" {
            "Layers"
        } else if LEGACY_SCENES.contains(&info.id) {
            LEGACY
        } else {
            "Classic"
        };
        let entry = VisualizerEntry {
            id: info.id,
            name: info.name,
            group,
            params: info.params,
        };
        if group == LEGACY {
            legacy.push(entry);
        } else {
            out.push(entry);
        }
    }
    for (id, name, params) in LAYER_VISUALIZERS {
        let params = crate::layer_params(id).unwrap_or(params);
        out.push(VisualizerEntry {
            id,
            name,
            group: "Layers",
            params,
        });
    }
    for scene in styles::all() {
        out.push(VisualizerEntry {
            id: scene.id,
            name: scene.name,
            group: scene.pack,
            params: scene.params,
        });
    }
    out.extend(legacy);
    out
}
