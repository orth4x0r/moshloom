//! Everything a clip can draw, for the UI's browser and the clip editor: the web scenes, the layer
//! visualizers and the style packs, with the names of their eight parameters.

use crate::styles;
use mosh_shaders::VISUALIZERS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisualizerEntry {
    pub id: &'static str,
    pub name: &'static str,
    /// Browser group: "Classic" (the web app's scenes), "Layers" (fluid, particles, camera, text,
    /// overlay video), or a style pack's name.
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

/// Every visualizer id a clip may use, in browser order.
pub fn catalog() -> Vec<VisualizerEntry> {
    let mut out = Vec::new();
    for info in VISUALIZERS {
        let group = if info.id == "fluid" {
            "Layers"
        } else {
            "Classic"
        };
        out.push(VisualizerEntry {
            id: info.id,
            name: info.name,
            group,
            params: info.params,
        });
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
    out
}
