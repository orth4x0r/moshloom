//! The style packs and the visualizer catalog: ids are unique, every body compiles on the style
//! prelude, and the catalog lists everything a clip may draw.

use std::collections::HashSet;

use mosh_shaders::{VISUALIZERS, compile_wgsl};
use mosh_visuals::catalog::catalog;
use mosh_visuals::styles::{self, prelude};

const LAYER_IDS: [&str; 5] = ["fluid", "overlay", "camera", "text", "particles"];

#[test]
fn the_style_prelude_exposes_the_clip_parameters() {
    let body = "@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {\n    return vec4f(lut(P(0) + U.T), P(7));\n}\n";
    compile_wgsl(&format!("{}\n{body}", prelude())).expect("a body using P(i) and lut compiles");
}

#[test]
fn style_ids_are_unique_and_do_not_clash() {
    let mut seen: HashSet<&str> = VISUALIZERS.iter().map(|v| v.id).collect();
    seen.extend(LAYER_IDS);
    for scene in styles::all() {
        assert!(
            seen.insert(scene.id),
            "style scene id {:?} is already taken",
            scene.id
        );
        assert!(
            !scene.name.is_empty() && !scene.pack.is_empty(),
            "{} needs a name and a pack",
            scene.id
        );
    }
}

#[test]
fn every_style_body_compiles() {
    let prelude = prelude();
    for scene in styles::all() {
        compile_wgsl(&format!("{prelude}\n{}", scene.body))
            .unwrap_or_else(|error| panic!("style scene {} does not compile: {error}", scene.id));
    }
}

#[test]
fn the_catalog_lists_every_visualizer_once() {
    let entries = catalog();
    let ids: Vec<&str> = entries.iter().map(|e| e.id).collect();
    let unique: HashSet<&str> = ids.iter().copied().collect();
    assert_eq!(
        ids.len(),
        unique.len(),
        "duplicate ids in the catalog: {ids:?}"
    );
    for info in VISUALIZERS {
        assert!(
            unique.contains(info.id),
            "{} missing from the catalog",
            info.id
        );
    }
    for id in LAYER_IDS {
        assert!(unique.contains(id), "{id} missing from the catalog");
    }
    assert_eq!(entries.len(), VISUALIZERS.len() + 4 + styles::all().count());
}

/// A new clip's palette is 0 ("the scene's default"); style scenes have no engine.c default and
/// used to index past the 16-entry table (a panic in `Scenes::draw`).
#[test]
fn style_scenes_draw_with_the_default_palette() {
    use mosh_core::{AnalysisFrame, ClipParams};
    use mosh_render::{FrameRecorder, Gpu, GpuPreference, LayerTarget};
    use mosh_visuals::scenes::Scenes;
    let Some(scene) = styles::all().next() else { return };
    let gpu = Gpu::new(GpuPreference::Integrated).expect("a Vulkan device");
    let mut scenes = Scenes::new(&gpu).expect("scenes");
    let dst = LayerTarget::new(&gpu, 64, 36);
    let params = ClipParams { palette: 0, ..ClipParams::default() };
    let mut rec = FrameRecorder::new(&gpu);
    scenes
        .draw(&mut rec, &dst, scene.id, &AnalysisFrame::default(), &params, 0.5)
        .expect("draw");
    gpu.wait_frame(rec.submit());
}
