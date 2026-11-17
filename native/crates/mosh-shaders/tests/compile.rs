//! Task 1.12 gate: every extracted module compiles, the web custom-scene template still compiles
//! on top of `web_prelude()`, a broken shader reports its line, the uniform block matches the web
//! layout, and the visualizer list is complete.

use std::mem::{offset_of, size_of};
use std::path::{Path, PathBuf};

use mosh_core::{AnalysisFrame, ClipParams};
use mosh_shaders::{
    Binding, BindingKind, LayerUniforms, VISUALIZERS, WEB_UNIFORM_BYTES, compile_custom, compile_wgsl,
    layer_prelude, layer_uniforms, spirv, uniform_bytes, visualizer_shader, web_prelude,
};

/// `WG_TEMPLATE` from gpu.js (line 231), copied byte for byte. Pinned here on purpose: if the web
/// template changes, this test fails until the prelude still accepts it unchanged.
const WEB_TEMPLATE: &str = r#"// Custom visual. Uniforms: U.T (audio-driven time), U.hue, U.bass, U.mid, U.high, U.pulse (1 on the beat, decaying),
// U.beat, U.dens (Density slider 0..1), U.vari (Variant), U.res.  Helpers: XY(pos) centred coords, lut(v) palette colour,
// mel(i) spectrum band 0..23, h1(x)/h2(vec2) hashes, fin(rgb).
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let r = length(q);
  let v = sin(r * (10. + 30. * U.dens) - U.T * 3.) + sin(atan2(q.y, q.x) * 5. + U.T);
  let c = lut(v * 0.2 + U.hue) * (0.6 + 0.4 * U.pulse);
  return fin(c);
}"#;

/// The broken custom shader from `gpu_test.mjs` ("bad shader ->"): `nope` is on line 3 of the
/// body the user typed, and the web app reports it as line 3 after rebasing past the prelude.
const WEB_BROKEN_BODY: &str = "@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  return fin(vec3f(q.x, nope, 0.));
}";

fn wgsl_files() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("wgsl");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.expect("directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "wgsl"))
        .collect();
    files.sort();
    files
}

#[test]
fn every_wgsl_file_compiles() {
    let files = wgsl_files();
    assert!(!files.is_empty(), "wgsl/ is empty: run node scripts/extract_wgsl.mjs");
    let mut compiled = 0;
    let mut words = 0;
    for path in &files {
        let name = path.file_stem().and_then(|s| s.to_str()).expect("utf-8 file name");
        let src = std::fs::read_to_string(path).expect("read wgsl");
        let runtime = compile_wgsl(&src).unwrap_or_else(|e| panic!("{name}: {e}"));
        let build_time = spirv(name).unwrap_or_else(|| panic!("spirv(\"{name}\") is missing"));
        assert_eq!(
            build_time,
            runtime.as_slice(),
            "build-time SPIR-V for {name} differs from a runtime compile of the same file"
        );
        // A module with no entry point at all would still compile; every file here has one.
        assert!(build_time.len() > 20, "{name}: only {} words", build_time.len());
        compiled += 1;
        words += build_time.len();
    }
    println!("wgsl: {compiled} modules compiled, {words} SPIR-V words total");
    assert_eq!(compiled, files.len());
    assert!(compiled >= 33, "expected every extracted module, got {compiled}");
}

#[test]
fn every_wgsl_file_has_a_module_entry() {
    let files = wgsl_files();
    let names: Vec<&str> = files
        .iter()
        .map(|p| p.file_stem().and_then(|s| s.to_str()).expect("utf-8 name"))
        .collect();
    for name in &names {
        assert!(spirv(name).is_some(), "spirv(\"{name}\")");
    }
    assert_eq!(mosh_shaders::modules().len(), files.len(), "one module entry per wgsl file");
    for m in mosh_shaders::modules() {
        assert!(names.contains(&m.name), "{} has no wgsl file", m.name);
    }
}

#[test]
fn web_template_compiles_unchanged() {
    let prelude = web_prelude();
    // The prelude is the web app's own text: uniform struct, the six bindings, the helper
    // functions and the full-screen vertex stage.
    for needle in [
        "struct SU {",
        "@group(0) @binding(0) var<uniform> U: SU;",
        "@group(0) @binding(1) var lin: sampler;",
        "@group(0) @binding(2) var palT: texture_2d<f32>;",
        "@group(0) @binding(3) var caT: texture_2d<f32>;",
        "@group(0) @binding(4) var melT: texture_2d<f32>;",
        "@group(0) @binding(5) var rdT: texture_2d<f32>;",
        "@group(0) @binding(6) var mediaT: texture_2d<f32>;",
        "fn lut(",
        "fn mel(",
        "fn melh(",
        "fn XY(",
        "fn ca(",
        "fn rule(",
        "fn fin(",
        "@vertex fn vs(",
    ] {
        assert!(prelude.contains(needle), "web_prelude() is missing {needle:?}");
    }
    let src = format!("{prelude}{WEB_TEMPLATE}");
    let words = compile_wgsl(&src).expect("WG_TEMPLATE must compile on web_prelude() unchanged");
    assert!(words.len() > 100, "web template: {} words", words.len());
    // The web prelude declares the web uniform struct, not the superset: p0..p7 are added on top
    // by layer_prelude(), and a web shader must not see them.
    let uses_p0 = format!(
        "{prelude}@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {{ return vec4f(U.p0); }}"
    );
    assert!(
        compile_wgsl(&uses_p0).is_err(),
        "web_prelude() must keep the web struct: U.p0 belongs to layer_prelude()"
    );
}

#[test]
fn broken_shader_reports_line() {
    let err = compile_custom(WEB_BROKEN_BODY).expect_err("nope must not compile");
    assert_eq!(err.line, Some(3), "line {} ({})", err.line.unwrap_or(0), err.message);
    assert!(err.column.is_some(), "no column for {err:?}");
    assert!(err.message.contains("nope"), "message does not name the symbol: {}", err.message);
    let shown = err.to_string();
    assert!(shown.contains("line 3"), "display does not carry the line: {shown}");
    // The same mistake in a shader passed straight to compile_wgsl() keeps its absolute line.
    let src = format!("{}{WEB_BROKEN_BODY}", web_prelude());
    let raw = compile_wgsl(&src).expect_err("nope must not compile");
    let prelude_lines = web_prelude().matches('\n').count() as u32;
    assert_eq!(raw.line, Some(prelude_lines + 3));
}

#[test]
fn uniforms_size_multiple_of_16() {
    assert_eq!(size_of::<LayerUniforms>() % 16, 0, "uniform block must be 16-byte aligned");
    assert_eq!(size_of::<LayerUniforms>(), mosh_shaders::LAYER_UNIFORM_BYTES);
    assert_eq!(size_of::<LayerUniforms>(), 240);
    // A web shader compiled on web_prelude() reads the first WEB_UNIFORM_BYTES bytes, so every
    // field up to `mel` must sit exactly where gpu.js's layout table puts it.
    assert_eq!(WEB_UNIFORM_BYTES, 208);
    assert_eq!(offset_of!(LayerUniforms, res), 0);
    assert_eq!(offset_of!(LayerUniforms, t), 8);
    assert_eq!(offset_of!(LayerUniforms, hue), 12);
    assert_eq!(offset_of!(LayerUniforms, bass), 16);
    assert_eq!(offset_of!(LayerUniforms, mid), 20);
    assert_eq!(offset_of!(LayerUniforms, high), 24);
    assert_eq!(offset_of!(LayerUniforms, pulse), 28);
    assert_eq!(offset_of!(LayerUniforms, vari), 32);
    assert_eq!(offset_of!(LayerUniforms, dens), 36);
    assert_eq!(offset_of!(LayerUniforms, beat), 40);
    assert_eq!(offset_of!(LayerUniforms, frame), 44);
    assert_eq!(offset_of!(LayerUniforms, pal), 48);
    assert_eq!(offset_of!(LayerUniforms, aux0), 52);
    assert_eq!(offset_of!(LayerUniforms, aux1), 56);
    assert_eq!(offset_of!(LayerUniforms, aux2), 60);
    assert_eq!(offset_of!(LayerUniforms, l0), 64);
    assert_eq!(offset_of!(LayerUniforms, l1), 80);
    assert_eq!(offset_of!(LayerUniforms, l2), 96);
    assert_eq!(offset_of!(LayerUniforms, mel), 112);
    assert_eq!(offset_of!(LayerUniforms, p), WEB_UNIFORM_BYTES);
    // ... and the two WGSL structs must agree with the Rust one, on both sides of the superset.
    assert_eq!(uniform_bytes("layer_prelude"), Some(240), "layer_prelude struct size");
    assert_eq!(uniform_bytes("custom_template"), Some(208), "web struct size");
    assert_eq!(uniform_bytes("scene00_tunnel"), Some(208), "scene struct size");
}

#[test]
fn visualizer_list_has_17_entries() {
    assert_eq!(VISUALIZERS.len(), 17, "16 web scenes plus fluid");
    let mut ids: Vec<&str> = VISUALIZERS.iter().map(|v| v.id).collect();
    ids.sort_unstable();
    let mut unique = ids.clone();
    unique.dedup();
    assert_eq!(ids, unique, "duplicate visualizer ids");
    for v in VISUALIZERS {
        assert!(!v.id.is_empty() && !v.name.is_empty(), "{v:?}");
        assert!(v.params.len() == 8, "{}: {} params", v.id, v.params.len());
        for (i, p) in v.params.iter().enumerate() {
            assert!(!p.is_empty(), "{}: p{i} has no name", v.id);
            assert!(p.len() <= 32, "{}: p{i} name is too long", v.id);
            assert_eq!(p.trim(), *p, "{}: p{i} has stray whitespace", v.id);
        }
    }
    assert_eq!(VISUALIZERS[16].id, "fluid");
    // Every scene visualizer names a module that exists, so a layer can be built from the list.
    let mut scenes = 0;
    for v in VISUALIZERS {
        if let Some(module) = visualizer_shader(v.id) {
            assert!(spirv(module).is_some(), "{} -> {module} has no SPIR-V", v.id);
            scenes += 1;
        }
    }
    assert_eq!(scenes, 16, "the 16 web scenes have one fragment module each");
    assert_eq!(visualizer_shader("fluid"), None, "fluid is seven passes, not one module");
}

#[test]
fn custom_scene_can_read_p0_to_p7() {
    let body = r#"@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let s = U.p0 + U.p1 + U.p2 + U.p3 + U.p4 + U.p5 + U.p6 + U.p7;
  return fin(vec3f(q.x * s, q.y, U.dens));
}"#;
    let src = format!("{}{body}", layer_prelude());
    let words = compile_wgsl(&src).expect("a native custom scene reads p0..p7");
    assert!(words.len() > 100);
    // p0..p7 are the only difference from the web struct, and they sit after it.
    assert_eq!(uniform_bytes("layer_prelude"), Some(240));
    for needle in ["struct LU {", "var<uniform> U: LU;", "p0: f32", "p7: f32"] {
        assert!(layer_prelude().contains(needle), "layer_prelude() is missing {needle:?}");
    }
}

#[test]
fn layer_uniforms_fill_the_web_prefix() {
    let frame = AnalysisFrame {
        time_s: 12.5,
        ppq: 32.25,
        beat_phase: 0.25,
        bar_phase: 0.56,
        beat_index: 32,
        bar_index: 8,
        tempo_bpm: 128.0,
        playing: true,
        bands: [0.1, 0.2, 0.3],
        mel: core::array::from_fn(|i| i as f32 / 24.0),
        onset: 0.5,
        loudness_db: -14.0,
        loudness: 0.7,
        brightness: 0.4,
        intensity: 0.6,
        level: 3,
        ..Default::default()
    };
    let clip = ClipParams { p: [1.0, 0.25, 0.5, 2.0, 0.75, 0.0, 0.0, 0.0], palette: 3, seed: 7 };
    let u = layer_uniforms(&frame, &clip, 12.5, [1920.0, 1080.0]);
    assert_eq!(u.res, [1920.0, 1080.0]);
    assert_eq!(u.t, 12.5);
    assert_eq!(u.bass, 0.1);
    assert_eq!(u.mid, 0.2);
    assert_eq!(u.high, 0.3);
    assert_eq!(u.pal, 3.0);
    assert_eq!(u.dens, clip.p[1]);
    assert_eq!(u.hue, clip.p[2]);
    assert_eq!(u.vari, clip.p[3]);
    assert_eq!(u.p, clip.p);
    assert_eq!(u.beat, 32.25);
    assert_eq!(u.frame, 12.5 * 60.0);
    assert_eq!((u.aux0, u.aux1, u.aux2), (-1.0, 0.0, 0.0));
    // pulse is 1 on the beat and decays to 0 by the next one.
    assert!((u.pulse - 0.5625).abs() < 1e-6, "pulse {}", u.pulse);
    // 24 mel bands fill the six vec4s the way gpuSU writes them.
    assert_eq!(u.mel[0][0], 0.0);
    assert_eq!(u.mel[1][1], 5.0 / 24.0);
    assert_eq!(u.mel[5][3], 23.0 / 24.0);
    assert_eq!(u.mel[0][3], 3.0 / 24.0);
    // l0..l2 are opaque colours for the layer's accents.
    for c in [u.l0, u.l1, u.l2] {
        assert_eq!(c[3], 1.0);
        assert!(c.iter().all(|v| (0.0..=1.0).contains(v)), "{c:?}");
    }
    // A stopped transport has no beat pulse.
    let stopped = AnalysisFrame { playing: false, ..frame };
    assert_eq!(layer_uniforms(&stopped, &clip, 1.0, [1.0, 1.0]).pulse, 0.0);
    // The struct is a plain byte block for the renderer.
    assert_eq!(bytemuck::bytes_of(&u), u.as_bytes());
    assert_eq!(u.as_bytes().len(), 240);
}

#[test]
fn scene_bindings_match_the_web_bind_group() {
    let scene = mosh_shaders::bindings("scene00_tunnel").expect("scene bindings");
    let expected = [
        Binding { group: 0, binding: 0, kind: BindingKind::Uniform },
        Binding { group: 0, binding: 1, kind: BindingKind::Sampler },
        Binding { group: 0, binding: 2, kind: BindingKind::Texture },
        Binding { group: 0, binding: 3, kind: BindingKind::Texture },
        Binding { group: 0, binding: 4, kind: BindingKind::Texture },
        Binding { group: 0, binding: 5, kind: BindingKind::Texture },
        Binding { group: 0, binding: 6, kind: BindingKind::Texture },
    ];
    assert_eq!(scene, &expected);
    // The feedback pass binds seven slots: uniform, sampler, then five textures.
    let feedback = mosh_shaders::bindings("fx_feedback").expect("feedback bindings");
    assert_eq!(feedback.len(), 7, "{feedback:?}");
    assert_eq!(feedback[0].kind, BindingKind::Uniform);
    assert!(feedback[1..].iter().all(|b| b.kind != BindingKind::Uniform));
    // The fluid passes are four textures and one sampler behind a uniform.
    let fluid = mosh_shaders::bindings("fluid_jacobi").expect("fluid bindings");
    assert_eq!(fluid.len(), 6, "{fluid:?}");
    assert_eq!(fluid[3].kind, BindingKind::Texture);
    // The copy pass has no uniform at all.
    assert_eq!(mosh_shaders::bindings("fx_copy").expect("copy bindings").len(), 1);
}
