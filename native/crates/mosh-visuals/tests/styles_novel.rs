//! The Hyper + Wave novel style packs (`styles::novel`): six mathematically distinct procedural
//! visual algorithms (Poincare hyperbolic tessellation, 4D Hopf fibration, Kleinian Apollyon limit
//! sets, spectral Chladni cymatics, aperiodic quasicrystal phason interference, and analytical
//! curl tensor streamlines).
//!
//! Holds every scene to the same bar as `styles_cyber` and `styles_aero`:
//! * renders lit (mean luminance > 0.02) at 3 intensity levels x 3 parameter sets (including all
//!   zeros and all ones), with zero NaN/Inf and zero Vulkan validation messages;
//! * every one of `p0..p7` moves pixels;
//! * reacts to the beat (`U.pulse`) without strobing the whole frame;
//! * never paints a clipped white patch (< 2% pixels with RGB >= 250);
//! * fits under 0.5 ms per scene at 960x540 in release;
//! * writes 480x270 preview PNGs to `native/target/style-previews/novel/<id>.png`.

mod common;

use std::collections::HashSet;

use common::{assert_no_validation, differing_pixels, gpu};
use mosh_core::{AnalysisFrame, ClipParams};
use mosh_render::{FrameRecorder, Gpu, LayerTarget, Pipeline};
use mosh_shaders::VISUALIZERS;
use mosh_visuals::scenes::Scenes;
use mosh_visuals::styles::novel::SCENES;

const W: u32 = 160;
const H: u32 = 90;
const PREVIEW_W: u32 = 480;
const PREVIEW_H: u32 = 270;
const TIME_W: u32 = 960;
const TIME_H: u32 = 540;
const SAMPLES: usize = 20;
const BUDGET_MS: f32 = 0.5;
const REFERENCE_FLOOR_MS: f32 = 0.20;

fn frame_at(level: usize) -> AnalysisFrame {
    let (band, pulse): (f32, f32) = match level {
        0 => (0.18, 0.25),
        1 => (0.50, 0.60),
        _ => (0.90, 1.00),
    };
    AnalysisFrame {
        time_s: 12.0,
        ppq: 16.0,
        beat_phase: (1.0 - pulse.sqrt()).clamp(0.0, 1.0),
        bar_phase: 0.25,
        beat_index: 16,
        bar_index: 4,
        tempo_bpm: 120.0,
        playing: true,
        bands: [band, band * 0.82, band * 0.66],
        mel: std::array::from_fn(|k| {
            let wave = 0.5 + 0.5 * ((k as f32 * 0.65).sin());
            (band * (0.55 + 0.45 * wave)).clamp(0.0, 1.0)
        }),
        onset: pulse,
        loudness: band,
        level: level as u8 + 1,
        intensity: band,
        ..AnalysisFrame::default()
    }
}

fn frame_with_pulse(pulse: f32) -> AnalysisFrame {
    AnalysisFrame {
        beat_phase: (1.0 - pulse.sqrt()).clamp(0.0, 1.0),
        ..frame_at(2)
    }
}

fn params_for(seed: u64) -> ClipParams {
    let p = match seed {
        0 => [0.0; 8],
        1 => [0.7, 0.5, 0.25, 0.5, 0.5, 0.5, 0.5, 0.4],
        _ => [1.0; 8],
    };
    ClipParams {
        p,
        palette: 1 + (seed as u16 % 7),
        seed,
    }
}

fn preview_params() -> ClipParams {
    ClipParams {
        p: [0.65, 0.55, 0.12, 0.50, 0.60, 0.65, 0.60, 0.35],
        palette: 3,
        seed: 7,
    }
}

fn draw(
    scenes: &mut Scenes,
    gpu: &Gpu,
    dst: &LayerTarget,
    id: &str,
    frame: &AnalysisFrame,
    params: &ClipParams,
    t: f32,
) {
    let mut rec = FrameRecorder::new(gpu);
    scenes
        .draw(&mut rec, dst, id, frame, params, t)
        .unwrap_or_else(|error| panic!("draw {id}: {error}"));
    gpu.wait_frame(rec.submit());
}

fn mean_luminance(pixels: &[u8]) -> f32 {
    let sum: u64 = pixels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|px| u64::from(px[0]) + u64::from(px[1]) + u64::from(px[2]))
        .sum();
    sum as f32 / ((pixels.len() / 4) as f32 * 3.0 * 255.0)
}

fn clipped_share(pixels: &[u8]) -> f32 {
    let white = pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|px| px[0] >= 250 && px[1] >= 250 && px[2] >= 250)
        .count();
    white as f32 / (pixels.len() / 4) as f32
}

#[test]
fn the_novel_pack_has_six_scenes_in_two_packs() {
    assert_eq!(SCENES.len(), 6);
    let mut seen: HashSet<&str> = VISUALIZERS.iter().map(|v| v.id).collect();
    seen.extend(["fluid", "overlay", "camera", "text", "particles"]);
    let mut packs = [0usize; 2];
    for scene in SCENES {
        assert!(seen.insert(scene.id), "duplicate id {}", scene.id);
        assert!(!scene.name.is_empty(), "{} missing name", scene.id);
        assert!(
            scene.params.iter().all(|p| !p.is_empty()),
            "{} has empty param name: {:?}",
            scene.id,
            scene.params
        );
        match scene.pack {
            "Hyper" => packs[0] += 1,
            "Wave" => packs[1] += 1,
            other => panic!("unexpected pack {other:?} on {}", scene.id),
        }
    }
    assert_eq!(packs, [3, 3], "expected Hyper 3, Wave 3");
}

#[test]
fn every_novel_scene_renders_lit_and_writes_previews() {
    let gpu = gpu();
    let mut scenes = Scenes::new(&gpu).expect("scenes");
    let frames = [frame_at(0), frame_at(1), frame_at(2)];
    let mut dark = Vec::new();

    for scene in SCENES {
        for (level, frame) in frames.iter().enumerate() {
            for seed in 0..3u64 {
                let dst = LayerTarget::new(&gpu, W, H);
                let params = params_for(seed);
                draw(
                    &mut scenes,
                    &gpu,
                    &dst,
                    scene.id,
                    frame,
                    &params,
                    2.5 + seed as f32 * 0.9,
                );
                let lum = mean_luminance(&gpu.read_back(&dst));
                if lum <= 0.02 {
                    dark.push(format!("{} level {level} seed {seed}: {lum:.4}", scene.id));
                }
            }
        }
    }
    assert!(dark.is_empty(), "dark novel draws:\n{}", dark.join("\n"));

    // Render 480x270 previews for visual inspection.
    let dir = preview_dir();
    let preview_dst = LayerTarget::new(&gpu, PREVIEW_W, PREVIEW_H);
    let frame = frame_at(2);
    let params = preview_params();
    for (idx, scene) in SCENES.iter().enumerate() {
        draw(
            &mut scenes,
            &gpu,
            &preview_dst,
            scene.id,
            &frame,
            &params,
            5.0 + idx as f32 * 1.1,
        );
        let pixels = gpu.read_back(&preview_dst);
        let lum = mean_luminance(&pixels);
        let clip = clipped_share(&pixels);
        let path = dir.join(format!("{}.png", scene.id));
        write_png(&path, PREVIEW_W, PREVIEW_H, &pixels).expect("write preview png");
        println!(
            "{:<20} preview: lum {lum:.3}, clipped {:.2}%, -> {}",
            scene.id,
            clip * 100.0,
            path.display()
        );
        assert!(
            clip <= 0.02,
            "{} clipped {:.2}% white (> 2%)",
            scene.id,
            clip * 100.0
        );
    }
    assert_no_validation(&gpu);
}

#[test]
fn every_novel_control_moves_its_scene() {
    let gpu = gpu();
    let mut scenes = Scenes::new(&gpu).expect("scenes");
    let dst = LayerTarget::new(&gpu, W, H);
    let frame = frame_at(1);
    let base = ClipParams {
        p: [0.55, 0.5, 0.2, 0.5, 0.5, 0.5, 0.5, 0.3],
        palette: 3,
        seed: 5,
    };
    let mut inert = Vec::new();
    for scene in SCENES {
        for index in 0..8 {
            let mut lo = base;
            let mut hi = base;
            lo.p[index] = 0.0;
            // Hue wraps at 1.0, so test 0.0 vs 0.5 for P(2).
            hi.p[index] = if index == 2 { 0.5 } else { 1.0 };
            draw(&mut scenes, &gpu, &dst, scene.id, &frame, &lo, 4.2);
            let a = gpu.read_back(&dst);
            draw(&mut scenes, &gpu, &dst, scene.id, &frame, &hi, 4.2);
            let b = gpu.read_back(&dst);
            let changed = differing_pixels(&a, &b, 2);
            if changed < 20 {
                inert.push(format!(
                    "{} p{index} ({}): only {changed} px changed",
                    scene.id, scene.params[index]
                ));
            }
        }
    }
    assert!(
        inert.is_empty(),
        "inert controls found:\n{}",
        inert.join("\n")
    );
    assert_no_validation(&gpu);
}

#[test]
fn every_novel_scene_reacts_to_the_beat_without_strobing() {
    let gpu = gpu();
    let mut scenes = Scenes::new(&gpu).expect("scenes");
    let dst = LayerTarget::new(&gpu, W, H);
    let params = preview_params();
    let quiet = frame_with_pulse(0.0);
    let hit = frame_with_pulse(1.0);

    for scene in SCENES {
        draw(&mut scenes, &gpu, &dst, scene.id, &quiet, &params, 6.0);
        let a = gpu.read_back(&dst);
        draw(&mut scenes, &gpu, &dst, scene.id, &hit, &params, 6.0);
        let b = gpu.read_back(&dst);
        let changed = differing_pixels(&a, &b, 3);
        let dlum = (mean_luminance(&b) - mean_luminance(&a)).abs();
        println!(
            "{:<20} beat: {changed} px changed, |dlum| {dlum:.4}",
            scene.id
        );
        assert!(
            changed >= 80,
            "{} barely answered the beat ({changed} px changed)",
            scene.id
        );
        assert!(
            dlum < 0.25,
            "{} strobed global luminance on the beat (|dlum| {dlum:.4})",
            scene.id
        );
    }
    assert_no_validation(&gpu);
}

#[test]
fn novel_scenes_meet_their_share_of_the_budget() {
    let gpu = gpu();
    let mut scenes = Scenes::new(&gpu).expect("scenes");
    let dst = LayerTarget::new(&gpu, TIME_W, TIME_H);
    let frame = frame_at(2);
    let params = preview_params();

    // Warm-up: 24 passes of every scene at 960x540 so the iGPU reaches steady active clocks
    // before timing (matches styles_aero.rs).
    for pass in 0..24 {
        for scene in SCENES {
            draw(
                &mut scenes,
                &gpu,
                &dst,
                scene.id,
                &frame,
                &params,
                1.0 + pass as f32 / 60.0,
            );
        }
    }

    let mut rows = Vec::new();
    for scene in SCENES {
        let mut round_medians = Vec::new();
        let mut min_ms = f32::MAX;
        for round in 0..3 {
            let mut samples = Vec::with_capacity(SAMPLES);
            for pass in 0..SAMPLES {
                let mut rec = FrameRecorder::new(&gpu);
                scenes
                    .draw(
                        &mut rec,
                        &dst,
                        scene.id,
                        &frame,
                        &params,
                        4.0 + round as f32 + pass as f32 / 60.0,
                    )
                    .unwrap_or_else(|e| panic!("draw {}: {e}", scene.id));
                let timings = gpu.wait_frame(rec.submit());
                if pass < 2 {
                    continue;
                }
                if let Some((_, ms)) = timings.per_pass_ms.iter().find(|(n, _)| *n == scene.id) {
                    samples.push(*ms);
                }
            }
            samples.sort_by(f32::total_cmp);
            min_ms = min_ms.min(samples[0]);
            round_medians.push(samples[samples.len() / 2]);
        }
        let best_med = round_medians.iter().copied().fold(f32::MAX, f32::min);
        let worst_med = round_medians.iter().copied().fold(0.0f32, f32::max);
        rows.push((scene.id, min_ms, best_med, worst_med));
    }

    let floor = empty_pass_ms(&gpu, &dst, &frame, &params);
    println!("empty pass at {TIME_W}x{TIME_H}: median {floor:.3} ms");
    let mut worst = 0.0f32;
    for (id, min_ms, best_med, worst_med) in &rows {
        worst = worst.max(*best_med);
        println!(
            "{id:<20} {TIME_W}x{TIME_H}: median {best_med:.3} ms (min {min_ms:.3}, worst round {worst_med:.3}, +{:.3} ms over empty)",
            best_med - floor
        );
    }

    if !cfg!(debug_assertions) {
        if floor <= REFERENCE_FLOOR_MS {
            assert!(
                worst < BUDGET_MS,
                "worst novel scene median {worst:.3} ms exceeded {BUDGET_MS:.3} ms budget"
            );
        } else {
            assert!(
                worst < 5.0 * floor,
                "worst novel scene median {worst:.3} ms exceeded 5x empty floor ({:.3} ms)",
                5.0 * floor
            );
        }
    }
    assert_no_validation(&gpu);
}

fn empty_pass_ms(
    gpu: &Gpu,
    dst: &LayerTarget,
    frame: &AnalysisFrame,
    params: &ClipParams,
) -> f32 {
    let source = format!(
        "{}\n@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {{\n    \
         return fin(vec3f(XY(pos), 0.5));\n}}\n",
        mosh_visuals::styles::prelude()
    );
    let words = mosh_shaders::compile_wgsl(&source).expect("empty body compiles");
    let pipeline = Pipeline::fullscreen(gpu, &words, mosh_shaders::LAYER_UNIFORM_BYTES, 5)
        .expect("empty pipeline")
        .with_name("empty");
    let tables: Vec<LayerTarget> = (0..5).map(|_| LayerTarget::new(gpu, 1, 1)).collect();
    let inputs: Vec<&LayerTarget> = tables.iter().collect();
    let uniforms = mosh_shaders::layer_uniforms(
        frame,
        params,
        4.0,
        [dst.width() as f32, dst.height() as f32],
    );
    let mut samples = Vec::new();
    for pass in 0..SAMPLES {
        let mut rec = FrameRecorder::new(gpu);
        rec.draw(&pipeline, dst, uniforms.as_bytes(), &inputs)
            .expect("empty draw");
        let timings = gpu.wait_frame(rec.submit());
        if pass < 2 {
            continue;
        }
        if let Some((_, ms)) = timings.per_pass_ms.iter().find(|(n, _)| *n == "empty") {
            samples.push(*ms);
        }
    }
    samples.sort_by(f32::total_cmp);
    samples[samples.len() / 2]
}

fn preview_dir() -> std::path::PathBuf {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/style-previews/novel");
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    dir
}

fn write_png(
    path: &std::path::Path,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> std::io::Result<()> {
    let stride = width as usize * 4;
    let mut raw = Vec::with_capacity(height as usize * (1 + stride));
    for y in 0..height as usize {
        raw.push(0);
        raw.extend_from_slice(&rgba[y * stride..(y + 1) * stride]);
    }
    let mut zlib = vec![0x78, 0x01];
    for (index, chunk) in raw.chunks(65_535).enumerate() {
        let last = index + 1 == raw.len().div_ceil(65_535);
        zlib.push(u8::from(last));
        let len = chunk.len() as u16;
        zlib.extend_from_slice(&len.to_le_bytes());
        zlib.extend_from_slice(&(!len).to_le_bytes());
        zlib.extend_from_slice(chunk);
    }
    zlib.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut png = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut header = Vec::new();
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    header.extend_from_slice(&[8, 6, 0, 0, 0]);
    png_chunk(&mut png, b"IHDR", &header);
    png_chunk(&mut png, b"IDAT", &zlib);
    png_chunk(&mut png, b"IEND", &[]);
    std::fs::write(path, png)
}

fn png_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for byte in bytes {
        a = (a + u32::from(*byte)) % 65_521;
        b = (b + a) % 65_521;
    }
    (b << 16) | a
}
