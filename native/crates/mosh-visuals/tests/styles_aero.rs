//! The Y2K + Frutiger Aero style pack (`styles::aero`): eight scenes that must draw lit at three
//! intensity levels and three parameter sets, answer `p0`, answer the beat without strobing the
//! whole frame, keep both validation layers quiet, and fit in less than 0.5 ms of the frame at
//! 960x540 on the integrated GPU.
//!
//! Everything is offscreen and short on purpose (the reference machine is shared with other
//! workers): 160x90 for correctness, 480x270 for the previews a human looks at, 960x540 only for
//! the one timing report.
//!
//! # The previews
//!
//! [`aero_scenes_meet_their_share_of_the_budget`] writes one 480x270 PNG per scene to
//! `native/target/style-previews/aero/<id>.png`. They are build output, not committed, and the PR
//! lists their paths so the coordinator can show them to the human.
//!
//! # Palettes 1..=7, never 0, and why
//!
//! `Scenes::draw` resolves a clip palette of 0 through `scenes::DEFPAL`, whose sixteen entries
//! cover the sixteen classic scenes only; a style scene sits at index 16 or beyond, so a style
//! clip with palette 0 panics with `index out of bounds` (found while writing this pack, reported
//! on the PR: the fallback lives in `scenes.rs`, which is V-scenes' file, not this lane's). These
//! tests therefore pass 1, 3 and 6, which is the same coverage of `lut()`/accent rotation.

mod common;

use common::{assert_no_validation, differing_pixels, gpu};
use mosh_core::{AnalysisFrame, ClipParams};
use mosh_render::{FrameRecorder, Gpu, LayerTarget, Pipeline};
use mosh_visuals::scenes::Scenes;
use mosh_visuals::styles::aero;

const W: u32 = 160;
const H: u32 = 90;
const PREVIEW_W: u32 = 480;
const PREVIEW_H: u32 = 270;
const TIME_W: u32 = 960;
const TIME_H: u32 = 540;
/// Measured passes per scene after two warm-ups.
const SAMPLES: usize = 20;
/// The style pack's share of the 6 ms frame budget at 1080p, measured at 960x540 (a quarter of the
/// pixels, so a generous stand-in for the 1080p cost of one of three layers).
const BUDGET_MS: f32 = 0.5;
/// The empty-pass cost of a machine in its reference power state, from the committed numbers for
/// the web scenes (0.10-0.32 ms per pass at 960x540). Above this, absolute milliseconds measure
/// the clock and the shared machine, not the shader.
const REFERENCE_FLOOR_MS: f32 = 0.20;

fn scene_ids() -> Vec<&'static str> {
    aero::SCENES.iter().map(|scene| scene.id).collect()
}

/// Three intensity levels: a quiet bar, a mid bar and a drop (the frames `tests/scenes.rs` uses).
/// `beat_phase` 0 is the downbeat (`pulse` 1), 1 the next beat (`pulse` 0).
fn frame_at(level: usize) -> AnalysisFrame {
    let (band, phase): (f32, f32) = match level {
        0 => (0.18, 0.75),
        1 => (0.5, 0.35),
        _ => (0.9, 0.0),
    };
    AnalysisFrame {
        time_s: 12.0,
        ppq: 16.0,
        beat_phase: phase,
        bar_phase: 0.25,
        beat_index: 16,
        bar_index: 4,
        tempo_bpm: 120.0,
        playing: true,
        bands: [band, band * 0.82, band * 0.66],
        mel: std::array::from_fn(|k| (band * (1.0 - k as f32 / 30.0)).max(0.0)),
        onset: 1.0 - phase,
        loudness: band,
        level: level as u8 + 1,
        intensity: band,
        ..AnalysisFrame::default()
    }
}

/// Three parameter sets spread over all eight controls. The palette is never 0 (see the module
/// docs).
fn params_for(seed: u64) -> ClipParams {
    let r = |k: u64| {
        ((seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(k * 1442695040888963407))
            >> 40) as f32
            / (1u64 << 24) as f32
    };
    ClipParams {
        p: [
            0.7 + r(1) * 0.8,          // p0, the scene clock
            0.2 + r(2) * 0.7,          // p1
            0.1 + r(3) * 0.8,          // p2
            0.2 + r(4) * 0.7,          // p3
            0.15 + r(5) * 0.8,         // p4
            0.1 + r(6) * 0.85,         // p5
            0.05 + r(7) * 0.9,         // p6
            0.25 + r(8) * 0.7,         // p7
        ],
        palette: match seed % 3 {
            0 => 1,
            1 => 3,
            _ => 6,
        },
        seed,
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
    let mut recorder = FrameRecorder::new(gpu);
    scenes
        .draw(&mut recorder, dst, id, frame, params, t)
        .unwrap_or_else(|error| panic!("draw {id}: {error}"));
    gpu.wait_frame(recorder.submit());
}

/// The showcase parameter set the previews are drawn with: `p2` (Hue) is 0, which is the pack's
/// own palette (the aqua of the Aero scenes, the magenta-and-teal of the Y2K ones); a seed's random
/// hue would rotate them somewhere else and the PNG would not show the scene as designed. The
/// palette is 3, never 0 (see the module docs).
fn preview_params() -> ClipParams {
    ClipParams {
        p: [0.9, 0.6, 0.0, 0.55, 0.7, 0.6, 0.6, 0.6],
        palette: 3,
        seed: 7,
    }
}

/// Mean of the three colour channels over the whole frame, 0..1 (the gate's "lit" number).
fn mean_luminance(pixels: &[u8]) -> f32 {
    let sum: u64 = pixels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|pixel| u64::from(pixel[0]) + u64::from(pixel[1]) + u64::from(pixel[2]))
        .sum();
    let count = (pixels.len() / 4) as f32;
    sum as f32 / (count * 3.0 * 255.0)
}

/// Each scene at 3 intensity levels x 3 parameter sets: mean luminance > 0.02, read back.
#[test]
fn every_aero_scene_renders_lit() {
    let gpu = gpu();
    let mut scenes = Scenes::new(&gpu).expect("scenes");
    let ids = scene_ids();
    assert!(
        (6..=8).contains(&ids.len()),
        "the pack is six to eight scenes, got {}: {ids:?}",
        ids.len()
    );

    let frames = [frame_at(0), frame_at(1), frame_at(2)];
    let mut dark = Vec::new();
    let mut report = Vec::new();
    for id in &ids {
        for (level, frame) in frames.iter().enumerate() {
            for seed in 0..3u64 {
                let dst = LayerTarget::new(&gpu, W, H);
                let params = params_for(seed);
                draw(
                    &mut scenes,
                    &gpu,
                    &dst,
                    id,
                    frame,
                    &params,
                    2.0 + seed as f32 * 0.7,
                );
                let luminance = mean_luminance(&gpu.read_back(&dst));
                report.push(format!(
                    "{id:<22} level {level} seed {seed}: mean luminance {luminance:.4}"
                ));
                if luminance <= 0.02 {
                    dark.push(format!("{id} level {level} seed {seed}: {luminance:.4}"));
                }
            }
        }
    }
    for line in &report {
        println!("{line}");
    }
    assert!(
        dark.is_empty(),
        "{} of {} aero draws were not lit (> 0.02):\n{}",
        dark.len(),
        report.len(),
        dark.join("\n")
    );
    assert_no_validation(&gpu);
}

/// `p0` is the scene clock: 0 freezes the motion, 1 runs it, and every scene's pixels differ.
/// The other seven controls are printed for the PR (and to show they are not dead), because a
/// scene's `params` names are a promise to the clip editor.
#[test]
fn every_aero_control_moves_its_scene() {
    let gpu = gpu();
    let mut scenes = Scenes::new(&gpu).expect("scenes");
    let frame = frame_at(2);
    let base = params_for(1);
    let dst = LayerTarget::new(&gpu, W, H);
    let other = LayerTarget::new(&gpu, W, H);

    let mut frozen = Vec::new();
    let mut rows = Vec::new();
    for id in scene_ids() {
        let mut counts = [0usize; 8];
        for (param, slot) in counts.iter_mut().enumerate() {
            let mut low = base;
            let mut high = base;
            low.p[param] = 0.0;
            // `p2` is Hue, which is cyclic, so its two test values are 0 and a half turn: 0 and 1
            // are the same colour and would report "this control does nothing".
            high.p[param] = if param == 2 { 0.5 } else { 1.0 };
            // `p0` is a clock, so the two draws must be at a non-zero time for it to move pixels.
            draw(&mut scenes, &gpu, &dst, id, &frame, &low, 3.0);
            draw(&mut scenes, &gpu, &other, id, &frame, &high, 3.0);
            *slot = differing_pixels(&gpu.read_back(&dst), &gpu.read_back(&other), 2);
        }
        rows.push(format!("{id:<22} p0..p7 changed pixels: {counts:?}"));
        if counts[0] == 0 {
            frozen.push(id);
        }
    }
    for row in &rows {
        println!("{row}");
    }
    assert!(
        frozen.is_empty(),
        "p0 must move every scene, frozen: {frozen:?}"
    );
    assert_no_validation(&gpu);
}

/// The beat: same frame, same parameters, `pulse` 1 vs 0. Every scene changes, and no scene is a
/// full-frame strobe: the frame's mean luminance may not swing by more than a third, so the
/// flashes stay inside the score and the Strobe transition.
#[test]
fn aero_scenes_answer_the_beat_without_strobing_the_frame() {
    let gpu = gpu();
    let mut scenes = Scenes::new(&gpu).expect("scenes");
    let id_dst = LayerTarget::new(&gpu, W, H);
    let off_dst = LayerTarget::new(&gpu, W, H);
    let on = frame_at(2); // beat_phase 0 -> pulse 1
    let mut off = on;
    off.beat_phase = 1.0; // pulse 0
    let params = params_for(2);

    let mut quiet = Vec::new();
    let mut rows = Vec::new();
    for id in scene_ids() {
        draw(&mut scenes, &gpu, &id_dst, id, &on, &params, 4.0);
        draw(&mut scenes, &gpu, &off_dst, id, &off, &params, 4.0);
        let lit = gpu.read_back(&id_dst);
        let dark = gpu.read_back(&off_dst);
        let changed = differing_pixels(&lit, &dark, 2);
        let swing = (mean_luminance(&lit) - mean_luminance(&dark)).abs();
        rows.push(format!(
            "{id:<22} pulse: {changed} of {} pixels differ, mean luminance swing {swing:.3}",
            W * H
        ));
        // A whole frame is 14400 pixels at this size: 1 % is 144.
        if changed < (W * H / 100) as usize || swing > 0.33 {
            quiet.push(format!("{id}: {changed} pixels, swing {swing:.3}"));
        }
    }
    for row in &rows {
        println!("{row}");
    }
    assert!(
        quiet.is_empty(),
        "these scenes do not answer the beat, or strobe the whole frame:\n{}",
        quiet.join("\n")
    );
    assert_no_validation(&gpu);
}

/// One 480x270 PNG per scene under `native/target/style-previews/aero/`, then GPU milliseconds
/// per pass at 960x540: the median must be under 0.5 ms in a release build (a debug build prints
/// and asserts nothing, because the shader compiler is not what is being measured).
#[test]
fn aero_scenes_meet_their_share_of_the_budget() {
    let gpu = gpu();
    let mut scenes = Scenes::new(&gpu).expect("scenes");
    let ids = scene_ids();
    let frame = frame_at(2);
    let params = params_for(1);
    let showcase = preview_params();

    // Previews first, at a size a human can read without a magnifying glass.
    let previews = preview_dir();
    println!("previews: {}", previews.display());
    let preview = LayerTarget::new(&gpu, PREVIEW_W, PREVIEW_H);
    for (index, id) in ids.iter().enumerate() {
        draw(
            &mut scenes,
            &gpu,
            &preview,
            id,
            &frame,
            &showcase,
            6.0 + index as f32 * 0.9,
        );
        let path = previews.join(format!("{id}.png"));
        write_png(&path, PREVIEW_W, PREVIEW_H, &gpu.read_back(&preview))
            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        println!("  {id:<22} {}", path.display());
    }

    // Warm-up: 30 frames at the measured size, so the iGPU is at a steady clock. A cold GPU
    // reports 3-4x these numbers, and this machine is shared with other workers.
    let dst = LayerTarget::new(&gpu, TIME_W, TIME_H);
    for pass in 0..30 {
        draw(
            &mut scenes,
            &gpu,
            &dst,
            ids[0],
            &frame,
            &params,
            1.0 + pass as f32 / 60.0,
        );
    }

    let mut rows = Vec::new();
    for id in &ids {
        let mut samples = Vec::new();
        for pass in 0..SAMPLES {
            let t = 4.0 + pass as f32 / 60.0;
            let mut recorder = FrameRecorder::new(&gpu);
            scenes
                .draw(&mut recorder, &dst, id, &frame, &params, t)
                .expect("timed draw");
            let timings = gpu.wait_frame(recorder.submit());
            if pass < 2 {
                continue; // the first passes of a scene warm its own caches
            }
            let ms = timings
                .per_pass_ms
                .iter()
                .find(|(name, _)| name == id)
                .map(|(_, ms)| *ms)
                .unwrap_or_else(|| {
                    panic!("no timestamp for the {id} pass: {:?}", timings.per_pass_ms)
                });
            samples.push(ms);
        }
        samples.sort_by(f32::total_cmp);
        rows.push((*id, samples[0], samples[samples.len() / 2]));
    }
    let floor = empty_pass_ms(&gpu, &dst, &frame, &params);
    println!(
        "GPU ms per aero pass at {TIME_W}x{TIME_H} on {}",
        gpu.adapter_name()
    );
    println!("  {:<22} {:>7} {:>7} {:>7}", "scene", "min", "median", "above");
    println!("  {:<22} {:>7} {:>7} {:>7}", "empty pass", floor, floor, 0.0);
    for (id, min, median) in &rows {
        println!(
            "  {id:<22} {min:>7.3} {median:>7.3} {:>7.3}",
            median - floor
        );
    }
    let worst = rows
        .iter()
        .map(|(_, _, median)| *median)
        .fold(0.0f32, f32::max);
    let worst_above = rows
        .iter()
        .map(|(_, _, median)| median - floor)
        .fold(0.0f32, f32::max);
    println!(
        "  worst median {worst:.3} ms, {worst_above:.3} ms above the empty pass; target < \
         {BUDGET_MS:.3} ms"
    );
    // The budget is an absolute number, and an absolute number only means anything on a machine in
    // its reference power state: on this shared laptop the empty pass alone has measured 0.15 ms
    // and 0.34 ms an hour apart, which moves every scene by 3.7x (see `scenes/perf.rs`, which
    // asserts a ratio for the same reason). Below the reference floor, assert the budget itself.
    // Above it the empty pass costs more than the whole per-scene budget, so the gate that still
    // bites is the pack's weight against the one cost every pass pays on this machine: under five
    // empty passes, which is what the 0.5 ms budget is on a reference-state machine.
    if cfg!(debug_assertions) {
        println!("  debug build: printed, not asserted (build with --release for the budget)");
    } else if floor <= REFERENCE_FLOOR_MS {
        assert!(
            worst < BUDGET_MS,
            "every aero scene must cost under {BUDGET_MS} ms at {TIME_W}x{TIME_H} (empty pass \
             {floor:.3} ms), worst median {worst:.3} ms"
        );
    } else {
        println!(
            "  machine not in its reference state: the empty pass costs {floor:.3} ms, over the \
             {REFERENCE_FLOOR_MS:.2} ms reference floor; asserting under five empty passes"
        );
        assert!(
            worst < 5.0 * floor,
            "every aero scene must cost under five empty passes ({:.3} ms here) at \
             {TIME_W}x{TIME_H}, worst median {worst:.3} ms",
            5.0 * floor
        );
    }
    assert_no_validation(&gpu);
}

/// The GPU milliseconds of one empty pass at the measured size: a style body that only writes the
/// gradient. Everything a pass pays before its first instruction (render-pass setup, target
/// traffic, timestamp granularity) is in here.
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
    let words = mosh_shaders::compile_wgsl(&source).expect("the empty style body compiles");
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
        let mut recorder = FrameRecorder::new(gpu);
        recorder
            .draw(&pipeline, dst, uniforms.as_bytes(), &inputs)
            .expect("empty draw");
        let timings = gpu.wait_frame(recorder.submit());
        if pass < 2 {
            continue;
        }
        let ms = timings
            .per_pass_ms
            .iter()
            .find(|(name, _)| name == "empty")
            .map(|(_, ms)| *ms)
            .unwrap_or_else(|| panic!("no timestamp for the empty pass: {:?}", timings.per_pass_ms));
        samples.push(ms);
    }
    samples.sort_by(f32::total_cmp);
    samples[samples.len() / 2]
}

/// `native/target/style-previews/aero`, created on demand. `CARGO_MANIFEST_DIR` is the crate.
fn preview_dir() -> std::path::PathBuf {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/style-previews/aero");
    std::fs::create_dir_all(&dir).unwrap_or_else(|error| panic!("{}: {error}", dir.display()));
    dir
}

fn write_png(
    path: &std::path::Path,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> std::io::Result<()> {
    assert_eq!(
        rgba.len(),
        (width as usize) * (height as usize) * 4,
        "read-back of the wrong size"
    );
    // PNG scanlines, filter byte 0 on each: the compressor below is `stored` only, so a preview
    // is about 520 KB at 480x270. That is fine for build output; no image crate is in the
    // workspace and this file may not add one.
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
    header.extend_from_slice(&[8, 6, 0, 0, 0]); // 8 bits, RGBA, deflate, no filter, no interlace
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
