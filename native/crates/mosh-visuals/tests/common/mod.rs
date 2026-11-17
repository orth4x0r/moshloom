//! Offscreen Vulkan fixtures shared by the mosh-visuals integration tests.
//!
//! No window is opened: every test renders into `LayerTarget`s and reads the result back. The
//! test pictures are WGSL compiled with `naga` here, the same way `mosh-render`'s tests build
//! their shaders; the passes under test are the real `fx_effects`/`tr_mix`/`tr_copy` pipelines.

#![allow(dead_code)]

use mosh_render::{FrameRecorder, Gpu, GpuPreference, LayerTarget, Pipeline};

/// One device per test. The reference machine's integrated GPU is the default; a machine without
/// one (a VM, CI) falls back to whatever it has, so nothing here pins an adapter name.
pub fn gpu() -> Gpu {
    let gpu = Gpu::new(GpuPreference::Integrated)
        .or_else(|_| Gpu::new(GpuPreference::ByName(String::new())))
        .expect("a Vulkan device");
    println!(
        "adapter: {} | validation layers: {} | timestamp period: {} ns",
        gpu.adapter_name(),
        if gpu.validation_enabled() {
            "on"
        } else {
            "unavailable"
        },
        gpu.timestamp_period_ns()
    );
    gpu
}

/// The gate's "zero validation messages": printed as a number, then asserted.
pub fn assert_no_validation(gpu: &Gpu) {
    let messages = gpu.validation_messages();
    println!(
        "validation: {} message(s), layer {}",
        messages.len(),
        if gpu.validation_enabled() {
            "enabled"
        } else {
            "unavailable"
        }
    );
    assert!(
        messages.is_empty(),
        "{} validation message(s):\n{}",
        messages.len(),
        messages.join("\n")
    );
}

/// A deterministic test picture: gradients in x, y and diagonally, so mirror, fold, spin, zoom,
/// crush, streak and echo all move it. Alpha is 1.
pub const PATTERN_WGSL: &str = r#"
struct Params {
    res: vec2<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = pos.xy / params.res;
    let r = fract(uv.x * 7.0);
    let g = fract(uv.y * 5.0);
    let b = fract((uv.x + uv.y) * 3.0);
    return vec4<f32>(r, g, b, 1.0);
}
"#;

/// A flat colour, for a transition's "from" and "to" pictures.
pub const SOLID_WGSL: &str = r#"
struct Params {
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return params.color;
}
"#;

fn spirv(source: &str) -> Vec<u32> {
    let module = naga::front::wgsl::parse_str(source).expect("test shader parses");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator.validate(&module).expect("test shader validates");
    let options = naga::back::spv::Options {
        lang_version: (1, 3),
        ..Default::default()
    };
    naga::back::spv::write_vec(&module, &info, &options, None).expect("test shader compiles")
}

/// The pattern pipeline; its uniform is `res` (8 bytes).
pub fn pattern_pipeline(gpu: &Gpu) -> Pipeline {
    Pipeline::fullscreen(gpu, &spirv(PATTERN_WGSL), 16, 0)
        .expect("pattern pipeline")
        .with_name("pattern")
}

/// The solid pipeline; its uniform is `color` (16 bytes).
pub fn solid_pipeline(gpu: &Gpu) -> Pipeline {
    Pipeline::fullscreen(gpu, &spirv(SOLID_WGSL), 16, 0)
        .expect("solid pipeline")
        .with_name("solid")
}

fn floats(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Draws the pattern into `target`.
pub fn draw_pattern(gpu: &Gpu, pipeline: &Pipeline, target: &LayerTarget) {
    let mut recorder = FrameRecorder::new(gpu);
    let uniforms = floats(&[target.width() as f32, target.height() as f32]);
    recorder
        .draw(pipeline, target, &uniforms, &[])
        .expect("draw pattern");
    gpu.wait_frame(recorder.submit());
}

/// Draws `color` (0..1 per channel) into `target`.
pub fn draw_solid(gpu: &Gpu, pipeline: &Pipeline, target: &LayerTarget, color: [f32; 4]) {
    let mut recorder = FrameRecorder::new(gpu);
    recorder
        .draw(pipeline, target, &floats(&color), &[])
        .expect("draw solid");
    gpu.wait_frame(recorder.submit());
}

/// Mean luminance of an RGBA8 read-back, 0..255.
pub fn mean_luma(pixels: &[u8]) -> f32 {
    if pixels.is_empty() {
        return 0.0;
    }
    let total: f32 = pixels
        .chunks(4)
        .map(|pixel| {
            0.3 * f32::from(pixel[0]) + 0.59 * f32::from(pixel[1]) + 0.11 * f32::from(pixel[2])
        })
        .sum();
    total / (pixels.len() / 4) as f32
}

/// How many pixels differ by more than `tolerance` in some RGB channel.
pub fn differing_pixels(a: &[u8], b: &[u8], tolerance: i32) -> usize {
    assert_eq!(a.len(), b.len(), "read-backs of different sizes");
    a.chunks(4)
        .zip(b.chunks(4))
        .filter(|(left, right)| {
            (0..3).any(|channel| {
                (i32::from(left[channel]) - i32::from(right[channel])).abs() > tolerance
            })
        })
        .count()
}

/// Mean absolute per-channel difference, 0..255, for the "it changed" tests.
pub fn mean_difference(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len(), "read-backs of different sizes");
    let total: f32 = a
        .chunks(4)
        .zip(b.chunks(4))
        .map(|(left, right)| {
            (0..3)
                .map(|channel| (f32::from(left[channel]) - f32::from(right[channel])).abs())
                .sum::<f32>()
        })
        .sum();
    total / (a.len() / 4 * 3) as f32
}
