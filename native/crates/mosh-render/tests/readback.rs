//! `AsyncReadback`: a small RGBA8 picture of an image two frames late, with no wait.
//!
//! Offscreen: the readback is driven through `record_target`, which submits the same commands
//! `Stage::present` adds to a frame (and waits, because a test has no frame loop). What is
//! checked: the two-frame delay and the tags, the colours (linear 16F in, sRGB 8-bit out), the
//! orientation, that nothing is allocated per picture, and what a picture costs the thread that
//! asks for it, against the blocking `Gpu::read_back` it replaces.

use std::time::{Duration, Instant};

use mosh_render::{
    AsyncReadback, FrameRecorder, Gpu, GpuPreference, LayerTarget, Pipeline, READBACK_DELAY,
};

/// Left half red, right half green; blue runs along `uv.y`.
const SPLIT_WGSL: &str = r#"
struct Params {
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let left = step(uv.x, 0.5);
    return vec4<f32>(left * params.color.r, (1.0 - left) * params.color.g, uv.y, 1.0);
}
"#;

const SOLID_WGSL: &str = r#"
struct Params {
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
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

fn gpu() -> Gpu {
    Gpu::new(GpuPreference::Integrated)
        .or_else(|_| Gpu::new(GpuPreference::ByName(String::new())))
        .expect("a Vulkan device")
}

fn color_uniform(color: [f32; 4]) -> Vec<u8> {
    color.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn draw(gpu: &Gpu, pipeline: &Pipeline, target: &LayerTarget, color: [f32; 4]) {
    let mut recorder = FrameRecorder::new(gpu);
    recorder
        .draw(pipeline, target, &color_uniform(color), &[])
        .expect("draw");
    gpu.wait_frame(recorder.submit());
}

fn pixel(rgba: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let at = ((y * width + x) * 4) as usize;
    [rgba[at], rgba[at + 1], rgba[at + 2], rgba[at + 3]]
}

fn close(a: [u8; 4], b: [u8; 4], tolerance: i32) -> bool {
    a.iter()
        .zip(b)
        .all(|(a, b)| (i32::from(*a) - i32::from(b)).abs() <= tolerance)
}

fn median(values: &mut [Duration]) -> f64 {
    values.sort();
    values[values.len() / 2].as_secs_f64() * 1_000.0
}

#[test]
fn a_picture_arrives_two_frames_later_with_its_tag() {
    let gpu = gpu();
    let solid = Pipeline::fullscreen(&gpu, &spirv(SOLID_WGSL), 16, 0).expect("solid");
    let target = LayerTarget::new_single(&gpu, 320, 180);
    let mut readback = AsyncReadback::new(&gpu, 96, 54).expect("readback");
    assert_eq!(readback.size(), (96, 54));
    assert_eq!(READBACK_DELAY, 2);

    // Nothing recorded: nothing comes out, however many frames pass.
    assert!(readback.begin_frame().is_none());
    assert!(readback.begin_frame().is_none());

    // Frame 3 records picture 7; it is not there in frame 4 and it is in frame 5.
    assert!(readback.begin_frame().is_none());
    draw(&gpu, &solid, &target, [0.25, 0.5, 0.75, 1.0]);
    readback.record_target(&gpu, &target, 7).expect("record");
    assert!(readback.begin_frame().is_none(), "one frame later is too early");
    let picture = readback.begin_frame().expect("two frames later");
    assert_eq!(picture.tag, 7);
    assert_eq!((picture.width, picture.height), (96, 54));
    assert_eq!(picture.rgba.len(), 96 * 54 * 4);
    assert!(readback.begin_frame().is_none(), "a picture is handed out once");

    // Linear 0.25 / 0.5 / 0.75 comes out sRGB-encoded and opaque, on every pixel.
    for (index, rgba) in picture.rgba.as_chunks::<4>().0.iter().enumerate() {
        assert!(
            close(*rgba, [137, 188, 225, 255], 2),
            "pixel {index} is {rgba:?}"
        );
    }

    // A picture every frame: each one comes out exactly two frames later, in order.
    let mut taken = Vec::new();
    for frame in 0..10u64 {
        if let Some(picture) = readback.begin_frame() {
            taken.push((frame, picture.tag));
        }
        let level = frame as f32 / 10.0;
        draw(&gpu, &solid, &target, [level, level, level, 1.0]);
        readback.record_target(&gpu, &target, 100 + frame).expect("record");
    }
    assert_eq!(
        taken,
        (2..10).map(|frame| (frame, 100 + frame - 2)).collect::<Vec<_>>()
    );
    assert_eq!(readback.counts(), (11, 9));
    let messages = gpu.validation_messages();
    assert!(messages.is_empty(), "validation: {messages:?}");
}

#[test]
fn a_large_source_is_scaled_down_the_right_way_up() {
    let gpu = gpu();
    let split = Pipeline::fullscreen(&gpu, &spirv(SPLIT_WGSL), 16, 0).expect("split");
    // 1920x1080 into 96x54 is the two-step path; 160x90 into 96x54 is the direct one.
    for (width, height) in [(1920, 1080), (160, 90)] {
        let target = LayerTarget::new_single(&gpu, width, height);
        draw(&gpu, &split, &target, [1.0, 1.0, 0.0, 1.0]);
        let mut readback = AsyncReadback::new(&gpu, 96, 54).expect("readback");
        readback.begin_frame();
        readback.record_target(&gpu, &target, 1).expect("record");
        readback.begin_frame();
        let picture = readback.begin_frame().expect("a picture");
        let what = format!("{width}x{height}");

        // Left is red, right is green: not mirrored.
        let left = pixel(&picture.rgba, 96, 10, 27);
        let right = pixel(&picture.rgba, 96, 85, 27);
        assert!(left[0] > 250 && left[1] < 5, "{what}: left {left:?}");
        assert!(right[1] > 250 && right[0] < 5, "{what}: right {right:?}");
        // Top and bottom are the way the blocking readback has them: not flipped. (Which end
        // of `uv.y` is the top row is the vertex stage's business; the two readbacks must agree.)
        let reference = gpu.read_back(&target);
        let reference_top = pixel(&reference, width, width / 8, height / 20);
        let reference_bottom = pixel(&reference, width, width / 8, height - height / 20);
        let top = pixel(&picture.rgba, 96, 12, 2);
        let bottom = pixel(&picture.rgba, 96, 12, 51);
        assert!(
            (i32::from(top[2]) - i32::from(bottom[2])).abs() > 120,
            "{what}: top {top:?} bottom {bottom:?}"
        );
        assert_eq!(
            top[2] > bottom[2],
            reference_top[2] > reference_bottom[2],
            "{what}: top {top:?} bottom {bottom:?}, reference {reference_top:?} {reference_bottom:?}"
        );
        // The edge between the halves is where it belongs, within a pixel of the middle.
        let near_left = pixel(&picture.rgba, 96, 46, 27);
        let near_right = pixel(&picture.rgba, 96, 49, 27);
        assert!(near_left[0] > 250 && near_right[1] > 250, "{what}: the split moved");
    }
    let messages = gpu.validation_messages();
    assert!(messages.is_empty(), "validation: {messages:?}");
}

/// The point of the type: what a picture costs the thread that asks for it. Before: the blocking
/// `Gpu::read_back` (allocate, submit, wait, convert). After: recording a few commands into a
/// frame that is submitted anyway, and a memcpy two frames later. Run with `--release
/// --nocapture` for the numbers the README quotes.
#[test]
fn a_picture_costs_a_memcpy_not_a_wait() {
    let gpu = gpu();
    let solid = Pipeline::fullscreen(&gpu, &spirv(SOLID_WGSL), 16, 0).expect("solid");
    let stage = LayerTarget::new_single(&gpu, 1920, 1080);
    let small = LayerTarget::new_single(&gpu, 160, 90);
    draw(&gpu, &solid, &stage, [0.2, 0.4, 0.6, 1.0]);
    draw(&gpu, &solid, &small, [0.2, 0.4, 0.6, 1.0]);

    // Before: a blocking readback of a picture-sized target.
    let mut blocking = Vec::new();
    for _ in 0..30 {
        let started = Instant::now();
        let pixels = gpu.read_back(&small);
        blocking.push(started.elapsed());
        std::hint::black_box(pixels.len());
    }

    // After: 1920x1080 down to 160x90, every frame.
    let mut readback = AsyncReadback::new(&gpu, 160, 90).expect("readback");
    let allocations = gpu.allocation_count();
    let (mut record, mut take) = (Vec::new(), Vec::new());
    let mut pictures = 0;
    for frame in 0..200u64 {
        let started = Instant::now();
        let picture = readback.begin_frame();
        let took = started.elapsed();
        if let Some(picture) = picture {
            take.push(took);
            pictures += 1;
            std::hint::black_box(picture.rgba.len());
        }
        record.push(readback.record_target(&gpu, &stage, frame).expect("record"));
    }
    assert_eq!(pictures, 198);
    assert_eq!(
        gpu.allocation_count(),
        allocations,
        "a picture allocates nothing on the device"
    );

    let before = median(&mut blocking);
    let (record, take) = (median(&mut record), median(&mut take));
    println!(
        "readback of a 160x90 picture, {}: blocking Gpu::read_back median {before:.3} ms; async: record {record:.4} ms + take {take:.4} ms = {:.4} ms of the caller's time, no wait",
        gpu.adapter_name(),
        record + take
    );
    // The budget is for the optimised build the app ships; an unoptimised one gets slack for
    // its byte loop.
    let budget = if cfg!(debug_assertions) { 0.8 } else { 0.2 };
    assert!(
        record + take < budget,
        "an async picture cost {:.4} ms of CPU (budget {budget} ms)",
        record + take
    );
    let messages = gpu.validation_messages();
    assert!(messages.is_empty(), "validation: {messages:?}");
}
