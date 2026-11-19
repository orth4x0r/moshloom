//! Offscreen Vulkan tests for `mosh-render` (Task 1.7).
//!
//! No window is opened: every test renders into layer targets and reads the result back. The
//! one-ish GPU test that pins the reference machine's adapter is `#[ignore]`d, because another
//! machine legitimately has no integrated Radeon.
//!
//! Test shaders are WGSL, compiled to SPIR-V with `naga` here — the same toolchain mosh-shaders
//! uses for the real scenes.

use std::time::{Duration, Instant};

use mosh_core::show::BlendMode;
use mosh_render::{
    Compositor, FlashLimiter, FrameRecorder, Gpu, GpuPreference, LayerTarget, OutputImage, Pipeline,
    RenderError, SwapchainAction, SwapchainPolicy, MAX_UNIFORM_BYTES,
};

/// A fragment shader that writes its uniform `color`.
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

/// A feedback shader: sample the previous frame and add `amount`.
const FEEDBACK_WGSL: &str = r#"
struct Params {
    amount: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var source_sampler: sampler;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(source, source_sampler, uv) + params.amount;
}
"#;

/// A shader whose uniform block is the crate's whole cap: 256 `vec4`s, 4096 bytes. It returns the
/// last one, so a draw proves the end of a full-size block actually reached the GPU.
const BIG_UNIFORM_WGSL: &str = r#"
struct Params {
    values: array<vec4<f32>, 256>,
};

@group(0) @binding(0) var<uniform> params: Params;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return params.values[255];
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

/// One device per test. The reference machine's integrated GPU is the default; a machine without
/// one (CI, a VM) falls back to whatever single GPU it has, so only the adapter-name test is
/// machine-specific.
fn gpu() -> Gpu {
    let gpu = Gpu::new(GpuPreference::Integrated)
        .or_else(|_| Gpu::new(GpuPreference::ByName(String::new())))
        .expect("a Vulkan device");
    println!(
        "adapter: {} | validation layers: {} | timestamp period: {} ns",
        gpu.adapter_name(),
        if gpu.validation_enabled() { "on" } else { "unavailable" },
        gpu.timestamp_period_ns()
    );
    gpu
}

fn assert_no_validation_messages(gpu: &Gpu) {
    let messages = gpu.validation_messages();
    assert!(
        messages.is_empty(),
        "{} validation message(s):\n{}",
        messages.len(),
        messages.join("\n")
    );
    println!(
        "validation: {} message(s), layer {}",
        messages.len(),
        if gpu.validation_enabled() { "enabled" } else { "unavailable" }
    );
}

fn color_uniform(color: [f32; 4]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16);
    for value in color {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn first_pixel(pixels: &[u8]) -> [u8; 4] {
    [pixels[0], pixels[1], pixels[2], pixels[3]]
}

fn assert_pixel(pixel: [u8; 4], expected: [u8; 4], tolerance: i32, what: &str) {
    for channel in 0..4 {
        let difference = i32::from(pixel[channel]) - i32::from(expected[channel]);
        assert!(
            difference.abs() <= tolerance,
            "{what}: channel {channel} is {}, expected {} +-{tolerance} (full pixel {pixel:?})",
            pixel[channel],
            expected[channel]
        );
    }
}

/// Every pixel has `expected`, within `tolerance` (a half-float round trip and an 8-bit
/// conversion are not exact).
fn assert_all_pixels(pixels: &[u8], expected: [u8; 4], tolerance: i32, what: &str) {
    assert_eq!(pixels.len() % 4, 0, "RGBA8 readback is 4 bytes per pixel");
    for (index, pixel) in pixels.chunks(4).enumerate() {
        assert_pixel(
            [pixel[0], pixel[1], pixel[2], pixel[3]],
            expected,
            tolerance,
            &format!("{what} at pixel {index}"),
        );
    }
}

#[test]
#[ignore = "reference machine only: it has an integrated AMD Radeon and a discrete RX 6700S"]
fn picks_integrated_gpu_by_default() {
    let integrated = Gpu::new(GpuPreference::Integrated).expect("integrated GPU");
    println!("integrated: {}", integrated.adapter_name());
    assert!(
        integrated.adapter_name().contains("Radeon(TM) Graphics"),
        "the default must be the integrated GPU, got {}",
        integrated.adapter_name()
    );

    let discrete = Gpu::new(GpuPreference::Discrete).expect("discrete GPU");
    println!("discrete: {}", discrete.adapter_name());
    assert!(
        discrete.adapter_name().contains("RX 6700S"),
        "asking for the discrete GPU must find it, got {}",
        discrete.adapter_name()
    );
    assert_ne!(integrated.adapter_name(), discrete.adapter_name());

    let by_name = Gpu::new(GpuPreference::ByName("Nonexistent GPU".into()));
    assert!(
        by_name.is_err(),
        "a name no GPU matches must fail instead of silently picking one"
    );
    assert_no_validation_messages(&integrated);
    assert_no_validation_messages(&discrete);
}

#[test]
fn solid_color_shader_reads_back() {
    let gpu = gpu();
    let pipeline = Pipeline::fullscreen(&gpu, &spirv(SOLID_WGSL), 16, 0).expect("solid pipeline");
    let target = LayerTarget::new(&gpu, 8, 8);

    let mut recorder = FrameRecorder::new(&gpu);
    recorder
        .draw(&pipeline, &target, &color_uniform([0.25, 0.5, 0.75, 1.0]), &[])
        .expect("draw solid");
    let timings = gpu.wait_frame(recorder.submit());

    let pixels = gpu.read_back(&target);
    assert_eq!(pixels.len(), 8 * 8 * 4);
    assert_all_pixels(&pixels, [64, 128, 191, 255], 1, "solid colour");
    println!(
        "solid colour readback: {:?}, frame {:.3} ms",
        first_pixel(&pixels),
        timings.frame_ms
    );
    assert_no_validation_messages(&gpu);
}

/// The cap is 4096 bytes because the text layer's block is 1008 and one more field would not fit
/// in 1024. A pipeline at the cap must create, take a full-size uniform slice out of the ring and
/// draw; one byte past it must still be `UniformTooLarge`, not a silent truncation.
#[test]
fn uniform_block_at_the_cap_creates_and_draws() {
    assert_eq!(
        MAX_UNIFORM_BYTES, 4096,
        "the cap the layer crates are written against"
    );

    // What the cap costs: the uniform ring is the only allocation a fresh device makes, and its
    // stride is `align_up(MAX_UNIFORM_BYTES, minUniformBufferOffsetAlignment)`. Vulkan guarantees
    // that alignment is at most 256, so 4096 is the stride on every device: 4096 B x 256 draws x
    // 2 frame slots = 2 MiB, where 1024 used to cost 512 KiB. It is allocated once, not per frame.
    let fresh = gpu();
    assert_eq!(
        fresh.allocation_count(),
        1,
        "the uniform ring is a fresh device's only allocation"
    );
    assert!(
        fresh.allocated_bytes() >= MAX_UNIFORM_BYTES * 2 * 256,
        "the ring holds the cap for all 256 draws of both frame slots: {} B",
        fresh.allocated_bytes()
    );
    println!(
        "uniform ring: {} B (the 1024-byte cap cost 512 KiB)",
        fresh.allocated_bytes()
    );

    let gpu = fresh;
    let size = MAX_UNIFORM_BYTES as usize;
    let pipeline = Pipeline::fullscreen(&gpu, &spirv(BIG_UNIFORM_WGSL), size, 0)
        .expect("a pipeline with a 4096-byte uniform block");
    let target = LayerTarget::new(&gpu, 8, 8);

    // The shader reads the block's last vec4; everything before it is padding that a real layer
    // fills with its own fields.
    let mut uniforms = vec![0u8; size];
    let last = size - 16;
    for (index, value) in [0.25f32, 0.5, 0.75, 1.0].iter().enumerate() {
        uniforms[last + index * 4..last + index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }

    let mut recorder = FrameRecorder::new(&gpu);
    recorder
        .draw(&pipeline, &target, &uniforms, &[])
        .expect("draw with a 4096-byte uniform block");
    let timings = gpu.wait_frame(recorder.submit());

    let pixels = gpu.read_back(&target);
    assert_all_pixels(&pixels, [64, 128, 191, 255], 1, "4096-byte uniform");
    println!(
        "4096-byte uniform readback: {:?}, frame {:.3} ms, live device memory {} B in {} allocations",
        first_pixel(&pixels),
        timings.frame_ms,
        gpu.allocated_bytes(),
        gpu.allocation_count()
    );
    assert_no_validation_messages(&gpu);

    let beyond = Pipeline::fullscreen(&gpu, &spirv(SOLID_WGSL), size + 16, 0);
    match beyond {
        Err(RenderError::UniformTooLarge { max, .. }) => assert_eq!(max, MAX_UNIFORM_BYTES),
        Err(other) => panic!("wrong error for a block past the cap: {other:?}"),
        Ok(_) => panic!("a uniform block past the cap must be refused, not truncated"),
    }
}

#[test]
fn feedback_pingpong_accumulates() {
    let gpu = gpu();
    let pipeline = Pipeline::fullscreen(&gpu, &spirv(FEEDBACK_WGSL), 16, 1).expect("feedback");
    let target = LayerTarget::new(&gpu, 8, 8);

    // The target starts cleared to zero, so five frames of +0.1 land on 0.5.
    for frame in 0..5 {
        let mut recorder = FrameRecorder::new(&gpu);
        recorder
            .draw(&pipeline, &target, &color_uniform([0.1, 0.1, 0.1, 0.0]), &[&target])
            .unwrap_or_else(|error| panic!("feedback draw {frame}: {error}"));
        gpu.wait_frame(recorder.submit());
    }

    let pixels = gpu.read_back(&target);
    // 0.5 * 255 = 127.5; half floats store 0.0999756 for 0.1, so allow +-3.
    assert_all_pixels(&pixels, [128, 128, 128, 0], 3, "five accumulated frames");
    println!("feedback after 5 frames: {:?}", first_pixel(&pixels));
    assert_no_validation_messages(&gpu);
}

#[test]
fn composite_add_and_normal() {
    let gpu = gpu();
    let solid = Pipeline::fullscreen(&gpu, &spirv(SOLID_WGSL), 16, 0).expect("solid pipeline");
    let red = LayerTarget::new(&gpu, 8, 8);
    let green = LayerTarget::new(&gpu, 8, 8);
    let out = OutputImage::new(&gpu, 8, 8);

    let mut recorder = FrameRecorder::new(&gpu);
    recorder
        .draw(&solid, &red, &color_uniform([1.0, 0.0, 0.0, 1.0]), &[])
        .expect("draw red");
    recorder
        .draw(&solid, &green, &color_uniform([0.0, 1.0, 0.0, 1.0]), &[])
        .expect("draw green");
    gpu.wait_frame(recorder.submit());

    let mut compositor = Compositor::new(&gpu).expect("compositor");

    compositor
        .composite(&[(&red, BlendMode::Normal, 1.0)], &out)
        .expect("composite one layer");
    let one = gpu.read_back_output(&out);
    assert_all_pixels(&one, [255, 0, 0, 255], 1, "one layer, normal");

    compositor
        .composite(
            &[(&red, BlendMode::Normal, 1.0), (&green, BlendMode::Normal, 0.5)],
            &out,
        )
        .expect("composite normal");
    let normal = gpu.read_back_output(&out);
    assert_all_pixels(&normal, [128, 128, 0, 255], 2, "green at 0.5 over red");

    compositor
        .composite(
            &[(&red, BlendMode::Normal, 1.0), (&green, BlendMode::Add, 1.0)],
            &out,
        )
        .expect("composite add");
    let add = gpu.read_back_output(&out);
    assert_all_pixels(&add, [255, 255, 0, 255], 1, "green added to red");

    // Luma key discards dark texels: a black layer must leave the red alone.
    let black = LayerTarget::new(&gpu, 8, 8);
    let mut recorder = FrameRecorder::new(&gpu);
    recorder
        .draw(&solid, &black, &color_uniform([0.0, 0.0, 0.0, 1.0]), &[])
        .expect("draw black");
    gpu.wait_frame(recorder.submit());
    compositor
        .composite(
            &[(&red, BlendMode::Normal, 1.0), (&black, BlendMode::LumaKey, 1.0)],
            &out,
        )
        .expect("composite luma key");
    let keyed = gpu.read_back_output(&out);
    assert_all_pixels(&keyed, [255, 0, 0, 255], 1, "black luma-keyed away");

    println!(
        "composite: one {:?} normal {:?} add {:?} lumakey {:?}",
        first_pixel(&one),
        first_pixel(&normal),
        first_pixel(&add),
        first_pixel(&keyed)
    );
    assert_no_validation_messages(&gpu);
}

#[test]
fn two_frames_in_flight_submit_before_waiting() {
    let gpu = gpu();
    let pipeline = Pipeline::fullscreen(&gpu, &spirv(SOLID_WGSL), 16, 0)
        .expect("solid pipeline")
        .with_name("solid");
    let first_target = LayerTarget::new(&gpu, 8, 8);
    let second_target = LayerTarget::new(&gpu, 8, 8);

    // Two frames submitted before either is waited on: the two frame slots and the one timeline
    // semaphore have to let both run.
    let mut first = FrameRecorder::new(&gpu);
    first
        .draw(&pipeline, &first_target, &color_uniform([1.0, 0.0, 0.0, 1.0]), &[])
        .expect("frame A draw");
    let ticket_a = first.submit();

    let mut second = FrameRecorder::new(&gpu);
    second
        .draw(&pipeline, &second_target, &color_uniform([0.0, 0.0, 1.0, 1.0]), &[])
        .expect("frame B draw");
    let ticket_b = second.submit();

    let timings_a = gpu.wait_frame(ticket_a);
    let timings_b = gpu.wait_frame(ticket_b);

    assert_all_pixels(&gpu.read_back(&first_target), [255, 0, 0, 255], 1, "frame A");
    assert_all_pixels(&gpu.read_back(&second_target), [0, 0, 255, 255], 1, "frame B");
    assert_eq!(timings_a.per_pass_ms.len(), 1);
    assert_eq!(timings_b.per_pass_ms.len(), 1);
    println!(
        "two frames in flight: {:.4} ms then {:.4} ms",
        timings_a.frame_ms, timings_b.frame_ms
    );
    assert_no_validation_messages(&gpu);
}

#[test]
fn flash_limiter_400ms() {
    let mut limiter = FlashLimiter::new();
    let start = Instant::now();
    assert!(limiter.allow(start), "the first flash is always allowed");
    assert!(!limiter.allow(start + Duration::from_millis(1)));
    assert!(!limiter.allow(start + Duration::from_millis(399)));
    assert!(limiter.allow(start + Duration::from_millis(400)));
    assert!(!limiter.allow(start + Duration::from_millis(799)));
    assert!(limiter.allow(start + Duration::from_millis(800)));

    // A 16 flashes per second strobe for one second: two more starts, not sixteen.
    let mut limiter = FlashLimiter::new();
    assert!(limiter.allow(start));
    let mut starts = 0;
    for step in 1..=100u64 {
        if limiter.allow(start + Duration::from_millis(step * 10)) {
            starts += 1;
        }
    }
    assert_eq!(starts, 2, "400 ms apart at 16 flashes/s is at most 3 per second");
    println!("flash limiter: {starts} starts in 1000 ms of a 16 Hz strobe");
}

#[test]
fn timestamps_reported() {
    let gpu = gpu();
    let pipeline = Pipeline::fullscreen(&gpu, &spirv(SOLID_WGSL), 16, 0)
        .expect("solid pipeline")
        .with_name("solid");
    let target = LayerTarget::new(&gpu, 64, 64);

    let mut recorder = FrameRecorder::new(&gpu);
    for _ in 0..3 {
        recorder
            .draw(&pipeline, &target, &color_uniform([0.5, 0.5, 0.5, 1.0]), &[])
            .expect("draw");
    }
    let timings = gpu.wait_frame(recorder.submit());

    assert_eq!(timings.per_pass_ms.len(), 3, "one timing per recorded pass");
    for (name, milliseconds) in &timings.per_pass_ms {
        assert!(milliseconds.is_finite(), "{name} is not finite");
        assert!(*milliseconds >= 0.0, "{name} is negative");
    }
    assert!(
        timings.frame_ms.is_finite() && timings.frame_ms > 0.0,
        "frame time must be positive on a GPU with timestamps, got {}",
        timings.frame_ms
    );
    let sum: f32 = timings.per_pass_ms.iter().map(|(_, ms)| *ms).sum();
    assert!(
        timings.frame_ms + 0.05 >= sum,
        "frame span {} ms is shorter than its passes {} ms",
        timings.frame_ms,
        sum
    );
    println!(
        "timestamps: frame {:.4} ms, passes {:?} (period {} ns)",
        timings.frame_ms,
        timings
            .per_pass_ms
            .iter()
            .map(|(name, ms)| format!("{name} {ms:.4} ms"))
            .collect::<Vec<_>>(),
        gpu.timestamp_period_ns()
    );
    assert_no_validation_messages(&gpu);
}

#[test]
fn surface_lost_recreates_swapchain() {
    // No window and no GPU: the recovery policy is the part that can be tested offscreen, and
    // it is the part the stage runs in the field.
    let mut policy = SwapchainPolicy::new();
    assert!(!policy.is_valid());
    assert!(policy.complete_recreate([1920, 1080]));
    assert!(policy.is_valid());
    assert_eq!(policy.recreations(), 1);

    // An out-of-date acquire: rebuild the swapchain, not the surface.
    assert_eq!(
        policy.acquire_result(Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR)),
        SwapchainAction::Recreate {
            rebuild_surface: false
        }
    );
    assert!(policy.needs_recreate());
    assert!(!policy.is_valid());
    assert!(!policy.begin_recreate(), "the surface is still fine");
    assert!(policy.complete_recreate([1920, 1080]));
    assert!(policy.is_valid());
    assert_eq!(policy.recreations(), 2);
    assert_eq!(policy.surface_recreations(), 0);
    assert_eq!(
        policy.acquire_result(Ok(())),
        SwapchainAction::Continue,
        "the frame after a rebuild acquires normally"
    );

    // A lost surface (monitor unplugged mid-set): the surface must be rebuilt as well.
    assert_eq!(
        policy.present_result(Err(ash::vk::Result::ERROR_SURFACE_LOST_KHR)),
        SwapchainAction::Recreate {
            rebuild_surface: true
        }
    );
    assert!(policy.needs_recreate());
    assert!(policy.begin_recreate(), "the surface must be rebuilt");
    assert!(policy.complete_recreate([1920, 1080]));
    assert_eq!(policy.surface_recreations(), 1);
    assert!(policy.take_surface_lost(), "the stage reports the loss once");
    assert!(!policy.take_surface_lost(), "and only once");

    // Suboptimal present: keep going now, rebuild before the next acquire.
    assert_eq!(
        policy.present_result(Err(ash::vk::Result::SUBOPTIMAL_KHR)),
        SwapchainAction::Recreate {
            rebuild_surface: false
        }
    );
    assert!(!policy.begin_recreate(), "the surface survives a suboptimal present");
    assert!(policy.complete_recreate([1920, 1080]));

    // Minimized, or the monitor went away: no 0x0 swapchain, and no panic.
    assert_eq!(policy.resize(0, 0), SwapchainAction::SkipFrame);
    assert!(!policy.is_valid());
    assert!(policy.needs_recreate());
    assert_eq!(
        policy.acquire_result(Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR)),
        SwapchainAction::Recreate {
            rebuild_surface: false
        }
    );
    assert!(!policy.begin_recreate(), "a resize does not lose the surface");
    assert!(
        !policy.complete_recreate([0, 0]),
        "a zero extent is not a swapchain"
    );
    assert!(!policy.is_valid());
    assert!(policy.needs_recreate(), "and the next frame tries again");

    // A resize that arrives with a real size, then a hundred fault/recover cycles: still sane.
    assert_eq!(
        policy.resize(2560, 1440),
        SwapchainAction::Recreate {
            rebuild_surface: false
        }
    );
    assert!(!policy.begin_recreate(), "a resize does not lose the surface");
    assert!(policy.complete_recreate([2560, 1440]));
    assert_eq!(policy.extent(), [2560, 1440]);
    let before = policy.recreations();
    for _ in 0..100 {
        policy.present_result(Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR));
        assert!(!policy.begin_recreate());
        assert!(policy.complete_recreate([2560, 1440]));
    }
    assert_eq!(policy.recreations(), before + 100);
    assert!(policy.is_valid());

    println!(
        "swapchain policy: {} recreations ({} after a lost surface), {} skipped frames",
        policy.recreations(),
        policy.surface_recreations(),
        policy.skipped_frames()
    );
}
