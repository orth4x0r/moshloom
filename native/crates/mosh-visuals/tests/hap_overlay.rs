//! HAP overlay video with transparency, offscreen (Task 2.8's alpha follow-up).
//!
//! A HAP frame is BCn blocks: these tests build the blocks directly (there is no HAP encoder on
//! this machine), hand them to `OverlayLayer::set_hap_frame`, and read the layer target back.
//! Every pixel assertion therefore covers the whole path: block upload, the GPU's block decode,
//! HAP Q's YCoCg conversion, alpha premultiplication, and the compositor's "over".
//!
//! Row order: like `tests/overlay.rs`, a layer target is V-mirrored relative to the stage (`read_back`
//! row 0 is the audience's bottom row), and a HAP frame's first block row is its top. The pictures
//! asserted here are symmetric top to bottom, so their row order does not matter; the orientation
//! is pinned through the compositor's output in `tests/orientation.rs`.

use std::ops::Deref;
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use mosh_core::BlendMode;
use mosh_render::upload::BlockFormat;
use mosh_render::{
    Compositor, FrameRecorder, Gpu, GpuPreference, LayerTarget, OutputImage, Pipeline,
};
use mosh_visuals::overlay::OverlayLayer;

/// One GPU test at a time in this binary: `mosh_render::upload::allocation_count` is process-wide
/// and the allocation test compares it before and after a loop.
static GPU_LOCK: Mutex<()> = Mutex::new(());

struct TestGpu {
    gpu: Gpu,
    _guard: MutexGuard<'static, ()>,
}

impl Deref for TestGpu {
    type Target = Gpu;

    fn deref(&self) -> &Gpu {
        &self.gpu
    }
}

fn gpu() -> TestGpu {
    let guard = GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let gpu = Gpu::new(GpuPreference::Integrated)
        .or_else(|_| Gpu::new(GpuPreference::ByName(String::new())))
        .expect("a Vulkan device");
    println!(
        "adapter: {} | validation layers: {} | textureCompressionBC: {}",
        gpu.adapter_name(),
        if gpu.validation_enabled() {
            "on"
        } else {
            "unavailable"
        },
        gpu.texture_compression_bc()
    );
    TestGpu { gpu, _guard: guard }
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
        if gpu.validation_enabled() {
            "enabled"
        } else {
            "unavailable"
        }
    );
}

// ---------------------------------------------------------------------------------------------
// Fixtures: solid BCn blocks
// ---------------------------------------------------------------------------------------------

fn rgb565(rgb: [u8; 3]) -> u16 {
    (u16::from(rgb[0] >> 3) << 11) | (u16::from(rgb[1] >> 2) << 5) | u16::from(rgb[2] >> 3)
}

/// The 5-bit and 6-bit expansions a BC1 block decodes with, so an expectation computed here
/// matches the picture the GPU draws instead of the colour the encoder was asked for.
fn expand5(value: u8) -> u8 {
    let quantised = value >> 3;
    (quantised << 3) | (quantised >> 2)
}

fn expand6(value: u8) -> u8 {
    let quantised = value >> 2;
    (quantised << 2) | (quantised >> 4)
}

/// One solid 4x4 BC1 block: equal endpoints, every index 0.
fn bc1_block(rgb: [u8; 3]) -> [u8; 8] {
    let colour = rgb565(rgb);
    let (lo, hi) = (colour as u8, (colour >> 8) as u8);
    [lo, hi, lo, hi, 0, 0, 0, 0]
}

/// One solid 4x4 BC3 block: straight alpha, colour as BC1.
fn bc3_block(rgba: [u8; 4]) -> [u8; 16] {
    let mut block = [0u8; 16];
    block[0] = rgba[3];
    block[1] = rgba[3];
    block[8..].copy_from_slice(&bc1_block([rgba[0], rgba[1], rgba[2]]));
    block
}

/// One solid 4x4 BC4 block: two equal endpoints, every 3-bit index 0.
fn bc4_block(alpha: u8) -> [u8; 8] {
    [alpha, alpha, 0, 0, 0, 0, 0, 0]
}

/// A BC3 payload split into vertical zones at `x` (a multiple of the block width). `zones` is
/// `(block x range, rgba)`; every block gets one solid colour.
fn bc3_zones(width: u32, height: u32, zones: &[(std::ops::Range<u32>, [u8; 4])]) -> Vec<u8> {
    let blocks_x = width.div_ceil(4);
    let blocks_y = height.div_ceil(4);
    let mut payload = Vec::with_capacity((blocks_x * blocks_y * 16) as usize);
    for _ in 0..blocks_y {
        for block_x in 0..blocks_x {
            let rgba = zones
                .iter()
                .find(|(range, _)| range.contains(&block_x))
                .map(|(_, rgba)| *rgba)
                .expect("every block column is covered by a zone");
            payload.extend_from_slice(&bc3_block(rgba));
        }
    }
    payload
}

fn bc1_solid(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
    let blocks = width.div_ceil(4) * height.div_ceil(4);
    bc1_block(rgb).repeat(blocks as usize)
}

fn bc4_zones(width: u32, height: u32, zones: &[(std::ops::Range<u32>, u8)]) -> Vec<u8> {
    let blocks_x = width.div_ceil(4);
    let blocks_y = height.div_ceil(4);
    let mut payload = Vec::with_capacity((blocks_x * blocks_y * 8) as usize);
    for _ in 0..blocks_y {
        for block_x in 0..blocks_x {
            let alpha = zones
                .iter()
                .find(|(range, _)| range.contains(&block_x))
                .map(|(_, alpha)| *alpha)
                .expect("every block column is covered by a zone");
            payload.extend_from_slice(&bc4_block(alpha));
        }
    }
    payload
}

/// What the HAP Q shader should produce for a solid YCoCg block: FFmpeg's `ycocg2rgba(scaled)`
/// applied to the values a BC1 block actually decodes to.
fn ycocg_expected(rgba: [u8; 4]) -> [u8; 3] {
    let red = f32::from(expand5(rgba[0]));
    let green = f32::from(expand6(rgba[1]));
    let blue = f32::from(expand5(rgba[2]));
    let scale = (blue / 8.0).floor() + 1.0;
    let luma = f32::from(rgba[3]);
    let co = (red - 128.0) / scale;
    let cg = (green - 128.0) / scale;
    let clamp = |value: f32| value.clamp(0.0, 255.0).round() as u8;
    [
        clamp(luma + co - cg),
        clamp(luma + cg),
        clamp(luma - co - cg),
    ]
}

// ---------------------------------------------------------------------------------------------
// Helpers: a solid background layer and a composite
// ---------------------------------------------------------------------------------------------

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

fn color_uniform(color: [f32; 4]) -> Vec<u8> {
    color.iter().flat_map(|value| value.to_le_bytes()).collect()
}

/// A layer target filled with one opaque colour: the "background" of a composite.
fn solid_layer(gpu: &Gpu, width: u32, height: u32, color: [f32; 4]) -> LayerTarget {
    let pipeline = Pipeline::fullscreen(gpu, &spirv(SOLID_WGSL), 16, 0).expect("solid pipeline");
    let target = LayerTarget::new(gpu, width, height);
    let mut recorder = FrameRecorder::new(gpu);
    recorder
        .draw(&pipeline, &target, &color_uniform(color), &[])
        .expect("draw the background");
    gpu.wait_frame(recorder.submit());
    target
}

/// Composites `layers` bottom first and reads the result back as RGBA8.
fn composite(gpu: &Gpu, layers: &[(&LayerTarget, BlendMode, f32)], out: &OutputImage) -> Vec<u8> {
    let mut compositor = Compositor::new(gpu).expect("compositor");
    compositor
        .composite(layers, out)
        .expect("composite the layers");
    gpu.read_back_output(out)
}

fn pixel_at(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let offset = ((y * width + x) * 4) as usize;
    [
        pixels[offset],
        pixels[offset + 1],
        pixels[offset + 2],
        pixels[offset + 3],
    ]
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

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

/// HAP Alpha (`Hap5`, BC3): the transparent half of the frame must draw nothing, so the layer
/// underneath shows through. The middle zone is half transparent, which pins premultiplication:
/// 50% red over blue is (128, 0, 127), not (255, 0, 127) and not (128, 0, 255).
#[test]
fn hap5_alpha_draws_only_its_opaque_half() {
    let gpu = gpu();
    if !gpu.texture_compression_bc() {
        println!("skip: this device has no textureCompressionBC, so HAP cannot be uploaded");
        return;
    }
    let (width, height) = (64u32, 36u32);
    let zones = |opaque: [u8; 4], half: [u8; 4], clear: [u8; 4]| {
        vec![(0..5u32, opaque), (5..10u32, half), (10..16u32, clear)]
    };
    let payload = bc3_zones(
        width,
        height,
        &zones([255, 0, 0, 255], [255, 0, 0, 128], [255, 0, 0, 0]),
    );

    let mut overlay = OverlayLayer::new(&gpu).expect("overlay");
    assert!(overlay.hap_supported(), "the device reported BC support");
    overlay
        .set_hap_frame(&gpu, 0, width, height, BlockFormat::Bc3, &payload, None)
        .expect("upload the HAP frame");
    let layer = LayerTarget::new(&gpu, width, height);
    {
        let mut recorder = FrameRecorder::new(&gpu);
        overlay
            .draw(&mut recorder, 0, &layer)
            .expect("draw the HAP layer");
        let timings = gpu.wait_frame(recorder.submit());
        let names: Vec<&str> = timings
            .per_pass_ms
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        assert_eq!(
            names,
            vec!["overlay.upload", "overlay.hap"],
            "the HAP copy and cover pass are timed as frame passes"
        );
    }

    // The layer target holds premultiplied colour: the transparent zone is (0,0,0,0), and the
    // half-transparent one is red * 0.5.
    let layer_pixels = gpu.read_back(&layer);
    assert_pixel(
        pixel_at(&layer_pixels, width, 8, 8),
        [255, 0, 0, 255],
        3,
        "opaque zone on the layer",
    );
    assert_pixel(
        pixel_at(&layer_pixels, width, 36, 8),
        [128, 0, 0, 128],
        6,
        "half-transparent zone is premultiplied",
    );
    assert_pixel(
        pixel_at(&layer_pixels, width, 60, 8),
        [0, 0, 0, 0],
        0,
        "transparent zone draws nothing",
    );

    // Over an opaque blue background: the same three zones, and the cut-out shows the blue.
    let background = solid_layer(&gpu, width, height, [0.0, 0.0, 1.0, 1.0]);
    let out = OutputImage::new(&gpu, width, height);
    let pixels = composite(
        &gpu,
        &[
            (&background, BlendMode::Normal, 1.0),
            (&layer, BlendMode::Normal, 1.0),
        ],
        &out,
    );

    for y in [2u32, 17, 31] {
        assert_pixel(
            pixel_at(&pixels, width, 8, y),
            [255, 0, 0, 255],
            4,
            "opaque half over the background",
        );
        assert_pixel(
            pixel_at(&pixels, width, 36, y),
            [128, 0, 127, 255],
            8,
            "half-transparent red over blue",
        );
        assert_pixel(
            pixel_at(&pixels, width, 60, y),
            [0, 0, 255, 255],
            4,
            "transparent half shows the background",
        );
    }

    println!(
        "HAP5 alpha 64x36: layer (opaque {:?}, half {:?}, clear {:?}) -> composite (opaque {:?}, \
         half {:?}, clear/bg {:?})",
        pixel_at(&layer_pixels, width, 8, 8),
        pixel_at(&layer_pixels, width, 36, 8),
        pixel_at(&layer_pixels, width, 60, 8),
        pixel_at(&pixels, width, 8, 8),
        pixel_at(&pixels, width, 36, 8),
        pixel_at(&pixels, width, 60, 8)
    );
    assert_no_validation_messages(&gpu);
}

/// HAP Q (`HapY`, scaled YCoCg in a BC3 texture): one solid block per column with known
/// YCoCg values, checked against the arithmetic FFmpeg's decoder uses.
#[test]
fn hap_q_ycocg_converts_to_rgb() {
    let gpu = gpu();
    if !gpu.texture_compression_bc() {
        println!("skip: this device has no textureCompressionBC, so HAP cannot be uploaded");
        return;
    }
    let (width, height) = (32u32, 16u32);
    // Two blocks per row, each a solid YCoCg value: (Co, Cg, scale byte, Y).
    let colours = [[178u8, 108, 0, 100], [128, 128, 0, 128]];
    let mut payload = Vec::new();
    for _ in 0..height.div_ceil(4) {
        for block_x in 0..width.div_ceil(4) {
            let colour = colours[(block_x % 2) as usize];
            payload.extend_from_slice(&bc3_block(colour));
        }
    }
    let mut overlay = OverlayLayer::new(&gpu).expect("overlay");
    overlay
        .set_hap_frame(
            &gpu,
            0,
            width,
            height,
            BlockFormat::Bc3YCoCg,
            &payload,
            None,
        )
        .expect("upload HAP Q");
    let layer = LayerTarget::new(&gpu, width, height);
    {
        let mut recorder = FrameRecorder::new(&gpu);
        overlay.draw(&mut recorder, 0, &layer).expect("draw HAP Q");
        gpu.wait_frame(recorder.submit());
    }
    let pixels = gpu.read_back(&layer);

    for (index, colour) in colours.iter().enumerate() {
        let expected = ycocg_expected(*colour);
        // Every block is 4 pixels wide, so block 0 starts at x = 0.
        let got = pixel_at(&pixels, width, index as u32 * 4 + 1, 8);
        assert_pixel(
            got,
            [expected[0], expected[1], expected[2], 255],
            4,
            &format!("YCoCg block {colour:?}"),
        );
    }
    println!(
        "HAP Q 32x16: {:?} -> {:?}, {:?} -> {:?}",
        colours[0],
        ycocg_expected(colours[0]),
        colours[1],
        ycocg_expected(colours[1])
    );
    assert_no_validation_messages(&gpu);
}

/// HAP Q Alpha (`HapM`): the colour plane is YCoCg, the alpha is a separate BC4 texture. The
/// transparent half must composite to the background, and both planes must upload as blocks.
#[test]
fn hap_q_alpha_uses_the_separate_bc4_plane() {
    let gpu = gpu();
    if !gpu.texture_compression_bc() {
        println!("skip: this device has no textureCompressionBC, so HAP cannot be uploaded");
        return;
    }
    let (width, height) = (64u32, 36u32);
    // A YCoCg grey (Y = 200, no chroma) everywhere: the picture is the same across the frame, so
    // what the assertion sees is the alpha plane alone.
    let colour =
        bc3_block([128, 128, 0, 200]).repeat((width.div_ceil(4) * height.div_ceil(4)) as usize);
    let alpha = bc4_zones(width, height, &[(0..8u32, 255), (8..16u32, 0)]);

    let mut overlay = OverlayLayer::new(&gpu).expect("overlay");
    overlay
        .set_hap_frame(
            &gpu,
            0,
            width,
            height,
            BlockFormat::Bc3YCoCg,
            &colour,
            Some(&alpha),
        )
        .expect("upload HAP Q Alpha");
    let layer = LayerTarget::new(&gpu, width, height);
    {
        let mut recorder = FrameRecorder::new(&gpu);
        overlay
            .draw(&mut recorder, 0, &layer)
            .expect("draw HAP Q Alpha");
        gpu.wait_frame(recorder.submit());
    }
    let layer_pixels = gpu.read_back(&layer);
    let grey = ycocg_expected([128, 128, 0, 200]);
    assert_pixel(
        pixel_at(&layer_pixels, width, 8, 8),
        [grey[0], grey[1], grey[2], 255],
        4,
        "opaque half of the alpha plane",
    );
    assert_pixel(
        pixel_at(&layer_pixels, width, 60, 8),
        [0, 0, 0, 0],
        0,
        "transparent half of the alpha plane",
    );

    let background = solid_layer(&gpu, width, height, [0.0, 1.0, 0.0, 1.0]);
    let out = OutputImage::new(&gpu, width, height);
    let pixels = composite(
        &gpu,
        &[
            (&background, BlendMode::Normal, 1.0),
            (&layer, BlendMode::Normal, 1.0),
        ],
        &out,
    );
    assert_pixel(
        pixel_at(&pixels, width, 8, 20),
        [grey[0], grey[1], grey[2], 255],
        4,
        "opaque half over green",
    );
    assert_pixel(
        pixel_at(&pixels, width, 60, 20),
        [0, 255, 0, 255],
        4,
        "transparent half shows green",
    );
    println!(
        "HAP Q Alpha 64x36: colour {:?} + BC4 alpha, layer {:?} / {:?}, composite {:?} / {:?}",
        [128u8, 128, 0, 200],
        pixel_at(&layer_pixels, width, 8, 8),
        pixel_at(&layer_pixels, width, 60, 8),
        pixel_at(&pixels, width, 8, 20),
        pixel_at(&pixels, width, 60, 20)
    );
    assert_no_validation_messages(&gpu);
}

/// A `HapA` frame (RGTC1, alpha only) has no colour plane: it draws a white silhouette.
#[test]
fn hap_alpha_only_draws_a_white_silhouette() {
    let gpu = gpu();
    if !gpu.texture_compression_bc() {
        println!("skip: this device has no textureCompressionBC, so HAP cannot be uploaded");
        return;
    }
    let (width, height) = (32u32, 16u32);
    let alpha = bc4_zones(width, height, &[(0..4u32, 255), (4..8u32, 0)]);
    let mut overlay = OverlayLayer::new(&gpu).expect("overlay");
    overlay
        .set_hap_frame(&gpu, 0, width, height, BlockFormat::Bc4, &alpha, None)
        .expect("upload HapA");
    let layer = LayerTarget::new(&gpu, width, height);
    {
        let mut recorder = FrameRecorder::new(&gpu);
        overlay.draw(&mut recorder, 0, &layer).expect("draw HapA");
        gpu.wait_frame(recorder.submit());
    }
    let pixels = gpu.read_back(&layer);
    assert_pixel(
        pixel_at(&pixels, width, 4, 8),
        [255, 255, 255, 255],
        2,
        "opaque half of a HapA frame",
    );
    assert_pixel(
        pixel_at(&pixels, width, 28, 8),
        [0, 0, 0, 0],
        0,
        "transparent half of a HapA frame",
    );
    println!(
        "HapA 32x16: {:?} / {:?}",
        pixel_at(&pixels, width, 4, 8),
        pixel_at(&pixels, width, 28, 8)
    );
    assert_no_validation_messages(&gpu);
}

/// The cost side of the budget: a 1080p HAP layer allocates nothing after its first frame, its
/// recorded CPU work stays under the 0.2 ms the BGRA overlay path is held to, and it holds a
/// fraction of the GPU memory a BGRA overlay of the same size does.
#[test]
fn hap_1080p_memory_and_cpu_per_frame() {
    let gpu = gpu();
    if !gpu.texture_compression_bc() {
        println!("skip: this device has no textureCompressionBC, so HAP cannot be uploaded");
        return;
    }
    let (width, height) = (1920u32, 1080u32);
    let payload = bc1_solid(width, height, [200, 120, 40]);

    // First frame: allocations happen here.
    let mut overlay = OverlayLayer::new(&gpu).expect("overlay");
    let dst = LayerTarget::new(&gpu, width, height);
    let before_hap = gpu.allocated_bytes();
    overlay
        .set_hap_frame(&gpu, 0, width, height, BlockFormat::Bc1, &payload, None)
        .expect("first HAP upload");
    {
        let mut recorder = FrameRecorder::new(&gpu);
        overlay
            .draw(&mut recorder, 0, &dst)
            .expect("first HAP draw");
        gpu.wait_frame(recorder.submit());
    }
    let hap_bytes = gpu.allocated_bytes() - before_hap;

    // A BGRA overlay of the same size, for the comparison: 1920x1080 x 4 bytes per image and per
    // staging buffer.
    let mut bgra_overlay = OverlayLayer::new(&gpu).expect("overlay");
    let bgra = vec![0u8; (width * height * 4) as usize];
    let before_bgra = gpu.allocated_bytes();
    bgra_overlay
        .set_frame(&gpu, 0, width, height, &bgra)
        .expect("first BGRA upload");
    {
        let mut recorder = FrameRecorder::new(&gpu);
        bgra_overlay
            .draw(&mut recorder, 0, &dst)
            .expect("first BGRA draw");
        gpu.wait_frame(recorder.submit());
    }
    let bgra_bytes = gpu.allocated_bytes() - before_bgra;
    // Everything the two overlays allocated happened above; the loop must add nothing.
    let allocations_after_first = mosh_render::upload::allocation_count();

    let frames = 60;
    let mut upload_samples = Vec::with_capacity(frames);
    let mut record_samples = Vec::with_capacity(frames);
    for _ in 0..frames {
        let start = Instant::now();
        overlay
            .set_hap_frame(&gpu, 0, width, height, BlockFormat::Bc1, &payload, None)
            .expect("upload");
        upload_samples.push(start.elapsed().as_secs_f64() * 1000.0);

        let start = Instant::now();
        let mut recorder = FrameRecorder::new(&gpu);
        let claim = start.elapsed().as_secs_f64() * 1000.0;
        overlay.draw(&mut recorder, 0, &dst).expect("draw");
        record_samples.push(start.elapsed().as_secs_f64() * 1000.0 - claim);
        gpu.wait_frame(recorder.submit());
    }
    let allocations = mosh_render::upload::allocation_count();
    assert_eq!(
        allocations, allocations_after_first,
        "60 uploads of one HAP size must reuse the image, the staging buffers and the pipeline"
    );

    record_samples.sort_by(f64::total_cmp);
    let record_min = record_samples[0];
    let record_median = record_samples[record_samples.len() / 2];
    upload_samples.sort_by(f64::total_cmp);
    let upload_min = upload_samples[0];
    let upload_median = upload_samples[upload_samples.len() / 2];

    let pixels = gpu.read_back(&dst);
    assert_pixel(
        pixel_at(&pixels, width, 16, 16),
        [expand5(200), expand6(120), expand5(40), 255],
        4,
        "1080p HAP1 frame",
    );

    println!(
        "1920x1080 HAP1: {} bytes of BC1 per frame, GPU memory {:.2} MB vs {:.2} MB for BGRA \
         ({:.0}%); upload memcpy {upload_min:.3} / {upload_median:.3} ms min/median, record \
         (copy + cover pass, no fence) {record_min:.3} / {record_median:.3} ms, allocations \
         {} -> {allocations}",
        payload.len(),
        hap_bytes as f64 / (1024.0 * 1024.0),
        bgra_bytes as f64 / (1024.0 * 1024.0),
        100.0 * hap_bytes as f64 / bgra_bytes as f64,
        allocations_after_first,
    );
    assert!(
        hap_bytes * 4 < bgra_bytes,
        "a 1080p BC1 layer must cost well under a quarter of the BGRA one: {hap_bytes} vs \
         {bgra_bytes} bytes"
    );
    assert!(
        record_min < 1.0,
        "the HAP layer's own CPU cost per frame must stay well inside the 2 ms render-thread \
         budget, min {record_min:.3} ms (median {record_median:.3} ms; the minimum is the pass's \
         own cost, the median moves with whatever else the machine is doing)"
    );
    assert_no_validation_messages(&gpu);
}
