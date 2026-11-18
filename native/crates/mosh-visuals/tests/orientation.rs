//! Orientation through the *real output path*: a source's top row must reach the top of the
//! picture an audience sees. Owner: the overlay / HAP / camera / text sources.
//!
//! Every visualizer draws into a `LayerTarget`, and a layer target is V-mirrored relative to the
//! stage: `Compositor` samples a layer with `fullscreen.vert`'s uv, whose y is 0 at the top of the
//! output image and 1 at its bottom, while a layer target's row 0 is that output's *bottom* row.
//! The stage's present pass adds no second flip (`Stage::present_image` composites straight into
//! the swapchain). So a test that reads a **layer** cannot say whether the audience sees a picture
//! upright — an upside-down source and a correct one look the same there, only mirrored.
//!
//! These tests follow the path a frame takes: draw into a `LayerTarget`, composite with
//! mosh-render's `Compositor` into an `OutputImage`, and read that back with
//! `Gpu::read_back_output`, whose row 0 is the top of the stage. The fixture is asymmetric in both
//! axes (red band on top, blue band on the bottom, a yellow box in the upper third on the left),
//! so a V-flip, an H-flip or a transposition each fail a different assertion.
//!
//! The camera case lives in `src/camera/tests.rs`: a captured frame can only be injected through
//! `CameraPass`, which is crate-private, so that test is a unit test in the crate and asserts the
//! same things against the same output path.

use std::ops::Deref;
use std::sync::{Mutex, MutexGuard};

use mosh_core::{AnalysisFrame, BlendMode, ClipParams};
use mosh_render::upload::BlockFormat;
use mosh_render::{Compositor, FrameRecorder, Gpu, GpuPreference, LayerTarget, OutputImage};
use mosh_visuals::overlay::OverlayLayer;
use mosh_visuals::text::TextLayer;

/// One GPU test at a time in this binary: several short offscreen tests in parallel fight for the
/// same integrated GPU, and this file's tests each build a device.
static GPU_LOCK: Mutex<()> = Mutex::new(());

/// A device, held under the test lock for as long as it lives.
struct TestGpu {
    /// Declared first so it is dropped first: the device goes away before the lock is released.
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
        "adapter: {} | validation layers: {}",
        gpu.adapter_name(),
        if gpu.validation_enabled() {
            "on"
        } else {
            "unavailable"
        }
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
// The fixture: the test video the coordinator's windowed run used
// ---------------------------------------------------------------------------------------------

const RED: [u8; 3] = [255, 0, 0];
const GREEN: [u8; 3] = [0, 255, 0];
const BLUE: [u8; 3] = [0, 0, 255];
const YELLOW: [u8; 3] = [255, 255, 0];

/// The source's size, the boxes in it, and the target's size. The target is the same aspect, so
/// cover scaling shows the whole frame and every fraction below is exact at half scale.
const SRC: (u32, u32) = (640, 360);
const SIZE: (u32, u32) = (320, 180);

/// The colour of one source pixel: red rows 0-59, blue rows 300-359, a yellow box at x 20-139,
/// y 100-139, and green everywhere else. The boxes are block-aligned, so a BCn fixture built from
/// whole 4x4 blocks reproduces it exactly.
fn source_colour(x: u32, y: u32) -> [u8; 3] {
    if y < 60 {
        RED
    } else if y >= 300 {
        BLUE
    } else if (20..140).contains(&x) && (100..140).contains(&y) {
        YELLOW
    } else {
        GREEN
    }
}

/// A decoded frame as `mosh-video` hands it over (BGRA, rows tightly packed, top row first).
fn bgra_frame(width: u32, height: u32) -> Vec<u8> {
    let mut bytes = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let rgb = source_colour(x, y);
            let at = ((y * width + x) * 4) as usize;
            bytes[at..at + 4].copy_from_slice(&[rgb[2], rgb[1], rgb[0], 255]);
        }
    }
    bytes
}

fn rgb565(rgb: [u8; 3]) -> u16 {
    (u16::from(rgb[0] >> 3) << 11) | (u16::from(rgb[1] >> 2) << 5) | u16::from(rgb[2] >> 3)
}

/// One solid 4x4 BC1 block: equal endpoints, every index 0 (so it decodes to the block's colour).
fn bc1_block(rgb: [u8; 3]) -> [u8; 8] {
    let colour = rgb565(rgb);
    let (lo, hi) = (colour as u8, (colour >> 8) as u8);
    [lo, hi, lo, hi, 0, 0, 0, 0]
}

/// The same fixture as a HAP1 (BC1) payload: one solid block per 4x4 cell, block rows top to
/// bottom, exactly the layout `OverlayLayer::set_hap_frame` takes.
fn bc1_frame(width: u32, height: u32) -> Vec<u8> {
    let blocks_x = width.div_ceil(4);
    let blocks_y = height.div_ceil(4);
    let mut payload = Vec::with_capacity((blocks_x * blocks_y * 8) as usize);
    for block_y in 0..blocks_y {
        for block_x in 0..blocks_x {
            // Every cell in the fixture is one solid 4x4 block, so the block's top-left pixel is
            // the whole block's colour.
            let rgb = source_colour(block_x * 4, block_y * 4);
            payload.extend_from_slice(&bc1_block(rgb));
        }
    }
    payload
}

fn pixel(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let at = ((y * width + x) * 4) as usize;
    [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]]
}

/// Composites one layer into an `OutputImage` and reads it back: the path the stage takes, minus
/// the swapchain. Row 0 of the result is the top of the picture the audience sees.
fn composite_output(gpu: &Gpu, layer: &LayerTarget, width: u32, height: u32) -> Vec<u8> {
    let out = OutputImage::new(gpu, width, height);
    let mut compositor = Compositor::new(gpu).expect("compositor");
    compositor
        .composite(&[(layer, BlendMode::Normal, 1.0)], &out)
        .expect("composite the layer");
    gpu.read_back_output(&out)
}

fn distance(pixel: [u8; 4], rgb: [u8; 3]) -> i32 {
    (0..3)
        .map(|channel| (i32::from(pixel[channel]) - i32::from(rgb[channel])).abs())
        .max()
        .unwrap_or(0)
}

/// The fixture's orientation on the stage, asserted on an output read-back (row 0 = the stage's
/// top): red across the top, blue across the bottom, and the marker in the upper third on the
/// left. `tolerance` leaves room for a half-float round trip and a BC1 block decode.
fn assert_upright(pixels: &[u8], width: u32, height: u32, tolerance: i32, what: &str) {
    let red_top = pixel(pixels, width, width / 2, 2);
    let blue_bottom = pixel(pixels, width, width / 2, height - 3);
    let marker_left = pixel(pixels, width, width / 12, height / 3);
    let middle = pixel(pixels, width, width / 2, height / 2);
    println!(
        "{what}: top {:?}, bottom {:?}, marker {:?}, middle {:?}",
        red_top, blue_bottom, marker_left, middle
    );

    assert!(
        distance(red_top, RED) <= tolerance,
        "{what}: output row 0 is {red_top:?}, and the source's top band is red: the picture is \
         upside down (or not drawn)"
    );
    assert!(
        distance(blue_bottom, BLUE) <= tolerance,
        "{what}: the output's last rows are {blue_bottom:?}, and the source's bottom band is \
         blue: the picture is upside down"
    );
    assert!(
        distance(marker_left, YELLOW) <= tolerance,
        "{what}: the output at ({}, {}) is {marker_left:?}, and the source's marker is yellow",
        width / 12,
        height / 3
    );
    assert!(
        distance(middle, GREEN) <= tolerance,
        "{what}: the output's middle is {middle:?}, and the source's middle band is green"
    );

    // The marker is only in the upper third on the left. Both checks are implied by the bands
    // above, but they name the two mistakes a band swap can hide.
    let mirrored_down = pixel(pixels, width, width / 12, height - height / 3);
    assert!(
        distance(mirrored_down, YELLOW) > 60,
        "{what}: the marker is also at y = {} (the lower third): the picture is mirrored \
         vertically, not just band-swapped at the edge",
        height - height / 3
    );
    let mirrored_right = pixel(pixels, width, width - width / 12, height / 3);
    assert!(
        distance(mirrored_right, YELLOW) > 60,
        "{what}: the marker is also on the right (x = {}): the picture is mirrored horizontally",
        width - width / 12
    );
}

// ---------------------------------------------------------------------------------------------
// Overlay video, decoded BGRA
// ---------------------------------------------------------------------------------------------

/// A decoded BGRA frame's top row lands at the top of the stage. This is the defect the
/// coordinator saw in a windowed run: the video played upside down, red band at the bottom.
#[test]
fn overlay_bgra_is_upright_on_the_stage() {
    let gpu = gpu();
    let (src_width, src_height) = SRC;
    let (width, height) = SIZE;

    let mut overlay = OverlayLayer::new(&gpu).expect("overlay");
    overlay
        .set_frame(
            &gpu,
            0,
            src_width,
            src_height,
            &bgra_frame(src_width, src_height),
        )
        .expect("upload the frame");
    let layer = LayerTarget::new(&gpu, width, height);
    {
        let mut recorder = FrameRecorder::new(&gpu);
        overlay
            .draw(&mut recorder, 0, &layer)
            .expect("draw the overlay");
        gpu.wait_frame(recorder.submit());
    }

    let pixels = composite_output(&gpu, &layer, width, height);
    assert_upright(
        &pixels,
        width,
        height,
        3,
        "overlay BGRA 640x360 into 320x180",
    );
    assert_no_validation_messages(&gpu);
}

// ---------------------------------------------------------------------------------------------
// Overlay video, HAP1
// ---------------------------------------------------------------------------------------------

/// A HAP frame's first block row lands at the top of the stage, through the block upload, the
/// GPU's BC1 decode and the HAP cover shader.
#[test]
fn hap_overlay_is_upright_on_the_stage() {
    let gpu = gpu();
    if !gpu.texture_compression_bc() {
        println!("skip: this device has no textureCompressionBC, so HAP cannot be uploaded");
        return;
    }
    let (src_width, src_height) = SRC;
    let (width, height) = SIZE;

    let mut overlay = OverlayLayer::new(&gpu).expect("overlay");
    assert!(overlay.hap_supported(), "the device reported BC support");
    overlay
        .set_hap_frame(
            &gpu,
            0,
            src_width,
            src_height,
            BlockFormat::Bc1,
            &bc1_frame(src_width, src_height),
            None,
        )
        .expect("upload the HAP frame");
    let layer = LayerTarget::new(&gpu, width, height);
    {
        let mut recorder = FrameRecorder::new(&gpu);
        overlay
            .draw(&mut recorder, 0, &layer)
            .expect("draw the HAP overlay");
        gpu.wait_frame(recorder.submit());
    }

    let pixels = composite_output(&gpu, &layer, width, height);
    assert_upright(&pixels, width, height, 4, "HAP1 640x360 into 320x180");
    assert_no_validation_messages(&gpu);
}

// ---------------------------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------------------------

fn is_ink(pixel: [u8; 4]) -> bool {
    pixel[0] > 2 || pixel[1] > 2 || pixel[2] > 2 || pixel[3] > 2
}

/// An `L` drawn on the stage has its foot at the bottom of the output and its stem on the left.
/// The text layer is the layer that already drew in audience coordinates; this pins the same
/// output-path orientation for it, so the shared convention has one test covering every layer.
#[test]
fn text_is_upright_on_the_stage() {
    let gpu = gpu();
    let (width, height) = (400u32, 300u32);

    let mut layer = TextLayer::new(&gpu).expect("text layer");
    layer.set_text(&gpu, 0, "L");
    let dst = LayerTarget::new(&gpu, width, height);
    let params = ClipParams {
        // size 0.7, the clean look, a thin outline; no jitter, glitch, pop or scroll.
        p: [0.7, 0.85, 0.5, 0.0, 0.0, 0.35, 0.0, 0.0],
        palette: 0,
        seed: 0,
    };
    {
        let mut recorder = FrameRecorder::new(&gpu);
        layer
            .draw(
                &mut recorder,
                0,
                &dst,
                &AnalysisFrame::default(),
                &params,
                0.0,
            )
            .expect("draw the text");
        gpu.wait_frame(recorder.submit());
    }

    let pixels = composite_output(&gpu, &dst, width, height);
    let quarter = height / 4;
    let ink_in = |y0: u32, y1: u32| -> usize {
        (y0..y1)
            .map(|y| {
                (0..width)
                    .filter(|x| is_ink(pixel(&pixels, width, *x, y)))
                    .count()
            })
            .sum()
    };
    let top = ink_in(0, quarter);
    let bottom = ink_in(height - quarter, height);
    let stem = (0..width / 2)
        .map(|x| {
            (0..height)
                .filter(|y| is_ink(pixel(&pixels, width, x, *y)))
                .count()
        })
        .sum::<usize>();
    let arm = (width / 2..width)
        .map(|x| {
            (0..height)
                .filter(|y| is_ink(pixel(&pixels, width, x, *y)))
                .count()
        })
        .sum::<usize>();
    println!(
        "L on the output: top quarter {top} lit px, bottom quarter {bottom}, left half {stem}, \
         right half {arm}"
    );

    assert!(
        bottom > top * 3 / 2,
        "the L's foot must be in the output's last rows (the audience's bottom): bottom quarter \
         {bottom}, top quarter {top}. This is the upside-down defect"
    );
    assert!(
        stem > arm * 3 / 2,
        "the L's stem must be on the left: left half {stem}, right half {arm}"
    );
    assert_no_validation_messages(&gpu);
}
