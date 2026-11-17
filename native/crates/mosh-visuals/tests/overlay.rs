//! Overlay video layers, offscreen (Task 2.8).
//!
//! A decoded BGRA frame (what `mosh-video` hands over) is uploaded into a texture and drawn into
//! a layer with aspect-preserving cover scaling. Every test renders into a layer target and reads
//! it back with `Gpu::read_back`, so no window is opened.
//!
//! # Row order, which the assertions depend on
//!
//! `Gpu::read_back` returns image row 0 first, and the compositor plus the stage's present pass
//! flip the picture once each, so a layer target's row 0 is the **top** of the picture an audience
//! sees. A frame whose row 0 is the top of the video therefore reads back with its row 0 first.
//! Every expected pixel below is written in that orientation.

use std::ops::Deref;
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use mosh_core::{AnalysisFrame, BlendMode, ClipParams, LayerState};
use mosh_render::{FrameRecorder, Gpu, GpuPreference, LayerTarget};
use mosh_visuals::overlay::OverlayLayer;
use mosh_visuals::{LayerSlot, Visuals};

/// One GPU test at a time in this binary. `mosh_render::upload::allocation_count` is process-wide,
/// and the allocation test compares it before and after a loop: another test allocating at the
/// same time would make that comparison meaningless. Serialising also keeps five short GPU tests
/// from fighting each other (and the other lanes) for the machine.
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

/// One device per test; the reference machine's integrated GPU, or whatever single GPU a machine
/// without one has (a VM, CI).
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

/// `mosh-video` hands over Media Foundation's RGB32 layout: byte 0 is blue, byte 2 is red.
fn bgra(rgb: [u8; 3]) -> [u8; 4] {
    [rgb[2], rgb[1], rgb[0], 255]
}

const RED: [u8; 3] = [255, 0, 0];
const GREEN: [u8; 3] = [0, 255, 0];
const BLUE: [u8; 3] = [0, 0, 255];
const YELLOW: [u8; 3] = [255, 255, 0];
const CYAN: [u8; 3] = [0, 255, 255];
const MAGENTA: [u8; 3] = [255, 0, 255];

/// A 160x90 checkerboard of 20x15 cells, red where `(column + row)` is even, green otherwise.
/// Ten cells across and six down: both counts are even, so this pattern is not symmetric under a
/// vertical or horizontal flip, and a flipped draw fails the assertions below.
fn checkerboard(width: u32, height: u32) -> Vec<u8> {
    let mut bgra_bytes = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let cell = (x / 20, y / 15);
            let colour = if (cell.0 + cell.1) % 2 == 0 {
                RED
            } else {
                GREEN
            };
            let offset = ((y * width + x) * 4) as usize;
            bgra_bytes[offset..offset + 4].copy_from_slice(&bgra(colour));
        }
    }
    bgra_bytes
}

/// A 160x90 frame of six 15-row bands: red, green, blue, yellow, cyan, magenta, top to bottom.
fn bands(width: u32, height: u32) -> Vec<u8> {
    let colours = [RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA];
    let band_height = height / 6;
    let mut bgra_bytes = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        let colour = colours[(y / band_height).min(5) as usize];
        for x in 0..width {
            let offset = ((y * width + x) * 4) as usize;
            bgra_bytes[offset..offset + 4].copy_from_slice(&bgra(colour));
        }
    }
    bgra_bytes
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

fn overlay_state() -> LayerState {
    LayerState {
        layer: 0,
        clip: Some(1),
        visualizer: Some("overlay".to_string()),
        params: ClipParams::default(),
        effects: Vec::new(),
        transition: None,
        blend: BlendMode::Normal,
        opacity: 1.0,
        seed: 0,
    }
}

/// One frame of the facade, through the same `"overlay"` chain the app runs.
fn draw_through_facade(
    visuals: &mut Visuals,
    gpu: &Gpu,
    slot: &mut LayerSlot,
) {
    let frame = AnalysisFrame::default();
    let mut recorder = FrameRecorder::new(gpu);
    visuals
        .draw_layer(&mut recorder, slot, &overlay_state(), &frame, 0.0)
        .expect("draw the overlay layer");
    gpu.wait_frame(recorder.submit());
}

/// A 160x90 video into a 64x36 target: the aspects match, so nothing is cropped and every one of
/// the 8x6 cells lands on an exact 8x6 pixel block. This is the "cover scaling" baseline: the
/// other test is the one that crops.
#[test]
fn checkerboard_cover_crop() {
    let gpu = gpu();
    let (video_width, video_height) = (160u32, 90u32);
    let (target_width, target_height) = (64u32, 36u32);
    let frame = checkerboard(video_width, video_height);

    let mut visuals = Visuals::new(&gpu, target_width, target_height).expect("visuals");
    visuals
        .overlay
        .set_frame(&gpu, 0, video_width, video_height, &frame)
        .expect("upload the checkerboard");
    let mut slot = LayerSlot::new(&gpu, 0, target_width, target_height);
    draw_through_facade(&mut visuals, &gpu, &mut slot);

    let pixels = gpu.read_back(slot.output());
    assert_eq!(pixels.len(), (target_width * target_height * 4) as usize);

    for row in 0..6u32 {
        for column in 0..8u32 {
            let expected = if (column + row) % 2 == 0 {
                [255, 0, 0, 255]
            } else {
                [0, 255, 0, 255]
            };
            let x = column * 8 + 4;
            let y = row * 6 + 3;
            let got = pixel_at(&pixels, target_width, x, y);
            assert_pixel(
                got,
                expected,
                // A half-float round trip and an 8-bit conversion are not exact.
                2,
                &format!("cell ({column},{row}) at ({x},{y})"),
            );
        }
    }
    // The corners pin the orientation: a vertically flipped draw would put the last row's cells
    // here instead, and the 6-row/10-column cell counts make that swap visible.
    assert_pixel(
        pixel_at(&pixels, target_width, 2, 2),
        [255, 0, 0, 255],
        2,
        "top-left cell",
    );
    assert_pixel(
        pixel_at(&pixels, target_width, 61, 1),
        [0, 255, 0, 255],
        2,
        "top-right cell",
    );
    assert_pixel(
        pixel_at(&pixels, target_width, 2, 33),
        [0, 255, 0, 255],
        2,
        "bottom-left cell",
    );

    println!(
        "checkerboard 160x90 -> 64x36: cells at 8x6 px, top-left {:?}",
        pixel_at(&pixels, target_width, 2, 2)
    );
    assert_no_validation_messages(&gpu);
}

/// A target **wider** than the video's aspect fills its width and crops the video's top and
/// bottom: 160x90 into 96x36 scales by 0.6 to 96x54 and keeps the middle 36 rows, so the outer
/// 15 rows of the video (the red and magenta bands) never appear.
#[test]
fn wider_target_crops_top_bottom() {
    let gpu = gpu();
    let (video_width, video_height) = (160u32, 90u32);
    let (target_width, target_height) = (96u32, 36u32);
    let frame = bands(video_width, video_height);

    let mut overlay = OverlayLayer::new(&gpu).expect("overlay");
    overlay
        .set_frame(&gpu, 0, video_width, video_height, &frame)
        .expect("upload the bands");
    let dst = LayerTarget::new(&gpu, target_width, target_height);
    {
        let mut recorder = FrameRecorder::new(&gpu);
        overlay
            .draw(&mut recorder, 0, &dst)
            .expect("draw the bands");
        gpu.wait_frame(recorder.submit());
    }

    let pixels = gpu.read_back(&dst);
    // The crop is centred: the four surviving bands cover 36 rows in four 9-row blocks.
    for (y, colour, name) in [
        (4u32, GREEN, "green band, top"),
        (13, BLUE, "blue band"),
        (22, YELLOW, "yellow band"),
        (31, CYAN, "cyan band, bottom"),
    ] {
        for x in [1u32, target_width / 2, target_width - 2] {
            assert_pixel(
                pixel_at(&pixels, target_width, x, y),
                [colour[0], colour[1], colour[2], 255],
                2,
                &format!("{name} at ({x},{y})"),
            );
        }
    }
    // The cropped-away bands must be gone everywhere, not merely absent from the sample points.
    for y in 0..target_height {
        for x in 0..target_width {
            let pixel = pixel_at(&pixels, target_width, x, y);
            for (cropped, name) in [(RED, "red"), (MAGENTA, "magenta")] {
                let expected = [cropped[0], cropped[1], cropped[2], 255];
                let distance = (0..3)
                    .map(|channel| {
                        (i32::from(pixel[channel]) - i32::from(expected[channel])).abs()
                    })
                    .max()
                    .unwrap_or(0);
                assert!(
                    distance > 24,
                    "{name} band from the cropped region appears at ({x},{y}): {pixel:?}"
                );
            }
        }
    }

    println!(
        "wider target 96x36 from 160x90: rows green/blue/yellow/cyan {:?} {:?} {:?} {:?}",
        pixel_at(&pixels, target_width, 48, 4),
        pixel_at(&pixels, target_width, 48, 13),
        pixel_at(&pixels, target_width, 48, 22),
        pixel_at(&pixels, target_width, 48, 31)
    );
    assert_no_validation_messages(&gpu);
}

/// A reused texture: 600 uploads and draws of one size allocate nothing after the first, and the
/// staging buffer, the image and the pipeline are the same objects throughout.
#[test]
fn no_new_allocations_after_first() {
    let gpu = gpu();
    let mut overlay = OverlayLayer::new(&gpu).expect("overlay");
    let dst = LayerTarget::new(&gpu, 64, 36);
    let frame = checkerboard(160, 90);

    overlay
        .set_frame(&gpu, 0, 160, 90, &frame)
        .expect("first upload");
    {
        let mut recorder = FrameRecorder::new(&gpu);
        overlay.draw(&mut recorder, 0, &dst).expect("first draw");
        gpu.wait_frame(recorder.submit());
    }
    let allocations_after_first = mosh_render::upload::allocation_count();

    let start = Instant::now();
    for _ in 0..600 {
        overlay
            .set_frame(&gpu, 0, 160, 90, &frame)
            .expect("upload");
        let mut recorder = FrameRecorder::new(&gpu);
        overlay.draw(&mut recorder, 0, &dst).expect("draw");
        gpu.wait_frame(recorder.submit());
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let allocations = mosh_render::upload::allocation_count();

    assert_eq!(
        allocations, allocations_after_first,
        "600 uploads of one size must reuse the image, the staging buffer and the pipeline"
    );
    // A first frame, then 600 more draws: the picture is still right after all of them.
    let pixels = gpu.read_back(&dst);
    assert_pixel(
        pixel_at(&pixels, 64, 2, 2),
        [255, 0, 0, 255],
        2,
        "checkerboard after 601 draws",
    );

    println!(
        "600 uploads + draws of 160x90: {} -> {allocations} GPU allocations in this process ({} \
         new), {:.3} ms per frame",
        allocations_after_first,
        allocations - allocations_after_first,
        elapsed_ms / 600.0
    );
    assert_no_validation_messages(&gpu);
}

/// A size change reallocates the image and the staging buffer once, and nothing after that: a
/// clip that switches from a small file to a larger one costs one allocation pair, not one per
/// frame. The draw after the change also exercises the new image's `UNDEFINED` starting layout.
#[test]
fn size_change_reallocates_once() {
    let gpu = gpu();
    let mut overlay = OverlayLayer::new(&gpu).expect("overlay");
    let dst = LayerTarget::new(&gpu, 64, 36);
    let small = checkerboard(160, 90);
    let large = checkerboard(320, 180);

    overlay
        .set_frame(&gpu, 0, 160, 90, &small)
        .expect("first upload");
    {
        let mut recorder = FrameRecorder::new(&gpu);
        overlay.draw(&mut recorder, 0, &dst).expect("first draw");
        gpu.wait_frame(recorder.submit());
    }
    let allocations_at_160x90 = mosh_render::upload::allocation_count();

    for _ in 0..3 {
        overlay
            .set_frame(&gpu, 0, 320, 180, &large)
            .expect("upload after the size change");
        let mut recorder = FrameRecorder::new(&gpu);
        overlay.draw(&mut recorder, 0, &dst).expect("draw");
        gpu.wait_frame(recorder.submit());
    }
    let allocations_at_320x180 = mosh_render::upload::allocation_count();
    assert_eq!(
        allocations_at_320x180 - allocations_at_160x90,
        2,
        "one image and one staging buffer for the new size, not one pair per frame"
    );

    let pixels = gpu.read_back(&dst);
    // The same checkerboard generator makes 20x15 cells, so 320x180 holds 16x12 of them and the
    // 64x36 target shows all of them at 4x3 px each.
    for row in 0..12u32 {
        for column in 0..16u32 {
            let expected = if (column + row) % 2 == 0 {
                [255, 0, 0, 255]
            } else {
                [0, 255, 0, 255]
            };
            assert_pixel(
                pixel_at(&pixels, 64, column * 4 + 2, row * 3 + 1),
                expected,
                2,
                &format!("320x180 into 64x36, cell ({column},{row})"),
            );
        }
    }

    println!(
        "size change 160x90 -> 320x180: {} -> {allocations_at_320x180} GPU allocations (+{}), \
         picture intact",
        allocations_at_160x90,
        allocations_at_320x180 - allocations_at_160x90
    );
    assert_no_validation_messages(&gpu);
}

/// A layer that never got a frame draws transparent black rather than a stale or garbage picture.
#[test]
fn missing_frame_draws_black() {
    let gpu = gpu();
    let mut overlay = OverlayLayer::new(&gpu).expect("overlay");
    let dst = LayerTarget::new(&gpu, 32, 18);
    {
        let mut recorder = FrameRecorder::new(&gpu);
        overlay.draw(&mut recorder, 3, &dst).expect("draw with no frame");
        gpu.wait_frame(recorder.submit());
    }

    let pixels = gpu.read_back(&dst);
    for y in 0..18u32 {
        for x in 0..32u32 {
            assert_pixel(
                pixel_at(&pixels, 32, x, y),
                [0, 0, 0, 0],
                0,
                &format!("missing frame at ({x},{y})"),
            );
        }
    }
    println!("missing frame: 32x18 target is transparent black");
    assert_no_validation_messages(&gpu);
}

/// Time per 1920x1080 upload plus draw, printed for the PR body. Not a budget assertion: the
/// design's 6 ms/frame covers the whole stage, and this is one overlay layer of it.
#[test]
fn upload_and_draw_1080p_time() {
    let gpu = gpu();
    let (width, height) = (1920u32, 1080u32);
    let mut overlay = OverlayLayer::new(&gpu).expect("overlay");
    let dst = LayerTarget::new(&gpu, width, height);
    let frame = checkerboard(width, height);

    // Two warm-up frames: the first upload allocates and the first draw builds descriptors.
    for _ in 0..2 {
        overlay
            .set_frame(&gpu, 0, width, height, &frame)
            .expect("warm-up upload");
        let mut recorder = FrameRecorder::new(&gpu);
        overlay.draw(&mut recorder, 0, &dst).expect("warm-up draw");
        gpu.wait_frame(recorder.submit());
    }

    let frames = 20;
    let mut upload_ms = 0.0;
    let mut draw_ms = 0.0;
    let mut redraw_ms = 0.0;
    for _ in 0..frames {
        let start = Instant::now();
        overlay
            .set_frame(&gpu, 0, width, height, &frame)
            .expect("upload");
        upload_ms += start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        let mut recorder = FrameRecorder::new(&gpu);
        overlay.draw(&mut recorder, 0, &dst).expect("draw");
        gpu.wait_frame(recorder.submit());
        draw_ms += start.elapsed().as_secs_f64() * 1000.0;

        // The same picture again, with no new frame: the transfer is skipped and only the cover
        // pass and the one-shot's fence wait remain, which is what this split is here to show.
        let start = Instant::now();
        let mut recorder = FrameRecorder::new(&gpu);
        overlay.draw(&mut recorder, 0, &dst).expect("redraw");
        gpu.wait_frame(recorder.submit());
        redraw_ms += start.elapsed().as_secs_f64() * 1000.0;
    }
    let upload_ms = upload_ms / frames as f64;
    let draw_ms = draw_ms / frames as f64;
    let redraw_ms = redraw_ms / frames as f64;

    assert!(upload_ms.is_finite() && draw_ms.is_finite() && redraw_ms.is_finite());
    let pixels = gpu.read_back(&dst);
    assert_pixel(
        pixel_at(&pixels, width, 10, 10),
        [255, 0, 0, 255],
        2,
        "1080p checkerboard",
    );

    println!(
        "1920x1080 overlay, {} frames, {} MB per frame: upload (CPU -> mapped staging) {upload_ms:.3} \
         ms, draw (transfer + cover pass + fence) {draw_ms:.3} ms, redraw (cover pass + fence, no \
         transfer) {redraw_ms:.3} ms",
        frames,
        (width * height * 4) / (1024 * 1024)
    );
    assert_no_validation_messages(&gpu);
}
