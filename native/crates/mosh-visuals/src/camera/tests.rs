//! Camera layer tests, offscreen: no window is opened, and the GPU tests need no camera.
//!
//! * The parameter math (`zoom`/`pan`/`mirror`/`posterize`, the uniform block) is pure and tested
//!   in `params.rs`.
//! * The GPU tests here draw a **synthetic** frame through exactly the path a captured one takes:
//!   `VideoTexture::upload` of a BGRA frame, then `CameraPass::record`, which is what
//!   [`super::CameraLayer::draw`] does after it takes a frame from the capture thread. So the
//!   shader, the uniform block and the cover scaling are covered without a webcam.
//! * `zero_validation_messages`-style checks: every GPU test asserts `Gpu::validation_messages()`
//!   is empty and prints the count.
//! * [`default_camera_for_three_seconds`] is `#[ignore]`d: it opens the real default camera.

use std::ops::Deref;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use mosh_core::{BlendMode, ClipParams};
use mosh_render::upload::VideoTexture;
use mosh_render::{Compositor, FrameRecorder, Gpu, GpuPreference, LayerTarget, OutputImage};

use super::pass::CameraPass;
use super::*;

/// One GPU test at a time in this binary: several tests opening a Vulkan device at once on the
/// machines this runs on is more contention than the tests need.
static GPU_LOCK: Mutex<()> = Mutex::new(());

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

fn assert_no_validation(gpu: &Gpu) {
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

/// A clip with exactly these `p0..p7` values, everything else default.
fn params(p: [f32; 8]) -> ClipParams {
    ClipParams {
        p,
        ..ClipParams::default()
    }
}

fn bgra(rgb: [u8; 3]) -> [u8; 4] {
    [rgb[2], rgb[1], rgb[0], 255]
}

const RED: [u8; 3] = [255, 0, 0];
const GREEN: [u8; 3] = [0, 255, 0];
const BLUE: [u8; 3] = [0, 0, 255];
const GREY: [u8; 3] = [128, 128, 128];

/// A 160x90 checkerboard of 20x15 cells, red where `(column + row)` is even, green otherwise.
fn checkerboard(width: u32, height: u32) -> Vec<u8> {
    let mut bytes = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let cell = (x / 20, y / 15);
            let colour = if (cell.0 + cell.1) % 2 == 0 {
                RED
            } else {
                GREEN
            };
            let offset = ((y * width + x) * 4) as usize;
            bytes[offset..offset + 4].copy_from_slice(&bgra(colour));
        }
    }
    bytes
}

/// Three equal vertical bands: red, green, blue, left to right.
fn bands(width: u32, height: u32) -> Vec<u8> {
    let colours = [RED, GREEN, BLUE];
    let mut bytes = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let colour = colours[((x * 3) / width).min(2) as usize];
            let offset = ((y * width + x) * 4) as usize;
            bytes[offset..offset + 4].copy_from_slice(&bgra(colour));
        }
    }
    bytes
}

/// One flat colour.
fn solid(width: u32, height: u32, colour: [u8; 3]) -> Vec<u8> {
    let mut bytes = vec![0u8; (width * height * 4) as usize];
    for pixel in bytes.as_chunks_mut::<4>().0 {
        pixel.copy_from_slice(&bgra(colour));
    }
    bytes
}

const YELLOW: [u8; 3] = [255, 255, 0];

/// The orientation fixture, the same one `tests/orientation.rs` uses: a red band on the top 60
/// rows, a blue band on the bottom 60, a yellow box at x 20-139 / y 100-139 (upper third, left)
/// and green elsewhere. Asymmetric in both axes, so a V-flip, an H-flip or a transpose each fail a
/// different assertion.
fn orientation_frame(width: u32, height: u32) -> Vec<u8> {
    let mut bytes = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let colour = if y < 60 {
                RED
            } else if y >= height - 60 {
                BLUE
            } else if (20..140).contains(&x) && (100..140).contains(&y) {
                YELLOW
            } else {
                GREEN
            };
            let offset = ((y * width + x) * 4) as usize;
            bytes[offset..offset + 4].copy_from_slice(&bgra(colour));
        }
    }
    bytes
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

/// A pixel in *audience* coordinates (row 0 = the top of the stage).
///
/// A layer target's read-back is V-mirrored relative to the stage: the compositor samples a layer
/// with uv.y = 0 at the top of the output, so a layer's row 0 is the audience's bottom row, and the
/// present pass adds no second flip (`tests/orientation.rs` has the output-path assertions). The
/// cover-geometry tests below read the layer directly and work in audience order.
fn stage_pixel(pixels: &[u8], width: u32, height: u32, x: u32, y: u32) -> [u8; 4] {
    pixel_at(pixels, width, x, height - 1 - y)
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

/// Draws `frame` through the camera pass exactly as a captured one is drawn: upload into a
/// `VideoTexture`, then the pass's recorded copy and cover draw inside one frame.
fn draw_synthetic(
    gpu: &Gpu,
    width: u32,
    height: u32,
    frame: &[u8],
    dst: &LayerTarget,
    clip: &ClipParams,
) -> Result<(), mosh_render::RenderError> {
    let pass = CameraPass::new(gpu)?;
    let mut texture = VideoTexture::new(gpu, width, height)?;
    texture.upload(gpu, width, height, frame)?;
    let mut recorder = FrameRecorder::new(gpu);
    pass.record(&mut recorder, &mut texture, dst, clip)?;
    gpu.wait_frame(recorder.submit());
    Ok(())
}

/// A 160x90 checkerboard into a 64x36 target: the aspects match, so every one of the 8x6 cells
/// lands on an exact 8x6 pixel block and the cover geometry is pinned by the corners. The picture
/// is read in audience order (see [`stage_pixel`]); the source's orientation through the output
/// path is `captured_frame_is_upright_on_the_stage`.
#[test]
fn synthetic_frame_draws_cover_scaled_and_unflipped() {
    let gpu = gpu();
    let (video_width, video_height) = (160u32, 90u32);
    let (target_width, target_height) = (64u32, 36u32);
    let dst = LayerTarget::new(&gpu, target_width, target_height);

    draw_synthetic(
        &gpu,
        video_width,
        video_height,
        &checkerboard(video_width, video_height),
        &dst,
        &params([0.0; 8]),
    )
    .expect("draw the checkerboard");

    let pixels = gpu.read_back(&dst);
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
            assert_pixel(
                stage_pixel(&pixels, target_width, target_height, x, y),
                expected,
                2,
                &format!("cell ({column},{row}) at ({x},{y})"),
            );
        }
    }
    // The corners pin the orientation: a horizontally flipped draw would swap the left and right
    // cells, and a cover crop that took the wrong rows would move the 6 cell rows.
    assert_pixel(
        stage_pixel(&pixels, target_width, target_height, 2, 2),
        [255, 0, 0, 255],
        2,
        "top-left cell",
    );
    assert_pixel(
        stage_pixel(&pixels, target_width, target_height, 61, 1),
        [0, 255, 0, 255],
        2,
        "top-right cell",
    );
    assert_pixel(
        stage_pixel(&pixels, target_width, target_height, 2, 33),
        [0, 255, 0, 255],
        2,
        "bottom-left cell",
    );
    println!(
        "synthetic 160x90 -> 64x36: top-left {:?}, top-right {:?}",
        stage_pixel(&pixels, target_width, target_height, 2, 2),
        stage_pixel(&pixels, target_width, target_height, 61, 1)
    );
    assert_no_validation(&gpu);
}

/// The camera picture's orientation through the real output path. A synthetic frame's top band
/// must be on the top of the composited output, and its marker must stay in the upper third on the
/// left: the same fixture and the same assertions `tests/orientation.rs` runs for overlay BGRA and
/// HAP. This case lives here because a captured frame can only be injected through `CameraPass`,
/// which is crate-private.
#[test]
fn captured_frame_is_upright_on_the_stage() {
    let gpu = gpu();
    let (width, height) = (640u32, 360u32);
    let (out_width, out_height) = (320u32, 180u32);
    let dst = LayerTarget::new(&gpu, out_width, out_height);
    draw_synthetic(
        &gpu,
        width,
        height,
        &orientation_frame(width, height),
        &dst,
        &params([0.0; 8]),
    )
    .expect("draw the fixture");

    let out = OutputImage::new(&gpu, out_width, out_height);
    let mut compositor = Compositor::new(&gpu).expect("compositor");
    compositor
        .composite(&[(&dst, BlendMode::Normal, 1.0)], &out)
        .expect("composite the camera layer");
    let pixels = gpu.read_back_output(&out);

    // Output row 0 is the top of the stage: red band first, blue band last, marker upper-left.
    let top = pixel_at(&pixels, out_width, out_width / 2, 2);
    let bottom = pixel_at(&pixels, out_width, out_width / 2, out_height - 3);
    let marker = pixel_at(&pixels, out_width, out_width / 12, out_height / 3);
    let middle = pixel_at(&pixels, out_width, out_width / 2, out_height / 2);
    let lower = pixel_at(
        &pixels,
        out_width,
        out_width / 12,
        out_height - out_height / 3,
    );
    println!(
        "camera on the output: top {top:?}, bottom {bottom:?}, marker {marker:?}, middle {middle:?}"
    );
    assert_pixel(
        top,
        [255, 0, 0, 255],
        3,
        "output row 0 is the frame's red top band",
    );
    assert_pixel(
        bottom,
        [0, 0, 255, 255],
        3,
        "the output's last rows are the frame's blue bottom band",
    );
    assert_pixel(
        marker,
        [255, 255, 0, 255],
        3,
        "the marker is in the upper third, left",
    );
    assert_pixel(middle, [0, 255, 0, 255], 3, "the frame's middle is green");
    assert_pixel(
        lower,
        [0, 255, 0, 255],
        3,
        "the marker is not repeated in the lower third (that would be a vertical mirror)",
    );
    assert_no_validation(&gpu);
}

/// `mirror` swaps left and right and nothing else.
#[test]
fn mirror_swaps_left_and_right() {
    let gpu = gpu();
    let dst = LayerTarget::new(&gpu, 96, 54);
    let frame = bands(160, 90);

    draw_synthetic(&gpu, 160, 90, &frame, &dst, &params([0.0; 8])).expect("plain");
    let plain = gpu.read_back(&dst);
    assert_pixel(
        pixel_at(&plain, 96, 8, 27),
        [255, 0, 0, 255],
        2,
        "plain left",
    );
    assert_pixel(
        pixel_at(&plain, 96, 87, 27),
        [0, 0, 255, 255],
        2,
        "plain right",
    );

    draw_synthetic(
        &gpu,
        160,
        90,
        &frame,
        &dst,
        &params([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    )
    .expect("mirrored");
    let mirrored = gpu.read_back(&dst);
    assert_pixel(
        pixel_at(&mirrored, 96, 8, 27),
        [0, 0, 255, 255],
        2,
        "mirrored left is the plain right",
    );
    assert_pixel(
        pixel_at(&mirrored, 96, 87, 27),
        [255, 0, 0, 255],
        2,
        "mirrored right is the plain left",
    );
    println!(
        "mirror: left {:?} -> {:?}, right {:?} -> {:?}",
        pixel_at(&plain, 96, 8, 27),
        pixel_at(&mirrored, 96, 8, 27),
        pixel_at(&plain, 96, 87, 27),
        pixel_at(&mirrored, 96, 87, 27)
    );
    assert_no_validation(&gpu);
}

/// `zoom` punches in: at 4x a 3-band frame shows only its centre band. `pan x` slides the framed
/// picture to the frame's edge at full pan.
#[test]
fn zoom_shows_the_centre_and_pan_reaches_the_edge() {
    let gpu = gpu();
    let dst = LayerTarget::new(&gpu, 90, 90);
    let frame = bands(160, 90);

    draw_synthetic(
        &gpu,
        160,
        90,
        &frame,
        &dst,
        &params([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    .expect("zoomed");
    let zoomed = gpu.read_back(&dst);
    for x in [2u32, 45, 88] {
        assert_pixel(
            pixel_at(&zoomed, 90, x, 45),
            [0, 255, 0, 255],
            2,
            &format!("4x zoom, column {x}: the centre band fills the target"),
        );
    }

    draw_synthetic(
        &gpu,
        160,
        90,
        &frame,
        &dst,
        &params([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    .expect("zoomed and panned right");
    let panned = gpu.read_back(&dst);
    assert_pixel(
        pixel_at(&panned, 90, 88, 45),
        [0, 0, 255, 255],
        2,
        "4x zoom, full pan right: the right band reaches the target's right edge",
    );
    println!(
        "zoom 4x: centre {:?} .. {:?}; pan right edge {:?}",
        pixel_at(&zoomed, 90, 2, 45),
        pixel_at(&zoomed, 90, 88, 45),
        pixel_at(&panned, 90, 88, 45)
    );
    assert_no_validation(&gpu);
}

/// The grade: exposure multiplies, contrast pivots at mid grey, tint moves the green/magenta axis
/// and posterize quantises to the requested number of levels.
#[test]
fn exposure_contrast_tint_and_posterize_grade_the_picture() {
    let gpu = gpu();
    let dst = LayerTarget::new(&gpu, 16, 9);
    let grey = solid(160, 90, GREY);

    // Exposure +2 stops (4x) pushes mid grey to white; -2 stops (0.25x) to 32.
    draw_synthetic(
        &gpu,
        160,
        90,
        &grey,
        &dst,
        &params([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    .expect("bright");
    let bright = pixel_at(&gpu.read_back(&dst), 16, 8, 4);
    assert_pixel(
        bright,
        [255, 255, 255, 255],
        2,
        "exposure +2 stops on mid grey",
    );

    draw_synthetic(
        &gpu,
        160,
        90,
        &grey,
        &dst,
        &params([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]),
    )
    .expect("dark");
    let dark = pixel_at(&gpu.read_back(&dst), 16, 8, 4);
    assert_pixel(dark, [32, 32, 32, 255], 2, "exposure -2 stops on mid grey");

    // Contrast 2x around mid grey leaves 128 where it is, and flat 0 makes everything that grey.
    let edge = solid(160, 90, [200, 200, 200]);
    draw_synthetic(
        &gpu,
        160,
        90,
        &edge,
        &dst,
        &params([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    )
    .expect("contrast");
    assert_pixel(
        pixel_at(&gpu.read_back(&dst), 16, 8, 4),
        [255, 255, 255, 255],
        2,
        "contrast 2x pushes 200 to white",
    );
    draw_synthetic(
        &gpu,
        160,
        90,
        &edge,
        &dst,
        &params([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
    )
    .expect("flat contrast");
    assert_pixel(
        pixel_at(&gpu.read_back(&dst), 16, 8, 4),
        [128, 128, 128, 255],
        2,
        "contrast 0 is flat mid grey",
    );

    // Tint +1 is magenta: red and blue up by half, green down by half (128 * 1.5 = 192).
    draw_synthetic(
        &gpu,
        160,
        90,
        &grey,
        &dst,
        &params([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    )
    .expect("tint");
    let tinted = pixel_at(&gpu.read_back(&dst), 16, 8, 4);
    assert_pixel(
        tinted,
        [192, 64, 192, 255],
        2,
        "tint +1 on mid grey is magenta",
    );

    // Posterize with 33 levels leaves 255 and 0 alone and keeps 128 at its own step.
    let mut ramp = vec![0u8; (160 * 90 * 4) as usize];
    for y in 0..90 {
        for x in 0..160 {
            let value = match (x * 3) / 160 {
                0 => 255u8,
                1 => 128,
                _ => 0,
            };
            let offset = ((y * 160 + x) * 4) as usize;
            ramp[offset..offset + 4].copy_from_slice(&bgra([value, value, value]));
        }
    }
    draw_synthetic(
        &gpu,
        160,
        90,
        &ramp,
        &dst,
        &params([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
    )
    .expect("posterize");
    let posterized = gpu.read_back(&dst);
    assert_pixel(
        pixel_at(&posterized, 16, 2, 4),
        [255, 255, 255, 255],
        2,
        "posterized white band",
    );
    assert_pixel(
        pixel_at(&posterized, 16, 8, 4),
        [128, 128, 128, 255],
        2,
        "posterized mid band",
    );
    assert_pixel(
        pixel_at(&posterized, 16, 14, 4),
        [0, 0, 0, 255],
        2,
        "posterized black band",
    );

    println!(
        "grade: exposure {bright:?}/{dark:?}, tint {tinted:?}, posterize bands {:?} {:?} {:?}",
        pixel_at(&posterized, 16, 2, 4),
        pixel_at(&posterized, 16, 8, 4),
        pixel_at(&posterized, 16, 14, 4)
    );
    assert_no_validation(&gpu);
}

/// A camera layer is drawn inside the frame: one upload pass and one camera pass in the frame's
/// own timings (a pass recorded in a submission of its own could not appear there), and the same
/// texture is redrawn when no new picture arrived (a 30 fps camera against a 60 fps render).
///
/// The global `submission_count` is not asserted here: other GPU tests in this binary submit at the
/// same time, so a process-wide counter would race. The frame's per-pass list is the evidence.
#[test]
fn camera_draws_inside_the_frame() {
    let gpu = gpu();
    let dst = LayerTarget::new(&gpu, 64, 36);
    let frame = checkerboard(160, 90);
    let pass = CameraPass::new(&gpu).expect("camera pass");
    let mut texture = VideoTexture::new(&gpu, 160, 90).expect("texture");

    // Warm-up: allocates the texture and the pipeline.
    texture
        .upload(&gpu, 160, 90, &frame)
        .expect("warm-up upload");
    {
        let mut recorder = FrameRecorder::new(&gpu);
        pass.record(&mut recorder, &mut texture, &dst, &params([0.0; 8]))
            .expect("warm-up draw");
        gpu.wait_frame(recorder.submit());
    }

    // A frame with a new picture: the staged copy, then the camera pass.
    texture.upload(&gpu, 160, 90, &frame).expect("upload");
    let mut recorder = FrameRecorder::new(&gpu);
    pass.record(&mut recorder, &mut texture, &dst, &params([0.0; 8]))
        .expect("draw");
    let timings = gpu.wait_frame(recorder.submit());
    let names: Vec<&str> = timings
        .per_pass_ms
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(
        names,
        vec!["overlay.upload", super::pass::PASS_NAME],
        "the frame times the copy and the camera pass"
    );

    // No new picture (the render thread is faster than the camera): no upload pass, and the last
    // picture is redrawn instead of the layer going black.
    let mut recorder = FrameRecorder::new(&gpu);
    pass.record(&mut recorder, &mut texture, &dst, &params([0.0; 8]))
        .expect("redraw");
    let redraw = gpu.wait_frame(recorder.submit());
    let redraw_names: Vec<&str> = redraw
        .per_pass_ms
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(
        redraw_names,
        vec![super::pass::PASS_NAME],
        "a redraw uploads nothing and draws the last picture"
    );
    assert_pixel(
        stage_pixel(&gpu.read_back(&dst), 64, 36, 2, 2),
        [255, 0, 0, 255],
        2,
        "the redrawn picture is still the checkerboard",
    );
    println!(
        "camera frame path: passes with a new frame {:?}, on a redraw {:?}",
        timings
            .per_pass_ms
            .iter()
            .map(|(name, ms)| format!("{name} {ms:.3} ms"))
            .collect::<Vec<_>>(),
        redraw
            .per_pass_ms
            .iter()
            .map(|(name, ms)| format!("{name} {ms:.3} ms"))
            .collect::<Vec<_>>()
    );
    assert_no_validation(&gpu);
}

/// A layer that was never given a camera, or whose camera is missing, draws transparent black and
/// never returns an error: a blown take costs a black layer, not a stopped show.
#[test]
fn missing_camera_draws_black_without_errors() {
    let gpu = gpu();
    let mut layer = CameraLayer::new(&gpu).expect("camera layer");
    let dst = LayerTarget::new(&gpu, 32, 18);

    // No source at all.
    {
        let mut recorder = FrameRecorder::new(&gpu);
        layer
            .draw(&mut recorder, 3, &dst, &params([0.0; 8]))
            .expect("draw a layer with no camera");
        gpu.wait_frame(recorder.submit());
    }
    let pixels = gpu.read_back(&dst);
    for y in 0..18u32 {
        for x in 0..32u32 {
            assert_pixel(
                pixel_at(&pixels, 32, x, y),
                [0, 0, 0, 0],
                0,
                &format!("no camera at ({x},{y})"),
            );
        }
    }

    // A camera name that does not exist: the capture thread reports the reason, the render thread
    // keeps drawing black.
    layer.set_source(0, Some("L-camera: no such camera"));
    for _ in 0..3 {
        let mut recorder = FrameRecorder::new(&gpu);
        layer
            .draw(&mut recorder, 0, &dst, &params([0.0; 8]))
            .expect("draw a missing camera");
        gpu.wait_frame(recorder.submit());
    }
    let pixels = gpu.read_back(&dst);
    assert_pixel(
        pixel_at(&pixels, 32, 16, 9),
        [0, 0, 0, 0],
        0,
        "missing camera",
    );

    // The reason shows up in the capture's stats within the retry interval, without the render
    // thread ever seeing an error.
    let deadline = Instant::now() + Duration::from_secs(5);
    let stats = loop {
        let stats = layer.slots[0]
            .as_ref()
            .expect("the slot was created by set_source")
            .capture
            .stats();
        if stats.last_error.is_some() || Instant::now() > deadline {
            break stats;
        }
        std::thread::sleep(Duration::from_millis(25));
    };
    assert!(
        stats.last_error.is_some(),
        "a missing camera is reported through CameraCapture::stats"
    );
    assert!(
        !stats.open,
        "nothing is open while the name does not match a device"
    );
    println!(
        "missing camera: {} attempt(s), last error {:?}",
        stats.reopen_attempts, stats.last_error
    );
    assert_no_validation(&gpu);
}

/// The camera names Windows reports, printed. No GPU and no capture: this must never panic, even
/// on a machine with no cameras at all.
#[test]
fn devices_lists_without_panicking() {
    let devices = CameraLayer::devices();
    println!("{} camera(s):", devices.len());
    for (index, name) in devices.iter().enumerate() {
        println!("  {index}: {name}");
        assert!(
            !name.trim().is_empty(),
            "an unnamed device still gets a label"
        );
    }
}

/// The camera shader compiles, and what it declares matches what `CameraPass` builds: a 48-byte
/// uniform block at binding 0, the frame at binding 1 and the pipeline's immutable sampler at
/// binding 2. This is the pure (no GPU, no camera) half of the shader gate; `params.rs` pins the
/// uniform's contents.
#[test]
fn the_camera_shader_matches_the_uniform_block_and_bindings() {
    let source = include_str!("overlay_camera.wgsl");
    let words = mosh_shaders::compile_wgsl(source).expect("the camera shader compiles");
    assert!(words.len() > 20, "only {} SPIR-V words", words.len());

    let module = naga::front::wgsl::parse_str(source).expect("the camera shader parses");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    );
    validator
        .validate(&module)
        .expect("the camera shader validates");

    let mut uniform_bytes = None;
    let mut bindings = Vec::new();
    for (_, var) in module.global_variables.iter() {
        let Some(binding) = var.binding else { continue };
        match var.space {
            naga::AddressSpace::Uniform => {
                if let naga::TypeInner::Struct { span, .. } = module.types[var.ty].inner {
                    uniform_bytes = Some(span);
                }
            }
            naga::AddressSpace::Handle => match module.types[var.ty].inner {
                naga::TypeInner::Image { .. } | naga::TypeInner::Sampler { .. } => {
                    bindings.push(binding.binding);
                }
                _ => {}
            },
            _ => {}
        }
    }
    bindings.sort_unstable();
    assert_eq!(
        uniform_bytes,
        Some(super::params::UNIFORM_BYTES as u32),
        "the shader's uniform block must be the {} bytes the CPU writes",
        super::params::UNIFORM_BYTES
    );
    assert_eq!(
        bindings,
        vec![1, 2],
        "the frame is binding 1 and the pipeline's immutable sampler is binding 2"
    );
    println!(
        "camera shader: {} SPIR-V words, uniform {} bytes, texture/sampler bindings {bindings:?}",
        words.len(),
        super::params::UNIFORM_BYTES
    );
}

/// Render-thread cost of a 1080p camera frame: the upload memcpy (the handover from the capture
/// thread) and the recorded passes (the staged copy and the camera draw, no fence wait).
///
/// The camera pass is the overlay cover pass plus one uniform block and one sample, so this
/// measures both in the same run and compares them instead of hard-coding a number that depends on
/// how loaded the machine is: the camera layer must cost no more than an overlay video layer plus
/// a tenth of a millisecond, and must stay well inside the README's 2 ms render-thread budget.
/// (The committed overlay budget test hard-codes 0.2 ms; on a loaded machine that assertion fails
/// on this checkout today, which is exactly why the comparison is relative.)
#[test]
fn camera_cpu_time_per_frame_1080p() {
    let gpu = gpu();
    let (width, height) = (1920u32, 1080u32);
    let dst = LayerTarget::new(&gpu, width, height);
    let frame = checkerboard(width, height);
    let pass = CameraPass::new(&gpu).expect("camera pass");
    let mut texture = VideoTexture::new(&gpu, width, height).expect("texture");
    // The baseline: the overlay cover pass the camera pass is modelled on.
    let mut cover = mosh_render::upload::CoverPass::new(
        &gpu,
        mosh_shaders::spirv("overlay_video").expect("the overlay_video shader"),
    )
    .expect("overlay cover pass");
    let mut cover_texture = VideoTexture::new(&gpu, width, height).expect("overlay texture");

    // Two warm-up frames each: the first allocates, the first draw builds the descriptor set.
    for _ in 0..2 {
        texture
            .upload(&gpu, width, height, &frame)
            .expect("warm-up upload");
        let mut recorder = FrameRecorder::new(&gpu);
        pass.record(&mut recorder, &mut texture, &dst, &params([0.0; 8]))
            .expect("warm-up draw");
        gpu.wait_frame(recorder.submit());

        cover_texture
            .upload(&gpu, width, height, &frame)
            .expect("overlay warm-up upload");
        let mut recorder = FrameRecorder::new(&gpu);
        cover
            .record(&mut recorder, &mut cover_texture, &dst)
            .expect("overlay warm-up draw");
        gpu.wait_frame(recorder.submit());
    }

    let frames = 30;
    let mut store_samples = Vec::with_capacity(frames);
    let mut record_samples = Vec::with_capacity(frames);
    let mut cover_samples = Vec::with_capacity(frames);
    for _ in 0..frames {
        let start = Instant::now();
        texture.upload(&gpu, width, height, &frame).expect("upload");
        store_samples.push(start.elapsed().as_secs_f64() * 1000.0);

        let mut recorder = FrameRecorder::new(&gpu);
        let start = Instant::now();
        pass.record(&mut recorder, &mut texture, &dst, &params([0.0; 8]))
            .expect("draw");
        record_samples.push(start.elapsed().as_secs_f64() * 1000.0);
        gpu.wait_frame(recorder.submit());

        cover_texture
            .upload(&gpu, width, height, &frame)
            .expect("overlay upload");
        let mut recorder = FrameRecorder::new(&gpu);
        let start = Instant::now();
        cover
            .record(&mut recorder, &mut cover_texture, &dst)
            .expect("overlay draw");
        cover_samples.push(start.elapsed().as_secs_f64() * 1000.0);
        gpu.wait_frame(recorder.submit());
    }
    let stats = |samples: &mut Vec<f64>| {
        samples.sort_by(f64::total_cmp);
        (samples[0], samples[samples.len() / 2])
    };
    let (store_min, store_median) = stats(&mut store_samples);
    let (record_min, record_median) = stats(&mut record_samples);
    let (cover_min, cover_median) = stats(&mut cover_samples);
    assert_pixel(
        stage_pixel(&gpu.read_back(&dst), width, height, 10, 10),
        [255, 0, 0, 255],
        2,
        "1080p checkerboard",
    );
    assert!(
        record_min <= cover_min + 0.1,
        "the camera pass must cost no more than the overlay cover pass plus 0.1 ms: camera \
         {record_min:.3} ms, overlay {cover_min:.3} ms (medians {record_median:.3} / \
         {cover_median:.3})"
    );
    assert!(
        record_min < 1.0,
        "the camera layer must stay well inside the 2 ms render-thread budget, min {record_min:.3} ms"
    );
    println!(
        "1920x1080, {frames} frames, {} MB per frame: camera record (copy + draw, no fence) \
         {record_min:.3} / {record_median:.3} ms min/median, overlay cover pass {cover_min:.3} / \
         {cover_median:.3} ms in the same run, camera upload memcpy {store_min:.3} / \
         {store_median:.3} ms",
        (width * height * 4) / (1024 * 1024)
    );
    assert_no_validation(&gpu);
}

/// The real default camera: opens it, draws through the whole layer path and reports what came
/// back. `#[ignore]` because a machine may have no camera and because it takes a few seconds; run
/// it with `cargo test -p mosh-visuals --lib camera::tests::default_camera -- --ignored --nocapture`.
///
/// Windows can spend seconds negotiating RGB32 with a UVC camera's video processor, so the test
/// first draws while waiting (up to `OPEN_TIMEOUT`) for the first frame, then measures three
/// seconds of frames. That wait is on the capture thread, and the draws during it are black: this
/// test also proves the render thread keeps producing frames while a camera opens.
#[test]
#[ignore = "needs a webcam (or prints that there is none) and runs for a few seconds"]
fn default_camera_for_three_seconds() {
    const OPEN_TIMEOUT: Duration = Duration::from_secs(15);
    let gpu = gpu();
    let devices = CameraLayer::devices();
    println!("{} camera(s): {devices:?}", devices.len());
    if devices.is_empty() {
        println!("default_camera_for_three_seconds: no camera found, nothing captured");
        return;
    }

    let mut layer = CameraLayer::new(&gpu).expect("camera layer");
    let dst = LayerTarget::new(&gpu, 960, 540);
    let opened_at = Instant::now();
    layer.set_source(0, None);

    // Draw while the capture thread opens the device: every frame here is black, and none blocks.
    let mut draws_while_opening = 0u32;
    let mut first_frame_after = None;
    while opened_at.elapsed() < OPEN_TIMEOUT {
        let mut recorder = FrameRecorder::new(&gpu);
        layer
            .draw(&mut recorder, 0, &dst, &params([0.0; 8]))
            .expect("draw while the camera opens");
        gpu.wait_frame(recorder.submit());
        draws_while_opening += 1;
        let stats = layer.slots[0]
            .as_ref()
            .expect("the slot was created by set_source")
            .capture
            .stats();
        if stats.frames > 0 {
            first_frame_after = Some(opened_at.elapsed());
            break;
        }
        if stats.last_error.is_some() && !stats.open {
            // A retry is in flight; keep drawing, the capture thread tries again in two seconds.
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    let run = Duration::from_secs(3);
    let start = Instant::now();
    let mut record_samples = Vec::new();
    let mut draws = 0u32;
    while start.elapsed() < run {
        let mut recorder = FrameRecorder::new(&gpu);
        let at = Instant::now();
        layer
            .draw(&mut recorder, 0, &dst, &params([0.0; 8]))
            .expect("draw the camera");
        record_samples.push(at.elapsed().as_secs_f64() * 1000.0);
        gpu.wait_frame(recorder.submit());
        draws += 1;
        // A 2 ms breather keeps this near a camera's rate instead of spinning the GPU flat out.
        std::thread::sleep(Duration::from_millis(2));
    }
    let elapsed = start.elapsed().as_secs_f64();
    let stats = layer.slots[0]
        .as_ref()
        .expect("the slot was created by set_source")
        .capture
        .stats();
    let has_picture = layer.slots[0].as_ref().is_some_and(|slot| slot.has_picture);

    record_samples.sort_by(f64::total_cmp);
    let record_min = record_samples.first().copied().unwrap_or(f64::NAN);
    let record_median = record_samples
        .get(record_samples.len() / 2)
        .copied()
        .unwrap_or(f64::NAN);
    let record_max = record_samples.last().copied().unwrap_or(f64::NAN);
    let pixels = gpu.read_back(&dst);
    let mean = pixels
        .chunks(4)
        .map(|pixel| {
            u64::from((u32::from(pixel[0]) + u32::from(pixel[1]) + u32::from(pixel[2])) / 3)
        })
        .sum::<u64>() as f64
        / (pixels.len() / 4).max(1) as f64;

    println!(
        "default camera: first frame after {:?} ({draws_while_opening} black draws while opening, \
         {} open attempt(s))",
        first_frame_after, stats.reopen_attempts
    );
    println!(
        "default camera for {elapsed:.2} s: {} frame(s) captured ({:.1}/s), {draws} draws, \
         device {:?} at {:?}, open {}, picture {}, last error {:?}",
        stats.frames,
        stats.frames as f64 / elapsed,
        stats.device,
        stats.frame,
        stats.open,
        has_picture,
        stats.last_error
    );
    println!(
        "render thread per frame: record {record_min:.3} / {record_median:.3} / {record_max:.3} ms \
         min/median/max over {draws} frames; mean output luma {mean:.1} of 255"
    );
    if stats.frames > 0 {
        assert!(has_picture, "a captured frame reaches the layer texture");
        assert!(
            mean > 1.0,
            "the camera picture is not black (mean luma {mean:.1}); a covered or disabled camera \
             can legitimately deliver black frames"
        );
    } else {
        println!(
            "no frame arrived within {OPEN_TIMEOUT:?}: {}",
            stats.last_error.as_deref().unwrap_or("no error reported")
        );
    }
    assert_no_validation(&gpu);

    if let Some(slot) = layer.slots[0].as_mut() {
        slot.capture.shutdown();
    }
}
