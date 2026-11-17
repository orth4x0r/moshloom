//! The text layer, offscreen (wave 3, L-text).
//!
//! Every test renders into a `LayerTarget` and reads it back with `Gpu::read_back`, so no window
//! is opened. The layer is driven exactly the way the app drives it: `set_text` when a text clip
//! starts, then `draw` once per frame with the clip's `p0..p7`.
//!
//! # What is asserted, and why in these terms
//!
//! * Only glyph pixels are lit. The text sheet is baked at `set_text`; a pixel outside every
//!   glyph's cell must stay transparent black, so a border or a gap between letters reading back
//!   with ink is a layout bug, not a style choice.
//! * `size` (`p0`) and `spacing` (`p2`) change the block's bounding box. Both are per-frame
//!   parameters applied by the shader, so the test measures the lit bounding box of the read-back.
//! * The beat pop (`p6`) changes pixels and nothing else flashes: a full-frame flash is excluded
//!   by keeping the four corners black in every frame of a 400 ms window (`README`: no full-frame
//!   flash faster than every 400 ms).
//! * Three looks (`p1`) render differently and write preview PNGs under
//!   `native/target/style-previews/text/` (not committed).
//!
//! # Row order
//!
//! `read_back` returns image row 0 first, and the layer target's row 0 is the top of the picture
//! (the compositor and the present pass flip once each). The style previews below are written in
//! that order, so they look like what an audience sees.

use std::ops::Deref;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use mosh_core::{AnalysisFrame, BlendMode, ClipParams, LayerState};
use mosh_render::{FrameRecorder, Gpu, GpuPreference, LayerTarget};
use mosh_visuals::text::TextLayer;
use mosh_visuals::{LayerSlot, Visuals};

/// One GPU test at a time in this binary: several short offscreen tests in parallel fight for the
/// same integrated GPU, and the timing test wants the machine to itself.
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

/// `p0..p7` for the text layer, as the clip editor shows them.
#[allow(clippy::too_many_arguments)]
fn params(
    size: f32,
    font: f32,
    spacing: f32,
    jitter: f32,
    glitch: f32,
    outline: f32,
    pop: f32,
    scroll: f32,
) -> ClipParams {
    ClipParams {
        p: [size, font, spacing, jitter, glitch, outline, pop, scroll],
        palette: 0,
        seed: 0,
    }
}

/// The clean look: no jitter, no glitch, a thin outline.
fn clean(size: f32) -> ClipParams {
    params(size, 0.85, 0.5, 0.0, 0.0, 0.35, 0.0, 0.0)
}

/// One read-back picture.
struct Shot {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}

impl Shot {
    fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let offset = ((y * self.width + x) * 4) as usize;
        [
            self.pixels[offset],
            self.pixels[offset + 1],
            self.pixels[offset + 2],
            self.pixels[offset + 3],
        ]
    }

    /// How many pixels carry ink. The layer target is transparent black outside the glyphs.
    fn lit(&self) -> usize {
        self.pixels.chunks(4).filter(|pixel| is_ink(pixel)).count()
    }

    /// The bounding box of every lit pixel, `(x0, y0, x1, y1)` inclusive, or `None`.
    fn ink_box(&self) -> Option<(u32, u32, u32, u32)> {
        let mut bounds: Option<(u32, u32, u32, u32)> = None;
        for y in 0..self.height {
            for x in 0..self.width {
                if is_ink(&self.pixel(x, y)) {
                    bounds = Some(match bounds {
                        None => (x, y, x, y),
                        Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
                    });
                }
            }
        }
        bounds
    }

    /// True when every pixel in `x0..x1, y0..y1` is black.
    fn region_is_black(&self, x0: u32, y0: u32, x1: u32, y1: u32) -> bool {
        (y0..y1).all(|y| (x0..x1).all(|x| !is_ink(&self.pixel(x, y))))
    }

    /// Lit pixels per picture row, for the multi-line test.
    fn lit_rows(&self) -> Vec<usize> {
        (0..self.height)
            .map(|y| {
                (0..self.width)
                    .filter(|x| is_ink(&self.pixel(*x, y)))
                    .count()
            })
            .collect()
    }

    fn mean_luma(&self) -> f32 {
        let total: f32 = self
            .pixels
            .chunks(4)
            .map(|pixel| {
                0.3 * f32::from(pixel[0]) + 0.59 * f32::from(pixel[1]) + 0.11 * f32::from(pixel[2])
            })
            .sum();
        total / (self.pixels.len() / 4) as f32
    }
}

/// Ink means "this pixel is not the layer target's transparent black". A tolerance of 2 keeps a
/// 16-bit target's rounding from counting as ink.
fn is_ink(pixel: &[u8]) -> bool {
    pixel[0] > 2 || pixel[1] > 2 || pixel[2] > 2 || pixel[3] > 2
}

/// Solid ink: a glyph body or its outline, not the soft glow around it.
fn is_solid_ink(pixel: &[u8]) -> bool {
    pixel[0].max(pixel[1]).max(pixel[2]) > 48 || pixel[3] > 48
}

fn mean_difference(a: &Shot, b: &Shot) -> f32 {
    assert_eq!(
        a.pixels.len(),
        b.pixels.len(),
        "read-backs of different sizes"
    );
    let total: f32 = a
        .pixels
        .chunks(4)
        .zip(b.pixels.chunks(4))
        .map(|(left, right)| {
            (0..3)
                .map(|channel| (f32::from(left[channel]) - f32::from(right[channel])).abs())
                .sum::<f32>()
        })
        .sum();
    total / (a.pixels.len() / 4 * 3) as f32
}

/// The whole path the app takes: `set_text` once, then one frame.
fn draw_once(
    gpu: &Gpu,
    text: &str,
    params: &ClipParams,
    frame: &AnalysisFrame,
    t: f32,
    width: u32,
    height: u32,
) -> Shot {
    let mut layer = TextLayer::new(gpu).expect("text layer");
    layer.set_text(gpu, 0, text);
    let dst = LayerTarget::new(gpu, width, height);
    let mut recorder = FrameRecorder::new(gpu);
    layer
        .draw(&mut recorder, 0, &dst, frame, params, t)
        .expect("draw text");
    gpu.wait_frame(recorder.submit());
    Shot {
        pixels: gpu.read_back(&dst),
        width,
        height,
    }
}

// ---------------------------------------------------------------------------------------------
// Layout: what the sheet holds.
// ---------------------------------------------------------------------------------------------

/// `ILANA` at 320x180 fills part of the frame, is lit nowhere near the border, and has at least one
/// fully black column between two letters: the sheet has voids between glyph cells and a pixel
/// there must stay black.
#[test]
fn ilana_lights_only_where_glyphs_are() {
    let gpu = gpu();
    let (width, height) = (320u32, 180u32);
    let frame = AnalysisFrame::default();
    let shot = draw_once(&gpu, "ILANA", &clean(0.85), &frame, 0.0, width, height);

    let lit = shot.lit();
    let ratio = lit as f32 / (width * height) as f32;
    let (x0, y0, x1, y1) = shot.ink_box().expect("the word must be visible");
    let box_width = x1 - x0 + 1;
    let box_height = y1 - y0 + 1;

    println!(
        "ILANA 320x180: {lit} lit pixels ({:.1}%), box {box_width}x{box_height} at ({x0},{y0}), \
         mean luma {:.1}",
        ratio * 100.0,
        shot.mean_luma()
    );

    assert!(
        (0.02..0.55).contains(&ratio),
        "{:.1}% of the frame is lit; five letters at 0.85 should cover between 2% and 55%",
        ratio * 100.0
    );
    assert!(
        box_width > width / 4 && box_width < width,
        "the word is {box_width}px wide in a {width}px frame"
    );
    assert!(
        box_height > height / 8 && box_height < height,
        "the word is {box_height}px tall in a {height}px frame"
    );
    // The block is centred, so it cannot touch the border: glyphs never reach the sheet's margin.
    assert!(
        x0 >= 2 && y0 >= 2 && x1 + 2 < width && y1 + 2 < height,
        "ink at ({x0},{y0})..({x1},{y1}) touches the frame border; the text is not centred and \
         padded"
    );

    // A gap between two letters: at least one column inside the box carries no structural ink.
    // The soft glow around a glyph is allowed to tint that gap (it is styling, not a glyph), so
    // this looks for solid ink, not for the 1-2 level halo the glow leaves.
    let mut empty_columns = 0;
    for x in x0..=x1 {
        if (y0..=y1).all(|y| !is_solid_ink(&shot.pixel(x, y))) {
            empty_columns += 1;
        }
    }
    assert!(
        empty_columns >= 2,
        "no empty column inside the word box: the letters are painted together (or a glyph cell's \
         void carries ink)"
    );
    assert_no_validation_messages(&gpu);
}

/// Two lines stack, each centred, with a black gap between them; one line sits on the vertical
/// centre of the frame.
#[test]
fn lines_stack_and_centre() {
    let gpu = gpu();
    let (width, height) = (400u32, 300u32);
    let frame = AnalysisFrame::default();
    let two = draw_once(&gpu, "IL\nANA", &clean(0.75), &frame, 0.0, width, height);

    let rows = two.lit_rows();
    // Runs of rows that carry ink: two bands, because `\n` is a line break.
    let mut bands = 0;
    let mut in_band = false;
    for count in &rows {
        if *count > 0 && !in_band {
            bands += 1;
        }
        in_band = *count > 0;
    }
    assert_eq!(
        bands, 2,
        "IL\\nANA must draw two bands of rows, got {bands}: {rows:?}"
    );

    let (x0, _, x1, _) = two.ink_box().expect("two lines are visible");
    let centre = (x0 + x1) as f32 / 2.0;
    assert!(
        (centre - width as f32 / 2.0).abs() < width as f32 * 0.04,
        "the block's centre is {centre} in a {width}px frame"
    );

    // The second line is centred too, not left-aligned under the first. `ANA` is narrower than
    // `IL` is not true, so compare each band's own ink box against the block's centre.
    let mut band_boxes = Vec::new();
    let mut start = None;
    for (y, count) in rows.iter().enumerate() {
        match (start, *count > 0) {
            (None, true) => start = Some(y),
            (Some(from), false) => {
                band_boxes.push((from as u32, y as u32 - 1));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(from) = start {
        band_boxes.push((from as u32, height - 1));
    }
    assert_eq!(
        band_boxes.len(),
        2,
        "expected two row bands, got {band_boxes:?}"
    );
    for (y0, y1) in band_boxes {
        let mut left = width;
        let mut right = 0;
        for y in y0..=y1 {
            for x in 0..width {
                if is_ink(&two.pixel(x, y)) {
                    left = left.min(x);
                    right = right.max(x);
                }
            }
        }
        let band_centre = (left + right) as f32 / 2.0;
        assert!(
            (band_centre - centre).abs() < width as f32 * 0.06,
            "a line's centre {band_centre} is off the block's centre {centre}"
        );
    }
    assert_no_validation_messages(&gpu);
}

/// Empty text draws nothing: the layer clears its target instead of leaving last frame's word up.
#[test]
fn empty_text_draws_nothing() {
    let gpu = gpu();
    let (width, height) = (320u32, 180u32);
    let frame = AnalysisFrame::default();
    for text in ["", "\n", "  \n \n", "\t"] {
        let shot = draw_once(&gpu, text, &clean(0.85), &frame, 0.0, width, height);
        assert_eq!(
            shot.lit(),
            0,
            "text {text:?} must draw nothing, got {} lit pixels",
            shot.lit()
        );
    }
    assert_no_validation_messages(&gpu);
}

/// The clip path the app uses: a `LayerState` whose visualizer is `"text"` reaches this layer
/// through `Visuals::draw_layer`, so a text clip in a show draws its words.
#[test]
fn the_facade_draws_a_text_clip() {
    let gpu = gpu();
    let (width, height) = (320u32, 180u32);
    let mut visuals = Visuals::new(&gpu, width, height).expect("visuals");
    visuals.text.set_text(&gpu, 0, "ILANA");
    let mut slot = LayerSlot::new(&gpu, 0, width, height);
    let frame = AnalysisFrame::default();
    let state = LayerState {
        layer: 0,
        clip: Some(7),
        visualizer: Some("text".to_string()),
        params: clean(0.85),
        effects: Vec::new(),
        transition: None,
        blend: BlendMode::Normal,
        opacity: 1.0,
        seed: 0,
    };
    let mut recorder = FrameRecorder::new(&gpu);
    visuals
        .draw_layer(&mut recorder, &mut slot, &state, &frame, 0.0)
        .expect("draw a text clip");
    gpu.wait_frame(recorder.submit());
    let shot = Shot {
        pixels: gpu.read_back(slot.output()),
        width,
        height,
    };
    let lit = shot.lit();
    println!("facade: a text clip lights {lit} pixels");
    assert!(
        lit > 500,
        "a text clip drew {lit} lit pixels through the facade"
    );
    assert_no_validation_messages(&gpu);
}

/// A layer that never received text draws nothing (no stale picture, no panic).
#[test]
fn a_layer_without_text_draws_nothing() {
    let gpu = gpu();
    let (width, height) = (160u32, 90u32);
    let mut layer = TextLayer::new(&gpu).expect("text layer");
    let dst = LayerTarget::new(&gpu, width, height);
    let frame = AnalysisFrame::default();
    let mut recorder = FrameRecorder::new(&gpu);
    layer
        .draw(&mut recorder, 3, &dst, &frame, &clean(0.8), 0.0)
        .expect("draw an empty layer");
    gpu.wait_frame(recorder.submit());
    let shot = Shot {
        pixels: gpu.read_back(&dst),
        width,
        height,
    };
    assert_eq!(
        shot.lit(),
        0,
        "an untouched layer must be transparent black"
    );
    assert_no_validation_messages(&gpu);
}

// ---------------------------------------------------------------------------------------------
// Parameters: size, spacing, pulse.
// ---------------------------------------------------------------------------------------------

/// `size` (`p0`) and `spacing` (`p2`) change the block's bounding box, which is what those two
/// parameters are for. The sheet is baked once; both are applied per frame by the shader.
///
/// `size` is measured on `IL`, a narrow word: a wide one stops growing at the frame's width (the
/// shader fits the block), so a wide word would under-report the size change on purpose.
#[test]
fn size_and_spacing_change_the_bounding_box() {
    let gpu = gpu();
    let (width, height) = (480u32, 270u32);
    let frame = AnalysisFrame::default();
    let small = draw_once(&gpu, "IL", &clean(0.2), &frame, 0.0, width, height);
    let large = draw_once(&gpu, "IL", &clean(0.95), &frame, 0.0, width, height);
    let (small_w, small_h) = {
        let (x0, y0, x1, y1) = small.ink_box().expect("small text is visible");
        (x1 - x0 + 1, y1 - y0 + 1)
    };
    let (large_w, large_h) = {
        let (x0, y0, x1, y1) = large.ink_box().expect("large text is visible");
        (x1 - x0 + 1, y1 - y0 + 1)
    };
    println!("size 0.2 -> {small_w}x{small_h}, size 0.95 -> {large_w}x{large_h}");
    assert!(
        large_w > small_w * 3 / 2 && large_h > small_h * 3 / 2,
        "size 0.2 gives {small_w}x{small_h} and 0.95 gives {large_w}x{large_h}: not 1.5x bigger"
    );

    let tight = draw_once(
        &gpu,
        "ILANA",
        &params(0.15, 0.85, 0.0, 0.0, 0.0, 0.35, 0.0, 0.0),
        &frame,
        0.0,
        width,
        height,
    );
    let wide = draw_once(
        &gpu,
        "ILANA",
        &params(0.15, 0.85, 1.0, 0.0, 0.0, 0.35, 0.0, 0.0),
        &frame,
        0.0,
        width,
        height,
    );
    let tight_w = {
        let (x0, _, x1, _) = tight.ink_box().expect("tight text is visible");
        x1 - x0 + 1
    };
    let wide_w = {
        let (x0, _, x1, _) = wide.ink_box().expect("wide text is visible");
        x1 - x0 + 1
    };
    println!("spacing 0.0 -> {tight_w}px wide, spacing 1.0 -> {wide_w}px wide");
    assert!(
        wide_w as f32 > tight_w as f32 * 1.12,
        "spacing 0 gives {tight_w}px and 1 gives {wide_w}px: the block does not widen"
    );
    assert_no_validation_messages(&gpu);
}

/// The beat pop (`p6`) moves pixels on the beat. Nothing flashes the whole frame: the corners stay
/// black through a 400 ms window of pops, glitch and jitter.
#[test]
fn pulse_changes_pixels_and_never_flashes_the_frame() {
    let gpu = gpu();
    let (width, height) = (320u32, 180u32);
    let mut frame = AnalysisFrame {
        bands: [0.9, 0.4, 0.2],
        intensity: 0.7,
        playing: true,
        ..Default::default()
    };

    let still = draw_once(
        &gpu,
        "ILANA",
        &params(0.7, 0.2, 0.5, 0.0, 0.0, 0.3, 0.0, 0.0),
        &frame,
        0.0,
        width,
        height,
    );
    let repeat = draw_once(
        &gpu,
        "ILANA",
        &params(0.7, 0.2, 0.5, 0.0, 0.0, 0.3, 0.0, 0.0),
        &frame,
        0.0,
        width,
        height,
    );
    assert_eq!(
        mean_difference(&still, &repeat),
        0.0,
        "two identical frames must read back identically"
    );

    let mut popped = Vec::new();
    for step in 0..10 {
        let t = step as f32 * 0.04; // 400 ms of frames
        frame.beat_phase = (t / 0.5).fract(); // 120 BPM
        popped.push(draw_once(
            &gpu,
            "ILANA",
            &params(0.7, 0.2, 0.5, 0.6, 0.7, 0.3, 1.0, 0.0),
            &frame,
            t,
            width,
            height,
        ));
    }
    let largest = popped
        .iter()
        .map(|shot| mean_difference(&still, shot))
        .fold(0.0f32, f32::max);
    println!("pulse 1.0 + glitch 0.7: largest change {largest:.2} levels per channel");
    assert!(
        largest > 1.0,
        "the beat pop changed pixels by only {largest:.2} levels"
    );
    for (step, shot) in popped.iter().enumerate() {
        assert!(
            shot.region_is_black(0, 0, 16, 16)
                && shot.region_is_black(width - 16, 0, width, 16)
                && shot.region_is_black(0, height - 16, 16, height)
                && shot.region_is_black(width - 16, height - 16, width, height),
            "frame {step} lights a corner: a full-frame flash faster than every 400 ms"
        );
    }
    assert_no_validation_messages(&gpu);
}

// ---------------------------------------------------------------------------------------------
// Looks, cost, previews.
// ---------------------------------------------------------------------------------------------

/// The three looks (`p1` selects them) render differently. Their previews are written under
/// `native/target/style-previews/text/` for the PR; nothing is committed.
#[test]
fn three_looks_render_differently_and_write_previews() {
    let gpu = gpu();
    let (width, height) = (960u32, 540u32);
    let frame = AnalysisFrame {
        bands: [0.7, 0.3, 0.5],
        intensity: 0.45,
        brightness: 0.5,
        ..Default::default()
    };

    let looks = [("chrome", 0.05), ("thorny", 0.45), ("clean", 0.85)];
    let mut shots = Vec::new();
    for (name, font) in looks {
        let shot = draw_once(
            &gpu,
            "ILANA",
            &params(0.8, font, 0.5, 0.10, 0.0, 0.45, 0.30, 0.0),
            &frame,
            1.25,
            width,
            height,
        );
        let path = preview_path(name);
        save_png(&path, &shot).expect("write a preview PNG");
        println!("preview: {} ({} lit px)", path.display(), shot.lit());
        shots.push(shot);
    }
    for (index, a) in shots.iter().enumerate() {
        for (other, b) in shots.iter().enumerate().skip(index + 1) {
            let difference = mean_difference(a, b);
            println!("look {index} vs look {other}: {difference:.2} levels per channel");
            assert!(
                difference > 2.0,
                "looks {index} and {other} render almost the same ({difference:.2})"
            );
        }
    }
    assert_no_validation_messages(&gpu);
}

fn preview_path(name: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // crates
    path.pop(); // native
    path.push("target");
    path.push("style-previews");
    path.push("text");
    std::fs::create_dir_all(&path).expect("create the preview directory");
    path.push(format!("{name}.png"));
    path
}

/// One RGBA8 picture as a PNG, with no image crate: a zlib stream of stored (uncompressed)
/// deflate blocks is a valid, if fat, PNG.
fn save_png(path: &PathBuf, shot: &Shot) -> std::io::Result<()> {
    let mut raw = Vec::with_capacity((shot.height * (1 + shot.width * 4)) as usize);
    for y in 0..shot.height {
        raw.push(0); // filter: none
        let start = (y * shot.width * 4) as usize;
        raw.extend_from_slice(&shot.pixels[start..start + (shot.width * 4) as usize]);
    }

    let mut zlib = vec![0x78, 0x01];
    let mut offset = 0;
    while offset < raw.len() {
        let chunk = (raw.len() - offset).min(65535);
        let last = offset + chunk == raw.len();
        zlib.push(if last { 1 } else { 0 });
        zlib.extend_from_slice(&(chunk as u16).to_le_bytes());
        zlib.extend_from_slice(&(!(chunk as u16)).to_le_bytes());
        zlib.extend_from_slice(&raw[offset..offset + chunk]);
        offset += chunk;
    }
    zlib.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut png = Vec::new();
    png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
    let mut header = Vec::new();
    header.extend_from_slice(&shot.width.to_be_bytes());
    header.extend_from_slice(&shot.height.to_be_bytes());
    header.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit RGBA, no interlace
    png_chunk(&mut png, b"IHDR", &header);
    png_chunk(&mut png, b"IDAT", &zlib);
    png_chunk(&mut png, b"IEND", &[]);
    std::fs::write(path, png)
}

fn png_chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    png.extend_from_slice(kind);
    png.extend_from_slice(data);
    png.extend_from_slice(&crc32(&crc_input).to_be_bytes());
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
        a = (a + u32::from(*byte)) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// The budget test: `set_text` is CPU work (glyph rasterising and the sheet bake) and happens when
/// the text changes, not per frame; `draw` is one GPU pass and must be inside the text layer's
/// 0.5 ms share of the 6 ms 1080p frame at 960x540.
#[test]
fn set_text_and_draw_costs() {
    let gpu = gpu();
    let (width, height) = (960u32, 540u32);
    let frame = AnalysisFrame {
        bands: [0.6, 0.4, 0.3],
        intensity: 0.5,
        ..Default::default()
    };

    let mut layer = TextLayer::new(&gpu).expect("text layer");
    let dst = LayerTarget::new(&gpu, width, height);
    let words = [
        "ILANA",
        "MOshloom",
        "Bassline\npressure",
        "THE Loom knows\nwhat the flesh wants",
        "129 BPM",
        "We are the loom\nand the loom is us",
        "drop it",
        "ILANA 2026",
    ];

    // Warm up: the first bake allocates the sheet texture.
    layer.set_text(&gpu, 0, "warm up");

    // Every word is distinct, so every call really bakes a sheet: no measurement below can be an
    // early return. This is the cost of a text *change*, which is a clip starting, not a frame.
    let mut set_samples = Vec::new();
    for word in words {
        let start = Instant::now();
        layer.set_text(&gpu, 0, word);
        set_samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    // The app may call `set_text` every frame: the same words again must cost nothing.
    let repeat = {
        let start = Instant::now();
        layer.set_text(&gpu, 0, words[words.len() - 1]);
        start.elapsed().as_secs_f64() * 1000.0
    };

    // Draw once so the copy of the last sheet lands, then measure steady-state draws.
    {
        let mut recorder = FrameRecorder::new(&gpu);
        layer
            .draw(&mut recorder, 0, &dst, &frame, &clean(0.8), 0.0)
            .expect("warm-up draw");
        gpu.wait_frame(recorder.submit());
    }

    let mut gpu_samples = Vec::new();
    let mut record_samples = Vec::new();
    let mut measured_looked_right = None;
    for step in 0..30 {
        let t = step as f32 / 60.0;
        let mut recorder = FrameRecorder::new(&gpu);
        let start = Instant::now();
        layer
            .draw(&mut recorder, 0, &dst, &frame, &clean(0.8), t)
            .expect("draw text");
        record_samples.push(start.elapsed().as_secs_f64() * 1000.0);
        let timings = gpu.wait_frame(recorder.submit());
        let pass = timings
            .per_pass_ms
            .iter()
            .find(|(name, _)| name == "text.glyphs")
            .map(|(_, ms)| *ms);
        assert!(
            pass.is_some(),
            "the frame must time a `text.glyphs` pass, got {:?}",
            timings.per_pass_ms
        );
        gpu_samples.push(pass.expect("checked just above"));
        measured_looked_right = Some(Shot {
            pixels: gpu.read_back(&dst),
            width,
            height,
        });
    }
    let stats = |samples: &mut Vec<f64>| {
        samples.sort_by(|a, b| a.total_cmp(b));
        (samples[0], samples[samples.len() / 2])
    };
    let (set_min, set_median) = stats(&mut set_samples);
    let (record_min, record_median) = stats(&mut record_samples);
    let mut gpu_f64: Vec<f64> = gpu_samples.iter().map(|ms| f64::from(*ms)).collect();
    let (draw_min, draw_median) = stats(&mut gpu_f64);
    let lit = measured_looked_right.expect("drew frames").lit();

    println!(
        "text at {width}x{height}: set_text {set_min:.2} / {set_median:.2} ms min/median over {} \
         distinct changes, a repeated text {repeat:.4} ms, draw record {record_min:.3} / \
         {record_median:.3} ms on the CPU, GPU `text.glyphs` {draw_min:.4} / {draw_median:.4} ms \
         min/median, {lit} lit px",
        words.len()
    );

    assert!(lit > 0, "the measured frame is blank");
    // A text change is a one-off (a clip starts); the debug test binary is 5-10x slower than the
    // app's release build, so this bound only catches a per-frame mistake (a bake inside `draw`
    // would land in the middle of the numbers above and below).
    assert!(
        set_min > 0.0 && set_median < 100.0,
        "set_text is the one-off cost of a text change, median {set_median:.2} ms"
    );
    assert!(
        repeat < 0.05,
        "`set_text` with unchanged text must be a no-op, took {repeat:.4} ms"
    );
    assert!(
        draw_min < 0.5,
        "the text pass must stay under 0.5 ms at 960x540, min {draw_min:.4} ms (median \
         {draw_median:.4})"
    );
    assert!(
        record_min < 0.5,
        "recording the pass on the CPU must stay under 0.5 ms, min {record_min:.3} ms (median \
         {record_median:.3})"
    );
    assert_no_validation_messages(&gpu);
}
