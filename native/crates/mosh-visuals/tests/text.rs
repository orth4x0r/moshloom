//! The text layer, offscreen (wave 3, L-text; rework L-text-2).
//!
//! Every test renders into a `LayerTarget` and reads it back with `Gpu::read_back`, so no window
//! is opened. The layer is driven exactly the way the app drives it: `set_text` when a text clip
//! starts, then `draw` once per frame with the clip's `p0..p7`.
//!
//! # What is asserted, and why in these terms
//!
//! * A baked line is **one continuous distance field**. The CPU half of that proof lives in
//!   `text/sheet.rs::a_baked_line_has_no_cell_seams` (the field is 1-Lipschitz everywhere, so no
//!   glyph cell can cut it); the GPU half is `no_glyph_is_cut_at_a_cell_edge`, which renders
//!   `MW` and `ILANA` with the outline at maximum and looks for a straight vertical or horizontal
//!   step in the glow, which is what a clipped cell produced.
//! * Spacing comes from the font's advance widths plus kerning, not from a cell grid:
//!   `text/layout.rs::spacing_comes_from_the_font_advances` compares the line's width against
//!   `fontdue`'s own metrics (within 2%).
//! * Each look draws its own display face (Unbounded, Grenze Gotisch, Space Grotesk): the CPU
//!   test compares the baked boxes, `each_look_renders_its_own_face` compares the pictures.
//! * `set_text` is CPU work that happens when a text change starts, not per frame: the median
//!   keystroke and its p95 are bounded (the worst is printed, not asserted — one preempted sample
//!   on a busy machine is not a regression), and a first-use 20-character text must stay under
//!   60 ms. `draw` is one GPU pass and must be under 0.5 ms median at 960x540.
//! * The beat pop (`p6`) and glitch change pixels, and nothing flashes the whole frame: the four
//!   corners stay black through a 400 ms window (`README`: no full-frame flash faster than every
//!   400 ms).
//! * Previews of the three looks, with a word, that word as the windowed run spelled it
//!   (`ilanaVisual`) and a two-line phrase, are written under `native/target/style-previews/text/`
//!   for the PR; nothing is committed.
//!
//! # Row order
//!
//! A layer target is V-mirrored relative to what the stage shows: the compositor samples a layer
//! with `fullscreen.vert`'s uv, whose y is 0 at the top of the output image and 1 at its bottom,
//! while a layer target's row 0 is the output's *bottom* row, and the present pass adds no second
//! flip. `read_back` therefore returns the layer's memory row 0 first, which is the *bottom* of
//! the audience's picture. `the_word_is_upright_on_the_stage` follows the picture through the
//! compositor into an `OutputImage`, whose row 0 is the top of the stage, so it pins the direction
//! the audience sees; the preview PNGs are written flipped, in audience order.

use std::ops::Deref;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use mosh_core::{AnalysisFrame, BlendMode, ClipParams, LayerState};
use mosh_render::{Compositor, FrameRecorder, Gpu, GpuPreference, LayerTarget, OutputImage};
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

/// The clean look (`p1` = 0.85): no jitter, no glitch, a thin outline.
fn clean(size: f32) -> ClipParams {
    params(size, 0.85, 0.5, 0.0, 0.0, 0.35, 0.0, 0.0)
}

/// A still frame with audio bands, for looks that react to the music.
fn audio_frame() -> AnalysisFrame {
    AnalysisFrame {
        bands: [0.7, 0.3, 0.5],
        intensity: 0.45,
        brightness: 0.5,
        ..Default::default()
    }
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

    fn luma(&self, x: u32, y: u32) -> i32 {
        let pixel = self.pixel(x, y);
        (3 * i32::from(pixel[0]) + 6 * i32::from(pixel[1]) + i32::from(pixel[2])) / 10
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

    /// The mask of solid ink (glyph bodies and outlines, not the soft glow), one byte per pixel.
    fn solid_mask(&self) -> Vec<bool> {
        self.pixels.chunks(4).map(is_solid_ink).collect()
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

/// The same draw, followed through mosh-render's `Compositor` into an `OutputImage` and read back
/// from it: this is the path a frame takes to the stage, so row 0 of the result is the top of the
/// picture an audience sees (see `tests/orientation.rs` for why the layer read-back is not).
fn draw_once_on_stage(
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

    let out = OutputImage::new(gpu, width, height);
    let mut compositor = Compositor::new(gpu).expect("compositor");
    compositor
        .composite(&[(&dst, BlendMode::Normal, 1.0)], &out)
        .expect("composite the text layer");
    Shot {
        pixels: gpu.read_back_output(&out),
        width,
        height,
    }
}

// ---------------------------------------------------------------------------------------------
// Layout: what the sheet holds.
// ---------------------------------------------------------------------------------------------

/// The stage showed "ilanaVisual" mirrored top to bottom once. A layer target is V-mirrored
/// relative to the audience (`read_back` row 0 is the bottom of the picture), so the pass draws in
/// audience coordinates; this test follows the same picture through the compositor into an
/// `OutputImage`, which is the path a frame takes to the stage.
///
/// An `L` pins both axes: its stem is on the left, and its foot (the wide horizontal bar) is at the
/// *audience's* bottom, which is the output's *last* rows (and the layer's first rows). With the
/// shader drawing in layer coordinates instead, the foot lands in the output's first rows and this
/// test fails.
#[test]
fn the_word_is_upright_on_the_stage() {
    let gpu = gpu();
    let (width, height) = (400u32, 300u32);
    let frame = AnalysisFrame::default();
    let shot = draw_once_on_stage(&gpu, "L", &clean(0.7), &frame, 0.0, width, height);

    let quarter = height / 4;
    let ink_in = |y0: u32, y1: u32| -> usize {
        (y0..y1)
            .map(|y| (0..width).filter(|x| is_ink(&shot.pixel(*x, y))).count())
            .sum()
    };
    let top = ink_in(0, quarter);
    let bottom = ink_in(height - quarter, height);
    let stem = (0..width / 2)
        .map(|x| (0..height).filter(|y| is_ink(&shot.pixel(x, *y))).count())
        .sum::<usize>();
    let arm = (width / 2..width)
        .map(|x| (0..height).filter(|y| is_ink(&shot.pixel(x, *y))).count())
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

/// `ILANA` at 320x180 fills part of the frame, is lit nowhere near the border, and has at least one
/// fully black column between two letters: a pixel outside every glyph must stay black, so a
/// border or a gap between letters reading back with ink is a layout bug, not a style choice.
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
    let mut empty_columns = 0;
    for x in x0..=x1 {
        if (y0..=y1).all(|y| !is_solid_ink(&shot.pixel(x, y))) {
            empty_columns += 1;
        }
    }
    assert!(
        empty_columns >= 2,
        "no empty column inside the word box: the letters are painted together (or a glyph's \
         field carries ink where it should not)"
    );
    assert_no_validation_messages(&gpu);
}

/// Two lines stack, each centred, with a quiet gap between them; the block sits on the frame's
/// vertical centre.
#[test]
fn lines_stack_and_centre() {
    let gpu = gpu();
    let (width, height) = (400u32, 300u32);
    let frame = AnalysisFrame::default();
    let two = draw_once(&gpu, "IL\nANA", &clean(0.75), &frame, 0.0, width, height);

    // Bright body pixels, not the soft shadow: the shadow is dark and offset, so it must not
    // decide where a line is or how wide it is.
    let body_rows: Vec<usize> = (0..height)
        .map(|y| (0..width).filter(|x| two.luma(*x, y) > 100).count())
        .collect();
    let rows = body_rows;
    let widest = rows.iter().copied().max().unwrap_or(0);
    // A row belongs to a line only when it carries a real share of the picture: a single stray
    // pixel in the gap is antialiasing, not a line.
    let threshold = (widest / 8).max(2);
    let mut bands = 0;
    let mut in_band = false;
    for count in &rows {
        let lit = *count > threshold;
        if lit && !in_band {
            bands += 1;
        }
        in_band = lit;
    }
    assert_eq!(
        bands, 2,
        "IL\\nANA must draw two bands of rows (threshold {threshold}), got {bands}"
    );

    let (x0, _, x1, _) = two.ink_box().expect("two lines are visible");
    let centre = (x0 + x1) as f32 / 2.0;
    assert!(
        (centre - width as f32 / 2.0).abs() < width as f32 * 0.04,
        "the block's centre is {centre} in a {width}px frame"
    );

    // Each band is centred on the block's centre, not left-aligned under the widest line.
    let mut band_boxes = Vec::new();
    let mut start = None;
    for (y, count) in rows.iter().enumerate() {
        let lit = *count > threshold;
        match (start, lit) {
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
                if two.luma(x, y) > 100 {
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
// The clipping defect: no straight cut where a glyph cell used to end.
// ---------------------------------------------------------------------------------------------

/// The largest run of rows in one column where the picture *ends*: one side is transparent black
/// and the other carries glow. A clipped cell cut the glow along its whole height at a fixed x,
/// which is exactly such a run. A smooth field, a stem or a curve cannot produce one that long:
/// the glow's own gradient reaches zero gradually, so its last lit column is one or two rows.
fn longest_vertical_step(shot: &Shot) -> usize {
    let mut longest = 0;
    for x in 1..shot.width {
        let mut run = 0;
        for y in 0..shot.height {
            let left = shot.luma(x - 1, y);
            let right = shot.luma(x, y);
            let (low, high) = (left.min(right), left.max(right));
            // A cut: one side black, the other lit but still in the glow (not the body), and the
            // step is abrupt. A glow's own edge fades by a few levels per pixel, never 20+.
            if low <= 2 && (22..=80).contains(&high) {
                run += 1;
                longest = longest.max(run);
            } else {
                run = 0;
            }
        }
    }
    longest
}

/// The same for rows: a horizontal cut across a glyph cell.
fn longest_horizontal_step(shot: &Shot) -> usize {
    let mut longest = 0;
    for y in 1..shot.height {
        let mut run = 0;
        for x in 0..shot.width {
            let up = shot.luma(x, y - 1);
            let down = shot.luma(x, y);
            let (low, high) = (up.min(down), up.max(down));
            if low <= 2 && (22..=80).contains(&high) {
                run += 1;
                longest = longest.max(run);
            } else {
                run = 0;
            }
        }
    }
    longest
}

/// Render the words with the outline at maximum and the glow turned up, then prove the glow does
/// not end at a straight line: the old per-glyph cells cut it along every cell border, one whole
/// glyph tall, which is what this measures.
#[test]
fn no_glyph_is_cut_at_a_cell_edge() {
    let gpu = gpu();
    let (width, height) = (960u32, 540u32);
    let frame = AnalysisFrame {
        bands: [0.9, 0.4, 0.8],
        intensity: 0.8,
        ..Default::default()
    };
    // Chrome: the brightest glow, no barbs (the sigil look grows spikes by design), and the
    // outline at full. Jitter and glitch off, so the only structure is the glyphs and their halo.
    let params = params(0.75, 0.05, 0.5, 0.0, 0.0, 1.0, 0.3, 0.0);
    for text in ["MW", "ILANA", "AV"] {
        let shot = draw_once(&gpu, text, &params, &frame, 1.0, width, height);
        let vertical = longest_vertical_step(&shot);
        let horizontal = longest_horizontal_step(&shot);
        println!(
            "{text} at outline 1.0: longest straight glow step {vertical} rows / {horizontal} \
             columns"
        );
        assert!(
            vertical < 12,
            "{text}: the glow steps along a straight vertical line {vertical} rows long; a glyph \
             cell is cutting it"
        );
        assert!(
            horizontal < 12,
            "{text}: the glow steps along a straight horizontal line {horizontal} columns long"
        );
    }
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

/// The beat pop (`p6`) and the glitch (`p4`) move pixels. Nothing flashes the whole frame: the
/// corners stay black through a 400 ms window of pops, glitch and jitter.
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

/// The three looks (`p1` selects them) render differently *and* draw different faces: the outline
/// of an `I` in Unbounded is not the `I` of Grenze Gotisch or Space Grotesk, so their solid-ink
/// masks cannot agree.
#[test]
fn each_look_renders_its_own_face() {
    let gpu = gpu();
    let (width, height) = (640u32, 360u32);
    let frame = AnalysisFrame::default();
    let looks = [("chrome", 0.05), ("sigil", 0.45), ("clean", 0.85)];
    let mut shots = Vec::new();
    for (name, font) in looks {
        // No jitter, no glitch: the difference must be the typeface, not the effects.
        let shot = draw_once(
            &gpu,
            "ILANA",
            &params(0.7, font, 0.5, 0.0, 0.0, 0.3, 0.0, 0.0),
            &frame,
            1.0,
            width,
            height,
        );
        println!("{name}: {} lit px", shot.lit());
        shots.push(shot);
    }
    for (index, a) in shots.iter().enumerate() {
        for (other, b) in shots.iter().enumerate().skip(index + 1) {
            let difference = mean_difference(a, b);
            let mask_a = a.solid_mask();
            let mask_b = b.solid_mask();
            let shared = mask_a
                .iter()
                .zip(mask_b.iter())
                .filter(|(left, right)| **left && **right)
                .count();
            let union = mask_a
                .iter()
                .zip(mask_b.iter())
                .filter(|(left, right)| **left || **right)
                .count();
            let overlap = shared as f32 / union.max(1) as f32;
            println!(
                "look {index} vs {other}: {difference:.2} levels, solid-ink overlap {:.0}%",
                overlap * 100.0
            );
            assert!(
                difference > 2.0,
                "looks {index} and {other} render almost the same ({difference:.2})"
            );
            assert!(
                overlap < 0.85,
                "looks {index} and {other} share {:.0}% of their glyphs: the look is not using \
                 its face",
                overlap * 100.0
            );
        }
    }
    assert_no_validation_messages(&gpu);
}

/// Save the three looks at 960x540, each with a word and with a two-line phrase, under
/// `native/target/style-previews/text/` for the PR; nothing is committed. Also asserts the three
/// looks differ in the pictures, which is the visible half of "one face per look".
#[test]
fn three_looks_write_previews() {
    let gpu = gpu();
    let (width, height) = (960u32, 540u32);
    let frame = audio_frame();
    let looks = [("chrome", 0.05), ("sigil", 0.45), ("clean", 0.85)];
    // `ilanaVisual` is the text the windowed run broke into `ILANAVISU` / `AL`: the Sigil look
    // uppercases it, and it must come out as one line in every look. The two-line phrase is the
    // other half of the wrap contract.
    let shots = [
        ("ilana", "ILANA"),
        ("ilanavisual", "ilanaVisual"),
        ("phrase", "the loom\nis us"),
    ];

    for (name, font) in looks {
        for (shot_name, text) in shots {
            let shot = draw_once(
                &gpu,
                text,
                &params(0.8, font, 0.5, 0.10, 0.0, 0.45, 0.30, 0.0),
                &frame,
                1.25,
                width,
                height,
            );
            if shot_name == "phrase" {
                // Every pixel outside the glow's reach must be transparent black: a non-zero
                // margin is the sheet's rectangle showing, which is the defect this rework is
                // about.
                let mut outside = 0usize;
                let (x0, y0, x1, y1) = shot.ink_box().expect("the phrase is visible");
                for y in 0..shot.height {
                    for x in 0..shot.width {
                        let beyond = x + 40 < x0 || x > x1 + 40 || y + 40 < y0 || y > y1 + 40;
                        if beyond && shot.pixel(x, y)[3] != 0 {
                            outside += 1;
                        }
                    }
                }
                println!("{name}: {outside} lit pixels more than 40px outside the ink box");
                assert_eq!(outside, 0, "{name}: the sheet's margin shows as ink");
            }
            let path = preview_path(&format!("{name}-{shot_name}"));
            save_png(&path, &shot).expect("write a preview PNG");
            println!("preview: {} ({} lit px)", path.display(), shot.lit());
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
///
/// Rows are written in reverse: the read-back is the layer target's, and the compositor mirrors
/// it once on the way to the stage, so the last layer row is the top of what an audience sees.
fn save_png(path: &PathBuf, shot: &Shot) -> std::io::Result<()> {
    let mut raw = Vec::with_capacity((shot.height * (1 + shot.width * 4)) as usize);
    for y in (0..shot.height).rev() {
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

/// The budget test: `set_text` is the one-off CPU cost of a text change (a clip starting, a lyric
/// being typed) and must stay under 2 ms in release once the glyphs are cached, which is the cost
/// the app pays on the render thread (`mosh-app/src/shell.rs` calls `set_text` every frame; the
/// same-text path is a no-op). `draw` is one GPU pass and is measured against an empty pass of the
/// same pipeline on the same machine.
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

    // First use: every character is new to the face, so this is the hitch a clip pays once. One
    // face is baked (the look last drawn), not three.
    let cold_text = "ZYXWVUTSRQPONMLKJIHG";
    assert_eq!(cold_text.chars().count(), 20);
    layer.set_text(&gpu, 0, "warm up");
    let cold = {
        let start = Instant::now();
        layer.set_text(&gpu, 0, cold_text);
        start.elapsed().as_secs_f64() * 1000.0
    };

    // A line typed one character at a time, the way the editor changes a text clip, from a cold
    // cache: every keystroke may introduce a character, and the wrap keeps the em size — and so
    // the glyph cache — the same across the whole length range.
    let phrase = "the loom knows what the flesh wants and the loom is us and more";
    // The clip's first text arrives as one change (that is the cold case above, reported
    // separately); what follows is typing. Every later keystroke adds a character or two, which is
    // where the cache has to hold.
    let first_text: String = phrase.chars().take(18).collect();
    layer.set_text(&gpu, 0, &first_text);
    let mut typing = Vec::new();
    for len in 19..=59usize {
        let text: String = phrase.chars().take(len).collect();
        let start = Instant::now();
        layer.set_text(&gpu, 0, &text);
        typing.push((len, start.elapsed().as_secs_f64() * 1000.0));
    }

    // The app calls `set_text` every frame: the same words again must cost nothing. The phrase
    // is baked first (that is the typing loop's last state, a prefix), then the *next* call is
    // the no-op the app makes every frame.
    layer.set_text(&gpu, 0, phrase);
    let repeat = {
        let start = Instant::now();
        layer.set_text(&gpu, 0, phrase);
        start.elapsed().as_secs_f64() * 1000.0
    };
    layer.set_text(&gpu, 0, "the loom knows which");

    // The pass's floor on this machine: the same pipeline and target with a block so small that
    // almost every pixel takes the empty-space early-out. Measured the same way as the real pass,
    // so the two numbers share one GPU clock state.
    let tiny = params(0.02, 0.85, 0.5, 0.0, 0.0, 0.35, 0.0, 0.0);
    let mut floor_samples: Vec<f64> = Vec::new();
    let mut gpu_samples: Vec<f64> = Vec::new();
    let mut record_samples = Vec::new();
    for step in 0..20 {
        let t = step as f32 / 60.0;
        let mut recorder = FrameRecorder::new(&gpu);
        layer
            .draw(&mut recorder, 0, &dst, &frame, &tiny, t)
            .expect("draw the floor probe");
        let timings = gpu.wait_frame(recorder.submit());
        if let Some((_, ms)) = timings
            .per_pass_ms
            .iter()
            .find(|(name, _)| name == "text.glyphs")
        {
            floor_samples.push(f64::from(*ms));
        }

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
        gpu_samples.push(f64::from(pass.expect("checked just above")));
    }
    // One read-back after the timing loop: a read-back is its own submission and would contend
    // with the pass being timed.
    let measured = Shot {
        pixels: gpu.read_back(&dst),
        width,
        height,
    };

    let stats = |samples: &mut Vec<f64>| {
        samples.sort_by(|a, b| a.total_cmp(b));
        (samples[0], samples[samples.len() / 2])
    };
    // A percentile after sorting: the sample a given share of the run came in under. The p95 of
    // forty-one keystrokes is where the slow tail starts, which is the number a busy machine moves
    // and a regression moves first.
    let percentile = |samples: &mut Vec<f64>, share: f64| {
        samples.sort_by(|a, b| a.total_cmp(b));
        let at = ((samples.len() as f64 - 1.0) * share).round() as usize;
        samples[at.min(samples.len() - 1)]
    };
    let mut typing_ms: Vec<f64> = typing.iter().map(|(_, ms)| *ms).collect();
    let typing_p95 = percentile(&mut typing_ms, 0.95);
    let typing_worst = typing
        .iter()
        .map(|(len, ms)| (*len, *ms))
        .fold((0usize, 0.0f64), |worst, next| {
            if next.1 > worst.1 { next } else { worst }
        });
    let (typing_min, typing_median) = stats(&mut typing_ms);
    let (record_min, record_median) = stats(&mut record_samples);
    let (draw_min, draw_median) = stats(&mut gpu_samples);
    let (_, floor_median) = stats(&mut floor_samples);
    let lit = measured.lit();

    println!(
        "text at {width}x{height}: set_text typing 18..59 chars {typing_min:.3} / \\
         {typing_median:.3} ms min/median, p95 {typing_p95:.3}, worst {:.3} ms at {} chars; a \\
         first-use 20-character text ({cold_text:?}) {cold:.3} ms; a repeated text {repeat:.4} ms; \\
         draw record {record_min:.3} / {record_median:.3} ms on the CPU, GPU `text.glyphs` \\
         {draw_min:.4} / {draw_median:.4} ms min/median (an empty pass of the same pipeline: \\
         {floor_median:.4} ms), {lit} lit px",
        typing_worst.1,
        typing_worst.0,
    );

    assert!(lit > 0, "the measured frame is blank");
    // Release bounds. The median is the number the budget is stated in; p95 is the load-tolerant
    // half. Under the full release suite (every test in this binary, one thread) the worst
    // keystroke has measured 6.4 ms while the median stayed at 1.5-1.6, so one preempted sample
    // must not fail the run; the worst is printed above and never asserted. A real regression
    // moves the median and the p95 together. The debug test binary is 5-20x slower and gets its
    // own, much looser bounds that still catch a per-frame mistake (a bake inside `draw`).
    let (typing_median_bound, typing_p95_bound) = if cfg!(debug_assertions) {
        (60.0, 90.0)
    } else {
        (2.0, 4.0)
    };
    assert!(
        typing_median < typing_median_bound,
        "the median keystroke must stay under {typing_median_bound} ms, median {typing_median:.3} \
         ms (p95 {typing_p95:.3}, worst {:.3} at {} characters)",
        typing_worst.1,
        typing_worst.0
    );
    assert!(
        typing_p95 < typing_p95_bound,
        "95% of keystrokes must stay under {typing_p95_bound} ms, p95 {typing_p95:.3} ms (median \
         {typing_median:.3}, worst {:.3} at {} characters)",
        typing_worst.1,
        typing_worst.0
    );
    // The cold hitch is the price of a character the cache has never seen; it must not also pay
    // for two faces that will not be drawn, and it must not become a per-frame bake. 60 ms is
    // about five times the 9-11 ms a first-use text has measured here (8-21 ms across runs, #61);
    // a regression to 200+ ms fails.
    let cold_bound = if cfg!(debug_assertions) { 400.0 } else { 60.0 };
    assert!(
        cold < cold_bound,
        "a first-use 20-character text must stay under {cold_bound} ms, took {cold:.3}"
    );
    assert!(
        repeat < 0.05,
        "`set_text` with unchanged text must be a no-op, took {repeat:.4} ms"
    );
    // The median is the number the budget is stated in: 0.5 ms on a quiet machine. A pass of this
    // layer cannot cost less than a pass of the same pipeline, so the bound never goes below an
    // empty pass's measured cost — on a loaded machine that floor alone can be 0.6 ms.
    let draw_median_bound = if cfg!(debug_assertions) {
        1.0
    } else {
        0.5f64.max(floor_median * 1.2)
    };
    assert!(
        draw_median < draw_median_bound,
        "the text pass must stay under {draw_median_bound} ms median at 960x540, median \\
         {draw_median:.4} ms (min {draw_min:.4}, an empty pass measures {floor_median:.4})"
    );
    assert!(
        record_min < 0.5,
        "recording the pass on the CPU must stay under 0.5 ms, min {record_min:.3} ms (median \\
         {record_median:.3})"
    );
    // Reported, not budgeted: the sigil look's bake also grows its thorns from the composed field
    // (`text/thorn.rs`), a cost the two other looks do not pay. The pass is measured the way the
    // clean one is — a warm-up frame (the first draw after a text change also uploads the sheet and
    // binds the look), then ten frames, median — because one cold sample is not a number.
    let sigil_params = params(0.8, 0.45, 0.5, 0.0, 0.0, 0.45, 0.0, 0.0);
    let mut sigil_samples: Vec<f64> = Vec::new();
    for step in 0..11 {
        let mut recorder = FrameRecorder::new(&gpu);
        layer
            .draw(
                &mut recorder,
                0,
                &dst,
                &frame,
                &sigil_params,
                step as f32 / 60.0,
            )
            .expect("draw the sigil look");
        let timings = gpu.wait_frame(recorder.submit());
        if step == 0 {
            continue;
        }
        if let Some((_, ms)) = timings
            .per_pass_ms
            .iter()
            .find(|(name, _)| name == "text.glyphs")
        {
            sigil_samples.push(f64::from(*ms));
        }
    }
    assert_eq!(sigil_samples.len(), 10, "ten measured sigil frames");
    let (sigil_min, sigil_median) = stats(&mut sigil_samples);
    // The sigil pass costs more than the clean one: it evaluates up to twenty thorn strokes and a
    // handful of mark strokes per pixel over a wider region.
    let sigil_text = "the loom knows what the flesh wants";
    let sigil = {
        let start = Instant::now();
        layer.set_text(&gpu, 0, sigil_text);
        start.elapsed().as_secs_f64() * 1000.0
    };
    // The same characters, reordered: every glyph is cached, so what is left is the layout, the
    // compose and the thorn growth — the look's own cost.
    let warm_text = "wants the flesh what the loom knows";
    let warm = {
        let start = Instant::now();
        layer.set_text(&gpu, 0, warm_text);
        start.elapsed().as_secs_f64() * 1000.0
    };
    println!(
        "a sigil-look text change ({sigil_text:?}, thorns and marks included): {sigil:.3} ms cold, \
         {warm:.3} ms with every glyph already cached; the sigil pass {sigil_min:.4} / \
         {sigil_median:.4} ms min/median"
    );
    // A sigil text change pays for the glyphs and for the thorn and mark growth, and it must not
    // become a per-frame or per-edit stall. 25 ms is about twice the 9.5 ms measured here for
    // thirty-six characters, and the reviewer measured 9.2 ms at the 600-character, seven-line cap
    // — generous, but a 3x regression fails it.
    let sigil_bound = if cfg!(debug_assertions) { 400.0 } else { 25.0 };
    assert!(
        sigil < sigil_bound,
        "a sigil text change must stay under {sigil_bound} ms, took {sigil:.3} ms (cold)"
    );
    assert_no_validation_messages(&gpu);
}

/// The sigil look's marks are authored once, on the right of the block, and the shader draws the
/// left copy mirrored about the block's centre. A build where only one half drew was caught by
/// eye, so this measures both flanks.
///
/// Calibrated on `SIGIL` at size 0.15, the pair the review prototyped with. The glyph mass is the
/// ink a 5x5 erosion keeps (the letters are filled; a mark's stroke is 2-3px wide, so an erosion
/// deletes it), and the flanks are the columns either side of it. The assertion is that the left
/// flank's ink **mirrors** the right flank's, pixel for pixel about the block's centre: that is
/// what the shader is supposed to do, and thorns — which are asymmetric by design — cannot fake
/// it. The bare "leftmost ink is a few pixels left of the mass" predicate the review prototyped
/// does not survive here: with the left copy disabled the leftmost ink is 229 and the letters 251,
/// so the left thorns alone sit 22px outside the mass and the predicate passes.
///
/// The text and size are part of the calibration on purpose: other words grow a thorn that reaches
/// as far as a mark.
#[test]
fn the_mirrored_mark_draws_on_both_sides() {
    let gpu = gpu();
    let (width, height) = (960u32, 540u32);
    let frame = AnalysisFrame::default();
    let shot = draw_once(
        &gpu,
        "SIGIL",
        &params(0.15, 0.45, 0.5, 0.0, 0.0, 0.3, 0.0, 0.0),
        &frame,
        0.0,
        width,
        height,
    );
    // The silver body, not the red glow: the glow's red never passes 90, the metal's channels sit
    // well above 120.
    let body = |x: u32, y: u32| {
        shot.pixel(x, y)
            .iter()
            .take(3)
            .copied()
            .max()
            .unwrap_or(0)
            > 120
    };
    let filled = |x: u32, y: u32| {
        (y.saturating_sub(2)..=(y + 2).min(height - 1))
            .all(|row| (x.saturating_sub(2)..=(x + 2).min(width - 1)).all(|column| body(column, row)))
    };
    let mut mass_left = width;
    let mut mass_right = 0;
    for y in 0..height {
        for x in 2..width - 2 {
            if filled(x, y) {
                mass_left = mass_left.min(x);
                mass_right = mass_right.max(x);
            }
        }
    }
    println!(
        "SIGIL at 0.15 on {width}x{height}: glyph mass {mass_left}..={mass_right} \
         (centre {}), block centre {}",
        (mass_left + mass_right) / 2,
        width / 2
    );
    let leftmost = (0..width)
        .find(|x| (0..height).any(|y| body(*x, y)))
        .expect("ink");
    let rightmost = (0..width)
        .rfind(|x| (0..height).any(|y| body(*x, y)))
        .expect("ink");
    println!(
        "ink {leftmost}..={rightmost}: {}px left of the mass, {}px right of it",
        mass_left.saturating_sub(leftmost),
        rightmost.saturating_sub(mass_right)
    );
    assert!(
        mass_right > mass_left + 100,
        "the glyph mass is {mass_left}..={mass_right}: this calibration needs the letters to fill \
         the middle of the frame"
    );

    // Every lit column outside the mass, and how much of it the other flank mirrors.
    let flank = 40u32;
    let mirrored_ink = |outward: bool| -> (usize, usize) {
        let mut ink = 0usize;
        let mut pairs = 0usize;
        let columns = if outward {
            (mass_left.saturating_sub(flank)..mass_left).collect::<Vec<_>>()
        } else {
            (mass_right + 1..=(mass_right + flank).min(width - 1)).collect::<Vec<_>>()
        };
        for x in columns {
            for y in 0..height {
                if !body(x, y) {
                    continue;
                }
                ink += 1;
                // The block is centred at width/2, so x mirrors to width - 1 - x (a pixel centre
                // at x + 0.5 maps to width - x - 0.5). One pixel of slack either way covers the
                // rounding of the two sides' rasterisation.
                let mirror = width - 1 - x;
                let found = (y.saturating_sub(1)..=(y + 1).min(height - 1)).any(|row| {
                    (mirror.saturating_sub(1)..=(mirror + 1).min(width - 1))
                        .any(|column| body(column, row))
                });
                pairs += usize::from(found);
            }
        }
        (ink, pairs)
    };
    let (left_ink, left_pairs) = mirrored_ink(true);
    let (right_ink, right_pairs) = mirrored_ink(false);
    println!(
        "{flank} flank columns each side: left {left_ink} lit, {left_pairs} with a mirror; \
         right {right_ink} lit, {right_pairs} with a mirror"
    );
    // With the mirrored copy drawn, the two marks are the same ink on both sides, so most of a
    // flank's ink finds its reflection. With the left copy missing, only thorns are left there and
    // almost nothing pairs.
    assert!(
        left_pairs * 2 > left_ink && left_ink > 100,
        "the left flank has {left_ink} lit pixels and only {left_pairs} of them mirror the right \
         flank: the mirrored mark is not drawing"
    );
    assert!(
        right_pairs * 2 > right_ink && right_ink > 100,
        "the right flank has {right_ink} lit pixels and only {right_pairs} of them mirror the left \
         flank: the authored mark is not drawing"
    );
    assert_no_validation_messages(&gpu);
}

#[test]
fn a_text_layer_fits_the_uniform_cap() {
    let gpu = gpu();
    // `Pipeline::fullscreen` refuses a block over `mosh_render::MAX_UNIFORM_BYTES` (4096 bytes)
    // with `UniformTooLarge`; the text block is 1008 — 112 bytes of scalars, 20 thorn strokes and
    // 8 mark strokes of 32 bytes. This constructor succeeding is the GPU-side assertion that it
    // fits; `text::tests::the_uniform_block_fits_the_render_crate_cap` pins the numbers.
    let _layer = TextLayer::new(&gpu).expect("the text pass's uniform block fits mosh-render's cap");
    println!(
        "TextLayer::new: a 1008-byte uniform block against a {}-byte cap",
        mosh_render::MAX_UNIFORM_BYTES
    );
    assert_no_validation_messages(&gpu);
}

/// A guard, not a witness: the pair stays two letters at the current dilation, and no committed
/// state of this layer welded them — a reviewer could not reproduce the "welded `at`" story from
/// any revision. What the test pins is the rule the story is about: the stroke dilation is a
/// fraction of the block, small enough that a pair with real side bearings keeps a column of empty
/// pixels between the letters. A thicker dilation, a looser kern or a coarser sheet would close
/// that column, and the failure says so.
#[test]
fn a_tight_pair_keeps_its_gap() {
    let gpu = gpu();
    let (width, height) = (960u32, 540u32);
    let frame = audio_frame();
    for text in ["at", "To", "AV"] {
        let shot = draw_once(
            &gpu,
            text,
            &params(0.9, 0.05, 0.5, 0.0, 0.0, 0.45, 0.0, 0.0),
            &frame,
            0.0,
            width,
            height,
        );
        let (x0, y0, x1, y1) = shot.ink_box().expect("the pair is visible");
        // Columns through the middle of the pair that carry no solid ink: a welded pair has none.
        let mut gaps = 0;
        for x in x0..=x1 {
            if (y0..=y1).all(|y| !is_solid_ink(&shot.pixel(x, y))) {
                gaps += 1;
            }
        }
        println!("{text:?}: {gaps} empty columns between the glyphs");
        assert!(
            gaps >= 2,
            "{text:?} welded its two glyphs together: no empty column inside the pair"
        );
    }
    assert_no_validation_messages(&gpu);
}

/// Long lines: 100 characters wrap into several lines and stay a readable size, and a text far
/// past the cap still draws a bounded block instead of a smear or nothing.
#[test]
fn long_lines_wrap_and_a_huge_text_is_capped() {
    let gpu = gpu();
    let (width, height) = (960u32, 540u32);
    let frame = AnalysisFrame::default();
    let params = params(0.6, 0.85, 0.5, 0.0, 0.0, 0.35, 0.0, 0.0);

    let hundred = "the loom knows what the flesh wants and the loom is us and the flesh is the loom".to_string();
    assert!(hundred.chars().count() >= 70);
    let shot = draw_once(&gpu, &hundred, &params, &frame, 0.0, width, height);
    let (x0, y0, x1, y1) = shot.ink_box().expect("a long line is visible");
    let box_height = y1 - y0 + 1;
    println!(
        "{} chars: box {}x{} at ({x0},{y0})",
        hundred.chars().count(),
        x1 - x0 + 1,
        box_height
    );
    assert!(
        box_height > height * 12 / 100,
        "a long line must wrap into a readable block, got {box_height}px tall in a {height}px frame"
    );
    assert!(
        x0 >= 2 && y0 >= 2 && x1 + 2 < width && y1 + 2 < height,
        "the wrapped block must stay inside the frame"
    );

    let huge = "lorem ipsum dolor sit amet ".repeat(100);
    let shot = draw_once(&gpu, &huge, &params, &frame, 0.0, width, height);
    let (x0, y0, x1, y1) = shot.ink_box().expect("a capped text still draws its visible part");
    println!(
        "{} chars: lit {} px, box {}x{}",
        huge.chars().count(),
        shot.lit(),
        x1 - x0 + 1,
        y1 - y0 + 1
    );
    assert!(
        x0 >= 2 && y0 >= 2 && x1 + 2 < width && y1 + 2 < height,
        "a capped text must not smear into the frame's edge"
    );
    assert!(
        (y1 - y0 + 1) < height * 90 / 100,
        "a capped text must be a bounded block"
    );
    assert_no_validation_messages(&gpu);
}
