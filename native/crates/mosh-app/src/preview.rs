//! Small pictures for the control window: the live preview of the stage, and clip thumbnails.
//!
//! Both are made on the render thread (the only thread that may touch the Vulkan device) and
//! sent to the window thread as plain RGBA bytes, where they become egui textures on the control
//! window's own OpenGL context. Nothing is shared between the two contexts.
//!
//! * **Preview**: after a stage frame is presented, the same layer stack is composited once more
//!   into a [`PREVIEW_WIDTH`]-wide image and read back, at most [`PREVIEW_FPS`] times a second
//!   and only while the control window asks for it (never while it is minimised).
//! * **Thumbnails**: one frame of a clip's scene drawn into a [`THUMB_SIZE`] target with the
//!   clip's parameters, on request, at most one per stage frame. The control window caches them
//!   by [`thumb_key`], so a clip is drawn once and again only when its look changes.
//!
//! **What they cost, and why they are this small.** `mosh-render`'s readback allocates a
//! buffer, submits a copy, waits for it and converts RGBA16F to RGBA8 one channel at a time out
//! of mapped device memory. On the reference machine (`tests/preview_gpu.rs`, release) that is
//! about 5 ms however small the picture is, plus about half a microsecond a pixel: 22 ms for
//! 256x144, 5.9 ms for 128x72, 5.5 ms for 96x54. The render thread waits for it, after the
//! frame has been presented, inside the slack a 60 fps frame leaves. So:
//!
//! * the pictures are exactly as large as the control window draws them (the top bar's preview
//!   slot is 86 px wide) and no larger;
//! * the preview runs at [`PREVIEW_FPS`], is off by default when the stage is a window beside
//!   the controls (the stage is its own preview there), and never runs while the control window
//!   is minimised, which is what it is during a set;
//! * a thumbnail is drawn once per look, one per frame at most, never in a frame that captured
//!   a preview.
//!
//! A readback into a buffer that is kept, in 8 bits, would take the 5 ms away. That is a change
//! to `mosh-render`, which is not this lane's crate: it is reported, not made.
//!
//! The sizes, the rate and the letterboxing are pure functions, tested without a GPU; the GPU
//! half is exercised offscreen in `tests/preview_gpu.rs`, which also prints what it costs.

use std::time::{Duration, Instant};

use mosh_core::{AnalysisFrame, BlendMode, ClipParams};
use mosh_render::{Compositor, FrameRecorder, Gpu, LayerTarget, OutputImage, RenderError};
use mosh_visuals::Visuals;

/// The preview's width in pixels; its height follows the stage's aspect.
pub const PREVIEW_WIDTH: u32 = 96;
/// The preview's refresh rate. A glance, not a monitor: the stage itself is the monitor.
pub const PREVIEW_FPS: u32 = 4;
/// A thumbnail's size: 16:9, as large as the detail panel draws it.
pub const THUMB_SIZE: (u32, u32) = (96, 54);

/// The preview image's size for a stage of `out` pixels: `max_width` wide (or the stage's own
/// width when that is smaller), the stage's aspect, even numbers, never smaller than 16 pixels
/// and never taller than twice its width (a portrait stage still fits the top bar).
pub fn preview_size(out: (u32, u32), max_width: u32) -> (u32, u32) {
    let (out_w, out_h) = (out.0.max(1), out.1.max(1));
    let width = max_width.min(out_w).max(16);
    let height = (u64::from(width) * u64::from(out_h) / u64::from(out_w)) as u32;
    let even = |value: u32| (value.max(16) + 1) & !1;
    (even(width), even(height).min(even(width) * 2))
}

/// The largest rectangle of `aspect` (width / height) that fits a `bounds` box, as
/// `(x, y, width, height)` inside the box: a 16:10 stage in a 16:9 slot gets bars at the sides,
/// never a stretched picture.
pub fn fit(aspect: f32, bounds: (f32, f32)) -> (f32, f32, f32, f32) {
    let (box_w, box_h) = (bounds.0.max(0.0), bounds.1.max(0.0));
    if aspect <= 0.0 || box_w == 0.0 || box_h == 0.0 {
        return (0.0, 0.0, box_w, box_h);
    }
    let (width, height) = if box_w / box_h > aspect {
        (box_h * aspect, box_h)
    } else {
        (box_w, box_w / aspect)
    };
    ((box_w - width) * 0.5, (box_h - height) * 0.5, width, height)
}

/// When the next preview frame is due.
#[derive(Clone, Copy, Debug)]
pub struct PreviewClock {
    period: Duration,
    last: Option<Instant>,
}

impl PreviewClock {
    pub fn new(fps: u32) -> PreviewClock {
        PreviewClock {
            period: Duration::from_secs_f64(1.0 / f64::from(fps.max(1))),
            last: None,
        }
    }

    /// Whether a frame is due at `now`; a due frame is counted as taken.
    pub fn due(&mut self, now: Instant) -> bool {
        if self
            .last
            .is_some_and(|last| now.saturating_duration_since(last) < self.period)
        {
            return false;
        }
        self.last = Some(now);
        true
    }
}

/// A small picture on its way to the control window: tightly packed RGBA8, top row first, in
/// display (sRGB) encoding with opaque alpha.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Picture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Linear light 0..255 to sRGB 0..255. The offscreen targets hold linear values (the stage's
/// swapchain does this encoding in hardware); egui textures are sRGB bytes.
fn srgb_table() -> &'static [u8; 256] {
    static TABLE: std::sync::OnceLock<[u8; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|index| {
            let linear = index as f32 / 255.0;
            let encoded = if linear <= 0.003_130_8 {
                linear * 12.92
            } else {
                1.055 * linear.powf(1.0 / 2.4) - 0.055
            };
            (encoded * 255.0).round().clamp(0.0, 255.0) as u8
        })
    })
}

/// Turn a linear RGBA8 readback into a display picture: sRGB-encoded, alpha forced opaque.
pub fn to_display(width: u32, height: u32, mut rgba: Vec<u8>) -> Picture {
    let table = srgb_table();
    for pixel in rgba.as_chunks_mut::<4>().0 {
        *pixel = [
            table[usize::from(pixel[0])],
            table[usize::from(pixel[1])],
            table[usize::from(pixel[2])],
            255,
        ];
    }
    Picture {
        width,
        height,
        rgba,
    }
}

/// The preview's GPU half: a compositor and a small output of its own, so the stage's
/// compositor, its swapchain and its timings are never touched.
///
/// Must be dropped before the [`Gpu`] it was built on (like every `mosh-render` object).
pub struct PreviewTap {
    compositor: Compositor,
    out: OutputImage,
    stage: (u32, u32),
}

impl PreviewTap {
    /// A tap for a stage of `stage` pixels.
    pub fn new(gpu: &Gpu, stage: (u32, u32)) -> Result<PreviewTap, RenderError> {
        let (width, height) = preview_size(stage, PREVIEW_WIDTH);
        Ok(PreviewTap {
            compositor: Compositor::new(gpu)?,
            out: OutputImage::new(gpu, width, height),
            stage,
        })
    }

    /// The stage size this tap was built for; a different one needs a new tap.
    pub fn stage(&self) -> (u32, u32) {
        self.stage
    }

    pub fn size(&self) -> (u32, u32) {
        (self.out.width(), self.out.height())
    }

    /// Composite `layers` (the stack the stage just presented) at preview size and read it back.
    ///
    /// Call between frames: the layer targets must not be in a recorder. This submits one small
    /// pass and one copy and waits for both, which is the preview's whole cost on the render
    /// thread.
    pub fn capture(
        &mut self,
        gpu: &Gpu,
        layers: &[(&LayerTarget, BlendMode, f32)],
    ) -> Result<Picture, RenderError> {
        self.compositor.composite(layers, &self.out)?;
        let rgba = gpu.read_back_output(&self.out);
        Ok(to_display(self.out.width(), self.out.height(), rgba))
    }
}

/// One clip thumbnail to draw: a visualizer and the parameters it is drawn with.
#[derive(Clone, Debug, PartialEq)]
pub struct ThumbRequest {
    pub key: u64,
    pub visualizer: String,
    pub params: ClipParams,
}

/// The cache key of a clip's look: its visualizer, its eight parameters in steps of 1/64 (a
/// slider nudge that changes nothing visible does not redraw the thumbnail), palette and seed.
pub fn thumb_key(visualizer: &str, params: &ClipParams) -> u64 {
    // FNV-1a: stable across runs, which `std`'s hasher is not promised to be.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    feed(visualizer.as_bytes());
    feed(&[0xff]);
    for value in params.p {
        feed(&[(value.clamp(0.0, 1.0) * 64.0).round() as u8]);
    }
    feed(&params.palette.to_le_bytes());
    feed(&params.seed.to_le_bytes());
    hash
}

/// Whether a thumbnail of `visualizer` can be drawn from the clip alone. A camera, a text and a
/// video depend on a source the thumbnail pass does not open; they get a labelled tile instead.
pub fn has_thumbnail(visualizer: &str) -> bool {
    !matches!(visualizer, "overlay" | "camera" | "text" | "particles")
        && !visualizer.starts_with("custom:")
}

/// The analysis frame thumbnails are drawn with: a loud, bright moment a little after a beat,
/// the same for every clip so two thumbnails differ only by their clips.
pub fn thumb_frame() -> AnalysisFrame {
    AnalysisFrame {
        time_s: 12.0,
        ppq: 16.25,
        beat_phase: 0.25,
        bar_phase: 0.062_5,
        beat_index: 16,
        bar_index: 4,
        tempo_bpm: 120.0,
        playing: true,
        bands: [0.8, 0.65, 0.5],
        mel: std::array::from_fn(|band| 0.8 * (1.0 - band as f32 / 30.0)),
        onset: 0.6,
        loudness: 0.8,
        brightness: 0.6,
        intensity: 0.75,
        level: 3,
        ..AnalysisFrame::default()
    }
}

/// The thumbnails' GPU half: one small scene target.
///
/// Must be dropped before the [`Gpu`] it was built on.
pub struct ThumbTap {
    target: LayerTarget,
}

impl ThumbTap {
    pub fn new(gpu: &Gpu) -> ThumbTap {
        ThumbTap {
            target: LayerTarget::new_single(gpu, THUMB_SIZE.0, THUMB_SIZE.1),
        }
    }

    /// Draw one frame of the requested scene and read it back. `None` for a visualizer that has
    /// no thumbnail (see [`has_thumbnail`]).
    ///
    /// Only the scene pass runs (no datamosh, effects or transition: those have history a single
    /// frame cannot show), so the stage's per-layer state is not touched.
    pub fn render(
        &mut self,
        gpu: &Gpu,
        visuals: &mut Visuals,
        request: &ThumbRequest,
    ) -> Result<Option<Picture>, RenderError> {
        if !has_thumbnail(&request.visualizer) {
            return Ok(None);
        }
        let frame = thumb_frame();
        let t = frame.time_s as f32;
        let mut recorder = FrameRecorder::new(gpu);
        if request.visualizer == "fluid" {
            visuals
                .fluid
                .draw(&mut recorder, &self.target, &frame, &request.params, t)?;
        } else {
            visuals.scenes.draw(
                &mut recorder,
                &self.target,
                &request.visualizer,
                &frame,
                &request.params,
                t,
            )?;
        }
        let ticket = recorder.submit();
        gpu.wait_frame(ticket);
        let rgba = gpu.read_back(&self.target);
        Ok(Some(to_display(THUMB_SIZE.0, THUMB_SIZE.1, rgba)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_preview_keeps_the_stage_aspect() {
        // 1080p: 96 x 54, exactly 16:9.
        assert_eq!(preview_size((1920, 1080), PREVIEW_WIDTH), (96, 54));
        // The laptop panel is 16:10.
        assert_eq!(preview_size((2560, 1600), PREVIEW_WIDTH), (96, 60));
        // A windowed stage that is smaller than the preview is not enlarged.
        assert_eq!(preview_size((64, 36), PREVIEW_WIDTH), (64, 36));
        // Odd sizes round up to even ones; degenerate sizes still give an image.
        assert_eq!(preview_size((1001, 563), 250), (250, 140));
        assert_eq!(preview_size((0, 0), PREVIEW_WIDTH), (16, 16));
        assert_eq!(preview_size((8, 8), PREVIEW_WIDTH), (16, 16));
        // A portrait stage does not make a tower: at most twice as tall as wide.
        assert_eq!(preview_size((1080, 3840), PREVIEW_WIDTH), (96, 192));
        // 96 x 54 RGBA is what crosses the thread boundary, four times a second: 83 KB.
        let (width, height) = preview_size((1920, 1080), PREVIEW_WIDTH);
        assert_eq!(width * height * 4 * PREVIEW_FPS, 82_944);
    }

    #[test]
    fn the_preview_is_letterboxed_never_stretched() {
        // A 16:9 picture in a 16:9 slot fills it.
        assert_eq!(fit(16.0 / 9.0, (160.0, 90.0)), (0.0, 0.0, 160.0, 90.0));
        // 16:10 in a 16:9 slot: bars at the sides.
        let (x, y, width, height) = fit(1.6, (160.0, 90.0));
        assert_eq!((y, height), (0.0, 90.0));
        assert!((width - 144.0).abs() < 1e-3 && (x - 8.0).abs() < 1e-3);
        // A wide picture in a tall slot: bars above and below.
        let (x, y, width, height) = fit(2.0, (100.0, 100.0));
        assert_eq!((x, width, height, y), (0.0, 100.0, 50.0, 25.0));
        // Nothing to fit into.
        assert_eq!(fit(1.6, (0.0, 90.0)), (0.0, 0.0, 0.0, 90.0));
    }

    #[test]
    fn the_preview_clock_holds_its_rate() {
        let mut clock = PreviewClock::new(PREVIEW_FPS);
        let start = Instant::now();
        let mut taken = 0;
        // Two seconds of 60 fps stage frames.
        for frame in 0..120u32 {
            if clock.due(start + Duration::from_micros(u64::from(frame) * 16_667)) {
                taken += 1;
            }
        }
        assert!((7..=9).contains(&taken), "{taken} previews in 2 s");
    }

    #[test]
    fn a_thumbnail_is_keyed_by_what_it_looks_like() {
        let params = ClipParams {
            p: [0.5; 8],
            palette: 1,
            seed: 7,
        };
        let key = thumb_key("tunnel", &params);
        assert_eq!(key, thumb_key("tunnel", &params), "stable");
        assert_ne!(key, thumb_key("loom", &params));
        assert_ne!(key, thumb_key("tunnel", &ClipParams { seed: 8, ..params }));
        assert_ne!(key, thumb_key("tunnel", &ClipParams { palette: 2, ..params }));
        // A nudge too small to see keeps the cached picture; a real move does not.
        let mut nudged = params;
        nudged.p[0] = 0.503;
        assert_eq!(key, thumb_key("tunnel", &nudged));
        nudged.p[0] = 0.56;
        assert_ne!(key, thumb_key("tunnel", &nudged));
        assert!(has_thumbnail("tunnel") && has_thumbnail("fluid"));
        assert!(!has_thumbnail("camera") && !has_thumbnail("text") && !has_thumbnail("overlay"));
    }

    #[test]
    fn readbacks_are_encoded_for_the_display() {
        // Linear mid grey (0.5) is sRGB 188; black and white stay; alpha is forced opaque.
        let picture = to_display(1, 3, vec![0, 128, 255, 0, 0, 0, 0, 10, 255, 255, 255, 200]);
        assert_eq!(picture.rgba[..4], [0, 188, 255, 255]);
        assert_eq!(picture.rgba[4..8], [0, 0, 0, 255]);
        assert_eq!(picture.rgba[8..], [255, 255, 255, 255]);
    }
}
