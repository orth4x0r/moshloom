//! Small pictures for the control window: the live preview of the stage, and clip thumbnails.
//!
//! Both are made on the render thread (the only thread that may touch the Vulkan device) and
//! sent to the window thread as plain RGBA bytes, where they become egui textures on the control
//! window's own OpenGL context. Nothing is shared between the two contexts.
//!
//! Neither one makes the render thread wait. They ride in the frame that is being drawn anyway
//! (`mosh_render::AsyncReadback`, through `Stage`):
//!
//! * **Preview**: the present submission blits the image it is about to present down to
//!   [`PREVIEW_WIDTH`] pixels and copies that into a mapped buffer; two frames later the bytes are
//!   picked up. [`PREVIEW_FPS`] times a second, and only while the control window is on screen
//!   and showing it.
//! * **Thumbnails**: one frame of a clip's scene is recorded into a [`THUMB_SIZE`] target at the
//!   end of a stage frame's own recorder, and read back the same way. [`ThumbQueue`] hands out
//!   at most one per stage frame, and none during the first frames after the renderer is built,
//!   so a show with fifty clips draws its thumbnails over a second instead of in one frame. The
//!   control window caches them by [`thumb_key`].
//!
//! What a picture costs the render thread is recording a draw and two blits and, two frames
//! later, a memcpy: measured in `mosh-render/tests/readback.rs` and `tests/preview_gpu.rs`. The
//! first version of this module used the blocking `Gpu::read_back` and cost 5.5 ms a picture.
//!
//! The sizes, the rate, the letterboxing and the queue are pure, tested without a GPU.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use mosh_core::{AnalysisFrame, ClipParams};

/// The preview's width in pixels; its height follows the stage's aspect. The top bar draws it
/// 86 points wide, which is 150 pixels on the reference laptop's 175% display.
pub const PREVIEW_WIDTH: u32 = 160;
/// The preview's refresh rate.
pub const PREVIEW_FPS: u32 = 10;
/// A thumbnail's size: 16:9, as large as the detail panel draws it on a high-DPI display.
pub const THUMB_SIZE: (u32, u32) = (128, 72);
/// Stage frames after the renderer is built in which no thumbnail is drawn: the first frames
/// are the ones that compile pipelines and fill caches, and they are late enough already.
pub const THUMB_WARMUP_FRAMES: u32 = 30;
/// Thumbnail requests waiting for a frame, at most; older ones are dropped.
pub const THUMB_QUEUE: usize = 96;
/// Stage frames a thumbnail may take to come back (it normally takes two) before it is given
/// up on: a frame that could not be presented took its picture with it.
pub const THUMB_PATIENCE_FRAMES: u32 = 30;

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

impl From<mosh_render::Readback> for Picture {
    fn from(readback: mosh_render::Readback) -> Picture {
        Picture {
            width: readback.width,
            height: readback.height,
            rgba: readback.rgba,
        }
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

/// Whether a thumbnail of `visualizer` can be drawn from the clip alone, without touching what
/// the stage is drawing. A camera, a text and a video depend on a source the thumbnail pass does
/// not open; the fluid and the particles are simulations the stage owns, and a thumbnail would
/// step them. Those get a tile in the clip's colour instead.
pub fn has_thumbnail(visualizer: &str) -> bool {
    !matches!(
        visualizer,
        "overlay" | "camera" | "text" | "particles" | "fluid"
    ) && !visualizer.starts_with("custom:")
}

/// The analysis frame a thumbnail is drawn with: a loud, bright moment a little after a beat,
/// the same for every clip so two thumbnails differ only by their clips.
///
/// The clock and the spectrum are `live`'s (the frame the stage is drawing): the scenes share
/// one spectrum history, written once per frame time, and a thumbnail must leave in it exactly
/// what the stage would have.
pub fn thumb_frame(live: &AnalysisFrame) -> AnalysisFrame {
    AnalysisFrame {
        beat_phase: 0.25,
        bar_phase: 0.062_5,
        playing: true,
        bands: [0.8, 0.65, 0.5],
        onset: 0.6,
        loudness: 0.8,
        brightness: 0.6,
        intensity: 0.75,
        level: 3,
        events: mosh_core::EventFlags::empty(),
        ..*live
    }
}

/// Thumbnails waiting to be drawn, handed out one stage frame at a time.
#[derive(Debug, Default)]
pub struct ThumbQueue {
    waiting: VecDeque<ThumbRequest>,
    /// Handed to a frame and not back yet: the key, and the frames it has been away.
    in_flight: Vec<(u64, u32)>,
    /// Stage frames since the renderer was (re)built.
    frames: u32,
    handed_out: u64,
}

impl ThumbQueue {
    pub fn new() -> ThumbQueue {
        ThumbQueue::default()
    }

    /// Queue a request. One that is already waiting is not queued twice; when the queue is
    /// full the oldest request gives way (the control window asks again for what it still shows).
    /// Returns the request that was dropped, so its asker can be told.
    pub fn push(&mut self, request: ThumbRequest) -> Option<ThumbRequest> {
        if self.waiting.iter().any(|other| other.key == request.key) {
            return None;
        }
        let dropped = if self.waiting.len() >= THUMB_QUEUE {
            self.waiting.pop_front()
        } else {
            None
        };
        self.waiting.push_back(request);
        dropped
    }

    /// The renderer was rebuilt: its first frames are warm-up again.
    pub fn renderer_rebuilt(&mut self) {
        self.frames = 0;
    }

    /// Called once per stage frame: the one thumbnail this frame may draw, if any.
    pub fn next_frame(&mut self) -> Option<ThumbRequest> {
        self.frames = self.frames.saturating_add(1);
        for (_, age) in &mut self.in_flight {
            *age += 1;
        }
        if self.frames <= THUMB_WARMUP_FRAMES {
            return None;
        }
        let request = self.waiting.pop_front()?;
        self.handed_out += 1;
        self.in_flight.push((request.key, 0));
        Some(request)
    }

    /// A thumbnail came back (or was refused): it is no longer on its way.
    pub fn settled(&mut self, key: u64) {
        self.in_flight.retain(|(other, _)| *other != key);
    }

    /// Thumbnails that were handed to a frame [`THUMB_PATIENCE_FRAMES`] ago and never came
    /// back (the frame was not presented, the renderer was rebuilt under them). They are
    /// forgotten here and returned, so whoever asked can ask again.
    pub fn lost(&mut self) -> Vec<u64> {
        let lost: Vec<u64> = self
            .in_flight
            .iter()
            .filter(|(_, age)| *age > THUMB_PATIENCE_FRAMES)
            .map(|(key, _)| *key)
            .collect();
        self.in_flight
            .retain(|(_, age)| *age <= THUMB_PATIENCE_FRAMES);
        lost
    }

    pub fn len(&self) -> usize {
        self.waiting.len()
    }

    pub fn is_empty(&self) -> bool {
        self.waiting.is_empty()
    }

    /// Thumbnails handed to a frame so far.
    pub fn handed_out(&self) -> u64 {
        self.handed_out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_preview_keeps_the_stage_aspect() {
        // 1080p: 160 x 90, exactly 16:9.
        assert_eq!(preview_size((1920, 1080), PREVIEW_WIDTH), (160, 90));
        // The laptop panel is 16:10.
        assert_eq!(preview_size((2560, 1600), PREVIEW_WIDTH), (160, 100));
        // A windowed stage that is smaller than the preview is not enlarged.
        assert_eq!(preview_size((64, 36), PREVIEW_WIDTH), (64, 36));
        // Odd sizes round up to even ones; degenerate sizes still give an image.
        assert_eq!(preview_size((1001, 563), 250), (250, 140));
        assert_eq!(preview_size((0, 0), PREVIEW_WIDTH), (16, 16));
        assert_eq!(preview_size((8, 8), PREVIEW_WIDTH), (16, 16));
        // A portrait stage does not make a tower: at most twice as tall as wide.
        assert_eq!(preview_size((1080, 3840), PREVIEW_WIDTH), (160, 320));
        // 160 x 90 RGBA is what crosses the thread boundary, ten times a second: 576 KB.
        let (width, height) = preview_size((1920, 1080), PREVIEW_WIDTH);
        assert_eq!(width * height * 4 * PREVIEW_FPS, 576_000);
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
        assert!((18..=21).contains(&taken), "{taken} previews in 2 s");
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
        assert!(has_thumbnail("tunnel") && has_thumbnail("swarm"));
        // Sources the pass does not open, and simulations the stage owns.
        for id in ["camera", "text", "overlay", "fluid", "particles", "custom:mine"] {
            assert!(!has_thumbnail(id), "{id}");
        }
    }

    #[test]
    fn a_thumbnail_leaves_the_stage_clock_and_spectrum_alone() {
        let live = AnalysisFrame {
            time_s: 91.5,
            ppq: 244.0,
            bar_index: 61,
            tempo_bpm: 163.0,
            mel: [0.3; 24],
            level: 1,
            intensity: 0.1,
            ..AnalysisFrame::default()
        };
        let frame = thumb_frame(&live);
        assert_eq!((frame.time_s, frame.ppq, frame.bar_index), (91.5, 244.0, 61));
        assert_eq!(frame.mel, live.mel);
        // And is bright whatever the music is doing, so thumbnails are comparable.
        assert_eq!(frame.level, 3);
        assert!(frame.intensity > 0.7 && frame.bands[0] > 0.7);
    }

    fn request(key: u64) -> ThumbRequest {
        ThumbRequest {
            key,
            visualizer: "tunnel".to_string(),
            params: ClipParams::default(),
        }
    }

    #[test]
    fn thumbnails_are_drawn_one_frame_at_a_time_after_the_warm_up() {
        // A show opens: fifty clips ask for a picture in the same instant.
        let mut queue = ThumbQueue::new();
        for key in 0..50 {
            assert!(queue.push(request(key)).is_none());
        }
        // Asked twice (two cells show the same look): drawn once.
        assert!(queue.push(request(7)).is_none());
        assert_eq!(queue.len(), 50);

        // The first frames after the renderer is built draw none.
        for frame in 0..THUMB_WARMUP_FRAMES {
            assert!(queue.next_frame().is_none(), "frame {frame} is warm-up");
        }
        // Then exactly one per frame, in the order they were asked for.
        let mut per_frame = Vec::new();
        for _ in 0..60 {
            per_frame.push(queue.next_frame().map(|request| request.key));
        }
        assert_eq!(
            per_frame[..50],
            (0..50).map(Some).collect::<Vec<_>>()[..],
            "one per frame, in order"
        );
        assert!(per_frame[50..].iter().all(Option::is_none), "then nothing");
        assert_eq!(queue.handed_out(), 50);
        assert!(queue.is_empty());

        // A rebuilt renderer warms up again before it draws thumbnails.
        queue.push(request(99));
        queue.renderer_rebuilt();
        assert!(queue.next_frame().is_none());

        // A picture that comes back is settled; one that never does is reported lost, once.
        let mut queue = ThumbQueue::new();
        queue.push(request(1));
        queue.push(request(2));
        for _ in 0..THUMB_WARMUP_FRAMES {
            queue.next_frame();
        }
        assert_eq!(queue.next_frame().map(|request| request.key), Some(1));
        assert_eq!(queue.next_frame().map(|request| request.key), Some(2));
        queue.next_frame();
        queue.settled(1);
        for _ in 0..THUMB_PATIENCE_FRAMES - 2 {
            queue.next_frame();
            assert!(queue.lost().is_empty(), "two frames is normal, thirty is patience");
        }
        queue.next_frame();
        queue.next_frame();
        assert_eq!(queue.lost(), vec![2]);
        assert!(queue.lost().is_empty());

        // A full queue drops its oldest request and says which.
        let mut queue = ThumbQueue::new();
        for key in 0..THUMB_QUEUE as u64 {
            queue.push(request(key));
        }
        assert_eq!(queue.push(request(1_000)).map(|dropped| dropped.key), Some(0));
        assert_eq!(queue.len(), THUMB_QUEUE);
    }
}
