//! Camera layer: a live webcam picture (e.g. the performer singing), so it can be moshed and
//! effected like any other layer. Owner: L-camera.
//!
//! The picture comes from [`mosh_video::camera`]: a capture thread owns the device and publishes
//! the newest frame; this module only takes it, uploads it and draws it. Nothing on the render
//! thread waits for the camera — [`CameraLayer::draw`] takes a frame if one is waiting and
//! otherwise redraws the last one, or clears the layer while there is no picture at all.
//!
//! # What a camera clip draws
//!
//! * Cover scaling, like overlay video: the camera's aspect fills the layer and the extra axis is
//!   cropped, centred.
//! * `p0..p7` = [`PARAMS`], applied by `overlay_camera.wgsl`: zoom, pan x, pan y, mirror, exposure,
//!   contrast, tint, posterize. The math is in `params.rs` and neutral at 0, so a default clip
//!   shows the whole frame.
//!
//! # Missing, busy and unplugged cameras
//!
//! The capture thread retries every `mosh_video::camera::RETRY_INTERVAL`, so a camera that is
//! missing, held by another app or unplugged mid-set recovers on its own. Until it produces a
//! picture the layer is transparent black and `draw` returns `Ok`: a blown take is a black layer,
//! never an error on the render thread or a stopped show. A device that goes away after it was
//! streaming clears the layer instead of freezing its last picture.
//!
//! # Cost
//!
//! One frame of a 1920x1080 camera is a memcpy into the frame's staging buffer (inside
//! [`VideoTexture::upload`]) plus two recorded passes through the frame's single submission, the
//! same path overlay video takes. Measured in the same run (`tests.rs`), the camera's two recorded
//! passes cost within 0.01 ms of an overlay layer's copy-and-cover pass (0.135 vs 0.129 ms minimum,
//! 0.163 vs 0.157 ms median at 1080p on the reference laptop), and `CameraLayer::draw` adds no fence
//! wait of its own.

use mosh_core::ClipParams;
use mosh_render::upload::VideoTexture;
use mosh_render::{FrameRecorder, Gpu, LayerTarget, RenderError};
use mosh_video::camera::{CameraCapture, CameraFrame};

mod params;
mod pass;
#[cfg(test)]
mod tests;

use pass::CameraPass;

/// What `p0..p7` mean for `camera` (shown in the clip editor).
pub const PARAMS: [&str; 8] = [
    "zoom",
    "pan x",
    "pan y",
    "mirror",
    "exposure",
    "contrast",
    "tint",
    "posterize",
];

/// A show has far fewer layers than this; the cap only stops a bogus index from growing the
/// per-layer vector without bound.
const MAX_LAYERS: usize = 32;

pub struct CameraLayer {
    /// The camera pipeline, shared by every camera layer in the frame.
    pass: CameraPass,
    /// One slot per layer index, created by [`CameraLayer::set_source`].
    slots: Vec<Option<CameraSlot>>,
}

/// One camera layer: its capture and the texture its newest frame lands in.
struct CameraSlot {
    capture: CameraCapture,
    /// Created by the first frame, reused for every frame of that size.
    texture: Option<VideoTexture>,
    /// True while `texture` holds a picture from the current, working capture: redrawing it is
    /// right when the render thread runs faster than the camera. Cleared when the capture fails,
    /// so an unplugged camera goes black instead of freezing.
    has_picture: bool,
}

impl CameraSlot {
    fn new(device: Option<&str>) -> CameraSlot {
        CameraSlot {
            capture: CameraCapture::open(device),
            texture: None,
            has_picture: false,
        }
    }

    /// Uploads one captured frame into this layer's texture, creating it (or resizing it through
    /// [`VideoTexture::upload`]) when the camera's picture size changed.
    fn store(&mut self, gpu: &Gpu, frame: &CameraFrame) -> Result<(), RenderError> {
        if self.texture.is_none() {
            self.texture = Some(VideoTexture::new(gpu, frame.w, frame.h)?);
        }
        let texture = self
            .texture
            .as_mut()
            .expect("the texture was created just above");
        let result = texture.upload(gpu, frame.w, frame.h, &frame.bgra);
        self.has_picture = result.is_ok();
        result
    }
}

impl CameraLayer {
    pub fn new(gpu: &Gpu) -> Result<CameraLayer, RenderError> {
        Ok(CameraLayer {
            pass: CameraPass::new(gpu)?,
            slots: Vec::new(),
        })
    }

    /// Which camera `layer` shows: `None` = the default camera. Called by the app when a camera
    /// clip starts; opening a device never blocks the render thread (capture runs on its own
    /// thread and the newest frame is taken each draw).
    ///
    /// Setting the same device again does nothing, so a clip that restarts does not reopen its
    /// camera. A layer index at or above the internal cap is ignored rather than allowed to grow
    /// the per-layer table.
    pub fn set_source(&mut self, layer: usize, device: Option<&str>) {
        if layer >= MAX_LAYERS {
            return;
        }
        let requested = normalize(device);
        if let Some(Some(slot)) = self.slots.get(layer)
            && slot.capture.requested() == requested
        {
            return;
        }
        if self.slots.len() <= layer {
            self.slots.resize_with(layer + 1, || None);
        }
        self.slots[layer] = Some(CameraSlot::new(requested));
    }

    /// The camera names Windows reports, for the clip editor.
    pub fn devices() -> Vec<String> {
        mosh_video::camera::devices()
    }

    /// Draw `layer`'s newest camera frame into `dst` (cover-scaled, `params` applied). Black
    /// until a frame has arrived; never waits for one.
    pub fn draw(
        &mut self,
        rec: &mut FrameRecorder<'_>,
        layer: usize,
        dst: &LayerTarget,
        params: &ClipParams,
    ) -> Result<(), RenderError> {
        let Some(slot) = self.slots.get_mut(layer).and_then(Option::as_mut) else {
            return rec.clear(dst);
        };

        // A failed capture has no picture: clear rather than freeze the last frame, and let the
        // capture thread's next attempt bring it back.
        if slot.capture.is_failed() {
            slot.has_picture = false;
            return rec.clear(dst);
        }

        if let Some(frame) = slot.capture.take_frame() {
            let stored = slot.store(rec.gpu(), &frame);
            // Hand the buffer back whatever happened: one allocation per failure, not per frame.
            slot.capture.recycle(frame);
            stored?;
        }

        let has_picture = slot.has_picture;
        match (slot.texture.as_mut(), has_picture) {
            (Some(texture), true) => self.pass.record(rec, texture, dst, params),
            _ => rec.clear(dst),
        }
    }
}

/// A device name as [`CameraCapture`] stores it: trimmed, with `""` meaning the default camera.
fn normalize(device: Option<&str>) -> Option<&str> {
    device.map(str::trim).filter(|name| !name.is_empty())
}
