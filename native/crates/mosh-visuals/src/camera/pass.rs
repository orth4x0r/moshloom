//! The camera pass: a captured BGRA frame in a [`VideoTexture`], drawn into a layer target with
//! cover scaling and the clip's parameters.
//!
//! One pipeline for the whole layer stack: the pipeline is per-draw state, not per-layer state, so
//! [`super::CameraLayer`] builds one and records it for every camera layer in the frame.
//!
//! # Why the shader is compiled here instead of extracted
//!
//! `mosh-shaders/wgsl/` is generated from the web app's `gpu.js` (`scripts/extract_wgsl.mjs`
//! deletes every file that is not in its expected set), and the web app has no camera visualizer,
//! so this layer's shader lives next to it as `overlay_camera.wgsl` and is compiled once with
//! `mosh_shaders::compile_wgsl` when the layer stack is built. A test (`tests.rs`) compiles and
//! reflects it, so a drift between the shader and [`UNIFORM_BYTES`] fails the build's test gate.

use mosh_core::ClipParams;
use mosh_render::upload::VideoTexture;
use mosh_render::{FrameRecorder, Gpu, LayerTarget, Pipeline, RenderError};

use super::params::{self, UNIFORM_BYTES};

/// The camera shader, `src/camera/overlay_camera.wgsl`.
const SHADER: &str = include_str!("overlay_camera.wgsl");

/// The name the frame's timings report for this pass.
pub(crate) const PASS_NAME: &str = "camera.frame";

pub(crate) struct CameraPass {
    pipeline: Pipeline,
}

impl CameraPass {
    pub(crate) fn new(gpu: &Gpu) -> Result<CameraPass, RenderError> {
        // A constant that a test compiles, so this only fails if the shader was edited badly.
        // `RenderError` has no shader variant: `Stage` is its one free-text error, the message the
        // app's stats overlay can show while the show keeps running with a black camera layer.
        let words = mosh_shaders::compile_wgsl(SHADER)
            .map_err(|error| RenderError::Stage(format!("camera shader: {error}")))?;
        let pipeline = Pipeline::fullscreen(gpu, &words, UNIFORM_BYTES, 1)?.with_name(PASS_NAME);
        Ok(CameraPass { pipeline })
    }

    /// Records the camera picture into `rec`'s frame: the frame's pending upload, then the cover
    /// draw. No submission and no CPU wait, exactly like `upload::CoverPass` for overlay video, so
    /// a camera layer costs the frame only the passes it records.
    pub(crate) fn record(
        &self,
        rec: &mut FrameRecorder<'_>,
        texture: &mut VideoTexture,
        dst: &LayerTarget,
        params: &ClipParams,
    ) -> Result<(), RenderError> {
        let uniforms = params::uniforms(
            params,
            (texture.width(), texture.height()),
            (dst.width(), dst.height()),
        );
        rec.upload_video(texture)?;
        rec.draw_video(&self.pipeline, dst, &uniforms, texture)
    }
}
