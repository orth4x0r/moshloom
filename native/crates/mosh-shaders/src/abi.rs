//! The per-layer uniform block: one `#[repr(C)]` struct every visualizer and effect reads, and
//! the two WGSL struct declarations that go with it.
//!
//! # Why it is a superset
//!
//! `gpu.js` declares `SU` (208 bytes: `res`, `T`, `hue`, `bass`, `mid`, `high`, `pulse`, `vari`,
//! `dens`, `beat`, `frame`, `pal`, `aux0..2`, `l0..l2`, `mel[6]`). A custom scene written for the
//! web app reads exactly those fields. `LayerUniforms` keeps that layout byte for byte and appends
//! `p0..p7` (32 bytes, total 240), so:
//!
//! * a web custom shader compiles against [`web_prelude()`] (the web `SU`, 208 bytes) and reads the
//!   right values from a 240-byte `LayerUniforms` buffer, because the shared prefix is identical;
//! * a native custom scene compiles against [`layer_prelude()`] (`LU`: `SU` plus `p0..p7`) and can
//!   read both halves.
//!
//! `tests/compile.rs::uniforms_size_multiple_of_16` pins every offset, and the build step reflects
//! both WGSL struct sizes out of naga, so a drifting field order fails the gate rather than the
//! stage.
//!
//! # Field meanings
//!
//! | field | meaning |
//! | --- | --- |
//! | `res` | render target size in pixels |
//! | `t` | audio-driven time in seconds (`U.T` in gpu.js) |
//! | `hue` | palette offset 0..1, from `p2` |
//! | `bass`, `mid`, `high` | normalised band levels 0..1 (`AnalysisFrame::bands`) |
//! | `pulse` | 1 on the beat, decaying to 0 by the next one; 0 when the transport is stopped |
//! | `vari` | variant selector, from `p3` (`U.vari`) |
//! | `dens` | density 0..1, from `p1` (`U.dens`) |
//! | `beat` | musical position in beats (ppq), fractional |
//! | `frame` | frame counter proxy, `t * 60` |
//! | `pal` | palette index 0..7, from `ClipParams::palette` |
//! | `aux0..2` | scene hooks: reaction-diffusion seeds (-1 = unset, as in gpu.js) and the media aspect ratio / palette switch, which the renderer overrides per visualizer |
//! | `l0..l2` | opaque accent colours for the layer (the palette texture is still authoritative for `lut()`) |
//! | `mel` | 24 mel bands packed into six vec4s, exactly as `gpuSU` writes them |
//! | `p0..p7` | generic clip parameters, documented per visualizer by `VISUALIZERS` |

use mosh_core::{AnalysisFrame, ClipParams};

/// Bytes a web custom shader reads: `SU` in gpu.js. `gpu.js`'s own layout table asserts the same
/// number, and `scripts/extract_wgsl.mjs` fails if it changes.
pub const WEB_UNIFORM_BYTES: usize = 208;

/// Bytes of [`LayerUniforms`]: the web block plus `p0..p7`.
pub const LAYER_UNIFORM_BYTES: usize = 240;

/// The per-layer uniform block. `#[repr(C)]`, all members `f32`, no padding holes, so
/// `bytemuck::bytes_of` gives the bytes for the uniform buffer directly.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LayerUniforms {
    pub res: [f32; 2],
    pub t: f32,
    pub hue: f32,
    pub bass: f32,
    pub mid: f32,
    pub high: f32,
    pub pulse: f32,
    pub vari: f32,
    pub dens: f32,
    pub beat: f32,
    pub frame: f32,
    pub pal: f32,
    pub aux0: f32,
    pub aux1: f32,
    pub aux2: f32,
    pub l0: [f32; 4],
    pub l1: [f32; 4],
    pub l2: [f32; 4],
    /// 24 mel bands as six vec4s (`mel[k / 4][k % 4]`), the order `gpuSU` writes.
    pub mel: [[f32; 4]; 6],
    /// The eight generic clip parameters, documented per visualizer in
    /// [`VISUALIZERS`](crate::VISUALIZERS).
    pub p: [f32; 8],
}

impl LayerUniforms {
    /// Bytes to upload. Always [`LAYER_UNIFORM_BYTES`].
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

/// Builds one frame's uniform block from the analysis frame, the clip's parameters and the render
/// target size.
///
/// `t` is audio-driven time in seconds (the web app's `U.T`): it is passed in, not read from
/// `AnalysisFrame::time_s`, because the renderer decides between musical time and wall time.
pub fn layer_uniforms(f: &AnalysisFrame, p: &ClipParams, t: f32, res: [f32; 2]) -> LayerUniforms {
    // The web app's pulse comes from the wasm engine's plan. The native port has no per-frame
    // plan, so pulse is derived from the beat phase: 1 on the downbeat, 0 by the next beat.
    let pulse = if f.playing {
        let phase = f.beat_phase.clamp(0.0, 1.0);
        (1.0 - phase) * (1.0 - phase)
    } else {
        0.0
    };
    let hue = p.p[2];
    let base = hue + f32::from(p.palette) * ACCENT_ROTATION;
    LayerUniforms {
        res,
        t,
        hue,
        bass: f.bands[0],
        mid: f.bands[1],
        high: f.bands[2],
        pulse,
        vari: p.p[3],
        dens: p.p[1],
        beat: f.ppq as f32,
        frame: t * 60.0,
        pal: f32::from(p.palette),
        // gpu.js writes aux0 = -1 ("no reaction-diffusion seed") every frame and the renderer sets
        // it per visualizer.
        aux0: -1.0,
        aux1: 0.0,
        aux2: 0.0,
        l0: accent(base, 0.75, 0.95),
        l1: accent(base + 1.0 / 3.0, 0.70, 1.0),
        l2: accent(base + 2.0 / 3.0, 0.60, 0.90),
        mel: pack_mel(&f.mel),
        p: p.p,
    }
}

/// How far one palette index rotates the layer's accent colours, so two clips with the same hue
/// and different palettes are not the same colour.
const ACCENT_ROTATION: f32 = 0.11;

fn pack_mel(mel: &[f32; 24]) -> [[f32; 4]; 6] {
    let mut out = [[0.0; 4]; 6];
    for (k, v) in mel.iter().enumerate() {
        out[k / 4][k % 4] = *v;
    }
    out
}

fn accent(hue: f32, saturation: f32, value: f32) -> [f32; 4] {
    let h = hue.rem_euclid(1.0) * 6.0;
    let sector = h.floor();
    let f = h - sector;
    let (p, q, t) = (value * (1.0 - saturation), value * (1.0 - saturation * f), value * (1.0 - saturation * (1.0 - f)));
    let (r, g, b) = match sector as i32 % 6 {
        0 => (value, t, p),
        1 => (q, value, p),
        2 => (p, value, t),
        3 => (p, q, value),
        4 => (t, p, value),
        _ => (value, p, q),
    };
    [r, g, b, 1.0]
}

/// The web app's prelude: uniform struct `SU`, the six bindings, the helper functions and the
/// full-screen vertex stage, extracted verbatim from `gpu.js` (`WG_HEAD + WG_VS`).
///
/// Compiling `web_prelude() + <a web custom shader>` is the web app's own custom-scene path, so a
/// shader written for the web app keeps working. [`compile_custom`](crate::compile_custom) does
/// exactly that and rebases error lines the way gpu.js does.
pub fn web_prelude() -> &'static str {
    include_str!("../wgsl/web_prelude.wgsl")
}

/// The same prelude with the native uniform struct `LU`: `SU`'s fields in the same order plus
/// `p0..p7`. Native visualizers and runtime-compiled native scenes build on this one.
pub fn layer_prelude() -> &'static str {
    include_str!("../wgsl/layer_prelude.wgsl")
}
