//! HAP: block-compressed overlay video with alpha (Task 2.8's transparency follow-up).
//!
//! HAP is the VJ standard for overlay video because the CPU never decodes a pixel: every frame is
//! a small number of BCn (DXT/BC) blocks, so a frame is parsed, Snappy-decompressed and uploaded
//! straight into a compressed GPU texture. HAP Alpha and HAP Q Alpha additionally carry alpha, so
//! an overlay can be a cut-out graphic instead of a luma-keyed rectangle.
//!
//! This module is the CPU half: one HAP sample (what a QuickTime `.mov` stores per frame) in, a
//! clone of the BCn payload plus its [`HapTextureFormat`] out. It is pure Rust, has no Windows
//! dependency, and is therefore tested on every machine; [`crate::mov`] reads the container.
//!
//! # The variants, as the sample itself declares them
//!
//! A HAP sample is a section header followed by either one texture section or a `0x0D`
//! multi-image container holding two. The header's low nibble names the texture format and the
//! high nibble names the compressor, so a sample is self-describing: the container's fourcc only
//! has to say "this track is HAP". The mapping is the one FFmpeg's `hapdec.c` implements
//! (`libavcodec/hap.h`), which is the reference decoder for the format:
//!
//! | fourcc | planes | texture section | result |
//! | --- | --- | --- | --- |
//! | `Hap1` | 1 | `0x0B` DXT1/BC1 | RGB, opaque |
//! | `Hap5` | 1 | `0x0E` DXT5/BC3 | RGBA, straight alpha in the block's alpha channel |
//! | `HapY` | 1 | `0x0F` DXT5 scaled YCoCg | RGB, needs the shader's YCoCg conversion, luma in the alpha channel |
//! | `HapA` | 1 | `0x01` RGTC1/BC4 | alpha only: a silhouette |
//! | `HapM` (HAP Q Alpha) | 2 | `0x0F` + `0x01` | scaled YCoCg colour plus a separate BC4 alpha plane |
//!
//! Compressors are `0xA` (raw BCn blocks), `0xB` (Snappy, the normal one) and `0xC` (a
//! decode-instructions container that splits the texture into independently compressed chunks).
//!
//! # Sizes
//!
//! HAP is 4x4 blocks, so a frame's texture is padded up to a multiple of 4 in each direction
//! (1920x1080 needs no padding; 160x90 becomes 160x92) and the container's display size is what
//! the renderer scales against. A 1920x1080 `Hap1` frame is 480x270 blocks x 8 bytes = 1.0 MB
//! uncompressed (vs 8.3 MB of BGRA), and a `Hap5`/`HapY` frame is twice that per plane.
//!
//! # Alpha is straight, not premultiplied
//!
//! HAP stores straight (un-premultiplied) colour with an alpha channel: `Hap5`'s BC3 alpha and
//! `HapM`'s separate BC4 plane. The overlay shader multiplies colour by alpha before it reaches
//! the layer target, because `mosh-render`'s `BlendMode::Normal` is premultiplied "over".
//!
//! Nothing here has been checked against a file written by a real HAP encoder: the tests build
//! their own `.mov` (there is no encoder on this machine and media is never downloaded).

use std::fmt;
use std::path::Path;

use crate::mov;
use crate::{VideoError, VideoInfo};

/// 8K RGBA equivalent: a real HAP frame never needs more, so a bad header cannot ask for a huge
/// allocation. A 7680x4320 BC3 plane is 33 MB, well under this.
const MAX_FRAME_BYTES: usize = 7680 * 4320 * 4;

/// Compressor nibbles of a section type byte.
const COMPRESSOR_NONE: u8 = 0xA0;
const COMPRESSOR_SNAPPY: u8 = 0xB0;
const COMPRESSOR_COMPLEX: u8 = 0xC0;

/// Texture-format nibbles of a section type byte.
const FORMAT_RGTC1: u8 = 0x01;
const FORMAT_RGB_DXT1: u8 = 0x0B;
const FORMAT_RGBA_DXT5: u8 = 0x0E;
const FORMAT_YCOCG_DXT5: u8 = 0x0F;

/// The multi-image container of a two-texture frame (`HapM` / HAP Q Alpha).
const SECTION_MULTI_IMAGE: u8 = 0x0D;

/// Sub-sections of a complex (chunked) section's decode-instructions container.
const SECTION_DECODE_INSTRUCTIONS: u8 = 0x01;
const SECTION_COMPRESSOR_TABLE: u8 = 0x02;
const SECTION_SIZE_TABLE: u8 = 0x03;
const SECTION_OFFSET_TABLE: u8 = 0x04;

/// Second-stage compressor codes inside a decode-instructions table (the low nibble of a
/// compressor nibble).
const CHUNK_NONE: u8 = 0x0A;
const CHUNK_SNAPPY: u8 = 0x0B;

/// Which BCn format a HAP plane is stored in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HapTextureFormat {
    /// DXT1 / BC1: 4x4 blocks of 8 bytes, RGB, always opaque.
    Bc1,
    /// DXT5 / BC3: 4x4 blocks of 16 bytes, RGBA with straight alpha.
    Bc3,
    /// DXT5 / BC3 carrying **scaled YCoCg**: 4x4 blocks of 16 bytes. Luma (Y) is in the alpha
    /// channel, Co in red, Cg in green, and blue holds the chroma scale; a shader has to convert
    /// it to RGB (see `mosh-render`'s `overlay_hap.wgsl`).
    Bc3YCoCg,
    /// RGTC1 / BC4: 4x4 blocks of 8 bytes, one channel: an alpha plane, or a whole alpha-only
    /// (`HapA`) frame.
    Bc4Alpha,
}

impl HapTextureFormat {
    /// Bytes one 4x4 block occupies.
    pub fn bytes_per_block(self) -> usize {
        match self {
            Self::Bc1 | Self::Bc4Alpha => 8,
            Self::Bc3 | Self::Bc3YCoCg => 16,
        }
    }

    /// True for the YCoCg variant, which a shader has to convert before display.
    pub fn is_ycocg(self) -> bool {
        self == Self::Bc3YCoCg
    }

    /// True when this plane is an alpha mask rather than a picture (`HapA`, or the second plane
    /// of `HapM`).
    pub fn is_alpha_only(self) -> bool {
        self == Self::Bc4Alpha
    }

    /// A name for error messages and test output.
    pub fn name(self) -> &'static str {
        match self {
            Self::Bc1 => "BC1 (DXT1)",
            Self::Bc3 => "BC3 (DXT5)",
            Self::Bc3YCoCg => "BC3 YCoCg (HAP Q)",
            Self::Bc4Alpha => "BC4 (RGTC1 alpha)",
        }
    }

    fn from_nibble(nibble: u8) -> Option<Self> {
        match nibble {
            FORMAT_RGB_DXT1 => Some(Self::Bc1),
            FORMAT_RGBA_DXT5 => Some(Self::Bc3),
            FORMAT_YCOCG_DXT5 => Some(Self::Bc3YCoCg),
            FORMAT_RGTC1 => Some(Self::Bc4Alpha),
            _ => None,
        }
    }

    /// Bytes a `width` x `height` frame of this format occupies, with the size rounded up to
    /// whole 4x4 blocks (which is what a HAP encoder writes).
    pub fn plane_bytes(self, width: u32, height: u32) -> usize {
        let blocks_x = usize::try_from(width.div_ceil(4)).unwrap_or(0);
        let blocks_y = usize::try_from(height.div_ceil(4)).unwrap_or(0);
        blocks_x
            .saturating_mul(blocks_y)
            .saturating_mul(self.bytes_per_block())
    }
}

/// Why a HAP sample could not be parsed. Never a panic: a broken overlay is an error the app
/// reports and carries on from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HapError {
    /// The sample is empty, or a section header needs more bytes than the sample has.
    Truncated { need: usize, have: usize },
    /// The low nibble of a section type is not a texture format this build knows (BC7 `Hap7`, for
    /// example: the payload is a BCn texture too, but the block format is not wired up here yet).
    UnknownFormat(u8),
    /// The high nibble of a section type is not a known compressor.
    UnknownCompressor(u8),
    /// A complex section's decode-instructions container is malformed.
    BadInstructions(&'static str),
    /// Snappy refused the payload.
    Snappy(String),
    /// A frame is larger than this build will allocate.
    TooLarge(usize),
    /// A two-texture frame has no colour plane, or two of them.
    NoColorPlane,
}

impl fmt::Display for HapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { need, have } => {
                write!(f, "HAP sample truncated: need {need} bytes, have {have}")
            }
            Self::UnknownFormat(nibble) => {
                write!(f, "unsupported HAP texture format 0x{nibble:02X}")
            }
            Self::UnknownCompressor(nibble) => {
                write!(f, "unsupported HAP compressor 0x{nibble:02X}")
            }
            Self::BadInstructions(reason) => write!(f, "bad HAP decode instructions: {reason}"),
            Self::Snappy(reason) => write!(f, "HAP Snappy decompression failed: {reason}"),
            Self::TooLarge(bytes) => write!(f, "HAP frame of {bytes} bytes is larger than 8K"),
            Self::NoColorPlane => write!(f, "two-texture HAP frame has no colour plane"),
        }
    }
}

impl std::error::Error for HapError {}

/// One decoded plane: its block format and tightly packed 4x4 blocks, top block row first.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HapPlane {
    pub format: Option<HapTextureFormat>,
    pub data: Vec<u8>,
}

/// One HAP frame, decompressed and ready to upload into a compressed texture.
///
/// The buffers are reused frame to frame by [`HapReader`]: compare `data.as_ptr()` to know
/// whether a new frame was written.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HapFrame {
    /// The colour (or sole) plane. `None` only before the first frame is parsed.
    pub format: Option<HapTextureFormat>,
    pub data: Vec<u8>,
    /// The second plane of a `HapM` (HAP Q Alpha) frame: a BC4 alpha mask.
    pub alpha: Option<HapPlane>,
}

impl HapFrame {
    /// True for a frame whose colour already carries alpha (`Hap5`) or whose alpha is a second
    /// plane (`HapM`).
    pub fn has_alpha(&self) -> bool {
        self.format == Some(HapTextureFormat::Bc3) || self.alpha.is_some()
    }
}

/// One parsed section header: the type byte and the body it covers.
struct Section<'a> {
    kind: u8,
    header_len: usize,
    body: &'a [u8],
}

/// Reads a section header at the start of `data` (3-byte little-endian body length plus a type
/// byte; a zero length means the length is the next four bytes, so the header is eight bytes).
///
/// The length is the size *after* the header, which is what FFmpeg's `ff_hap_parse_section_header`
/// reads and what FFmpeg's HAP encoder writes.
fn section(data: &[u8]) -> Result<Section<'_>, HapError> {
    if data.len() < 4 {
        return Err(HapError::Truncated {
            need: 4,
            have: data.len(),
        });
    }
    let mut length = u32::from_le_bytes([data[0], data[1], data[2], 0]) as usize;
    let kind = data[3];
    let mut header = 4usize;
    if length == 0 {
        if data.len() < 8 {
            return Err(HapError::Truncated {
                need: 8,
                have: data.len(),
            });
        }
        length = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        header = 8;
    }
    let end = header
        .checked_add(length)
        .filter(|end| *end <= data.len())
        .ok_or(HapError::Truncated {
            need: header.saturating_add(length),
            have: data.len(),
        })?;
    Ok(Section {
        kind,
        header_len: header,
        body: &data[header..end],
    })
}

/// Parses one HAP sample into a fresh [`HapFrame`].
pub fn parse_frame(sample: &[u8]) -> Result<HapFrame, HapError> {
    let mut frame = HapFrame::default();
    parse_frame_into(sample, &mut frame)?;
    Ok(frame)
}

/// Parses one HAP sample into `frame`, reusing its buffers: the steady state (one overlay, one
/// size) allocates nothing per frame.
pub fn parse_frame_into(sample: &[u8], frame: &mut HapFrame) -> Result<(), HapError> {
    let outer = section(sample)?;
    if outer.kind & 0x0F != SECTION_MULTI_IMAGE {
        let format = decode_into(outer.kind, outer.body, &mut frame.data)?;
        frame.format = Some(format);
        frame.alpha = None;
        return Ok(());
    }

    // Two textures (`HapM`, HAP Q Alpha): the colour plane, then the alpha plane.
    let mut format = None;
    let mut alpha: Option<HapPlane> = None;
    let mut at = 0usize;
    while at < outer.body.len() {
        let inner = section(&outer.body[at..])?;
        if inner.kind & 0x0F == FORMAT_RGTC1 {
            let plane = alpha.get_or_insert_with(HapPlane::default);
            plane.format = Some(decode_into(inner.kind, inner.body, &mut plane.data)?);
        } else {
            format = Some(decode_into(inner.kind, inner.body, &mut frame.data)?);
        }
        at += inner.header_len + inner.body.len();
    }
    let format = format.ok_or(HapError::NoColorPlane)?;
    frame.format = Some(format);
    frame.alpha = alpha;
    Ok(())
}

/// Decompresses one section body into `out` and returns the format its type byte declares.
fn decode_into(kind: u8, body: &[u8], out: &mut Vec<u8>) -> Result<HapTextureFormat, HapError> {
    let format =
        HapTextureFormat::from_nibble(kind & 0x0F).ok_or(HapError::UnknownFormat(kind & 0x0F))?;
    match kind & 0xF0 {
        COMPRESSOR_NONE => {
            check_size(body.len())?;
            out.clear();
            out.extend_from_slice(body);
        }
        COMPRESSOR_SNAPPY => decompress_snappy(body, out)?,
        COMPRESSOR_COMPLEX => decompress_chunked(body, out)?,
        other => return Err(HapError::UnknownCompressor(other)),
    }
    check_size(out.len())?;
    Ok(format)
}

/// Bounds check before a payload is allowed to become a frame buffer.
fn check_size(bytes: usize) -> Result<(), HapError> {
    if bytes > MAX_FRAME_BYTES {
        return Err(HapError::TooLarge(bytes));
    }
    Ok(())
}

/// Snappy block format, into `out`.
fn decompress_snappy(payload: &[u8], out: &mut Vec<u8>) -> Result<(), HapError> {
    let length = snap::raw::decompress_len(payload).map_err(|e| HapError::Snappy(e.to_string()))?;
    check_size(length)?;
    out.clear();
    out.resize(length, 0);
    snap::raw::Decoder::new()
        .decompress(payload, out)
        .map_err(|e| HapError::Snappy(e.to_string()))?;
    Ok(())
}

/// A complex (chunked) section: a decode-instructions container followed by the chunk data.
///
/// The container holds a compressor table (`0x02`), a compressed-size table (`0x03`) and an
/// optional offset table (`0x04`); the chunks themselves follow the container. Chunks are
/// concatenated in table order, each Snappy-decompressed into its slice of the frame buffer.
fn decompress_chunked(body: &[u8], out: &mut Vec<u8>) -> Result<(), HapError> {
    let instructions = section(body)?;
    if instructions.kind != SECTION_DECODE_INSTRUCTIONS {
        return Err(HapError::BadInstructions(
            "missing decode-instructions section",
        ));
    }
    // The chunks follow the decode-instructions container; their offsets are relative to that
    // point (FFmpeg's decoder reads them from the same position).
    let chunk_data = &body[instructions.header_len + instructions.body.len()..];

    let mut compressors: &[u8] = &[];
    let mut sizes: &[u8] = &[];
    let mut offsets: Option<&[u8]> = None;
    let mut at = 0usize;
    while at < instructions.body.len() {
        let table = section(&instructions.body[at..])?;
        match table.kind {
            SECTION_COMPRESSOR_TABLE => compressors = table.body,
            SECTION_SIZE_TABLE => sizes = table.body,
            SECTION_OFFSET_TABLE => offsets = Some(table.body),
            _ => {}
        }
        at += table.header_len + table.body.len();
    }
    if compressors.is_empty() || sizes.len() < compressors.len() * 4 {
        return Err(HapError::BadInstructions(
            "missing compressor or size table",
        ));
    }
    if let Some(table) = offsets
        && table.len() < compressors.len() * 4
    {
        return Err(HapError::BadInstructions("offset table too short"));
    }

    let read = |table: &[u8], index: usize| -> usize {
        let at = index * 4;
        u32::from_le_bytes([table[at], table[at + 1], table[at + 2], table[at + 3]]) as usize
    };

    // First pass: the uncompressed length of every chunk, so the destination is sized once.
    let mut total = 0usize;
    let mut running = 0usize;
    let mut plan: Vec<(usize, usize, usize, bool)> = Vec::with_capacity(compressors.len());
    for (index, &compressor) in compressors.iter().enumerate() {
        let size = read(sizes, index);
        let offset = match offsets {
            Some(table) => read(table, index),
            None => running,
        };
        let end = offset
            .checked_add(size)
            .filter(|end| *end <= chunk_data.len())
            .ok_or(HapError::BadInstructions("chunk runs past the frame"))?;
        let source = &chunk_data[offset..end];
        let snappy = match compressor {
            CHUNK_NONE => false,
            CHUNK_SNAPPY => true,
            other => return Err(HapError::UnknownCompressor(u16::from(other) as u8)),
        };
        let length = if snappy {
            snap::raw::decompress_len(source).map_err(|e| HapError::Snappy(e.to_string()))?
        } else {
            source.len()
        };
        total = total.checked_add(length).ok_or(HapError::TooLarge(total))?;
        check_size(total)?;
        plan.push((offset, size, length, snappy));
        running = running.saturating_add(size);
    }

    out.clear();
    out.resize(total, 0);
    let mut written = 0usize;
    for (offset, size, length, snappy) in plan {
        let target = &mut out[written..written + length];
        let source = &chunk_data[offset..offset + size];
        if snappy {
            snap::raw::Decoder::new()
                .decompress(source, target)
                .map_err(|e| HapError::Snappy(e.to_string()))?;
        } else {
            target.copy_from_slice(source);
        }
        written += length;
    }
    Ok(())
}

/// One open HAP clip: the demuxer's sample table plus the compressed-sample buffer. The decoded
/// frame belongs to the [`crate::VideoSource`] it feeds, so its buffers are reused across calls.
pub(crate) struct HapReader {
    file: mov::MovFile,
    /// Index of the next sample [`Self::read_frame`] returns.
    next: usize,
    /// The compressed sample currently being read; reused.
    sample: Vec<u8>,
}

impl HapReader {
    /// Opens `path` when it is a QuickTime/MP4 file with a HAP video track, and validates its
    /// first frame so a broken file fails at open instead of mid-set.
    ///
    /// `Ok(None)` means "not a HAP file" (the caller falls back to Media Foundation); `Err` means
    /// the file says it is HAP but cannot be read, and carries the reason.
    pub(crate) fn open_if_hap(path: &Path) -> Result<Option<(Self, VideoInfo)>, VideoError> {
        let Some(file) = mov::MovFile::open_hap(path)? else {
            return Ok(None);
        };
        let width = file.width;
        let height = file.height;
        if width == 0 || height == 0 || file.samples.is_empty() {
            return Err(VideoError::Unsupported(format!(
                "{} has a HAP track with no frames",
                path.display()
            )));
        }
        if width > 7680 || height > 4320 {
            return Err(VideoError::Unsupported(format!(
                "{} is {width}x{height}, larger than 8K",
                path.display()
            )));
        }
        let info = VideoInfo {
            w: width,
            h: height,
            fps: file.fps,
            duration_s: file.duration_s,
        };
        let mut reader = HapReader {
            file,
            next: 0,
            sample: Vec::new(),
        };
        // Decoding frame 0 costs a sample read and a Snappy pass at open, and turns a garbage
        // file into an error the app can show instead of a black overlay mid-set.
        let mut first = HapFrame::default();
        reader.read_frame(&mut first).map_err(|error| match error {
            VideoError::Decode(reason) => VideoError::Unsupported(format!(
                "{}: HAP frame 0 is unreadable: {reason}",
                path.display()
            )),
            other => other,
        })?;
        reader.next = 0;
        Ok(Some((reader, info)))
    }

    /// Moves the cursor to the last sample whose presentation time is at or before `t_s`.
    pub(crate) fn seek(&mut self, t_s: f64) {
        self.next = self.file.sample_at(t_s);
    }

    /// Reads the next sample into `frame` and returns its presentation time, or `None` at the end
    /// of the clip.
    pub(crate) fn read_frame(&mut self, frame: &mut HapFrame) -> Result<Option<f64>, VideoError> {
        if self.next >= self.file.samples.len() {
            return Ok(None);
        }
        let index = self.next;
        let pts_s = self.file.read_sample(index, &mut self.sample)?;
        parse_frame_into(&self.sample, frame)
            .map_err(|error| VideoError::Decode(format!("sample {index}: {error}")))?;

        let width = self.file.width;
        let height = self.file.height;
        let format = match frame.format {
            Some(format) => format,
            None => {
                return Err(VideoError::Decode(format!(
                    "sample {index}: no colour plane"
                )));
            }
        };
        // The container's fourcc and the sample's own section type must agree; FFmpeg's decoder
        // makes the same check, and a mismatch is a broken file, not a new variant.
        if let Some(expected) = expected_format(&self.file.codec) {
            if format != expected {
                return Err(VideoError::Decode(format!(
                    "sample {index}: {} says {}, sample is {}",
                    String::from_utf8_lossy(&self.file.codec),
                    expected.name(),
                    format.name()
                )));
            }
        } else if format != HapTextureFormat::Bc3YCoCg {
            return Err(VideoError::Decode(format!(
                "sample {index}: HapM colour plane is {}, expected {}",
                format.name(),
                HapTextureFormat::Bc3YCoCg.name()
            )));
        }
        if let Some(alpha) = frame.alpha.as_ref()
            && alpha.format != Some(HapTextureFormat::Bc4Alpha)
        {
            return Err(VideoError::Decode(format!(
                "sample {index}: alpha plane is not BC4"
            )));
        }

        let needed = format.plane_bytes(width, height);
        if frame.data.len() != needed {
            return Err(VideoError::Decode(format!(
                "sample {index}: {} decompressed to {} bytes, {width}x{height} needs {needed}",
                format.name(),
                frame.data.len()
            )));
        }
        if let Some(alpha) = frame.alpha.as_ref() {
            let needed = HapTextureFormat::Bc4Alpha.plane_bytes(width, height);
            if alpha.data.len() != needed {
                return Err(VideoError::Decode(format!(
                    "sample {index}: alpha plane is {} bytes, needs {needed}",
                    alpha.data.len()
                )));
            }
        }

        self.next += 1;
        Ok(Some(pts_s))
    }

    /// CPU memory the reader holds: the sample table, the compressed sample and itself. The
    /// decompressed frame belongs to the `VideoSource`.
    pub(crate) fn memory_bytes(&self) -> usize {
        self.file.memory_bytes() + self.sample.capacity() + std::mem::size_of::<Self>()
    }
}

/// The fourccs a HAP track can use, so opening a file can tell "HAP" from "some other QuickTime
/// codec" before it parses a frame.
pub(crate) fn is_hap_fourcc(fourcc: &[u8; 4]) -> bool {
    matches!(fourcc, b"Hap1" | b"Hap5" | b"HapY" | b"HapA" | b"HapM")
}

/// The texture format a fourcc's frames must use, or `None` for the two-plane `HapM` (whose
/// section headers carry both formats).
pub(crate) fn expected_format(fourcc: &[u8; 4]) -> Option<HapTextureFormat> {
    match fourcc {
        b"Hap1" => Some(HapTextureFormat::Bc1),
        b"Hap5" => Some(HapTextureFormat::Bc3),
        b"HapY" => Some(HapTextureFormat::Bc3YCoCg),
        b"HapA" => Some(HapTextureFormat::Bc4Alpha),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn section_bytes(kind: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + payload.len());
        let length = payload.len() as u32;
        out.extend_from_slice(&length.to_le_bytes()[..3]);
        out.push(kind);
        out.extend_from_slice(payload);
        out
    }

    fn snappy(bytes: &[u8]) -> Vec<u8> {
        snap::raw::Encoder::new().compress_vec(bytes).unwrap()
    }

    #[test]
    fn raw_bc1_section_parses() {
        let payload = vec![0x11u8; 128];
        let frame = parse_frame(&section_bytes(COMPRESSOR_NONE | FORMAT_RGB_DXT1, &payload))
            .expect("raw BC1");
        assert_eq!(frame.format, Some(HapTextureFormat::Bc1));
        assert_eq!(frame.data, payload);
        assert!(frame.alpha.is_none());
        assert!(!frame.has_alpha());
    }

    #[test]
    fn snappy_bc3_section_parses() {
        let payload: Vec<u8> = (0..256).map(|i| (i % 251) as u8).collect();
        let frame = parse_frame(&section_bytes(
            COMPRESSOR_SNAPPY | FORMAT_RGBA_DXT5,
            &snappy(&payload),
        ))
        .expect("snappy BC3");
        assert_eq!(frame.format, Some(HapTextureFormat::Bc3));
        assert_eq!(frame.data, payload);
        assert!(frame.has_alpha());
    }

    #[test]
    fn long_header_is_honoured() {
        let payload = vec![0x5Au8; 64];
        let mut sample = vec![0, 0, 0, COMPRESSOR_NONE | FORMAT_YCOCG_DXT5];
        sample.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        sample.extend_from_slice(&payload);
        let frame = parse_frame(&sample).expect("eight-byte header");
        assert_eq!(frame.format, Some(HapTextureFormat::Bc3YCoCg));
        assert_eq!(frame.data, payload);
        assert!(frame.format.unwrap().is_ycocg());
    }

    #[test]
    fn two_planes_parse_as_colour_and_alpha() {
        let colour = vec![0xC3u8; 64];
        let alpha = vec![0xA5u8; 32];
        let mut sample = section_bytes(COMPRESSOR_NONE | FORMAT_YCOCG_DXT5, &colour);
        sample.extend_from_slice(&section_bytes(
            COMPRESSOR_SNAPPY | FORMAT_RGTC1,
            &snappy(&alpha),
        ));
        let sample = section_bytes(COMPRESSOR_NONE | SECTION_MULTI_IMAGE, &sample);

        let frame = parse_frame(&sample).expect("two planes");
        assert_eq!(frame.format, Some(HapTextureFormat::Bc3YCoCg));
        assert_eq!(frame.data, colour);
        let plane = frame.alpha.as_ref().expect("alpha plane");
        assert_eq!(plane.format, Some(HapTextureFormat::Bc4Alpha));
        assert_eq!(plane.data, alpha);
        assert!(frame.has_alpha());
    }

    #[test]
    fn complex_chunked_sections_concatenate() {
        let first = vec![0x11u8; 40];
        let second = vec![0x22u8; 24];
        let first_chunk = snappy(&first);
        let second_chunk = snappy(&second);

        let mut tables = section_bytes(SECTION_COMPRESSOR_TABLE, &[CHUNK_SNAPPY, CHUNK_SNAPPY]);
        let mut sizes = Vec::new();
        sizes.extend_from_slice(&(first_chunk.len() as u32).to_le_bytes());
        sizes.extend_from_slice(&(second_chunk.len() as u32).to_le_bytes());
        tables.extend_from_slice(&section_bytes(SECTION_SIZE_TABLE, &sizes));
        let instructions = section_bytes(SECTION_DECODE_INSTRUCTIONS, &tables);

        let mut body = instructions;
        body.extend_from_slice(&first_chunk);
        body.extend_from_slice(&second_chunk);
        let frame = parse_frame(&section_bytes(COMPRESSOR_COMPLEX | FORMAT_RGBA_DXT5, &body))
            .expect("chunked");

        let mut expected = first;
        expected.extend_from_slice(&second);
        assert_eq!(frame.data, expected);
        assert_eq!(frame.format, Some(HapTextureFormat::Bc3));
    }

    #[test]
    fn offset_table_is_optional_and_honoured() {
        let first = vec![0x33u8; 16];
        let second = vec![0x44u8; 16];
        let chunks = [first.clone(), second.clone()];
        let mut payload = Vec::new();
        for chunk in &chunks {
            payload.extend_from_slice(chunk);
        }
        // Give the second chunk an explicit offset table entry pointing past the first.
        let mut tables = section_bytes(SECTION_COMPRESSOR_TABLE, &[CHUNK_NONE, CHUNK_NONE]);
        let mut sizes = Vec::new();
        for chunk in &chunks {
            sizes.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        }
        tables.extend_from_slice(&section_bytes(SECTION_SIZE_TABLE, &sizes));
        let mut offsets = Vec::new();
        offsets.extend_from_slice(&0u32.to_le_bytes());
        offsets.extend_from_slice(&(chunks[0].len() as u32).to_le_bytes());
        tables.extend_from_slice(&section_bytes(SECTION_OFFSET_TABLE, &offsets));
        let instructions = section_bytes(SECTION_DECODE_INSTRUCTIONS, &tables);

        let mut body = instructions;
        body.extend_from_slice(&payload);
        let frame = parse_frame(&section_bytes(COMPRESSOR_COMPLEX | FORMAT_RGB_DXT1, &body))
            .expect("offset table");
        let mut expected = first;
        expected.extend_from_slice(&second);
        assert_eq!(frame.data, expected);
    }

    #[test]
    fn truncated_and_unknown_samples_are_errors() {
        assert!(matches!(parse_frame(&[]), Err(HapError::Truncated { .. })));
        assert!(matches!(
            parse_frame(&[0x20, 0, 0, 0xB0, 1, 2, 3]),
            Err(HapError::Truncated { .. })
        ));
        // Format nibble 0x0C (BC7 in FFmpeg's table is not assigned a section type; 0x0C is
        // unknown here) is a clean error, not a wrong format.
        assert_eq!(
            parse_frame(&section_bytes(COMPRESSOR_NONE | 0x0C, &[0; 8])),
            Err(HapError::UnknownFormat(0x0C))
        );
        assert_eq!(
            parse_frame(&section_bytes(0xD0 | FORMAT_RGB_DXT1, &[0; 8])),
            Err(HapError::UnknownCompressor(0xD0))
        );
        assert_eq!(
            parse_frame(&section_bytes(
                COMPRESSOR_SNAPPY | FORMAT_RGB_DXT1,
                &[0xFF; 8]
            )),
            Err(HapError::Snappy(snap::Error::Header.to_string()))
        );
    }

    #[test]
    fn plane_bytes_pads_to_whole_blocks() {
        // 1920x1080 is exactly 480x270 blocks, so a 1080p frame needs no padding at all.
        assert_eq!(HapTextureFormat::Bc1.plane_bytes(1920, 1080), 480 * 270 * 8);
        assert_eq!(
            HapTextureFormat::Bc3.plane_bytes(1920, 1080),
            480 * 270 * 16
        );
        // 160x90 is 40x22.5 blocks, so the height rounds up to 23.
        assert_eq!(HapTextureFormat::Bc1.plane_bytes(160, 90), 40 * 23 * 8);
        assert_eq!(HapTextureFormat::Bc4Alpha.plane_bytes(5, 5), 2 * 2 * 8);
    }

    #[test]
    fn fourcc_table_matches_the_container() {
        for fourcc in [b"Hap1", b"Hap5", b"HapY", b"HapA", b"HapM"] {
            assert!(is_hap_fourcc(fourcc), "{fourcc:?}");
        }
        assert!(!is_hap_fourcc(b"avc1"));
        assert_eq!(expected_format(b"Hap1"), Some(HapTextureFormat::Bc1));
        assert_eq!(expected_format(b"Hap5"), Some(HapTextureFormat::Bc3));
        assert_eq!(expected_format(b"HapY"), Some(HapTextureFormat::Bc3YCoCg));
        assert_eq!(expected_format(b"HapA"), Some(HapTextureFormat::Bc4Alpha));
        assert_eq!(expected_format(b"HapM"), None);
    }
}
