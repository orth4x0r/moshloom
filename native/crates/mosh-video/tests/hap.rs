//! HAP overlay video, end to end through `VideoSource` (Task 2.8's transparency follow-up).
//!
//! There is no HAP encoder on this machine and media is never downloaded, so every fixture here
//! is built in the test: a tiny QuickTime `.mov` whose video track is HAP, with BC1/BC3/BC4
//! blocks of known colours and Snappy-compressed samples. What that cannot prove is that a file
//! written by a real encoder (Resolume, TouchDesigner, FFmpeg) parses — that is the one thing
//! this lane does not verify, and the PR says so.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use mosh_video::hap::{self, HapTextureFormat};
use mosh_video::{VideoError, VideoSource};

// ---------------------------------------------------------------------------------------------
// Fixtures: BCn blocks
// ---------------------------------------------------------------------------------------------

/// RGB565 the way BC1/BC3 colour blocks store their endpoints.
fn rgb565(rgb: [u8; 3]) -> u16 {
    (u16::from(rgb[0] >> 3) << 11) | (u16::from(rgb[1] >> 2) << 5) | u16::from(rgb[2] >> 3)
}

/// What a BC1 block decodes to when both endpoints are equal and every index selects endpoint 0:
/// one solid colour, quantised to 5-6-5. The GPU's expansion, so expectations match the picture.
fn rgb565_expand(colour: u16) -> [u8; 3] {
    let r = ((colour >> 11) & 0x1F) as u8;
    let g = ((colour >> 5) & 0x3F) as u8;
    let b = (colour & 0x1F) as u8;
    [
        (r << 3) | (r >> 2),
        (g << 2) | (g >> 4),
        (b << 3) | (b >> 2),
    ]
}

/// One solid 4x4 BC1 block: the bytes a HAP DXT1 payload is made of.
fn bc1_block(rgb: [u8; 3]) -> [u8; 8] {
    let colour = rgb565(rgb);
    let (lo, hi) = (colour as u8, (colour >> 8) as u8);
    [lo, hi, lo, hi, 0, 0, 0, 0]
}

/// One solid 4x4 BC3 block: straight alpha in the alpha block, colour as BC1.
fn bc3_block(rgba: [u8; 4]) -> [u8; 16] {
    let mut block = [0u8; 16];
    // Alpha endpoints equal, every 3-bit index 0: alpha is exactly `rgba[3]`.
    block[0] = rgba[3];
    block[1] = rgba[3];
    let colour = bc1_block([rgba[0], rgba[1], rgba[2]]);
    block[8..].copy_from_slice(&colour);
    block
}

/// One solid 4x4 BC4 block: two equal endpoints, every index 0.
fn bc4_block(alpha: u8) -> [u8; 8] {
    [alpha, alpha, 0, 0, 0, 0, 0, 0]
}

/// A `width` x `height` payload of one solid block colour, rounded up to whole 4x4 blocks.
fn bc1_solid(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
    let blocks = (width.div_ceil(4) * height.div_ceil(4)) as usize;
    let block = bc1_block(rgb);
    block.repeat(blocks)
}

/// A BC3 payload whose 4x4 blocks are split at `x`: `left` colour left of the split, `right`
/// colour right of it. Each block gets its own colour, so the split is exact at a block edge (put
/// it on a multiple of 4 and the frame has a crisp transparent half).
fn bc3_half_alpha(width: u32, height: u32, left: [u8; 4], right: [u8; 4]) -> Vec<u8> {
    let blocks_x = width.div_ceil(4);
    let blocks_y = height.div_ceil(4);
    let mut payload = Vec::with_capacity((blocks_x * blocks_y * 16) as usize);
    for _ in 0..blocks_y {
        for block_x in 0..blocks_x {
            let rgba = if block_x < blocks_x / 2 { left } else { right };
            payload.extend_from_slice(&bc3_block(rgba));
        }
    }
    payload
}

/// A `width` x `height` BC4 payload whose left blocks are opaque and right blocks transparent.
fn bc4_half(width: u32, height: u32) -> Vec<u8> {
    let blocks_x = width.div_ceil(4);
    let blocks_y = height.div_ceil(4);
    let mut payload = Vec::with_capacity((blocks_x * blocks_y * 8) as usize);
    for _ in 0..blocks_y {
        for block_x in 0..blocks_x {
            let alpha = if block_x < blocks_x / 2 { 255 } else { 0 };
            payload.extend_from_slice(&bc4_block(alpha));
        }
    }
    payload
}

// ---------------------------------------------------------------------------------------------
// Fixtures: one HAP sample and one `.mov`
// ---------------------------------------------------------------------------------------------

/// A HAP section: 3-byte little-endian body length, then the type byte.
fn section(kind: u8, payload: &[u8]) -> Vec<u8> {
    assert!(
        payload.len() < 0x00FF_FFFF,
        "use the eight-byte header in the real world"
    );
    let mut out = Vec::with_capacity(4 + payload.len());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes()[..3]);
    out.push(kind);
    out.extend_from_slice(payload);
    out
}

fn snappy(bytes: &[u8]) -> Vec<u8> {
    snap::raw::Encoder::new()
        .compress_vec(bytes)
        .expect("Snappy-compress a fixture")
}

/// Compressor nibbles, as FFmpeg's `hap.h` defines them.
const NONE: u8 = 0xA0;
const SNAPPY: u8 = 0xB0;

/// Texture-format nibbles from the same header.
const BC1: u8 = 0x0B;
const BC3: u8 = 0x0E;
const YCOCG: u8 = 0x0F;
const RGTC1: u8 = 0x01;

/// A one-plane HAP sample.
fn hap_sample(format: u8, payload: &[u8], compress: bool) -> Vec<u8> {
    if compress {
        section(SNAPPY | format, &snappy(payload))
    } else {
        section(NONE | format, payload)
    }
}

/// A `HapM` (HAP Q Alpha) sample: a `0x0D` container with the colour plane, then the alpha plane.
fn hap_sample_two_planes(colour: &[u8], alpha: &[u8]) -> Vec<u8> {
    let mut planes = section(SNAPPY | YCOCG, &snappy(colour));
    planes.extend_from_slice(&section(SNAPPY | RGTC1, &snappy(alpha)));
    section(NONE | 0x0D, &planes)
}

fn be32(value: u32) -> [u8; 4] {
    value.to_be_bytes()
}

fn append_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&be32(value));
}

/// A box: 32-bit size, type, body.
fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + body.len());
    append_u32(&mut out, (8 + body.len()) as u32);
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    out
}

/// A full box: size, type, version + flags, body.
fn full_box(kind: &[u8; 4], version_flags: u32, body: &[u8]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(4 + body.len());
    append_u32(&mut payload, version_flags);
    payload.extend_from_slice(body);
    boxed(kind, &payload)
}

/// A minimal QuickTime `.mov` with one HAP video track.
///
/// `samples_per_chunk` controls how `stsc`/`stco` describe the samples, so the demuxer's chunk
/// walk (rather than only its sample table) is exercised.
struct MovBuilder<'a> {
    fourcc: [u8; 4],
    width: u32,
    height: u32,
    /// Frames per second; the media timescale is 1000 and `stts` uses one entry.
    fps: u32,
    samples: &'a [Vec<u8>],
    samples_per_chunk: u32,
}

impl MovBuilder<'_> {
    fn build(&self) -> Vec<u8> {
        let mut ftyp = Vec::new();
        ftyp.extend_from_slice(b"qt  ");
        append_u32(&mut ftyp, 0x0000_0200);
        ftyp.extend_from_slice(b"qt  ");
        let ftyp = boxed(b"ftyp", &ftyp);

        // mdat first, so every stco offset is "the data start plus the samples before this chunk".
        let mut media = Vec::new();
        for sample in self.samples {
            media.extend_from_slice(sample);
        }
        let mdat = boxed(b"mdat", &media);
        let media_start = (ftyp.len() + 8) as u32;

        // Chunk layout: `samples_per_chunk` samples each, the last chunk short.
        let per_chunk = self.samples_per_chunk.max(1) as usize;
        let mut chunk_offsets = Vec::new();
        let mut offset = media_start;
        for chunk in self.samples.chunks(per_chunk) {
            chunk_offsets.push(offset);
            offset += chunk.iter().map(|sample| sample.len() as u32).sum::<u32>();
        }

        let delta = 1000 / self.fps;
        let duration = delta * self.samples.len() as u32;

        // stsd: one video sample entry with the HAP fourcc and the display size.
        let mut entry = Vec::new();
        entry.extend_from_slice(&[0u8; 6]); // reserved
        entry.extend_from_slice(&[0, 1]); // data reference index
        entry.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]); // version, revision, vendor
        entry.extend_from_slice(&[0u8; 8]); // temporal + spatial quality
        entry.extend_from_slice(&(self.width as u16).to_be_bytes());
        entry.extend_from_slice(&(self.height as u16).to_be_bytes());
        entry.extend_from_slice(&[0x00, 0x48, 0x00, 0x00]); // 72 dpi
        entry.extend_from_slice(&[0x00, 0x48, 0x00, 0x00]);
        entry.extend_from_slice(&[0, 0, 0, 0]); // reserved
        entry.extend_from_slice(&[0, 1]); // frame count
        entry.extend_from_slice(&[0u8; 32]); // compressor name
        entry.extend_from_slice(&[0, 24]); // depth
        entry.extend_from_slice(&[0xFF, 0xFF]); // pre_defined
        let size = (8 + entry.len()) as u32;
        let mut sample_entry = Vec::new();
        append_u32(&mut sample_entry, size);
        sample_entry.extend_from_slice(&self.fourcc);
        sample_entry.extend_from_slice(&entry);
        let stsd = full_box(b"stsd", 0, &{
            let mut body = Vec::new();
            append_u32(&mut body, 1);
            body.extend_from_slice(&sample_entry);
            body
        });

        let mut stts_body = Vec::new();
        append_u32(&mut stts_body, 1);
        append_u32(&mut stts_body, self.samples.len() as u32);
        append_u32(&mut stts_body, delta);
        let stts = full_box(b"stts", 0, &stts_body);

        // One stsc entry is enough for a fixture; real encoders interleave more carefully.
        let mut stsc_body = Vec::new();
        append_u32(&mut stsc_body, 1);
        append_u32(&mut stsc_body, 1);
        append_u32(&mut stsc_body, per_chunk as u32);
        append_u32(&mut stsc_body, 1);
        let stsc = full_box(b"stsc", 0, &stsc_body);

        let mut stsz_body = Vec::new();
        append_u32(&mut stsz_body, 0); // sizes vary per sample
        append_u32(&mut stsz_body, self.samples.len() as u32);
        for sample in self.samples {
            append_u32(&mut stsz_body, sample.len() as u32);
        }
        let stsz = full_box(b"stsz", 0, &stsz_body);

        let mut stco_body = Vec::new();
        append_u32(&mut stco_body, chunk_offsets.len() as u32);
        for offset in &chunk_offsets {
            append_u32(&mut stco_body, *offset);
        }
        let stco = full_box(b"stco", 0, &stco_body);

        let mut stbl_body = Vec::new();
        for atom in [stsd, stts, stsc, stsz, stco] {
            stbl_body.extend_from_slice(&atom);
        }
        let stbl = boxed(b"stbl", &stbl_body);

        let vmhd = full_box(b"vmhd", 1, &[0, 0, 0, 0, 0, 0, 0, 0]);
        let url = full_box(b"url ", 1, &[]);
        let mut dref_body = Vec::new();
        append_u32(&mut dref_body, 1);
        dref_body.extend_from_slice(&url);
        let dref = full_box(b"dref", 0, &dref_body);
        let dinf = boxed(b"dinf", &dref);
        let mut minf_body = Vec::new();
        minf_body.extend_from_slice(&vmhd);
        minf_body.extend_from_slice(&dinf);
        minf_body.extend_from_slice(&stbl);
        let minf = boxed(b"minf", &minf_body);

        let mut mdhd_body = Vec::new();
        append_u32(&mut mdhd_body, 0); // creation
        append_u32(&mut mdhd_body, 0); // modification
        append_u32(&mut mdhd_body, 1000); // timescale
        append_u32(&mut mdhd_body, duration);
        mdhd_body.extend_from_slice(&[0x55, 0xC4]); // language "und"
        mdhd_body.extend_from_slice(&[0, 0]); // quality
        let mdhd = full_box(b"mdhd", 0, &mdhd_body);

        let mut hdlr_body = Vec::new();
        append_u32(&mut hdlr_body, 0); // pre_defined
        hdlr_body.extend_from_slice(b"vide");
        hdlr_body.extend_from_slice(&[0u8; 12]);
        hdlr_body.extend_from_slice(b"VideoHandler\0");
        let hdlr = full_box(b"hdlr", 0, &hdlr_body);

        let mut mdia_body = Vec::new();
        mdia_body.extend_from_slice(&mdhd);
        mdia_body.extend_from_slice(&hdlr);
        mdia_body.extend_from_slice(&minf);
        let mdia = boxed(b"mdia", &mdia_body);

        let mut tkhd_body = Vec::new();
        append_u32(&mut tkhd_body, 0); // creation
        append_u32(&mut tkhd_body, 0); // modification
        append_u32(&mut tkhd_body, 1); // track id
        append_u32(&mut tkhd_body, 0); // reserved
        append_u32(&mut tkhd_body, duration);
        tkhd_body.extend_from_slice(&[0u8; 8]);
        tkhd_body.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // layer, group, volume, reserved
        tkhd_body.extend_from_slice(&[0u8; 36]); // matrix
        append_u32(&mut tkhd_body, self.width << 16);
        append_u32(&mut tkhd_body, self.height << 16);
        let tkhd = full_box(b"tkhd", 3, &tkhd_body);

        let mut trak_body = Vec::new();
        trak_body.extend_from_slice(&tkhd);
        trak_body.extend_from_slice(&mdia);
        let trak = boxed(b"trak", &trak_body);

        let mut mvhd_body = Vec::new();
        append_u32(&mut mvhd_body, 0);
        append_u32(&mut mvhd_body, 0);
        append_u32(&mut mvhd_body, 1000);
        append_u32(&mut mvhd_body, duration);
        append_u32(&mut mvhd_body, 0x0001_0000); // rate 1.0
        mvhd_body.extend_from_slice(&[1, 0, 0, 0]); // volume + reserved
        mvhd_body.extend_from_slice(&[0u8; 8]);
        mvhd_body.extend_from_slice(&[0u8; 36]);
        mvhd_body.extend_from_slice(&[0u8; 24]);
        append_u32(&mut mvhd_body, 2); // next track id
        let mvhd = full_box(b"mvhd", 0, &mvhd_body);

        let mut moov_body = Vec::new();
        moov_body.extend_from_slice(&mvhd);
        moov_body.extend_from_slice(&trak);
        let moov = boxed(b"moov", &moov_body);

        let mut file = Vec::new();
        file.extend_from_slice(&ftyp);
        file.extend_from_slice(&mdat);
        file.extend_from_slice(&moov);
        file
    }
}

/// Writes a fixture into the test's own temp directory (never `fixtures/`, which is frozen).
fn write_mov(name: &str, bytes: &[u8]) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("hap");
    fs::create_dir_all(&dir).expect("create the HAP fixture directory");
    let path = dir.join(name);
    fs::write(&path, bytes).expect("write the HAP fixture");
    path
}

fn hap_mov(
    name: &str,
    fourcc: [u8; 4],
    width: u32,
    height: u32,
    fps: u32,
    samples: &[Vec<u8>],
    samples_per_chunk: u32,
) -> PathBuf {
    let bytes = MovBuilder {
        fourcc,
        width,
        height,
        fps,
        samples,
        samples_per_chunk,
    }
    .build();
    write_mov(name, &bytes)
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

const RED: [u8; 3] = [255, 0, 0];
const GREEN: [u8; 3] = [0, 255, 0];
const BLUE: [u8; 3] = [0, 0, 255];
const WHITE: [u8; 3] = [255, 255, 255];

#[test]
fn hap1_file_decodes_by_time_and_loops() {
    let (width, height) = (16u32, 16u32);
    let colours = [RED, GREEN, BLUE, WHITE];
    let payloads: Vec<Vec<u8>> = colours
        .iter()
        .map(|colour| bc1_solid(width, height, *colour))
        .collect();
    // Frame 0 raw, then Snappy: both compressors on the same clip.
    let samples: Vec<Vec<u8>> = payloads
        .iter()
        .enumerate()
        .map(|(index, payload)| hap_sample(BC1, payload, index != 0))
        .collect();
    let path = hap_mov("hap1.mov", *b"Hap1", width, height, 25, &samples, 2);

    let mut source = VideoSource::open(&path).expect("open the HAP fixture");
    assert!(source.is_hap(), "video/Hap1 is a HAP clip");
    let info = source.info();
    assert_eq!((info.w, info.h), (width, height));
    assert!(
        (info.duration_s - 0.160).abs() < 1e-6,
        "4 frames at 25 fps: {info:?}"
    );
    assert!((info.fps - 25.0).abs() < 1e-6, "{info:?}");

    // Each frame is a solid colour, so a wrong time->sample mapping is a wrong colour.
    for (index, expected) in [RED, GREEN, BLUE, WHITE].iter().enumerate() {
        let t = index as f64 * 0.040;
        let frame = source.frame_at(t).expect("a frame inside the clip");
        assert!(
            (frame.pts_s - t).abs() < 1e-9,
            "pts {} for t {t}",
            frame.pts_s
        );
        let hap = frame.hap().expect("a HAP payload, not BGRA");
        assert_eq!(hap.format, Some(HapTextureFormat::Bc1));
        assert!(
            frame.bgra.is_empty(),
            "HAP frames are never decoded to RGBA"
        );
        assert_eq!(frame.hap_texture_size(), Some((16, 16)));
        assert_eq!(
            &hap.data, &payloads[index],
            "frame {index} is {:?}",
            expected
        );
        let block = rgb565_expand(u16::from_le_bytes([hap.data[0], hap.data[1]]));
        assert_eq!(
            block,
            rgb565_expand(rgb565(*expected)),
            "frame {index} colour"
        );
    }

    // Looping: past the duration is the first frame again, with its own timestamp.
    let wrapped = source.frame_at(info.duration_s + 0.005).expect("loop");
    assert_eq!(wrapped.pts_s, 0.0);
    assert_eq!(&wrapped.hap().unwrap().data, &payloads[0]);

    // Backwards seeks land on the covering frame, not on a stale one.
    let late = source.frame_at(0.120).expect("frame 4");
    assert_eq!(late.pts_s, 0.120);
    assert_eq!(&late.hap().unwrap().data, &payloads[3]);
    let early = source.frame_at(0.041).expect("frame 2");
    assert_eq!(early.pts_s, 0.040);
    assert_eq!(&early.hap().unwrap().data, &payloads[1]);

    println!(
        "HAP1 fixture {width}x{height}, {} frames, {} bytes per frame, {:.3} MB source memory",
        samples.len(),
        payloads[0].len(),
        source.memory_bytes() as f64 / (1024.0 * 1024.0)
    );
}

#[test]
fn hap5_file_keeps_straight_alpha() {
    let (width, height) = (16u32, 16u32);
    let clear = [0u8, 0, 0, 0];
    let opaque = [255u8, 0, 0, 255];
    let payload = bc3_half_alpha(width, height, opaque, clear);
    // The alpha values the fixture encodes, so a reader that drops alpha is caught.
    assert_eq!(payload.len(), (4 * 4 * 16) as usize);

    let path = hap_mov(
        "hap5.mov",
        *b"Hap5",
        width,
        height,
        30,
        &[hap_sample(BC3, &payload, true)],
        1,
    );
    let mut source = VideoSource::open(&path).expect("open Hap5");
    let frame = source.frame_at(0.0).expect("frame 0");
    let hap = frame.hap().expect("HAP payload");
    assert_eq!(hap.format, Some(HapTextureFormat::Bc3));
    assert!(hap.has_alpha(), "Hap5 carries alpha");
    assert_eq!(hap.data, payload);
    assert_eq!(hap.alpha, None, "Hap5's alpha is in the colour plane");

    // Left blocks are opaque, right blocks fully transparent: the GPU test draws exactly this.
    assert_eq!(hap.data[0], 255, "left block alpha");
    assert_eq!(hap.data[2 * 16], 0, "right block alpha");
    println!(
        "HAP5 fixture: BC3 {} bytes, left alpha 255, right alpha 0",
        hap.data.len()
    );
}

#[test]
fn hapm_file_carries_a_separate_alpha_plane() {
    let (width, height) = (16u32, 16u32);
    // YCoCg grey: Y = 128 in the alpha channel, Co/Cg = 0 (red/green 128), scale 1 (blue 0).
    let colour = bc3_block([128, 128, 0, 128]).repeat(16);
    let alpha = bc4_half(width, height);
    let sample = hap_sample_two_planes(&colour, &alpha);
    let path = hap_mov("hapm.mov", *b"HapM", width, height, 30, &[sample], 1);

    let mut source = VideoSource::open(&path).expect("open HapM");
    let frame = source.frame_at(0.0).expect("frame 0");
    let hap = frame.hap().expect("HAP payload");
    assert_eq!(hap.format, Some(HapTextureFormat::Bc3YCoCg));
    assert!(hap.has_alpha());
    assert_eq!(hap.data, colour);
    let plane = hap.alpha.as_ref().expect("alpha plane");
    assert_eq!(plane.format, Some(HapTextureFormat::Bc4Alpha));
    assert_eq!(plane.data, alpha);
    println!(
        "HAPM fixture: YCoCg {} bytes + BC4 alpha {} bytes",
        hap.data.len(),
        plane.data.len()
    );
}

#[test]
fn garbage_is_a_clean_error_not_a_panic() {
    fn open_error(path: &Path) -> VideoError {
        match VideoSource::open(path) {
            Err(error) => error,
            Ok(_) => panic!("{} opened as a clip", path.display()),
        }
    }

    // Not a QuickTime file at all: the HAP probe declines and Media Foundation reports the rest.
    let path = write_mov("not-a-movie.mov", &[0x13; 400]);
    assert!(!open_error(&path).to_string().is_empty());

    // A HAP track whose first sample declares a format this build does not know (0x0C).
    let sample = section(NONE | 0x0C, &[0u8; 32]);
    let path = hap_mov("unknown-format.mov", *b"Hap1", 8, 8, 30, &[sample], 1);
    match open_error(&path) {
        VideoError::Unsupported(reason) => {
            assert!(
                reason.contains("unsupported HAP texture format"),
                "{reason}"
            );
        }
        other => panic!("expected Unsupported for an unknown HAP format, got {other:?}"),
    }

    // A Hap5 track whose sample is BC1: the container and the sample disagree.
    let sample = hap_sample(BC1, &bc1_solid(8, 8, RED), false);
    let path = hap_mov("mismatched.mov", *b"Hap5", 8, 8, 30, &[sample], 1);
    match open_error(&path) {
        VideoError::Unsupported(reason) => {
            assert!(
                reason.contains("says") && reason.contains("sample is"),
                "{reason}"
            );
        }
        other => panic!("expected Unsupported for a fourcc mismatch, got {other:?}"),
    }

    // A truncated sample: the section claims more bytes than the file holds.
    let mut sample = section(NONE | BC1, &bc1_solid(8, 8, RED));
    sample.truncate(12);
    let path = hap_mov("truncated-sample.mov", *b"Hap1", 8, 8, 30, &[sample], 1);
    match open_error(&path) {
        VideoError::Unsupported(reason) => {
            assert!(reason.contains("truncated"), "{reason}");
        }
        other => panic!("expected Unsupported for a truncated sample, got {other:?}"),
    }

    // A HAP frame that decompresses to the wrong size for the clip's dimensions.
    let sample = hap_sample(BC1, &bc1_solid(8, 8, RED), false);
    let path = hap_mov("wrong-size.mov", *b"Hap1", 16, 16, 30, &[sample], 1);
    match open_error(&path) {
        VideoError::Unsupported(reason) => {
            assert!(reason.contains("needs"), "{reason}");
        }
        other => panic!("expected Unsupported for a wrong-sized frame, got {other:?}"),
    }

    // Missing file: the existing contract, unchanged by the HAP path.
    assert!(matches!(
        VideoSource::open(Path::new("fixtures/no-such-clip.mov")),
        Err(VideoError::NotFound(_))
    ));
    println!("garbage HAP fixtures: 5 files, 5 clean errors, no panic");
}

/// The measurement the PR quotes: CPU cost of one 1080p HAP frame, and how much of that is Snappy.
#[test]
fn hap_1080p_cpu_cost_and_payload_size() {
    let (width, height) = (1920u32, 1080u32);
    // A pattern with some structure so Snappy has something to do, but not trivially constant.
    let mut payload = Vec::with_capacity((width.div_ceil(4) * height.div_ceil(4) * 8) as usize);
    for block in 0..(width.div_ceil(4) * height.div_ceil(4)) {
        let shade = ((block * 37) % 256) as u8;
        payload.extend_from_slice(&bc1_block([shade, 255 - shade, shade / 2]));
    }
    let compressed = hap_sample(BC1, &payload, true);
    let path = hap_mov(
        "hap1-1080p.mov",
        *b"Hap1",
        width,
        height,
        30,
        &[compressed.clone(), compressed.clone()],
        1,
    );

    let open_start = Instant::now();
    let mut source = VideoSource::open(&path).expect("open the 1080p fixture");
    let open_ms = open_start.elapsed().as_secs_f64() * 1e3;
    let info = source.info();
    assert_eq!((info.w, info.h), (width, height));

    // Parse only (Snappy + section walking) into a reused frame, no file read: the per-frame CPU
    // cost a render thread pays on top of the read.
    let frames = 50;
    let mut reused = hap::HapFrame::default();
    let start = Instant::now();
    for _ in 0..frames {
        hap::parse_frame_into(&compressed, &mut reused).expect("parse");
        assert_eq!(reused.data.len(), payload.len());
    }
    let parse_ms = start.elapsed().as_secs_f64() * 1e3 / frames as f64;

    // And the whole path: alternate between the two frames so every call seeks, reads and
    // parses instead of being served from the frame already in the buffer.
    let start = Instant::now();
    for index in 0..frames {
        let t = if index % 2 == 0 { 0.0 } else { 1.5 / info.fps };
        let frame = source.frame_at(t).expect("a frame");
        assert_eq!(frame.hap().unwrap().data.len(), payload.len());
    }
    let read_parse_ms = start.elapsed().as_secs_f64() * 1e3 / frames as f64;

    assert!(
        parse_ms < 20.0,
        "1080p HAP parse took {parse_ms:.3} ms per frame even in a debug build, which is far \
         past Snappy's cost for 1 MB of blocks"
    );
    println!(
        "HAP 1080p: {} bytes of BC1 ({:.2} MB), Snappy sample {} bytes ({:.1}% of the blocks), \
         open + validate {open_ms:.2} ms, parse {parse_ms:.3} ms/frame, seek + read + parse \
         {read_parse_ms:.3} ms/frame, source memory {:.2} MB, upload per frame {} bytes",
        payload.len(),
        payload.len() as f64 / (1024.0 * 1024.0),
        compressed.len(),
        100.0 * compressed.len() as f64 / payload.len() as f64,
        source.memory_bytes() as f64 / (1024.0 * 1024.0),
        payload.len()
    );
}
