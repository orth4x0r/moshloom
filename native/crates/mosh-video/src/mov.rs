//! A minimal QuickTime/MP4 demuxer, enough to play a HAP track.
//!
//! HAP is not a container of its own: it lives in a QuickTime `.mov` (or an MP4) as a video track
//! whose sample description names one of the HAP fourccs, and each sample is one HAP frame. The
//! only boxes a player needs are `moov/trak/*/mdia/*`:
//!
//! * `mdhd` — the media timescale (ticks per second),
//! * `hdlr` — `vide` for the video track,
//! * `stsd` — the sample description: the codec fourcc and the display size,
//! * `stts` — time-to-sample: how many samples each duration covers,
//! * `stsc` — sample-to-chunk: how many samples each chunk holds,
//! * `stsz` — every sample's compressed size,
//! * `stco`/`co64` — every chunk's file offset.
//!
//! `stts` and `stsz` are turned into a per-sample table at open ([`MovFile::samples`]), so a lookup
//! by time is a binary search and a frame read is one `seek` plus one `read`. HAP samples are key
//! frames, so there is no index or decode order to care about. Edit lists are ignored: a HAP clip
//! starts at its first sample.
//!
//! Anything that is not a QuickTime/MP4 file, or has no HAP track, returns `Ok(None)` and the
//! caller falls back to Media Foundation. A file whose track *is* HAP but whose tables do not add
//! up is an `Unsupported` error with the reason, so a broken overlay is reported instead of
//! silently showing nothing for a whole set.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::VideoError;

/// A cap on a `moov` box this reader will hold in memory. Real ones are a few KB per minute of
/// video; 64 MB is far past anything a stage clip produces and stops a hostile header.
const MAX_MOOV_BYTES: u64 = 64 << 20;

/// One sample of a track: where it is in the file and when it is shown.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MovSample {
    pub(crate) offset: u64,
    pub(crate) size: u32,
    pub(crate) pts_s: f64,
}

/// One open file's HAP track.
pub(crate) struct MovFile {
    file: File,
    pub(crate) samples: Vec<MovSample>,
    /// Display size from `stsd`: the picture, not the padded texture.
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) codec: [u8; 4],
    pub(crate) duration_s: f64,
    pub(crate) fps: f64,
}

impl MovFile {
    /// Opens `path` if it is a QuickTime/MP4 file with a HAP video track.
    ///
    /// `Ok(None)` is "not HAP" (another codec, or not a QuickTime file at all); `Err` is a HAP
    /// track whose tables cannot be used.
    pub(crate) fn open_hap(path: &Path) -> Result<Option<MovFile>, VideoError> {
        if !path.is_file() {
            return Err(VideoError::NotFound(path.to_path_buf()));
        }
        let mut file = File::open(path).map_err(|_| VideoError::NotFound(path.to_path_buf()))?;
        let len = file
            .metadata()
            .map_err(|error| VideoError::Unsupported(format!("{}: {error}", path.display())))?
            .len();
        let Some(moov) = read_moov(&mut file, len) else {
            return Ok(None);
        };
        let Some(track) = hap_track(&moov)? else {
            return Ok(None);
        };

        let samples = build_samples(&track, path)?;
        if samples.is_empty() {
            return Err(VideoError::Unsupported(format!(
                "{}: the HAP track has no samples",
                path.display()
            )));
        }
        let duration_s =
            samples.last().map(|sample| sample.pts_s).unwrap_or(0.0) + track.sample_duration_s();
        let fps = if track.sample_duration_s() > 0.0 {
            1.0 / track.sample_duration_s()
        } else {
            30.0
        };
        Ok(Some(MovFile {
            file,
            samples,
            width: track.width,
            height: track.height,
            codec: track.codec,
            duration_s,
            fps,
        }))
    }

    /// Index of the last sample shown at or before `t_s` (0 when `t_s` is before the clip).
    pub(crate) fn sample_at(&self, t_s: f64) -> usize {
        if !t_s.is_finite() || t_s <= 0.0 {
            return 0;
        }
        // partition_point: the first sample whose pts is after `t_s`.
        let after = self
            .samples
            .partition_point(|sample| sample.pts_s <= t_s + 1e-9);
        after.saturating_sub(1)
    }

    /// Reads sample `index` into `buffer` and returns its presentation time.
    pub(crate) fn read_sample(
        &mut self,
        index: usize,
        buffer: &mut Vec<u8>,
    ) -> Result<f64, VideoError> {
        let Some(sample) = self.samples.get(index).copied() else {
            return Err(VideoError::Decode(format!("no sample {index}")));
        };
        buffer.clear();
        buffer.resize(sample.size as usize, 0);
        self.file
            .seek(SeekFrom::Start(sample.offset))
            .and_then(|_| self.file.read_exact(buffer))
            .map_err(|error| {
                VideoError::Decode(format!(
                    "sample {index} at {} ({} bytes): {error}",
                    sample.offset, sample.size
                ))
            })?;
        Ok(sample.pts_s)
    }

    /// CPU memory the demuxer holds: the sample table and the file handle.
    pub(crate) fn memory_bytes(&self) -> usize {
        self.samples.capacity() * std::mem::size_of::<MovSample>()
    }
}

/// The tables one track contributes, all borrowed from the `moov` buffer.
struct Track {
    codec: [u8; 4],
    width: u32,
    height: u32,
    timescale: u32,
    /// `(sample count, duration in timescale ticks)` per entry.
    stts: Vec<(u32, u32)>,
    sizes: Vec<u32>,
    /// `(first chunk number, samples in that chunk)` per entry, in file order.
    stsc: Vec<(u32, u32)>,
    chunks: Vec<u64>,
}

impl Track {
    /// The nominal sample duration in seconds, from the first `stts` entry.
    fn sample_duration_s(&self) -> f64 {
        let delta = self.stts.first().map(|entry| entry.1).unwrap_or(0);
        if delta == 0 || self.timescale == 0 {
            return 0.0;
        }
        f64::from(delta) / f64::from(self.timescale)
    }
}

/// Finds the first video track whose sample description is one of the HAP fourccs.
fn hap_track(moov: &[u8]) -> Result<Option<Track>, VideoError> {
    for (kind, trak) in Boxes::new(moov) {
        if &kind != b"trak" {
            continue;
        }
        let Some(mdia) = child(trak, b"mdia") else {
            continue;
        };
        // A HAP track is a video track. A file with an audio track that happens to have a HAP
        // fourcc is broken, not a clip.
        if let Some(hdlr) = child(mdia, b"hdlr") {
            let mut cursor = Cursor::new(hdlr);
            let _ = cursor.skip(8);
            let handler = cursor.bytes(4).unwrap_or(&[0; 4]);
            if handler != b"vide" {
                continue;
            }
        }
        let Some(minf) = child(mdia, b"minf") else {
            continue;
        };
        let Some(stbl) = child(minf, b"stbl") else {
            continue;
        };
        let Some(stsd) = child(stbl, b"stsd") else {
            continue;
        };
        let Some((codec, width, height)) = sample_description(stsd) else {
            continue;
        };
        if !crate::hap::is_hap_fourcc(&codec) {
            continue;
        }

        // From here on the file says it is HAP: a missing table is an error with a reason.
        let malformed = |what: &str| {
            VideoError::Unsupported(format!(
                "{} track is missing or has a bad {what}",
                String::from_utf8_lossy(&codec)
            ))
        };
        let timescale = child(mdia, b"mdhd")
            .and_then(media_timescale)
            .filter(|timescale| *timescale > 0)
            .ok_or_else(|| malformed("mdhd timescale"))?;
        let stts = child(stbl, b"stts")
            .and_then(time_to_sample)
            .filter(|entries| !entries.is_empty())
            .ok_or_else(|| malformed("stts"))?;
        let sizes = child(stbl, b"stsz")
            .and_then(sample_sizes)
            .filter(|sizes| !sizes.is_empty())
            .ok_or_else(|| malformed("stsz"))?;
        let stsc = child(stbl, b"stsc")
            .and_then(sample_to_chunk)
            .filter(|entries| !entries.is_empty())
            .ok_or_else(|| malformed("stsc"))?;
        let chunks = child(stbl, b"stco")
            .and_then(chunk_offsets32)
            .or_else(|| child(stbl, b"co64").and_then(chunk_offsets64))
            .filter(|offsets| !offsets.is_empty())
            .ok_or_else(|| malformed("stco/co64"))?;

        return Ok(Some(Track {
            codec,
            width,
            height,
            timescale,
            stts,
            sizes,
            stsc,
            chunks,
        }));
    }
    Ok(None)
}

/// Expands `stts` + `stsz` + `stsc` + chunk offsets into one row per sample.
fn build_samples(track: &Track, path: &Path) -> Result<Vec<MovSample>, VideoError> {
    let count = track.sizes.len();
    let mut times = Vec::with_capacity(count);
    let mut tick = 0u64;
    for &(samples, delta) in &track.stts {
        for _ in 0..samples {
            if times.len() == count {
                break;
            }
            times.push(tick as f64 / f64::from(track.timescale));
            tick = tick.saturating_add(u64::from(delta));
        }
    }
    while times.len() < count {
        times.push(tick as f64 / f64::from(track.timescale));
    }

    let mut samples = Vec::with_capacity(count);
    let mut next = 0usize;
    for (index, &chunk_offset) in track.chunks.iter().enumerate() {
        let chunk = index as u32 + 1;
        let per_chunk = track
            .stsc
            .iter()
            .rev()
            .find(|entry| entry.0 <= chunk)
            .map(|entry| entry.1)
            .unwrap_or(1) as usize;
        let mut offset = chunk_offset;
        for _ in 0..per_chunk {
            if next == count {
                break;
            }
            let size = track.sizes[next];
            samples.push(MovSample {
                offset,
                size,
                pts_s: times[next],
            });
            offset = offset.saturating_add(u64::from(size));
            next += 1;
        }
        if next == count {
            break;
        }
    }
    if samples.len() != count {
        return Err(VideoError::Unsupported(format!(
            "{}: stsc/stco place {} of {} HAP samples",
            path.display(),
            samples.len(),
            count
        )));
    }
    Ok(samples)
}

/// Reads the `moov` box's body, or `None` when the file is not a QuickTime/MP4 file.
fn read_moov(file: &mut File, len: u64) -> Option<Vec<u8>> {
    let mut at = 0u64;
    while at + 8 <= len {
        let mut head = [0u8; 16];
        file.seek(SeekFrom::Start(at)).ok()?;
        file.read_exact(&mut head[..8]).ok()?;
        let size32 = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
        let kind = [head[4], head[5], head[6], head[7]];
        let (size, header) = match size32 {
            0 => (len - at, 8u64),
            1 => {
                file.read_exact(&mut head[8..16]).ok()?;
                (
                    u64::from_be_bytes([
                        head[8], head[9], head[10], head[11], head[12], head[13], head[14],
                        head[15],
                    ]),
                    16u64,
                )
            }
            size => (u64::from(size), 8u64),
        };
        if size < header || at.saturating_add(size) > len {
            // Not an ISO base media file (or a truncated one): let Media Foundation try.
            return None;
        }
        if &kind == b"moov" {
            let body_len = size - header;
            if body_len > MAX_MOOV_BYTES {
                return None;
            }
            let mut body = vec![0u8; body_len as usize];
            file.seek(SeekFrom::Start(at + header)).ok()?;
            file.read_exact(&mut body).ok()?;
            return Some(body);
        }
        at += size;
    }
    None
}

/// A cursor over a box body, big-endian like every QuickTime field.
struct Cursor<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Cursor { data, at: 0 }
    }

    fn bytes(&mut self, count: usize) -> Option<&'a [u8]> {
        let end = self.at.checked_add(count)?;
        let slice = self.data.get(self.at..end)?;
        self.at = end;
        Some(slice)
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.bytes(1)?[0])
    }

    fn u16(&mut self) -> Option<u16> {
        let bytes = self.bytes(2)?;
        Some(u16::from_be_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Option<u32> {
        let bytes = self.bytes(4)?;
        Some(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> Option<u64> {
        let bytes = self.bytes(8)?;
        Some(u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn skip(&mut self, count: usize) -> Option<()> {
        self.bytes(count).map(|_| ())
    }
}

/// Walks the boxes of a buffer: `(type, body)`.
struct Boxes<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Boxes<'a> {
    fn new(data: &'a [u8]) -> Self {
        Boxes { data, at: 0 }
    }
}

impl<'a> Iterator for Boxes<'a> {
    type Item = ([u8; 4], &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.at + 8 > self.data.len() {
            return None;
        }
        let size32 = u32::from_be_bytes([
            self.data[self.at],
            self.data[self.at + 1],
            self.data[self.at + 2],
            self.data[self.at + 3],
        ]);
        let kind = [
            self.data[self.at + 4],
            self.data[self.at + 5],
            self.data[self.at + 6],
            self.data[self.at + 7],
        ];
        let (size, header) = match size32 {
            0 => ((self.data.len() - self.at) as u64, 8usize),
            1 => {
                if self.at + 16 > self.data.len() {
                    return None;
                }
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&self.data[self.at + 8..self.at + 16]);
                (u64::from_be_bytes(bytes), 16)
            }
            size => (u64::from(size), 8),
        };
        if size < header as u64 || size > (self.data.len() - self.at) as u64 {
            return None;
        }
        let body_start = self.at + header;
        let body_end = self.at + size as usize;
        self.at = body_end;
        Some((kind, &self.data[body_start..body_end]))
    }
}

/// The body of the first `kind` box inside `data`.
fn child<'a>(data: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
    Boxes::new(data)
        .find(|(box_kind, _)| box_kind == kind)
        .map(|(_, body)| body)
}

/// The codec fourcc and display size from an `stsd` box.
fn sample_description(stsd: &[u8]) -> Option<([u8; 4], u32, u32)> {
    let mut cursor = Cursor::new(stsd);
    cursor.skip(4)?; // version + flags
    let entries = cursor.u32()?;
    if entries == 0 {
        return None;
    }
    cursor.u32()?; // entry size
    let mut codec = [0u8; 4];
    codec.copy_from_slice(cursor.bytes(4)?);
    cursor.skip(6 + 2)?; // reserved + data reference index
    cursor.skip(2 + 2 + 12)?; // version, revision, vendor, temporal quality, spatial quality
    let width = u32::from(cursor.u16()?);
    let height = u32::from(cursor.u16()?);
    Some((codec, width, height))
}

/// The media timescale from an `mdhd` box (version 0 and 1).
fn media_timescale(mdhd: &[u8]) -> Option<u32> {
    let mut cursor = Cursor::new(mdhd);
    let version = cursor.u8()?;
    cursor.skip(3)?; // flags
    match version {
        0 => {
            cursor.skip(4 + 4)?; // creation + modification
            cursor.u32() // timescale
        }
        _ => {
            cursor.skip(8 + 8)?; // creation + modification (64-bit)
            cursor.u32()
        }
    }
}

/// `stts` entries: `(count, delta)`.
fn time_to_sample(stts: &[u8]) -> Option<Vec<(u32, u32)>> {
    let mut cursor = Cursor::new(stts);
    cursor.skip(4)?;
    let entries = cursor.u32()?;
    let mut out = Vec::new();
    for _ in 0..entries {
        let count = cursor.u32()?;
        let delta = cursor.u32()?;
        out.push((count, delta));
    }
    Some(out)
}

/// `stsz`: one size per sample (the constant-size form is expanded).
fn sample_sizes(stsz: &[u8]) -> Option<Vec<u32>> {
    let mut cursor = Cursor::new(stsz);
    cursor.skip(4)?;
    let constant = cursor.u32()?;
    let count = cursor.u32()?;
    if constant != 0 {
        return Some(vec![constant; count as usize]);
    }
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        out.push(cursor.u32()?);
    }
    Some(out)
}

/// `stsc`: `(first chunk, samples per chunk)`.
fn sample_to_chunk(stsc: &[u8]) -> Option<Vec<(u32, u32)>> {
    let mut cursor = Cursor::new(stsc);
    cursor.skip(4)?;
    let entries = cursor.u32()?;
    let mut out = Vec::new();
    for _ in 0..entries {
        let first_chunk = cursor.u32()?;
        let samples = cursor.u32()?;
        cursor.u32()?; // sample description index
        out.push((first_chunk, samples));
    }
    Some(out)
}

fn chunk_offsets32(stco: &[u8]) -> Option<Vec<u64>> {
    let mut cursor = Cursor::new(stco);
    cursor.skip(4)?;
    let entries = cursor.u32()?;
    let mut out = Vec::with_capacity(entries as usize);
    for _ in 0..entries {
        out.push(u64::from(cursor.u32()?));
    }
    Some(out)
}

fn chunk_offsets64(co64: &[u8]) -> Option<Vec<u64>> {
    let mut cursor = Cursor::new(co64);
    cursor.skip(4)?;
    let entries = cursor.u32()?;
    let mut out = Vec::with_capacity(entries as usize);
    for _ in 0..entries {
        out.push(cursor.u64()?);
    }
    Some(out)
}
