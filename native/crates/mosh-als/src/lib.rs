//! `mosh-als`: import an Ableton Live set (`.als`, gzipped XML) — Task 1.9 of
//! `docs/superpowers/plans/2026-11-17-native-port.md`.
//!
//! The set is read as a stream (`flate2` + `quick-xml`): a real arrangement is tens of MB of
//! XML, so nothing is ever loaded whole. What comes out is the part the show needs:
//!
//! * arrangement locators, as [`SongMarker`] in ppq (beats, f64) — the song markers of the
//!   human's one-long-arrangement set,
//! * time-signature changes, as [`SigChange`], decoded from Live's enum index,
//! * master tempo automation, as `(ppq, bpm)` points.
//!
//! The element paths were checked against Live 11.0.1 sets on the reference machine; the parser
//! matches by element name inside the enclosing context instead of by an exact path, because the
//! same data moves between Live versions:
//!
//! ```text
//! LiveSet/Locators/Locators/Locator            { Time (beats), Name }
//! LiveSet/MasterTrack/Tempo                     { Manual, AutomationTarget@Id }
//! LiveSet/MasterTrack/TimeSignature             { Manual, AutomationTarget@Id }
//! LiveSet/MasterTrack/AutomationEnvelopes/
//!     Envelopes/AutomationEnvelope              EnvelopeTarget/PointeeId == one of the target ids
//!         Automation/Events/FloatEvent          { Time (beats), Value }   -> tempo
//!         Automation/Events/EnumEvent           { Time (beats), Value }   -> time signature
//! ```
//!
//! Documented assumptions, all of them deliberate:
//!
//! * **Initial values.** Live writes the value in force before the arrangement starts at a large
//!   negative `Time` (observed: `-63072000`). Such a point means "the value at beat 0" and is
//!   reported at ppq `0.0`; a real event at `0.0` wins over it.
//! * **Order.** The envelopes are written *before* the `Tempo` / `TimeSignature` elements that
//!   carry the target ids, so envelopes are collected first and matched to the ids at the end.
//! * **No automation.** A master parameter with a `Manual` value and no envelope points yields a
//!   single point at ppq `0.0`, so `tempo_points` / `time_signatures` are never empty for a set
//!   that has a master track.
//! * **Duplicates.** Two points at the same ppq resolve to the last one written.
//! * **Junk.** Points with a non-finite value are dropped, as are time-signature enum values that
//!   [`decode_time_signature`] does not know: a guessed meter is worse than a missing one.

use std::cmp::Ordering;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use flate2::read::GzDecoder;
use mosh_core::{SigChange, SongMarker};
use quick_xml::Reader;
use quick_xml::XmlVersion;
use quick_xml::events::{BytesStart, Event};

/// Times at or below this are Live's "before the arrangement starts" sentinel, not beats.
const INITIAL_TIME_SENTINEL: f64 = -1.0e6;

/// Everything the show imports from a Live set.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AlsInfo {
    /// Arrangement locators, sorted by ppq. `start_ppq` is in beats, never bars.
    pub locators: Vec<SongMarker>,
    /// Time-signature changes, sorted by ppq. Always at least one entry for a real set.
    pub time_signatures: Vec<SigChange>,
    /// Master tempo automation as `(ppq, bpm)`, sorted by ppq.
    pub tempo_points: Vec<(f64, f64)>,
}

/// Why a set could not be read.
#[derive(Debug)]
pub enum AlsError {
    /// The file could not be opened or read.
    Io(std::io::Error),
    /// The file is not gzipped, so it is not an `.als`.
    NotGzip,
    /// The XML parsed but holds no `MasterTrack`, so it is not a Live set.
    NotALiveSet,
    /// The gzipped payload is not well-formed XML.
    Xml(quick_xml::Error),
}

impl fmt::Display for AlsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlsError::Io(e) => write!(f, "cannot read the set: {e}"),
            AlsError::NotGzip => {
                write!(
                    f,
                    "not a gzipped Live set (an .als file starts with the gzip magic bytes)"
                )
            }
            AlsError::NotALiveSet => {
                write!(
                    f,
                    "the XML has no <MasterTrack>, so it is not an Ableton Live set"
                )
            }
            AlsError::Xml(e) => write!(f, "malformed .als XML: {e}"),
        }
    }
}

impl std::error::Error for AlsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AlsError::Io(e) => Some(e),
            AlsError::Xml(e) => Some(e),
            AlsError::NotGzip | AlsError::NotALiveSet => None,
        }
    }
}

/// Read an `.als` file into the show's data.
pub fn read_als<P: AsRef<Path>>(path: P) -> Result<AlsInfo, AlsError> {
    let mut file = File::open(path.as_ref()).map_err(AlsError::Io)?;

    let mut magic = [0_u8; 2];
    if file.read_exact(&mut magic).is_err() || magic != *b"\x1f\x8b" {
        return Err(AlsError::NotGzip);
    }
    file.seek(SeekFrom::Start(0)).map_err(AlsError::Io)?;

    let gz = GzDecoder::new(BufReader::with_capacity(64 * 1024, file));
    let mut xml = Reader::from_reader(BufReader::with_capacity(64 * 1024, gz));
    xml.config_mut().trim_text(true);

    let mut parser = SetParser::default();
    let mut buf = Vec::new();
    loop {
        match xml.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                parser.on_open(&e)?;
                parser.on_attrs(&e)?;
            }
            Ok(Event::Empty(e)) => parser.on_attrs(&e)?,
            Ok(Event::End(e)) => parser.on_close(e.name().as_ref()),
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(e) => return Err(AlsError::Xml(e)),
        }
        buf.clear();
    }

    parser.finish()
}

/// Decode Live's `TimeSignature` enum index into `(numerator, denominator)`.
///
/// Live writes the master signature as an enum index, not as two numbers. The index is
/// `numerator * 50 + denominator_code`, where the denominator code counts up in powers of two
/// from `/2`: `0 -> /2`, `1 -> /4`, `2 -> /8`, `3 -> /16`.
///
/// This reproduces the two values observed in Live 11.0.1 sets on the reference machine:
/// `201 -> 4/4` and `302 -> 6/8` (the only set with a non-4/4 master held 6/8 clips). The other
/// entries of the table above are the same arithmetic and are covered by unit tests, but they
/// have not been seen in a real set. Code `4` is also read as `/16`: that is what the
/// "denominator / 4" reading of the same numbers gives, `/16` is the largest denominator Live's
/// UI offers, and both readings agree on the two confirmed values.
///
/// An index this decoder does not recognise returns `None`; the caller drops the point rather
/// than guessing a meter.
pub fn decode_time_signature(value: u32) -> Option<(u16, u16)> {
    let num = value / 50;
    let code = value % 50;
    let den: u16 = match code {
        0 => 2,
        1 => 4,
        2 => 8,
        3 | 4 => 16,
        _ => return None,
    };
    if num == 0 || num > u32::from(u16::MAX) {
        return None;
    }
    Some((num as u16, den))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MasterField {
    Tempo,
    TimeSignature,
}

#[derive(Default)]
struct LocatorDraft {
    time: Option<f64>,
    name: Option<String>,
}

struct EnvelopeDraft {
    pointee: Option<u64>,
    points: Vec<Point>,
}

struct Point {
    time: f64,
    value: f64,
}

#[derive(Default)]
struct SetParser {
    in_locators: usize,
    locator: Option<LocatorDraft>,
    locators: Vec<SongMarker>,

    saw_master: bool,
    in_master: bool,
    field: Option<MasterField>,
    tempo_target: Option<u64>,
    sig_target: Option<u64>,
    tempo_manual: Option<f64>,
    sig_manual: Option<u64>,

    envelope: Option<EnvelopeDraft>,
    envelopes: Vec<EnvelopeDraft>,
}

impl SetParser {
    /// Structural elements: the ones that open and close a region we care about.
    fn on_open(&mut self, e: &BytesStart<'_>) -> Result<(), AlsError> {
        let name = e.name();
        match name.as_ref() {
            "Locators" => self.in_locators += 1,
            "Locator" if self.in_locators > 0 => self.locator = Some(LocatorDraft::default()),
            "MasterTrack" => {
                self.saw_master = true;
                self.in_master = true;
                self.field = None;
                self.envelope = None;
                self.tempo_target = None;
                self.sig_target = None;
                self.tempo_manual = None;
                self.sig_manual = None;
            }
            "Tempo" if self.in_master => self.field = Some(MasterField::Tempo),
            "TimeSignature" if self.in_master => self.field = Some(MasterField::TimeSignature),
            "AutomationEnvelope" if self.in_master => {
                self.envelope = Some(EnvelopeDraft {
                    pointee: None,
                    points: Vec::new(),
                });
            }
            _ => {}
        }
        Ok(())
    }

    /// Attribute-carrying tags: `Start` and `Empty` both land here, since Live writes most of
    /// them self-closed.
    fn on_attrs(&mut self, e: &BytesStart<'_>) -> Result<(), AlsError> {
        let name = e.name();
        match name.as_ref() {
            "Time" => {
                if let Some(locator) = self.locator.as_mut()
                    && let Some(time) = attr_f64(e, "Value")?
                {
                    locator.time = Some(time);
                }
            }
            "Name" => {
                if let Some(locator) = self.locator.as_mut()
                    && let Some(name) = attr_str(e, "Value")?
                {
                    locator.name = Some(name);
                }
            }
            "Manual" => {
                if self.in_master {
                    match self.field {
                        Some(MasterField::Tempo) => {
                            if let Some(v) = attr_f64(e, "Value")? {
                                self.tempo_manual = Some(v);
                            }
                        }
                        Some(MasterField::TimeSignature) => {
                            if let Some(v) = attr_f64(e, "Value")? {
                                self.sig_manual = Some(v.round() as u64);
                            }
                        }
                        None => {}
                    }
                }
            }
            "AutomationTarget" => {
                if self.in_master
                    && let Some(id) = attr_u64(e, "Id")?
                {
                    match self.field {
                        Some(MasterField::Tempo) => self.tempo_target = Some(id),
                        Some(MasterField::TimeSignature) => self.sig_target = Some(id),
                        None => {}
                    }
                }
            }
            "PointeeId" => {
                if let Some(envelope) = self.envelope.as_mut()
                    && let Some(id) = attr_u64(e, "Value")?
                {
                    envelope.pointee = Some(id);
                }
            }
            "FloatEvent" | "EnumEvent" => {
                if let Some(envelope) = self.envelope.as_mut()
                    && let (Some(time), Some(value)) = (attr_f64(e, "Time")?, attr_f64(e, "Value")?)
                {
                    envelope.points.push(Point { time, value });
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn on_close(&mut self, name: &str) {
        match name {
            "Locators" => self.in_locators = self.in_locators.saturating_sub(1),
            "Locator" => {
                if let Some(draft) = self.locator.take() {
                    let name = draft.name.unwrap_or_default();
                    // A locator without a `Time` carries no position to speak of.
                    if let Some(time) = draft.time {
                        self.locators.push(SongMarker {
                            name,
                            start_ppq: time,
                        });
                    }
                }
            }
            "Tempo" if self.field == Some(MasterField::Tempo) => self.field = None,
            "TimeSignature" if self.field == Some(MasterField::TimeSignature) => self.field = None,
            "AutomationEnvelope" => {
                if let Some(envelope) = self.envelope.take() {
                    self.envelopes.push(envelope);
                }
            }
            "MasterTrack" => {
                self.in_master = false;
                self.field = None;
                self.envelope = None;
            }
            _ => {}
        }
    }

    fn finish(mut self) -> Result<AlsInfo, AlsError> {
        if !self.saw_master {
            return Err(AlsError::NotALiveSet);
        }

        let tempo_target = self.tempo_target;
        let sig_target = self.sig_target;

        let mut tempo_points: Vec<(f64, f64)> = Vec::new();
        let mut sig_points: Vec<(f64, f64)> = Vec::new();
        for envelope in &self.envelopes {
            if envelope.pointee.is_none() {
                continue;
            }
            if envelope.pointee == tempo_target {
                tempo_points.extend(
                    envelope
                        .points
                        .iter()
                        .filter(|p| p.value.is_finite())
                        .map(|p| (arrangement_time(p.time), p.value)),
                );
            } else if envelope.pointee == sig_target {
                sig_points.extend(
                    envelope
                        .points
                        .iter()
                        .filter(|p| p.value.is_finite())
                        .map(|p| (arrangement_time(p.time), p.value)),
                );
            }
        }

        if tempo_points.is_empty()
            && let Some(bpm) = self.tempo_manual
        {
            tempo_points.push((0.0, bpm));
        }
        sort_and_dedupe(&mut tempo_points, |p| p.0);

        let mut time_signatures: Vec<SigChange> = sig_points
            .into_iter()
            .filter_map(|(ppq, value)| {
                let index = value.round();
                if !(0.0..=f64::from(u32::MAX)).contains(&index) {
                    return None;
                }
                let (num, den) = decode_time_signature(index as u32)?;
                Some(SigChange { ppq, num, den })
            })
            .collect();
        if time_signatures.is_empty()
            && let Some(manual) = self.sig_manual
            && manual <= u64::from(u32::MAX)
            && let Some((num, den)) = decode_time_signature(manual as u32)
        {
            time_signatures.push(SigChange { ppq: 0.0, num, den });
        }
        sort_and_dedupe(&mut time_signatures, |s| s.ppq);

        sort_and_dedupe(&mut self.locators, |l| l.start_ppq);

        Ok(AlsInfo {
            locators: self.locators,
            time_signatures,
            tempo_points,
        })
    }
}

/// Live's "before the arrangement" sentinel becomes beat 0; every real position passes through.
fn arrangement_time(time: f64) -> f64 {
    if time <= INITIAL_TIME_SENTINEL {
        0.0
    } else {
        time
    }
}

fn sort_and_dedupe<T>(items: &mut Vec<T>, time: impl Fn(&T) -> f64) {
    items.sort_by(|a, b| time(a).partial_cmp(&time(b)).unwrap_or(Ordering::Equal));

    let mut deduped: Vec<T> = Vec::with_capacity(items.len());
    for item in items.drain(..) {
        let same_time = matches!(deduped.last(), Some(last) if time(last) == time(&item));
        if same_time {
            let last = deduped.len() - 1;
            deduped[last] = item;
        } else {
            deduped.push(item);
        }
    }
    *items = deduped;
}

fn attr_str(e: &BytesStart<'_>, key: &str) -> Result<Option<String>, AlsError> {
    match e.try_get_attribute(key) {
        Ok(Some(attr)) => Ok(Some(
            attr.normalized_value(XmlVersion::Implicit1_0)
                .map_err(AlsError::Xml)?
                .into_owned(),
        )),
        Ok(None) => Ok(None),
        Err(err) => Err(AlsError::Xml(quick_xml::Error::InvalidAttr(err))),
    }
}

fn attr_f64(e: &BytesStart<'_>, key: &str) -> Result<Option<f64>, AlsError> {
    Ok(attr_str(e, key)?.and_then(|v| v.trim().parse::<f64>().ok()))
}

fn attr_u64(e: &BytesStart<'_>, key: &str) -> Result<Option<u64>, AlsError> {
    Ok(attr_str(e, key)?.and_then(|v| v.trim().parse::<u64>().ok()))
}
