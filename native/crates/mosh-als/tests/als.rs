//! `mosh-als` tests, from Task 1.9 of docs/superpowers/plans/2026-11-17-native-port.md.
//!
//! Every fixture is generated here from an XML string and gzipped at run time: the repo never
//! holds a real Live set, and a real `.als` is 100 MB+ of uncompressed XML.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use flate2::Compression;
use flate2::write::GzEncoder;

use mosh_als::{AlsError, decode_time_signature, read_als};
use mosh_core::{SigChange, SongMarker};

/// A gzipped temporary file that deletes itself.
struct TempSet {
    path: PathBuf,
}

impl TempSet {
    fn new(gzipped: bool, body: &str) -> TempSet {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "mosh-als-test-{}-{}.als",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let file = File::create(&path).expect("create temp set");
        if gzipped {
            let mut gz = GzEncoder::new(file, Compression::fast());
            gz.write_all(body.as_bytes()).expect("gzip temp set");
            gz.finish().expect("finish gzip");
        } else {
            let mut file = file;
            file.write_all(body.as_bytes()).expect("write temp set");
        }
        TempSet { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempSet {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// A Live 11-shaped set: locators, a master tempo envelope and a master time-signature
/// envelope, with distractors that a naive "first envelope in the file" parser would take.
///
/// The master envelopes come *before* the `<Tempo>` / `<TimeSignature>` elements that carry the
/// target ids, which is the order Live itself writes (checked against Live 11.0.1 sets).
const SET: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Ableton MajorVersion="5" MinorVersion="11.0_433" Creator="Ableton Live 11.0.1d1">
  <LiveSet>
    <Locators>
      <Locators>
        <Locator Id="0">
          <Time Value="128" />
          <Name Value="Outro" />
        </Locator>
        <Locator Id="1">
          <Time Value="0" />
          <Name Value="Intro" />
        </Locator>
        <Locator Id="2">
          <Time Value="64" />
          <Name Value="Song &amp; Chorus" />
        </Locator>
      </Locators>
    </Locators>
    <Tracks>
      <AudioTrack Id="0">
        <Name>
          <EffectiveName Value="Track 1" />
        </Name>
        <AutomationEnvelopes>
          <Envelopes>
            <AutomationEnvelope Id="0">
              <EnvelopeTarget>
                <PointeeId Value="900" />
              </EnvelopeTarget>
              <Automation>
                <Events>
                  <FloatEvent Id="0" Time="-63072000" Value="0.85" />
                  <FloatEvent Id="1" Time="16" Value="0.2" />
                  <EnumEvent Id="2" Time="32" Value="198" />
                </Events>
              </Automation>
            </AutomationEnvelope>
          </Envelopes>
        </AutomationEnvelopes>
      </AudioTrack>
    </Tracks>
    <MasterTrack>
      <Name>
        <EffectiveName Value="Master" />
      </Name>
      <AutomationEnvelopes>
        <Envelopes>
          <AutomationEnvelope Id="0">
            <EnvelopeTarget>
              <PointeeId Value="7001" />
            </EnvelopeTarget>
            <Automation>
              <Events>
                <EnumEvent Id="0" Time="-63072000" Value="201" />
                <EnumEvent Id="1" Time="64" Value="302" />
              </Events>
            </Automation>
          </AutomationEnvelope>
          <AutomationEnvelope Id="1">
            <EnvelopeTarget>
              <PointeeId Value="7002" />
            </EnvelopeTarget>
            <Automation>
              <Events>
                <FloatEvent Id="0" Time="-63072000" Value="124.000252" />
                <FloatEvent Id="1" Time="64" Value="60" />
                <FloatEvent Id="2" Time="128" Value="140.5" />
                <FloatEvent Id="3" Time="128" Value="141" />
              </Events>
            </Automation>
          </AutomationEnvelope>
        </Envelopes>
      </AutomationEnvelopes>
      <SendsListWrapper LomId="0" />
      <Tempo>
        <LomId Value="0" />
        <Manual Value="124.000252" />
        <MidiControllerRange>
          <Min Value="60" />
          <Max Value="200" />
        </MidiControllerRange>
        <AutomationTarget Id="7002">
          <LockEnvelope Value="0" />
        </AutomationTarget>
        <ModulationTarget Id="3">
          <LockEnvelope Value="0" />
        </ModulationTarget>
      </Tempo>
      <TimeSignature>
        <LomId Value="0" />
        <Manual Value="201" />
        <AutomationTarget Id="7001">
          <LockEnvelope Value="0" />
        </AutomationTarget>
      </TimeSignature>
    </MasterTrack>
  </LiveSet>
</Ableton>
"#;

#[test]
fn reads_locators_in_beats() {
    let set = TempSet::new(true, SET);
    let info = read_als(set.path()).expect("read set");

    assert_eq!(
        info.locators,
        vec![
            SongMarker {
                name: "Intro".to_string(),
                start_ppq: 0.0,
            },
            SongMarker {
                name: "Song & Chorus".to_string(),
                start_ppq: 64.0,
            },
            SongMarker {
                name: "Outro".to_string(),
                start_ppq: 128.0,
            },
        ]
    );
}

#[test]
fn reads_time_signature_changes() {
    let set = TempSet::new(true, SET);
    let info = read_als(set.path()).expect("read set");

    // 201 is the enum index Live writes for 4/4 and 302 the one for 6/8 (see
    // `decode_time_signature`); the pre-roll sentinel time becomes ppq 0.
    assert_eq!(
        info.time_signatures,
        vec![
            SigChange {
                ppq: 0.0,
                num: 4,
                den: 4,
            },
            SigChange {
                ppq: 64.0,
                num: 6,
                den: 8,
            },
        ]
    );
}

#[test]
fn reads_tempo_automation_points() {
    let set = TempSet::new(true, SET);
    let info = read_als(set.path()).expect("read set");

    // The sentinel point is the initial tempo before beat 0, reported at 0; the two events at
    // beat 128 collapse to the last one.
    assert_eq!(
        info.tempo_points,
        vec![(0.0, 124.000252), (64.0, 60.0), (128.0, 141.0)]
    );
}

#[test]
fn ignores_automation_outside_the_master_track() {
    let set = TempSet::new(true, SET);
    let info = read_als(set.path()).expect("read set");

    // The track envelope (pointee 900) holds 0.85/0.2 floats and an enum 198; none of those
    // may leak into the master tempo or signature maps.
    assert_eq!(info.tempo_points.len(), 3);
    assert_eq!(info.time_signatures.len(), 2);
    assert!(info.tempo_points.iter().all(|(_, bpm)| *bpm > 59.0));
}

#[test]
fn decodes_time_signature_enum() {
    assert_eq!(decode_time_signature(201), Some((4, 4)));
    assert_eq!(decode_time_signature(302), Some((6, 8)));
    assert_eq!(decode_time_signature(200), Some((3, 4)));
    assert_eq!(decode_time_signature(198), Some((1, 4)));
    assert_eq!(decode_time_signature(202), Some((5, 4)));
    assert_eq!(decode_time_signature(303), Some((7, 8)));
    assert_eq!(decode_time_signature(308), Some((12, 8)));
    assert_eq!(decode_time_signature(100), Some((2, 2)));
    assert_eq!(decode_time_signature(396), Some((1, 16)));
    assert_eq!(decode_time_signature(495), None);
}

#[test]
fn undecodable_time_signature_events_are_dropped() {
    let body = r#"<?xml version="1.0" encoding="UTF-8"?>
<Ableton>
  <LiveSet>
    <MasterTrack>
      <AutomationEnvelopes>
        <Envelopes>
          <AutomationEnvelope Id="0">
            <EnvelopeTarget><PointeeId Value="1" /></EnvelopeTarget>
            <Automation><Events>
              <EnumEvent Id="0" Time="-63072000" Value="201" />
              <EnumEvent Id="1" Time="32" Value="495" />
            </Events></Automation>
          </AutomationEnvelope>
        </Envelopes>
      </AutomationEnvelopes>
      <TimeSignature>
        <Manual Value="201" />
        <AutomationTarget Id="1" />
      </TimeSignature>
    </MasterTrack>
  </LiveSet>
</Ableton>
"#;
    let set = TempSet::new(true, body);
    let info = read_als(set.path()).expect("read set");

    assert_eq!(
        info.time_signatures,
        vec![SigChange {
            ppq: 0.0,
            num: 4,
            den: 4,
        }]
    );
}

#[test]
fn set_without_tempo_automation_reports_the_manual_tempo() {
    let body = r#"<?xml version="1.0" encoding="UTF-8"?>
<Ableton>
  <LiveSet>
    <Locators>
      <Locators />
    </Locators>
    <MasterTrack>
      <Tempo>
        <Manual Value="98.5" />
        <AutomationTarget Id="42" />
      </Tempo>
      <TimeSignature>
        <Manual Value="302" />
        <AutomationTarget Id="43" />
      </TimeSignature>
    </MasterTrack>
  </LiveSet>
</Ableton>
"#;
    let set = TempSet::new(true, body);
    let info = read_als(set.path()).expect("read set");

    assert!(info.locators.is_empty());
    assert_eq!(info.tempo_points, vec![(0.0, 98.5)]);
    assert_eq!(
        info.time_signatures,
        vec![SigChange {
            ppq: 0.0,
            num: 6,
            den: 8,
        }]
    );
}

#[test]
fn not_gzip_returns_error() {
    let set = TempSet::new(false, "<Ableton><LiveSet /></Ableton>");
    let err = read_als(set.path()).expect_err("plain XML is not a Live set file");

    assert!(matches!(err, AlsError::NotGzip), "unexpected error: {err}");
}

#[test]
fn missing_master_track_returns_error() {
    let set = TempSet::new(
        true,
        r#"<?xml version="1.0"?><Something><Else Value="1" /></Something>"#,
    );
    let err = read_als(set.path()).expect_err("not a Live set");

    assert!(
        matches!(err, AlsError::NotALiveSet),
        "unexpected error: {err}"
    );
}
