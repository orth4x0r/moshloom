use mosh_core::*;
use std::mem::{align_of, size_of};

#[test]
fn ipc_sizes_are_frozen() {
    assert_eq!(size_of::<TransportRaw>(), 64);
    assert_eq!(size_of::<IpcHeader>(), 128);
    assert_eq!(size_of::<IpcBlock>(), 8320);
    assert_eq!(align_of::<IpcBlock>(), 64);
    assert_eq!(size_of::<IpcMidiEvent>(), 16);
    assert_eq!(BLOCKS_OFFSET % 64, 0);
    assert_eq!(REGION_BYTES, 128 + 256 * 8320 + 1024 * 16);
}

#[test]
fn roles_and_names() {
    assert_eq!(ShowRole::Master as u32, 0);
    assert_eq!(ShowRole::Control as u32, 16);
    assert_eq!(ShowRole::Master.mapping_name(), "Local\\moshloom.master.v1");
    assert_eq!(ShowRole::from_u32(16), Some(ShowRole::Control));
    assert_eq!(ShowRole::from_u32(5), None);
}

#[test]
fn transport_raw_roundtrip() {
    let t = Transport {
        tempo_bpm: 93.0,
        ppq: 6.5,
        bar_start_ppq: 4.0,
        sig_num: 3,
        sig_den: 4,
        playing: true,
        sample_pos: 1234,
    };
    assert_eq!(TransportRaw::from_transport(Some(&t)).to_transport(), Some(t));
    assert_eq!(TransportRaw::from_transport(None).to_transport(), None);
}

#[test]
fn project_default_and_ron_roundtrip() {
    let p = Project::default();
    assert_eq!(p.schema, 1);
    let s = ron::to_string(&p).unwrap();
    let q: Project = ron::from_str(&s).unwrap();
    assert_eq!(p, q);
}
