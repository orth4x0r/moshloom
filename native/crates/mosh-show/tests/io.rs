//! Project files: a RON round trip, a schema check and a missing file.

mod common;

use common::*;
use mosh_core::{
    ArrClip, EffectKind, EffectSlot, FollowAction, LaunchQuantize, ShowMode, SigChange,
};

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(dir).unwrap();
    dir.join(name)
}

fn full_project() -> mosh_core::Project {
    let mut a = following(1, FollowAction::RandomProcedural);
    a.react_to_events = true;
    a.effects = vec![EffectSlot { kind: EffectKind::Mosh, amount: 0.75, param: 0.5 }];
    a.params.palette = 7;
    a.transition_in = mosh_core::Transition { kind: mosh_core::TransitionKind::Melt, beats: 2.0 };

    let mut p = project(
        ShowMode::Autopilot,
        vec![a, clip(2)],
        grid(&[&[Some(1), Some(2)], &[Some(2), None]], LaunchQuantize::Phrase),
    );
    p.presets.push(preset("look", "fluid", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 42));
    p.arrangement.songs = vec![song("one", 0.0), song("two", 64.0)];
    p.arrangement.sig_map = vec![SigChange { ppq: 0.0, num: 4, den: 4 }, SigChange { ppq: 32.0, num: 3, den: 4 }];
    p.arrangement.lanes = vec![
        vec![ArrClip { clip: 1, start_ppq: 0.0, end_ppq: 32.0 }, ArrClip { clip: 2, start_ppq: 32.0, end_ppq: 64.0 }],
        vec![ArrClip { clip: 2, start_ppq: 8.0, end_ppq: 16.0 }],
    ];
    p
}

#[test]
fn project_roundtrip_ron() {
    let p = full_project();
    let path = tmp("roundtrip.ron");
    mosh_show::io::save(&p, &path).expect("save");
    let q = mosh_show::io::load(&path).expect("load");
    assert_eq!(p, q, "the project changed on the way through RON");
}

#[test]
fn wrong_schema_is_rejected() {
    let mut p = full_project();
    p.schema = 2;
    let path = tmp("schema2.ron");
    mosh_show::io::save(&p, &path).expect("save");
    match mosh_show::io::load(&path) {
        Err(mosh_show::ShowError::Schema { found, expected }) => {
            assert_eq!((found, expected), (2, mosh_core::PROJECT_SCHEMA));
        }
        other => panic!("expected a schema error, got {other:?}"),
    }
}

#[test]
fn missing_file_is_an_error() {
    let path = tmp("does-not-exist.ron");
    let _ = std::fs::remove_file(&path);
    match mosh_show::io::load(&path) {
        Err(mosh_show::ShowError::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound),
        other => panic!("expected an io error, got {other:?}"),
    }
}

#[test]
fn garbage_is_a_parse_error() {
    let path = tmp("garbage.ron");
    std::fs::write(&path, "this is not ron").unwrap();
    match mosh_show::io::load(&path) {
        Err(mosh_show::ShowError::Ron(_)) => {}
        other => panic!("expected a parse error, got {other:?}"),
    }
}
