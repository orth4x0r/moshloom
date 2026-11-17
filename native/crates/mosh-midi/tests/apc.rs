//! Decoding tests for the APC mini and the Moshloom Control plugin path.
//!
//! Everything here is pure: no MIDI port is opened, no pad is lit.

use mosh_core::{ControlEvent, MidiEvent};
use mosh_midi::{
    ApcModel, MidiLearn, PadColor, decode_apc_mini, decode_apc_mini_model, decode_control_note,
    pad_led_bytes,
};

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}

fn note(d1: u8, d2: u8) -> MidiEvent {
    MidiEvent { ppq: 0.0, status: 0x90, data1: d1, data2: d2 }
}

fn cc(d1: u8, d2: u8) -> MidiEvent {
    MidiEvent { ppq: 0.0, status: 0xB0, data1: d1, data2: d2 }
}

#[test]
fn pad_note_maps_to_launch() {
    // mk1: note 0 is the bottom-left pad, and the bottom row is the highest scene.
    assert_eq!(
        decode_apc_mini(&[0x90, 0, 127]),
        Some(ControlEvent::LaunchClip { layer: 0, scene: 7 })
    );
    assert_eq!(
        decode_apc_mini(&[0x90, 63, 100]),
        Some(ControlEvent::LaunchClip { layer: 7, scene: 0 })
    );
    assert_eq!(
        decode_apc_mini(&[0x90, 9, 1]),
        Some(ControlEvent::LaunchClip { layer: 1, scene: 6 })
    );
    // Any channel works: the mk1 may be set to another channel.
    assert_eq!(
        decode_apc_mini(&[0x9F, 8, 100]),
        Some(ControlEvent::LaunchClip { layer: 0, scene: 6 })
    );
    // A release is not a launch, and neither is a note-on with velocity 0.
    assert_eq!(decode_apc_mini(&[0x80, 0, 0]), None);
    assert_eq!(decode_apc_mini(&[0x90, 0, 0]), None);
}

#[test]
fn control_note_maps_to_launch() {
    assert_eq!(
        decode_control_note(&note(0, 100)),
        Some(ControlEvent::LaunchClip { layer: 0, scene: 0 })
    );
    assert_eq!(
        decode_control_note(&note(9, 100)),
        Some(ControlEvent::LaunchClip { layer: 1, scene: 1 })
    );
    assert_eq!(
        decode_control_note(&note(63, 1)),
        Some(ControlEvent::LaunchClip { layer: 7, scene: 7 })
    );
    // The plugin may send on any channel.
    assert_eq!(
        decode_control_note(&MidiEvent { status: 0x95, ..note(0, 100) }),
        Some(ControlEvent::LaunchClip { layer: 0, scene: 0 })
    );
    assert_eq!(decode_control_note(&MidiEvent { status: 0x80, ..note(0, 100) }), None);
    assert_eq!(decode_control_note(&note(0, 0)), None);
}

#[test]
fn scene_button_maps_to_scene() {
    // mk1 scene buttons are notes 82..=89, bottom to top.
    for n in 82..=89u8 {
        assert_eq!(
            decode_apc_mini(&[0x90, n, 127]),
            Some(ControlEvent::LaunchScene((n - 82) as usize)),
            "note {n}"
        );
    }
    // The Control plugin's scene notes are 64..=71 instead.
    for n in 64..=71u8 {
        assert_eq!(
            decode_control_note(&note(n, 100)),
            Some(ControlEvent::LaunchScene((n - 64) as usize)),
            "note {n}"
        );
    }
    // mk1 track buttons 64..=71 stop a layer.
    for n in 64..=71u8 {
        assert_eq!(
            decode_apc_mini(&[0x90, n, 127]),
            Some(ControlEvent::StopLayer((n - 64) as usize)),
            "note {n}"
        );
    }
}

#[test]
fn fader_maps_to_opacity() {
    assert_eq!(
        decode_apc_mini(&[0xB0, 48, 127]),
        Some(ControlEvent::SetOpacity { layer: 0, value: 1.0 })
    );
    match decode_apc_mini(&[0xB0, 55, 64]) {
        Some(ControlEvent::SetOpacity { layer, value }) => {
            assert_eq!(layer, 7);
            assert!(close(value, 64.0 / 127.0), "value {value}");
        }
        other => panic!("expected SetOpacity, got {other:?}"),
    }
    // CC 56 is the master fader: a bipolar intensity bias, -1..=1.
    assert_eq!(decode_apc_mini(&[0xB0, 56, 0]), Some(ControlEvent::IntensityBias(-1.0)));
    assert_eq!(decode_apc_mini(&[0xB0, 56, 127]), Some(ControlEvent::IntensityBias(1.0)));
    match decode_apc_mini(&[0xB0, 56, 64]) {
        Some(ControlEvent::IntensityBias(v)) => {
            assert!(close(v, 64.0 / 127.0 * 2.0 - 1.0), "bias {v}");
        }
        other => panic!("expected IntensityBias, got {other:?}"),
    }
    // Control plugin CC 1..=8 are parameters of layer 0.
    match decode_control_note(&cc(3, 64)) {
        Some(ControlEvent::SetParam { layer, index, value }) => {
            assert_eq!((layer, index), (0, 2));
            assert!(close(value, 64.0 / 127.0), "value {value}");
        }
        other => panic!("expected SetParam, got {other:?}"),
    }
    // Program change selects a song.
    let pc = MidiEvent { ppq: 0.0, status: 0xC0, data1: 3, data2: 0 };
    assert_eq!(decode_control_note(&pc), Some(ControlEvent::SelectSong(3)));
}

#[test]
fn led_message_bytes() {
    // y counts from the top, so y = 7 is the bottom row of the mk1 grid (note 0).
    assert_eq!(pad_led_bytes(ApcModel::Mk1, 0, 7, PadColor::Green), Some([0x90, 0, 1]));
    assert_eq!(pad_led_bytes(ApcModel::Mk1, 0, 7, PadColor::Red), Some([0x90, 0, 3]));
    assert_eq!(pad_led_bytes(ApcModel::Mk1, 0, 7, PadColor::Yellow), Some([0x90, 0, 5]));
    assert_eq!(pad_led_bytes(ApcModel::Mk1, 0, 7, PadColor::Off), Some([0x90, 0, 0]));
    // Blink is the same colour plus one.
    assert_eq!(pad_led_bytes(ApcModel::Mk1, 0, 7, PadColor::GreenBlink), Some([0x90, 0, 2]));
    assert_eq!(pad_led_bytes(ApcModel::Mk1, 0, 7, PadColor::RedBlink), Some([0x90, 0, 4]));
    assert_eq!(pad_led_bytes(ApcModel::Mk1, 0, 7, PadColor::YellowBlink), Some([0x90, 0, 6]));
    // The top row of the mk1 grid is notes 56..=63.
    assert_eq!(pad_led_bytes(ApcModel::Mk1, 7, 0, PadColor::Red), Some([0x90, 63, 3]));
    assert_eq!(pad_led_bytes(ApcModel::Mk1, 3, 3, PadColor::Green), Some([0x90, 35, 1]));
    // Out-of-grid coordinates light nothing.
    assert_eq!(pad_led_bytes(ApcModel::Mk1, 8, 0, PadColor::Green), None);
    assert_eq!(pad_led_bytes(ApcModel::Mk1, 0, 8, PadColor::Green), None);
}

#[test]
fn mk2_pad_origin_is_top_left() {
    assert_eq!(
        decode_apc_mini_model(ApcModel::Mk2, &[0x90, 0, 100]),
        Some(ControlEvent::LaunchClip { layer: 0, scene: 0 })
    );
    assert_eq!(
        decode_apc_mini_model(ApcModel::Mk2, &[0x90, 56, 100]),
        Some(ControlEvent::LaunchClip { layer: 0, scene: 7 })
    );
    assert_eq!(
        decode_apc_mini_model(ApcModel::Mk2, &[0x90, 7, 100]),
        Some(ControlEvent::LaunchClip { layer: 7, scene: 0 })
    );
    // The same physical pad is note 0 on the mk1 and note 56 on the mk2.
    assert_eq!(
        decode_apc_mini(&[0x90, 0, 100]),
        Some(ControlEvent::LaunchClip { layer: 0, scene: 7 })
    );
    // LEDs follow each model's own note convention, top-left origin for the mk2.
    assert_eq!(pad_led_bytes(ApcModel::Mk2, 0, 0, PadColor::Green), Some([0x90, 0, 1]));
    assert_eq!(pad_led_bytes(ApcModel::Mk2, 0, 7, PadColor::Green), Some([0x90, 56, 1]));
    assert_eq!(pad_led_bytes(ApcModel::Mk2, 7, 0, PadColor::Red), Some([0x90, 7, 2]));
}

#[test]
fn led_and_decode_agree_on_pad() {
    // set_pad(x, y) lights the pad that launches layer x of scene y, on both models.
    for model in [ApcModel::Mk1, ApcModel::Mk2] {
        for y in 0..8u8 {
            for x in 0..8u8 {
                let msg = pad_led_bytes(model, x, y, PadColor::Green).expect("in-grid pad");
                assert_eq!(
                    decode_apc_mini_model(model, &msg),
                    Some(ControlEvent::LaunchClip { layer: x as usize, scene: y as usize }),
                    "{model:?} ({x}, {y})"
                );
            }
        }
    }
}

#[test]
fn mk2_port_name_detected() {
    assert_eq!(ApcModel::from_port_name("APC MINI"), Some(ApcModel::Mk1));
    assert_eq!(ApcModel::from_port_name("APC MINI 0"), Some(ApcModel::Mk1));
    assert_eq!(ApcModel::from_port_name("APC mini mk2 MIDI 1"), Some(ApcModel::Mk2));
    assert_eq!(ApcModel::from_port_name("APC mini mk2 Control"), Some(ApcModel::Mk2));
    assert_eq!(ApcModel::from_port_name("Launchpad Mini MK3"), None);
    assert_eq!(ApcModel::from_port_name(""), None);
}

#[test]
fn malformed_messages_are_ignored() {
    let ignored: [&[u8]; 9] = [
        &[],
        &[0x90],
        &[0x90, 5],
        &[0x90, 200, 100], // note above 127
        &[0xB0, 48],       // fader without a value
        &[0xB0, 48, 200],  // value above 127
        &[0xF8],           // clock
        &[0xF0, 0x7D, 0x00],
        &[0xE0, 0x00, 0x40], // pitch bend
    ];
    for msg in ignored {
        assert_eq!(decode_apc_mini(msg), None, "{msg:?}");
    }
}

#[test]
fn midi_learn_round_trips_through_ron() {
    let mut learn = MidiLearn::default();
    learn.learn(0xB0, 20, ControlEvent::SetOpacity { layer: 2, value: 0.0 });
    learn.learn(0x90, 40, ControlEvent::LaunchClip { layer: 1, scene: 3 });
    learn.learn(0x90, 41, ControlEvent::Lock(true));

    let text = ron::to_string(&learn).expect("serialise MidiLearn");
    let back: MidiLearn = ron::from_str(&text).expect("deserialise MidiLearn");
    assert_eq!(back, learn);

    assert_eq!(
        back.map(0x90, 40, 100),
        Some(ControlEvent::LaunchClip { layer: 1, scene: 3 })
    );
    assert_eq!(
        back.map(0xB0, 20, 127),
        Some(ControlEvent::SetOpacity { layer: 2, value: 1.0 })
    );
    assert_eq!(back.map(0x90, 41, 127), Some(ControlEvent::Lock(true)));
    assert_eq!(back.map(0x90, 42, 127), None);
    assert_eq!(back.bindings().len(), 3);
}

#[test]
fn midi_learn_scales_continuous_templates() {
    let mut learn = MidiLearn::default();
    learn.learn(0xB0, 21, ControlEvent::SetParam { layer: 0, index: 4, value: 0.0 });
    learn.learn(0xB0, 22, ControlEvent::IntensityBias(0.0));

    assert!(matches!(
        learn.map(0xB0, 21, 64),
        Some(ControlEvent::SetParam { layer: 0, index: 4, value }) if close(value, 64.0 / 127.0)
    ));
    assert!(matches!(
        learn.map(0xB0, 22, 0),
        Some(ControlEvent::IntensityBias(v)) if close(v, -1.0)
    ));
    assert!(matches!(
        learn.map(0xB0, 22, 127),
        Some(ControlEvent::IntensityBias(v)) if close(v, 1.0)
    ));

    // One binding per (status, data1): learning the same control replaces the template.
    learn.learn(0xB0, 21, ControlEvent::SetOpacity { layer: 5, value: 0.0 });
    assert_eq!(learn.bindings().len(), 2);
    assert_eq!(learn.map(0xB0, 21, 0), Some(ControlEvent::SetOpacity { layer: 5, value: 0.0 }));
    assert!(learn.forget(0xB0, 21));
    assert_eq!(learn.map(0xB0, 21, 0), None);
    assert!(!learn.forget(0xB0, 21));
    assert_eq!(learn.bindings().len(), 1);
}

/// Lists the machine's MIDI ports. Run with:
/// `cargo test -p mosh-midi -- --ignored --nocapture`
#[test]
#[ignore = "needs a real MIDI device; lists the machine's ports, lights nothing"]
fn list_midi_ports() {
    match mosh_midi::port_names() {
        Ok(names) => {
            println!("midi input ports: {}", names.len());
            for name in &names {
                println!("  {name} (apc: {:?})", ApcModel::from_port_name(name));
            }
        }
        Err(e) => println!("midi init failed: {e}"),
    }
}
