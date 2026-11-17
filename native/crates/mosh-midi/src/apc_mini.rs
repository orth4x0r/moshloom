//! The Akai APC mini maps: pads, scene and track buttons, faders, pad LEDs.
//!
//! Both models are decoded here. The mk1 sends pads as notes 0..=63 with row 0 at the
//! bottom of the grid; the mk2 sends pads 0..=63 with row 0 at the top left. Everything
//! in this module is pure so the maps can be tested without a device.

use mosh_core::{ControlEvent, MidiEvent};

/// Notes 0..=63 are the 8x8 pad grid on both models.
pub const PAD_NOTES: std::ops::RangeInclusive<u8> = 0..=63;
/// mk1 scene buttons, bottom to top.
pub const SCENE_FIRST_NOTE: u8 = 82;
pub const SCENE_LAST_NOTE: u8 = 89;
/// mk1 track buttons, left to right.
pub const TRACK_FIRST_NOTE: u8 = 64;
pub const TRACK_LAST_NOTE: u8 = 71;
/// mk1 track faders, left to right.
pub const FADER_FIRST_CC: u8 = 48;
pub const FADER_LAST_CC: u8 = 55;
/// mk1 master fader.
pub const MASTER_CC: u8 = 56;

/// Note-on status before the channel nibble. The APC mini pads are note-on messages.
const NOTE_ON: u8 = 0x90;
const CONTROL_CHANGE: u8 = 0xB0;
const PROGRAM_CHANGE: u8 = 0xC0;

/// Which APC mini is attached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApcModel {
    /// APC mini (mk1): pads 0..=63 with row 0 at the bottom, four LED colours.
    Mk1,
    /// APC mini mk2: pads 0..=63 with row 0 at the top left, RGB LED velocities.
    Mk2,
}

impl ApcModel {
    /// Reads the model out of a MIDI port name, or `None` when the port is not an APC mini.
    pub fn from_port_name(name: &str) -> Option<ApcModel> {
        let lower = name.to_ascii_lowercase();
        if !lower.contains("apc mini") {
            return None;
        }
        if lower.contains("mk2") { Some(ApcModel::Mk2) } else { Some(ApcModel::Mk1) }
    }

    /// Note of the pad at grid position `(x, y)`, with `y` counted from the top row.
    ///
    /// `set_pad` and the decoders share this, so a lit pad is the pad that launches
    /// `LaunchClip { layer: x, scene: y }`.
    pub fn pad_note(self, x: u8, y: u8) -> Option<u8> {
        if x > 7 || y > 7 {
            return None;
        }
        Some(match self {
            // mk1 row 0 is the bottom row, so the top row (y = 0) is row 7.
            ApcModel::Mk1 => (7 - y) * 8 + x,
            // mk2 pads count from the top left.
            ApcModel::Mk2 => y * 8 + x,
        })
    }

    /// Grid position of a pad note, `y` counted from the top row.
    pub fn pad_position(self, note: u8) -> Option<(u8, u8)> {
        if note > 63 {
            return None;
        }
        let col = note % 8;
        let row = note / 8;
        let y = match self {
            ApcModel::Mk1 => 7 - row,
            ApcModel::Mk2 => row,
        };
        Some((col, y))
    }
}

/// Pad LED colour. The blink variants pulse; the APC mini still lights the same hue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadColor {
    Off,
    Green,
    Red,
    Yellow,
    GreenBlink,
    RedBlink,
    YellowBlink,
}

impl PadColor {
    /// mk1 LED velocity: 0 off, 1 green, 3 red, 5 yellow, and +1 blinks.
    pub fn mk1_velocity(self) -> u8 {
        match self {
            PadColor::Off => 0,
            PadColor::Green => 1,
            PadColor::Red => 3,
            PadColor::Yellow => 5,
            PadColor::GreenBlink => 2,
            PadColor::RedBlink => 4,
            PadColor::YellowBlink => 6,
        }
    }

    /// mk2 LED velocity: a palette index (off 0, green 1, red 2, yellow 3) with a blink bit.
    ///
    /// The mk2's palette table is not in any source we have, so this low-index convention is
    /// ours until it can be checked on hardware; the mk1 table above is the documented one.
    pub fn mk2_velocity(self) -> u8 {
        const BLINK: u8 = 0x40;
        match self {
            PadColor::Off => 0,
            PadColor::Green => 1,
            PadColor::Red => 2,
            PadColor::Yellow => 3,
            PadColor::GreenBlink => 1 | BLINK,
            PadColor::RedBlink => 2 | BLINK,
            PadColor::YellowBlink => 3 | BLINK,
        }
    }

    /// Whether the LED should pulse.
    pub fn is_blink(self) -> bool {
        matches!(self, PadColor::GreenBlink | PadColor::RedBlink | PadColor::YellowBlink)
    }
}

/// The note-on bytes that light the pad at `(x, y)`, `y` counted from the top row.
///
/// `None` when the position is off the 8x8 grid.
pub fn pad_led_bytes(model: ApcModel, x: u8, y: u8, c: PadColor) -> Option<[u8; 3]> {
    let note = model.pad_note(x, y)?;
    let velocity = match model {
        ApcModel::Mk1 => c.mk1_velocity(),
        ApcModel::Mk2 => c.mk2_velocity(),
    };
    Some([NOTE_ON, note, velocity])
}

/// Decodes one message from an APC mini mk1.
pub fn decode_apc_mini(msg: &[u8]) -> Option<ControlEvent> {
    decode_apc_mini_model(ApcModel::Mk1, msg)
}

/// Decodes one message from the given APC mini model.
///
/// The channel nibble is ignored: a controller moved to another MIDI channel keeps working.
/// Note-offs, note-on with velocity 0, MIDI clock and everything else outside the map decode
/// to `None`.
pub fn decode_apc_mini_model(model: ApcModel, msg: &[u8]) -> Option<ControlEvent> {
    let (&status, rest) = msg.split_first()?;
    let data1 = *rest.first()?;
    let data2 = *rest.get(1)?;
    if data1 > 127 || data2 > 127 {
        return None;
    }
    match status & 0xF0 {
        NOTE_ON if data2 > 0 => match data1 {
            0..=63 => {
                let (layer, scene) = model.pad_position(data1)?;
                Some(ControlEvent::LaunchClip {
                    layer: layer as usize,
                    scene: scene as usize,
                })
            }
            SCENE_FIRST_NOTE..=SCENE_LAST_NOTE => {
                Some(ControlEvent::LaunchScene((data1 - SCENE_FIRST_NOTE) as usize))
            }
            TRACK_FIRST_NOTE..=TRACK_LAST_NOTE => {
                Some(ControlEvent::StopLayer((data1 - TRACK_FIRST_NOTE) as usize))
            }
            _ => None,
        },
        CONTROL_CHANGE => decode_apc_fader(data1, data2),
        _ => None,
    }
}

/// mk1 faders: CC 48..=55 are layer opacity, CC 56 is the master intensity bias.
fn decode_apc_fader(cc: u8, value: u8) -> Option<ControlEvent> {
    match cc {
        FADER_FIRST_CC..=FADER_LAST_CC => Some(ControlEvent::SetOpacity {
            layer: (cc - FADER_FIRST_CC) as usize,
            value: normalise(value),
        }),
        // A fader is bipolar for intensity: bottom -1, centre 0, top +1.
        MASTER_CC => Some(ControlEvent::IntensityBias(bipolar(value))),
        _ => None,
    }
}

/// Decodes one note, CC or program change from the Moshloom Control plugin.
///
/// The plugin's layout is the same grid but with scenes counted from the top, so note
/// `n` (0..=63) launches layer `n % 8` of scene `n / 8`. Notes 64..=71 launch scenes,
/// CC 1..=8 are layer 0's parameters and a program change selects a song.
pub fn decode_control_note(e: &MidiEvent) -> Option<ControlEvent> {
    if e.data1 > 127 || e.data2 > 127 {
        return None;
    }
    match e.status & 0xF0 {
        NOTE_ON if e.data2 > 0 => match e.data1 {
            0..=63 => Some(ControlEvent::LaunchClip {
                layer: (e.data1 % 8) as usize,
                scene: (e.data1 / 8) as usize,
            }),
            TRACK_FIRST_NOTE..=TRACK_LAST_NOTE => {
                Some(ControlEvent::LaunchScene((e.data1 - TRACK_FIRST_NOTE) as usize))
            }
            _ => None,
        },
        CONTROL_CHANGE => match e.data1 {
            1..=8 => Some(ControlEvent::SetParam {
                layer: 0,
                index: (e.data1 - 1) as usize,
                value: normalise(e.data2),
            }),
            _ => None,
        },
        PROGRAM_CHANGE => Some(ControlEvent::SelectSong(e.data1 as usize)),
        _ => None,
    }
}

/// 0..=127 as 0..=1.
pub fn normalise(value: u8) -> f32 {
    value.min(127) as f32 / 127.0
}

/// 0..=127 as -1..=1.
pub fn bipolar(value: u8) -> f32 {
    normalise(value) * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_position_round_trips() {
        for model in [ApcModel::Mk1, ApcModel::Mk2] {
            for y in 0..8u8 {
                for x in 0..8u8 {
                    let note = model.pad_note(x, y).unwrap();
                    assert_eq!(model.pad_position(note), Some((x, y)), "{model:?}");
                }
            }
            assert_eq!(model.pad_note(8, 0), None);
            assert_eq!(model.pad_position(64), None);
        }
    }

    #[test]
    fn blink_colours_report_blink() {
        assert!(!PadColor::Green.is_blink());
        assert!(PadColor::GreenBlink.is_blink());
        assert!(PadColor::YellowBlink.is_blink());
        assert!(!PadColor::Off.is_blink());
    }
}
