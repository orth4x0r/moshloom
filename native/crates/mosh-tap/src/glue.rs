//! The only place that touches nih-plug's host types: they are converted here to the plain
//! structs the cores take, so the cores are testable without a VST3 host and without `mosh-ipc`.

use nih_plug::prelude::{NoteEvent, Transport as HostCtxTransport};

use crate::control::HostEvent;
use crate::host::HostTransport;

/// Ableton's transport for this block, or `None` when the host reports no position (offline
/// rendering without transport, for instance). The app then keeps the last tempo and free-runs.
pub fn host_transport(transport: &HostCtxTransport) -> Option<HostTransport> {
    // Without a beat position there is nothing to align visuals to.
    let ppq = transport.pos_beats()?;

    Some(HostTransport {
        tempo_bpm: transport.tempo.unwrap_or(0.0),
        ppq,
        bar_start_ppq: transport.bar_start_pos_beats().unwrap_or(ppq),
        sig_num: signature_value(transport.time_sig_numerator),
        sig_den: signature_value(transport.time_sig_denominator),
        playing: transport.playing,
        sample_pos: transport.pos_samples().unwrap_or(0),
    })
}

/// A missing or nonsensical time signature falls back to 4/4; bars must never divide by zero.
fn signature_value(value: Option<i32>) -> u16 {
    match value {
        Some(value) if (1..=u16::MAX as i32).contains(&value) => value as u16,
        _ => 4,
    }
}

/// The MIDI event for a host note event, or `None` for the events the app does not use
/// (polyphonic modulation, aftertouch, SysEx, ...).
pub fn host_event(event: &NoteEvent<()>) -> Option<HostEvent> {
    match *event {
        NoteEvent::NoteOn {
            timing,
            channel,
            note,
            velocity,
            ..
        } => Some(HostEvent::NoteOn {
            timing,
            channel,
            note,
            velocity,
        }),
        NoteEvent::NoteOff {
            timing,
            channel,
            note,
            ..
        } => Some(HostEvent::NoteOff {
            timing,
            channel,
            note,
        }),
        // A choke stops the note immediately; to the app it is the same message.
        NoteEvent::Choke {
            timing,
            channel,
            note,
            ..
        } => Some(HostEvent::NoteOff {
            timing,
            channel,
            note,
        }),
        NoteEvent::MidiCC {
            timing,
            channel,
            cc,
            value,
        } => Some(HostEvent::ControlChange {
            timing,
            channel,
            cc,
            value,
        }),
        NoteEvent::MidiProgramChange {
            timing,
            channel,
            program,
        } => Some(HostEvent::ProgramChange {
            timing,
            channel,
            program,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_time_signature_falls_back_to_four_four() {
        assert_eq!(signature_value(None), 4);
        assert_eq!(signature_value(Some(0)), 4);
        assert_eq!(signature_value(Some(-3)), 4);
        assert_eq!(signature_value(Some(3)), 3);
        assert_eq!(signature_value(Some(8)), 8);
    }

    #[test]
    fn unused_host_events_are_dropped() {
        let pressure = NoteEvent::<()>::PolyPressure {
            timing: 0,
            voice_id: None,
            channel: 0,
            note: 60,
            pressure: 0.5,
        };
        assert_eq!(host_event(&pressure), None);
    }

    #[test]
    fn note_events_become_control_events() {
        let on = NoteEvent::<()>::NoteOn {
            timing: 12,
            voice_id: None,
            channel: 3,
            note: 36,
            velocity: 0.5,
        };
        assert_eq!(
            host_event(&on),
            Some(HostEvent::NoteOn {
                timing: 12,
                channel: 3,
                note: 36,
                velocity: 0.5
            })
        );

        let cc = NoteEvent::<()>::MidiCC {
            timing: 4,
            channel: 0,
            cc: 48,
            value: 0.25,
        };
        assert_eq!(
            host_event(&cc),
            Some(HostEvent::ControlChange {
                timing: 4,
                channel: 0,
                cc: 48,
                value: 0.25
            })
        );
    }
}
