//! mosh-midi: the Akai APC mini and MIDI learn.
//!
//! See docs/superpowers/plans/2026-11-17-native-port.md, Task 1.10.
//!
//! The decoding is pure and lives in [`apc_mini`] and [`learn`]; the `midir` part here is a
//! thin port layer. A controller is only used while building and rehearsing a show, never
//! during a set, so nothing here is on the audio path.
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use mosh_midi::{Controller, PadColor};
//!
//! let mut ctl = Controller::open_apc_mini()?;
//! for event in ctl.poll() {
//!     println!("{event:?}");
//! }
//! ctl.set_pad(0, 0, PadColor::Green);
//! # Ok(())
//! # }
//! ```

pub mod apc_mini;
pub mod learn;

pub use apc_mini::{
    ApcModel, PadColor, bipolar, decode_apc_mini, decode_apc_mini_model, decode_control_note,
    normalise, pad_led_bytes,
};
pub use learn::{LearnBinding, MidiLearn};

use std::fmt;
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::time::Duration;

use midir::{Ignore, MidiInput, MidiInputConnection, MidiInputPort, MidiOutput, MidiOutputConnection};
use mosh_core::ControlEvent;

/// Name the app registers with the MIDI backend.
pub const CLIENT_NAME: &str = "moshloom";

/// Messages buffered between the MIDI backend thread and `poll`.
///
/// The backend callback never blocks; when the app has not polled for a while the oldest
/// messages are already gone. A controller sends a handful of messages per gesture, so a full
/// queue means the app is not running, not that the queue is small.
const QUEUE: usize = 256;

/// What can go wrong while opening a controller.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MidiError {
    /// No port name contained "APC MINI".
    NoApcMini,
    /// The MIDI backend itself could not be opened.
    Init(String),
    /// The APC mini port exists but could not be connected.
    Connect(String),
}

impl fmt::Display for MidiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MidiError::NoApcMini => f.write_str("no APC mini MIDI port found"),
            MidiError::Init(e) => write!(f, "MIDI backend unavailable: {e}"),
            MidiError::Connect(e) => write!(f, "APC mini port could not be opened: {e}"),
        }
    }
}

impl std::error::Error for MidiError {}

/// An open APC mini, its input queue and its learned controls.
pub struct Controller {
    model: ApcModel,
    learn: MidiLearn,
    /// Held for its lifetime: dropping the connection closes the port.
    _input: Option<MidiInputConnection<SyncSender<Vec<u8>>>>,
    output: Option<MidiOutputConnection>,
    rx: Receiver<Vec<u8>>,
}

impl Controller {
    /// Opens the first APC mini input and, when it exists, the matching output for pad LEDs.
    ///
    /// The output is optional: without it the pads cannot be lit, but launches still arrive.
    /// A machine with no APC mini gets [`MidiError::NoApcMini`].
    pub fn open_apc_mini() -> Result<Controller, MidiError> {
        let mut midi_in = MidiInput::new(CLIENT_NAME).map_err(|e| MidiError::Init(e.to_string()))?;
        // Only notes, CCs and program changes matter; skip sysex, clock and active sensing.
        midi_in.ignore(Ignore::All);

        let ports = midi_in.ports();
        let (port, model, port_name) = find_apc_port(&midi_in, &ports).ok_or(MidiError::NoApcMini)?;

        let (tx, rx) = sync_channel(QUEUE);
        let input = midi_in
            .connect(
                &port,
                CLIENT_NAME,
                move |_timestamp, message, tx: &mut SyncSender<Vec<u8>>| {
                    // Never block the MIDI backend thread.
                    if let Err(TrySendError::Full(_)) = tx.try_send(message.to_vec()) {
                        // Dropped: the app has not polled in a long time.
                    }
                },
                tx,
            )
            .map_err(|e| MidiError::Connect(e.kind().to_string()))?;

        let output = open_output(&port_name, model);

        Ok(Controller {
            model,
            learn: MidiLearn::default(),
            _input: Some(input),
            output,
            rx,
        })
    }

    /// Which APC model was found.
    pub fn model(&self) -> ApcModel {
        self.model
    }

    /// Every control event received since the last call.
    pub fn poll(&mut self) -> Vec<ControlEvent> {
        let mut events = Vec::new();
        while let Ok(message) = self.rx.try_recv() {
            if let Some(event) = self.decode(&message) {
                events.push(event);
            }
        }
        events
    }

    /// Lights the pad at `(x, y)`, with `y` counted from the top row.
    ///
    /// A learned binding fires the pad at the same position as `LaunchClip { layer: x, scene: y }`.
    /// Without an output port, or for a position off the grid, this does nothing.
    pub fn set_pad(&mut self, x: u8, y: u8, c: PadColor) {
        let Some(bytes) = pad_led_bytes(self.model, x, y, c) else {
            return;
        };
        if let Some(output) = self.output.as_mut() {
            // An unplugged controller is not an error: the LED simply does not change.
            let _ = output.send(&bytes);
        }
    }

    /// The learned controls.
    pub fn midi_learn(&self) -> &MidiLearn {
        &self.learn
    }

    /// Replaces the learned controls, e.g. with the ones loaded from a project.
    pub fn set_midi_learn(&mut self, learn: MidiLearn) {
        self.learn = learn;
    }

    /// The learned controls, for the editor that binds them.
    pub fn midi_learn_mut(&mut self) -> &mut MidiLearn {
        &mut self.learn
    }

    /// A learned binding wins over the built-in APC map.
    fn decode(&self, message: &[u8]) -> Option<ControlEvent> {
        if let (Some(&status), Some(&data1)) = (message.first(), message.get(1)) {
            let value = message.get(2).copied().unwrap_or(0);
            if let Some(event) = self.learn.map(status, data1, value) {
                return Some(event);
            }
        }
        decode_apc_mini_model(self.model, message)
    }
}

/// The machine's MIDI input port names, for a controller picker.
pub fn port_names() -> Result<Vec<String>, MidiError> {
    let midi_in = MidiInput::new(CLIENT_NAME).map_err(|e| MidiError::Init(e.to_string()))?;
    Ok(midi_in
        .ports()
        .iter()
        .filter_map(|p| midi_in.port_name(p).ok())
        .collect())
}

/// First port whose name names an APC mini.
fn find_apc_port(midi_in: &MidiInput, ports: &[MidiInputPort]) -> Option<(MidiInputPort, ApcModel, String)> {
    for port in ports {
        let Ok(name) = midi_in.port_name(port) else {
            continue;
        };
        if let Some(model) = ApcModel::from_port_name(&name) {
            return Some((port.clone(), model, name));
        }
    }
    None
}

/// The output port that belongs to the input port, if the device exposes one.
fn open_output(port_name: &str, model: ApcModel) -> Option<MidiOutputConnection> {
    let midi_out = MidiOutput::new(CLIENT_NAME).ok()?;
    let ports = midi_out.ports();
    let same_name = ports.iter().find(|p| {
        midi_out.port_name(p).is_ok_and(|name| name == port_name)
    });
    let port = same_name.or_else(|| {
        ports.iter().find(|p| {
            midi_out
                .port_name(p)
                .ok()
                .and_then(|name| ApcModel::from_port_name(&name))
                == Some(model)
        })
    })?;
    // Opening the output only enables LED feedback.
    midi_out.connect(port, CLIENT_NAME).ok()
}

/// How long the app should wait before polling an idle controller again.
///
/// The APC mini sends nothing while untouched, so the app's frame loop can poll at this
/// interval without missing a gesture.
pub const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(4);

#[cfg(test)]
mod tests {
    use super::*;

    fn headless() -> Controller {
        Controller {
            model: ApcModel::Mk1,
            learn: MidiLearn::default(),
            _input: None,
            output: None,
            rx: sync_channel(QUEUE).1,
        }
    }

    #[test]
    fn poll_without_ports_is_empty_and_set_pad_is_harmless() {
        let mut controller = headless();
        assert_eq!(controller.poll(), Vec::new());
        // No output port: lighting a pad is a no-op, not a panic.
        controller.set_pad(0, 0, PadColor::Green);
        controller.set_pad(9, 9, PadColor::Red);
    }

    #[test]
    fn learn_overrides_the_builtin_map() {
        let mut controller = headless();
        controller.midi_learn_mut().learn(0x90, 0, ControlEvent::LaunchScene(3));
        assert_eq!(controller.decode(&[0x90, 0, 100]), Some(ControlEvent::LaunchScene(3)));
        // Everything not learned still uses the APC map.
        assert_eq!(
            controller.decode(&[0x90, 1, 100]),
            Some(ControlEvent::LaunchClip { layer: 1, scene: 7 })
        );
    }

    #[test]
    fn error_messages_name_the_problem() {
        assert_eq!(MidiError::NoApcMini.to_string(), "no APC mini MIDI port found");
        assert!(MidiError::Init("x".into()).to_string().contains('x'));
        assert!(MidiError::Connect("y".into()).to_string().contains('y'));
    }
}
