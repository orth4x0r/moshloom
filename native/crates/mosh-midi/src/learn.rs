//! MIDI learn: bind any incoming `(status, data1)` pair to a control event.
//!
//! A binding stores a template event. Continuous templates (`SetParam`, `SetOpacity`,
//! `IntensityBias`) take the incoming data byte as their value; discrete ones (launches,
//! `Lock`, `BackToArrangement`, `SelectSong`) fire as stored. Bindings are plain data with
//! serde derives so a project can save and load them.

use mosh_core::ControlEvent;
use serde::{Deserialize, Serialize};

/// One learned control.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LearnBinding {
    /// Raw status byte, channel included.
    pub status: u8,
    /// Note or CC number.
    pub data1: u8,
    /// What the control does; continuous events take the incoming value.
    pub event: ControlEvent,
}

/// A set of learned controls, one per `(status, data1)` pair.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MidiLearn {
    bindings: Vec<LearnBinding>,
}

impl MidiLearn {
    /// Binds a control, replacing whatever it was bound to before.
    pub fn learn(&mut self, status: u8, data1: u8, template: ControlEvent) {
        let binding = LearnBinding { status, data1, event: template };
        match self.bindings.iter_mut().find(|b| b.status == status && b.data1 == data1) {
            Some(slot) => *slot = binding,
            None => self.bindings.push(binding),
        }
    }

    /// Removes a binding. Returns whether there was one.
    pub fn forget(&mut self, status: u8, data1: u8) -> bool {
        let before = self.bindings.len();
        self.bindings.retain(|b| b.status != status || b.data1 != data1);
        self.bindings.len() != before
    }

    /// The learned events, in the order they were first learned.
    pub fn bindings(&self) -> &[LearnBinding] {
        &self.bindings
    }

    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// The event a message triggers, with the value applied to continuous templates.
    ///
    /// `status` is matched exactly, so a binding made on one MIDI channel does not fire on
    /// another.
    pub fn map(&self, status: u8, data1: u8, value: u8) -> Option<ControlEvent> {
        let event = self
            .bindings
            .iter()
            .find(|b| b.status == status && b.data1 == data1)
            .map(|b| b.event)?;
        Some(with_value(event, value))
    }
}

/// Replaces a continuous template's value with what the control sent.
fn with_value(event: ControlEvent, value: u8) -> ControlEvent {
    let unit = crate::apc_mini::normalise(value);
    match event {
        ControlEvent::SetParam { layer, index, .. } => {
            ControlEvent::SetParam { layer, index, value: unit }
        }
        ControlEvent::SetOpacity { layer, .. } => {
            ControlEvent::SetOpacity { layer, value: unit }
        }
        ControlEvent::IntensityBias(_) => ControlEvent::IntensityBias(crate::apc_mini::bipolar(value)),
        discrete => discrete,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learn_and_forget_keep_one_binding_per_control() {
        let mut learn = MidiLearn::default();
        assert!(learn.is_empty());
        learn.learn(0x90, 36, ControlEvent::LaunchClip { layer: 2, scene: 4 });
        learn.learn(0x90, 36, ControlEvent::LaunchScene(6));
        assert_eq!(learn.bindings().len(), 1);
        assert_eq!(learn.map(0x90, 36, 100), Some(ControlEvent::LaunchScene(6)));
        assert!(learn.forget(0x90, 36));
        assert!(learn.is_empty());
        assert_eq!(learn.map(0x90, 36, 100), None);
    }

    #[test]
    fn channel_is_part_of_the_match() {
        let mut learn = MidiLearn::default();
        learn.learn(0x90, 36, ControlEvent::LaunchScene(0));
        assert!(learn.map(0x90, 36, 100).is_some());
        assert!(learn.map(0x91, 36, 100).is_none());
    }

    #[test]
    fn continuous_templates_take_the_message_value() {
        let mut learn = MidiLearn::default();
        learn.learn(0xB0, 7, ControlEvent::SetOpacity { layer: 3, value: 0.5 });
        assert_eq!(learn.map(0xB0, 7, 0), Some(ControlEvent::SetOpacity { layer: 3, value: 0.0 }));
        assert_eq!(learn.map(0xB0, 7, 127), Some(ControlEvent::SetOpacity { layer: 3, value: 1.0 }));
    }
}
