//! Where the stage output goes: a fullscreen window on the first non-primary monitor, or a
//! borderless window on the laptop screen when there is no second monitor.
//!
//! Review Focus item 5: the second monitor missing, or unplugged mid-set, must not crash the
//! show. The decision is a pure function over a monitor list, so it is tested without a window
//! manager; the shell re-runs it whenever the monitor list changes and moves the stage window
//! with the result.

/// One monitor as the shell sees it. `index` is the position in the shell's monitor list, which
/// is also the id handed to `mosh_render::MonitorId`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MonitorInfo {
    pub index: u32,
    pub name: String,
    /// The laptop panel, as the OS names it.
    pub primary: bool,
    pub width: u32,
    pub height: u32,
}

impl MonitorInfo {
    pub fn new(index: u32, name: impl Into<String>, primary: bool, width: u32, height: u32) -> Self {
        MonitorInfo {
            index,
            name: name.into(),
            primary,
            width,
            height,
        }
    }
}

/// What the stage window should be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StagePlacement {
    /// Borderless fullscreen on this monitor (idle UI-free output).
    Fullscreen(u32),
    /// No second monitor: a borderless window covering this monitor (the laptop screen).
    Laptop(u32),
    /// No monitors at all: the stage hides until one comes back.
    Hidden,
}

impl StagePlacement {
    /// The monitor to present on, if there is one.
    pub fn monitor(self) -> Option<u32> {
        match self {
            StagePlacement::Fullscreen(index) | StagePlacement::Laptop(index) => Some(index),
            StagePlacement::Hidden => None,
        }
    }

    /// True when the stage should take the whole monitor through a fullscreen window.
    pub fn is_fullscreen(self) -> bool {
        matches!(self, StagePlacement::Fullscreen(_))
    }

    /// True when there is nothing to present on.
    pub fn is_hidden(self) -> bool {
        matches!(self, StagePlacement::Hidden)
    }
}

/// Pick the stage monitor from whatever monitors exist.
///
/// The first non-primary monitor wins, in the OS order; with a single monitor (or a primary-only
/// list) the stage is a borderless window on it; with none the stage hides.
pub fn monitor_choice(monitors: &[MonitorInfo]) -> StagePlacement {
    if monitors.len() >= 2
        && let Some(second) = monitors.iter().find(|monitor| !monitor.primary)
    {
        return StagePlacement::Fullscreen(second.index);
    }
    match monitors.iter().find(|monitor| monitor.primary).or_else(|| monitors.first()) {
        Some(monitor) => StagePlacement::Laptop(monitor.index),
        None => StagePlacement::Hidden,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn laptop() -> MonitorInfo {
        MonitorInfo::new(0, "laptop", true, 1920, 1080)
    }

    fn external() -> MonitorInfo {
        MonitorInfo::new(1, "stage", false, 2560, 1440)
    }

    #[test]
    fn monitor_choice_picks_the_stage_monitor() {
        // No monitor at all: hide the stage, keep the show running.
        assert_eq!(monitor_choice(&[]), StagePlacement::Hidden);

        // One monitor: a borderless window on the laptop screen, never a fullscreen request
        // the OS would have to refuse.
        assert_eq!(monitor_choice(&[laptop()]), StagePlacement::Laptop(0));

        // Two monitors: fullscreen on the second, which is the stage.
        assert_eq!(
            monitor_choice(&[laptop(), external()]),
            StagePlacement::Fullscreen(1)
        );

        // The OS order is not the laptop-first order: the primary is found by flag, not by index.
        let second_first = [
            MonitorInfo::new(0, "stage", false, 2560, 1440),
            MonitorInfo::new(1, "laptop", true, 1920, 1080),
        ];
        assert_eq!(
            monitor_choice(&second_first),
            StagePlacement::Fullscreen(0)
        );

        // Two monitors, neither flagged primary: the first non-primary one still wins.
        let no_primary = [
            MonitorInfo::new(0, "a", false, 1280, 720),
            MonitorInfo::new(1, "b", false, 1280, 720),
        ];
        assert_eq!(monitor_choice(&no_primary), StagePlacement::Fullscreen(0));

        // One monitor that is not the primary (the laptop panel is off): still the only screen.
        let only = [MonitorInfo::new(3, "stage", false, 2560, 1440)];
        assert_eq!(monitor_choice(&only), StagePlacement::Laptop(3));
    }

    #[test]
    fn placement_reports_where_it_goes() {
        assert_eq!(StagePlacement::Fullscreen(2).monitor(), Some(2));
        assert!(StagePlacement::Fullscreen(2).is_fullscreen());
        assert_eq!(StagePlacement::Laptop(0).monitor(), Some(0));
        assert!(!StagePlacement::Laptop(0).is_fullscreen());
        assert_eq!(StagePlacement::Hidden.monitor(), None);
        assert!(StagePlacement::Hidden.is_hidden());
    }
}
