//! Where the two windows go.
//!
//! * **Two monitors**: the stage is fullscreen on the first non-primary one, the control window
//!   is on the laptop screen.
//! * **One monitor**: the stage is an ordinary window *beside* the control window, so the
//!   controls are never covered. `F11` makes the stage fullscreen and back, `Esc` leaves
//!   fullscreen and brings the control window to the front.
//!
//! [`monitor_choice`] picks the stage's monitor; [`window_plan`] turns that and the performer's
//! `F11` choice into the two windows' rectangles.
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
    /// No second monitor: the stage shares this monitor (the laptop screen) with the control
    /// window, as a window beside it unless the performer asks for fullscreen.
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

/// A window's client area, in physical pixels relative to its monitor's top-left corner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Area {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl Area {
    pub fn right(self) -> i32 {
        self.x + self.width as i32
    }

    pub fn bottom(self) -> i32 {
        self.y + self.height as i32
    }

    /// Whether two areas share any pixel.
    pub fn overlaps(self, other: Area) -> bool {
        self.x < other.right()
            && other.x < self.right()
            && self.y < other.bottom()
            && other.y < self.bottom()
    }
}

/// What the stage window should be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageWindow {
    /// Borderless fullscreen on this monitor.
    Fullscreen { monitor: u32 },
    /// An ordinary, resizable window on this monitor.
    Windowed { monitor: u32, area: Area },
    /// No monitor: nothing to present on.
    Hidden,
}

impl StageWindow {
    pub fn monitor(self) -> Option<u32> {
        match self {
            StageWindow::Fullscreen { monitor } | StageWindow::Windowed { monitor, .. } => {
                Some(monitor)
            }
            StageWindow::Hidden => None,
        }
    }

    pub fn is_fullscreen(self) -> bool {
        matches!(self, StageWindow::Fullscreen { .. })
    }
}

/// Where both windows go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowPlan {
    pub stage: StageWindow,
    /// The control window's monitor and area; `None` when there is no monitor at all.
    pub control: Option<(u32, Area)>,
    /// The stage and the control window are on the same monitor, so a fullscreen stage covers
    /// the controls (and `Esc` must bring them back).
    pub shared_monitor: bool,
}

/// The gap around and between windows, as a fraction of the monitor's width.
const MARGIN: f32 = 0.012;
/// Room left for the taskbar and the windows' own title bars, as a fraction of the height.
const CHROME: f32 = 0.10;
/// The control window's share of a monitor it shares with the stage.
const CONTROL_SHARE: f32 = 0.64;

/// Place the stage and the control window on whatever monitors exist.
///
/// `fullscreen` is the performer's `F11` choice: `None` until they press it, which means
/// "fullscreen when the stage has a monitor of its own, a window when it shares one".
pub fn window_plan(monitors: &[MonitorInfo], fullscreen: Option<bool>) -> WindowPlan {
    let placement = monitor_choice(monitors);
    let Some(stage_monitor) = placement.monitor() else {
        return WindowPlan {
            stage: StageWindow::Hidden,
            control: None,
            shared_monitor: false,
        };
    };
    let find = |index: u32| monitors.iter().find(|monitor| monitor.index == index);
    let stage_info = find(stage_monitor);
    // The control window goes on the laptop panel: the primary monitor, or with none flagged the
    // first monitor that is not the stage's.
    let control_info = if placement.is_fullscreen() {
        monitors
            .iter()
            .find(|monitor| monitor.primary && monitor.index != stage_monitor)
            .or_else(|| monitors.iter().find(|monitor| monitor.index != stage_monitor))
    } else {
        stage_info
    };
    let shared = control_info.map(|monitor| monitor.index) == Some(stage_monitor);
    let size = |info: Option<&MonitorInfo>| {
        info.map_or((1920, 1080), |monitor| {
            (monitor.width.max(640), monitor.height.max(480))
        })
    };

    let wants_fullscreen = fullscreen.unwrap_or(!shared);
    let (control, stage_area) = if shared {
        let (control, stage) = side_by_side(size(stage_info));
        (control, stage)
    } else {
        (alone(size(control_info)), centred(size(stage_info)))
    };
    WindowPlan {
        stage: if wants_fullscreen {
            StageWindow::Fullscreen {
                monitor: stage_monitor,
            }
        } else {
            StageWindow::Windowed {
                monitor: stage_monitor,
                area: stage_area,
            }
        },
        control: control_info.map(|monitor| (monitor.index, control)),
        shared_monitor: shared,
    }
}

/// One monitor for both: the control window on the left, the stage top right at 16:9.
fn side_by_side((width, height): (u32, u32)) -> (Area, Area) {
    let margin = (width as f32 * MARGIN).round() as i32;
    let top = (height as f32 * CHROME * 0.4).round() as i32;
    let usable = height as f32 * (1.0 - CHROME);
    let control_width = (width as f32 * CONTROL_SHARE).round() as u32;
    let control = Area {
        x: margin,
        y: top,
        width: control_width,
        height: usable.round() as u32,
    };
    let stage_x = control.right() + margin;
    let stage_width = (width as i32 - stage_x - margin).max(160) as u32;
    let stage_height = (stage_width * 9 / 16).min(usable.round() as u32);
    let stage = Area {
        x: stage_x,
        y: top,
        width: stage_width,
        height: stage_height,
    };
    (control, stage)
}

/// The control window with a monitor to itself: most of it, leaving the desktop's edge visible.
fn alone((width, height): (u32, u32)) -> Area {
    let margin = (width as f32 * MARGIN * 2.0).round() as i32;
    let top = (height as f32 * CHROME * 0.4).round() as i32;
    Area {
        x: margin,
        y: top,
        width: (width as i32 - margin * 2).max(560) as u32,
        height: (height as f32 * (1.0 - CHROME)).round() as u32,
    }
}

/// A windowed stage on its own monitor (the performer pressed `F11` to leave fullscreen): a
/// 16:9 window in the middle, 60% of the monitor's width.
fn centred((width, height): (u32, u32)) -> Area {
    let stage_width = (width as f32 * 0.6).round() as u32;
    let stage_height = (stage_width * 9 / 16).min((height as f32 * (1.0 - CHROME)) as u32);
    Area {
        x: (width as i32 - stage_width as i32) / 2,
        y: (height as i32 - stage_height as i32) / 2,
        width: stage_width,
        height: stage_height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_screen_puts_the_stage_beside_the_controls() {
        for (width, height) in [(1920, 1080), (2560, 1600), (1366, 768)] {
            let monitors = [MonitorInfo::new(0, "laptop", true, width, height)];
            let plan = window_plan(&monitors, None);
            let (monitor, control) = plan.control.expect("a control window");
            assert_eq!(monitor, 0);
            assert!(plan.shared_monitor);
            let StageWindow::Windowed { monitor, area } = plan.stage else {
                panic!("one screen must not start fullscreen: {:?}", plan.stage);
            };
            assert_eq!(monitor, 0);
            // The whole point: the stage never covers the controls, and both are on screen.
            assert!(!control.overlaps(area), "{width}x{height}: {control:?} {area:?}");
            for window in [control, area] {
                assert!(window.x >= 0 && window.y >= 0);
                assert!(window.right() <= width as i32, "{width}x{height}: {window:?}");
                assert!(window.bottom() <= height as i32, "{width}x{height}: {window:?}");
            }
            // The controls get most of the screen; the stage keeps the output's shape.
            assert!(control.width > area.width);
            assert!(control.width >= 800, "{width}x{height}: {control:?}");
            let aspect = area.width as f32 / area.height as f32;
            assert!((aspect - 16.0 / 9.0).abs() < 0.02, "{aspect}");

            // F11: fullscreen on the same screen, and the controls stay where they were.
            let full = window_plan(&monitors, Some(true));
            assert_eq!(full.stage, StageWindow::Fullscreen { monitor: 0 });
            assert_eq!(full.control, plan.control);
            assert!(full.shared_monitor, "Esc has to bring the controls back");
            // F11 again: back to the same window.
            assert_eq!(window_plan(&monitors, Some(false)), plan);
        }
    }

    #[test]
    fn a_second_monitor_gets_the_stage_fullscreen() {
        let monitors = [
            MonitorInfo::new(0, "laptop", true, 2560, 1600),
            MonitorInfo::new(1, "stage", false, 1920, 1080),
        ];
        let plan = window_plan(&monitors, None);
        assert_eq!(plan.stage, StageWindow::Fullscreen { monitor: 1 });
        let (monitor, control) = plan.control.expect("a control window");
        assert_eq!(monitor, 0, "the controls stay on the laptop");
        assert!(!plan.shared_monitor);
        assert!(control.right() <= 2560 && control.bottom() <= 1600);
        assert!(control.width >= 2000, "a screen to itself: {control:?}");

        // F11 on two monitors takes the stage out of fullscreen, on its own monitor.
        let windowed = window_plan(&monitors, Some(false));
        let StageWindow::Windowed { monitor, area } = windowed.stage else {
            panic!("F11 must leave fullscreen");
        };
        assert_eq!(monitor, 1);
        assert!(area.right() <= 1920 && area.bottom() <= 1080);
        assert_eq!(windowed.control, plan.control);

        // The OS order is not the laptop-first order.
        let swapped = [
            MonitorInfo::new(0, "stage", false, 1920, 1080),
            MonitorInfo::new(1, "laptop", true, 2560, 1600),
        ];
        let plan = window_plan(&swapped, None);
        assert_eq!(plan.stage, StageWindow::Fullscreen { monitor: 0 });
        assert_eq!(plan.control.map(|(monitor, _)| monitor), Some(1));

        // No monitor at all: nothing to place, nothing to crash on.
        let none = window_plan(&[], None);
        assert_eq!(none.stage, StageWindow::Hidden);
        assert_eq!(none.control, None);
    }

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
