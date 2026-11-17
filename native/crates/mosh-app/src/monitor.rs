//! Where the two windows go.
//!
//! * **Two monitors**: the stage is fullscreen on the first non-primary one, the control window
//!   fills the laptop screen's work area.
//! * **One monitor**: the stage is an ordinary window *beside* the control window, so the
//!   controls are never covered. The controls take the left 58% of the work area, the stage the
//!   rest, with a 16:9 picture. `F11` makes the stage fullscreen and back, `Esc` leaves
//!   fullscreen and brings the control window to the front.
//!
//! Everything is placed inside the monitor's **work area** (the desktop without the taskbar),
//! and a window's rectangle here is its **outer** rectangle, title bar and borders included:
//! [`Chrome`] is the difference to the client area. The first version placed client areas
//! against the whole monitor, and on the reference laptop the control window came out taller
//! than the screen.
//!
//! The performer's own arrangement wins over the plan: [`SavedLayout`] is the two windows'
//! last positions and sizes, kept per monitor setup ([`setup_key`]), and used again as long as
//! the window would still be on a screen ([`usable`]).
//!
//! [`monitor_choice`] picks the stage's monitor; [`window_plan`] turns that and the performer's
//! `F11` choice into the two windows' rectangles. All of it is pure, so it is tested without a
//! window manager; the shell re-runs it whenever the monitor list changes.

/// A rectangle in physical pixels: a monitor's work area relative to the monitor's top-left
/// corner, or a window's outer rectangle relative to the same corner.
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

    /// Whether `other` lies wholly inside this area.
    pub fn contains(self, other: Area) -> bool {
        other.x >= self.x
            && other.y >= self.y
            && other.right() <= self.right()
            && other.bottom() <= self.bottom()
    }

    /// The width and height of the part of `other` that is inside this area.
    pub fn visible_part(self, other: Area) -> (u32, u32) {
        let width = self.right().min(other.right()) - self.x.max(other.x);
        let height = self.bottom().min(other.bottom()) - self.y.max(other.y);
        (width.max(0) as u32, height.max(0) as u32)
    }

    /// The client size of a window whose outer rectangle is this area.
    pub fn inner(self, chrome: Chrome) -> (u32, u32) {
        (
            self.width.saturating_sub(chrome.width).max(1),
            self.height.saturating_sub(chrome.height).max(1),
        )
    }
}

/// What a window's frame adds to its client area: outer size minus inner size, in pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Chrome {
    pub width: u32,
    pub height: u32,
}

impl Chrome {
    /// A guess for a monitor's scale factor, used until a real window has been measured: a
    /// 31-point title bar and the 8-point resize borders Windows puts around a window (invisible,
    /// but part of its outer rectangle).
    pub fn estimate(scale: f64) -> Chrome {
        let scale = scale.clamp(1.0, 4.0);
        Chrome {
            width: (16.0 * scale).round() as u32,
            height: (39.0 * scale).round() as u32,
        }
    }
}

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
    /// The monitor's top-left corner on the desktop.
    pub x: i32,
    pub y: i32,
    /// The desktop without the taskbar, relative to the monitor's top-left corner.
    pub work: Area,
}

impl MonitorInfo {
    /// A monitor at the desktop's origin whose work area is all of it.
    pub fn new(index: u32, name: impl Into<String>, primary: bool, width: u32, height: u32) -> Self {
        MonitorInfo {
            index,
            name: name.into(),
            primary,
            width,
            height,
            x: 0,
            y: 0,
            work: Area {
                x: 0,
                y: 0,
                width,
                height,
            },
        }
    }

    /// The monitor's place on the desktop.
    pub fn at(mut self, x: i32, y: i32) -> Self {
        self.x = x;
        self.y = y;
        self
    }

    /// The work area, relative to the monitor's corner. One that is empty or not on the monitor
    /// (a bad answer from the OS) is ignored.
    pub fn with_work(mut self, work: Area) -> Self {
        let whole = Area {
            x: 0,
            y: 0,
            width: self.width,
            height: self.height,
        };
        if work.width >= 320 && work.height >= 240 && whole.contains(work) {
            self.work = work;
        }
        self
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
/// list) the stage shares it; with none the stage hides.
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

/// What the stage window should be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageWindow {
    /// Borderless fullscreen on this monitor.
    Fullscreen { monitor: u32 },
    /// An ordinary, resizable window on this monitor; `area` is its outer rectangle.
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
    /// The control window's monitor and outer rectangle; `None` when there is no monitor at all.
    pub control: Option<(u32, Area)>,
    /// The stage and the control window are on the same monitor, so a fullscreen stage covers
    /// the controls (and `Esc` must bring them back).
    pub shared_monitor: bool,
}

/// The control window's share of the work area's width when the stage is beside it.
pub const CONTROL_SHARE: f32 = 0.58;
/// The stage's picture.
const STAGE_ASPECT: (u32, u32) = (16, 9);

/// Place the stage and the control window on whatever monitors exist.
///
/// `fullscreen` is the performer's `F11` choice: `None` until they press it, which means
/// "fullscreen when the stage has a monitor of its own, a window when it shares one". `chrome`
/// is what a window frame adds to a client area, so the stage's *picture* is 16:9, not its frame.
pub fn window_plan(
    monitors: &[MonitorInfo],
    fullscreen: Option<bool>,
    chrome: Chrome,
) -> WindowPlan {
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
    let work = |info: Option<&MonitorInfo>| {
        info.map_or(
            Area {
                x: 0,
                y: 0,
                width: 1920,
                height: 1080,
            },
            |monitor| monitor.work,
        )
    };

    let wants_fullscreen = fullscreen.unwrap_or(!shared);
    let (control, stage_area) = if shared {
        side_by_side(work(stage_info), chrome)
    } else {
        (work(control_info), centred(work(stage_info), chrome))
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

/// The outer height of a window `outer_width` wide whose client area is 16:9, at most `limit`.
fn stage_height(outer_width: u32, chrome: Chrome, limit: u32) -> u32 {
    let picture = outer_width.saturating_sub(chrome.width).max(16);
    (picture * STAGE_ASPECT.1 / STAGE_ASPECT.0 + chrome.height).min(limit)
}

/// One monitor for both: the control window fills the left of the work area top to bottom, the
/// stage takes the rest of the width, top right, with a 16:9 picture.
fn side_by_side(work: Area, chrome: Chrome) -> (Area, Area) {
    let control_width = (work.width as f32 * CONTROL_SHARE).round() as u32;
    let control = Area {
        x: work.x,
        y: work.y,
        width: control_width,
        height: work.height,
    };
    let stage_width = work.width - control_width;
    let stage = Area {
        x: control.right(),
        y: work.y,
        width: stage_width,
        height: stage_height(stage_width, chrome, work.height),
    };
    (control, stage)
}

/// A windowed stage on its own monitor (the performer pressed `F11` to leave fullscreen): a
/// window in the middle of the work area, 60% of its width, with a 16:9 picture.
fn centred(work: Area, chrome: Chrome) -> Area {
    let width = (work.width as f32 * 0.6).round() as u32;
    let height = stage_height(width, chrome, work.height);
    Area {
        x: work.x + (work.width as i32 - width as i32) / 2,
        y: work.y + (work.height as i32 - height as i32) / 2,
        width,
        height,
    }
}

/// A window as the performer left it: its outer position on the desktop and its client size
/// (the two things the window system reports and takes back unchanged).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SavedWindow {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// Both windows as the performer left them, for one monitor setup.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SavedLayout {
    pub control: Option<SavedWindow>,
    /// The stage as a window; a fullscreen stage has no rectangle worth keeping.
    pub stage: Option<SavedWindow>,
}

/// Monitor setups remembered, at most (the laptop alone, the laptop and each venue's screen).
pub const LAYOUTS_KEPT: usize = 12;
/// The least of a saved window that must still be on a screen for it to be used again.
const VISIBLE_AT_LEAST: (u32, u32) = (240, 160);

/// A name for a monitor setup: every monitor's size, place and whether it is the primary one,
/// in a fixed order. The same monitors plugged in again give the same key; a different
/// projector, or the same one at another resolution, gives another.
pub fn setup_key(monitors: &[MonitorInfo]) -> String {
    let mut parts: Vec<String> = monitors
        .iter()
        .map(|monitor| {
            format!(
                "{}x{}@{},{}{}",
                monitor.width,
                monitor.height,
                monitor.x,
                monitor.y,
                if monitor.primary { "*" } else { "" }
            )
        })
        .collect();
    parts.sort();
    parts.join("|")
}

/// Whether a saved window can be used on these monitors: enough of it is inside some monitor's
/// work area to grab, and it is not larger than that work area (a window saved on a 4K screen
/// does not open across a laptop panel).
pub fn usable(window: SavedWindow, monitors: &[MonitorInfo], chrome: Chrome) -> bool {
    let outer = Area {
        x: window.x,
        y: window.y,
        width: window.width + chrome.width,
        height: window.height + chrome.height,
    };
    monitors.iter().any(|monitor| {
        let work = Area {
            x: monitor.x + monitor.work.x,
            y: monitor.y + monitor.work.y,
            ..monitor.work
        };
        let (width, height) = work.visible_part(outer);
        width >= VISIBLE_AT_LEAST.0.min(outer.width)
            && height >= VISIBLE_AT_LEAST.1.min(outer.height)
            && outer.width <= work.width + chrome.width
            && outer.height <= work.height + chrome.height
            // The title bar is what a window is dragged by: it must not be above the screen.
            && outer.y >= work.y - chrome.height as i32
    })
}

/// Put `layout` in the list under `key`, newest first, keeping [`LAYOUTS_KEPT`] setups.
pub fn remember(layouts: &mut Vec<(String, SavedLayout)>, key: &str, layout: SavedLayout) {
    layouts.retain(|(other, _)| other != key);
    layouts.insert(0, (key.to_string(), layout));
    layouts.truncate(LAYOUTS_KEPT);
}

/// The saved layout for a monitor setup, if there is one.
pub fn recall(layouts: &[(String, SavedLayout)], key: &str) -> Option<SavedLayout> {
    layouts
        .iter()
        .find(|(other, _)| other == key)
        .map(|(_, layout)| *layout)
}

/// The layouts as the text the app keeps in `%APPDATA%\ilanaVisual\windows.txt`: one setup a
/// line, `key <tab> control=x,y,w,h <tab> stage=x,y,w,h`.
pub fn format_layouts(layouts: &[(String, SavedLayout)]) -> String {
    let window = |name: &str, window: Option<SavedWindow>| match window {
        Some(window) => format!(
            "\t{name}={},{},{},{}",
            window.x, window.y, window.width, window.height
        ),
        None => String::new(),
    };
    layouts
        .iter()
        .map(|(key, layout)| {
            format!(
                "{key}{}{}\n",
                window("control", layout.control),
                window("stage", layout.stage)
            )
        })
        .collect()
}

/// Read [`format_layouts`]'s text back. A line or a field that does not parse is skipped: a
/// damaged file costs a remembered window, never a start.
pub fn parse_layouts(text: &str) -> Vec<(String, SavedLayout)> {
    let window = |value: &str| -> Option<SavedWindow> {
        let mut numbers = value.split(',').map(str::trim);
        let window = SavedWindow {
            x: numbers.next()?.parse().ok()?,
            y: numbers.next()?.parse().ok()?,
            width: numbers.next()?.parse().ok()?,
            height: numbers.next()?.parse().ok()?,
        };
        (numbers.next().is_none() && window.width > 0 && window.height > 0).then_some(window)
    };
    let mut layouts = Vec::new();
    for line in text.lines() {
        let mut fields = line.split('\t');
        let Some(key) = fields.next().filter(|key| !key.trim().is_empty()) else {
            continue;
        };
        let mut layout = SavedLayout::default();
        for field in fields {
            match field.split_once('=') {
                Some(("control", value)) => layout.control = window(value),
                Some(("stage", value)) => layout.stage = window(value),
                _ => {}
            }
        }
        if layouts.len() < LAYOUTS_KEPT && !layouts.iter().any(|(other, _)| other == key) {
            layouts.push((key.to_string(), layout));
        }
    }
    layouts
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

    /// The reference laptop: a 2560x1600 panel at 175%, the taskbar (48 points) at the bottom.
    fn reference() -> (MonitorInfo, Chrome) {
        let monitor = MonitorInfo::new(0, "laptop", true, 2560, 1600).with_work(Area {
            x: 0,
            y: 0,
            width: 2560,
            height: 1516,
        });
        (monitor, Chrome::estimate(1.75))
    }

    #[test]
    fn monitor_choice_picks_the_stage_monitor() {
        // No monitor at all: hide the stage, keep the show running.
        assert_eq!(monitor_choice(&[]), StagePlacement::Hidden);

        // One monitor: the stage shares the laptop screen, never a fullscreen request the OS
        // would have to refuse.
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

    #[test]
    fn both_windows_fit_the_reference_laptops_work_area() {
        // What the first windowed run got wrong: a control window 1504 px tall on a 1516 px work
        // area with its top 64 px down, and a stage only 853x530.
        let (monitor, chrome) = reference();
        let work = monitor.work;
        let plan = window_plan(std::slice::from_ref(&monitor), None, chrome);
        let (_, control) = plan.control.expect("a control window");
        let StageWindow::Windowed { area: stage, .. } = plan.stage else {
            panic!("one screen must not start fullscreen");
        };
        // Both wholly inside the work area (the taskbar is not covered), side by side.
        assert!(work.contains(control), "{control:?}");
        assert!(work.contains(stage), "{stage:?}");
        assert!(!control.overlaps(stage));
        // The controls: the left 58%, the full height.
        assert_eq!((control.x, control.y), (0, 0));
        assert_eq!((control.width, control.height), (1485, 1516));
        // The stage: all of the rest, and a 16:9 picture inside its frame.
        assert_eq!((stage.x, stage.y, stage.width), (1485, 0, 1075));
        let (picture_w, picture_h) = stage.inner(chrome);
        assert_eq!((picture_w, picture_h), (1047, 588));
        assert!((picture_w as f32 / picture_h as f32 - 16.0 / 9.0).abs() < 0.01);
        assert!(picture_w > 853, "wider than the first run's stage");
    }

    #[test]
    fn one_screen_puts_the_stage_beside_the_controls() {
        for (width, height, taskbar, scale) in [
            (1920, 1080, 48, 1.0),
            (2560, 1600, 84, 1.75),
            (1366, 768, 40, 1.0),
            (3840, 2160, 96, 2.0),
        ] {
            let work = Area {
                x: 0,
                y: 0,
                width,
                height: height - taskbar,
            };
            let monitors = [MonitorInfo::new(0, "laptop", true, width, height).with_work(work)];
            let chrome = Chrome::estimate(scale);
            let plan = window_plan(&monitors, None, chrome);
            let (monitor, control) = plan.control.expect("a control window");
            assert_eq!(monitor, 0);
            assert!(plan.shared_monitor);
            let StageWindow::Windowed { monitor, area } = plan.stage else {
                panic!("one screen must not start fullscreen: {:?}", plan.stage);
            };
            assert_eq!(monitor, 0);
            let what = format!("{width}x{height}: {control:?} {area:?}");
            // The whole point: the stage never covers the controls, and both are wholly on the
            // work area.
            assert!(!control.overlaps(area), "{what}");
            assert!(work.contains(control) && work.contains(area), "{what}");
            // The controls get 58% of the width and all of the height; the stage gets the rest
            // of the width, and its picture keeps the output's shape.
            let share = control.width as f32 / work.width as f32;
            assert!((share - CONTROL_SHARE).abs() < 0.001, "{what}");
            assert_eq!(control.height, work.height, "{what}");
            assert_eq!(control.width + area.width, work.width, "{what}");
            let (picture_w, picture_h) = area.inner(chrome);
            let aspect = picture_w as f32 / picture_h as f32;
            assert!((aspect - 16.0 / 9.0).abs() < 0.02, "{what}: {aspect}");

            // F11: fullscreen on the same screen, and the controls stay where they were.
            let full = window_plan(&monitors, Some(true), chrome);
            assert_eq!(full.stage, StageWindow::Fullscreen { monitor: 0 });
            assert_eq!(full.control, plan.control);
            assert!(full.shared_monitor, "Esc has to bring the controls back");
            // F11 again: back to the same window.
            assert_eq!(window_plan(&monitors, Some(false), chrome), plan);
        }

        // A taskbar on the left: the work area does not start at the monitor's corner.
        let work = Area {
            x: 72,
            y: 0,
            width: 1848,
            height: 1080,
        };
        let monitors = [MonitorInfo::new(0, "laptop", true, 1920, 1080).with_work(work)];
        let plan = window_plan(&monitors, None, Chrome::estimate(1.0));
        let (_, control) = plan.control.expect("a control window");
        assert_eq!(control.x, 72);
        assert!(work.contains(control));

        // A work area the OS got wrong (empty, or off the monitor) falls back to the monitor.
        let broken = MonitorInfo::new(0, "laptop", true, 1920, 1080).with_work(Area {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
        });
        assert_eq!(broken.work.width, 1920);
    }

    #[test]
    fn a_second_monitor_gets_the_stage_fullscreen() {
        let chrome = Chrome::estimate(1.0);
        let monitors = [
            MonitorInfo::new(0, "laptop", true, 2560, 1600).with_work(Area {
                x: 0,
                y: 0,
                width: 2560,
                height: 1516,
            }),
            MonitorInfo::new(1, "stage", false, 1920, 1080).at(2560, 0),
        ];
        let plan = window_plan(&monitors, None, chrome);
        assert_eq!(plan.stage, StageWindow::Fullscreen { monitor: 1 });
        let (monitor, control) = plan.control.expect("a control window");
        assert_eq!(monitor, 0, "the controls stay on the laptop");
        assert!(!plan.shared_monitor);
        // A screen to itself: the whole work area, and not a pixel of the taskbar.
        assert_eq!(control, monitors[0].work);

        // F11 on two monitors takes the stage out of fullscreen, on its own monitor.
        let windowed = window_plan(&monitors, Some(false), chrome);
        let StageWindow::Windowed { monitor, area } = windowed.stage else {
            panic!("F11 must leave fullscreen");
        };
        assert_eq!(monitor, 1);
        assert!(monitors[1].work.contains(area));
        assert_eq!(windowed.control, plan.control);

        // The OS order is not the laptop-first order.
        let swapped = [
            MonitorInfo::new(0, "stage", false, 1920, 1080),
            MonitorInfo::new(1, "laptop", true, 2560, 1600),
        ];
        let plan = window_plan(&swapped, None, chrome);
        assert_eq!(plan.stage, StageWindow::Fullscreen { monitor: 0 });
        assert_eq!(plan.control.map(|(monitor, _)| monitor), Some(1));

        // No monitor at all: nothing to place, nothing to crash on.
        let none = window_plan(&[], None, chrome);
        assert_eq!(none.stage, StageWindow::Hidden);
        assert_eq!(none.control, None);
    }

    #[test]
    fn window_positions_are_remembered_per_monitor_setup() {
        let (laptop, chrome) = reference();
        let alone = [laptop.clone()];
        let with_projector = [
            laptop.clone(),
            MonitorInfo::new(1, "projector", false, 1920, 1080).at(2560, 0),
        ];
        // Two setups, two keys; the same setup again, the same key, whatever the OS order.
        let key_alone = setup_key(&alone);
        let key_both = setup_key(&with_projector);
        assert_eq!(key_alone, "2560x1600@0,0*");
        assert_ne!(key_alone, key_both);
        let reversed = [with_projector[1].clone(), with_projector[0].clone()];
        assert_eq!(setup_key(&reversed), key_both);

        let mine = SavedLayout {
            control: Some(SavedWindow {
                x: 0,
                y: 0,
                width: 1700,
                height: 1400,
            }),
            stage: Some(SavedWindow {
                x: 1730,
                y: 40,
                width: 800,
                height: 450,
            }),
        };
        let gig = SavedLayout {
            control: Some(SavedWindow {
                x: 100,
                y: 50,
                width: 2000,
                height: 1300,
            }),
            stage: None,
        };
        let mut layouts = Vec::new();
        remember(&mut layouts, &key_alone, mine);
        remember(&mut layouts, &key_both, gig);
        assert_eq!(recall(&layouts, &key_alone), Some(mine));
        assert_eq!(recall(&layouts, &key_both), Some(gig));
        assert_eq!(recall(&layouts, "1024x768@0,0*"), None);

        // Through the file and back, unchanged; a newer layout for a setup replaces the old one.
        let text = format_layouts(&layouts);
        assert_eq!(parse_layouts(&text), layouts);
        let moved = SavedLayout {
            stage: None,
            ..mine
        };
        remember(&mut layouts, &key_alone, moved);
        assert_eq!(layouts.len(), 2);
        assert_eq!(layouts[0], (key_alone.clone(), moved), "newest first");
        // A damaged file loses a window, not the start.
        let damaged = format!("{key_alone}\tcontrol=1,2,three,4\tstage=5,6,700,400\n\n\tjunk\n");
        assert_eq!(
            parse_layouts(&damaged),
            vec![(
                key_alone.clone(),
                SavedLayout {
                    control: None,
                    stage: Some(SavedWindow {
                        x: 5,
                        y: 6,
                        width: 700,
                        height: 400
                    })
                }
            )]
        );
        // The list does not grow without end.
        for index in 0..40 {
            remember(&mut layouts, &format!("setup {index}"), mine);
        }
        assert_eq!(layouts.len(), LAYOUTS_KEPT);

        // A saved window is used again only where it can be seen and grabbed.
        let window = |x, y, width, height| SavedWindow {
            x,
            y,
            width,
            height,
        };
        assert!(usable(window(0, 0, 1700, 1400), &alone, chrome));
        assert!(usable(window(1730, 40, 800, 450), &alone, chrome));
        // On the projector, which is gone: not on the laptop alone, fine with it plugged in.
        assert!(!usable(window(2700, 100, 800, 450), &alone, chrome));
        assert!(usable(window(2700, 100, 800, 450), &with_projector, chrome));
        // Larger than the screen (saved on a bigger one), or with its title bar above it.
        assert!(!usable(window(0, 0, 3000, 1400), &alone, chrome));
        assert!(!usable(window(0, 0, 1700, 1580), &alone, chrome));
        assert!(!usable(window(100, -400, 800, 450), &alone, chrome));
        // Hanging off the right edge with a sliver showing: not enough to grab.
        assert!(!usable(window(2500, 100, 800, 450), &alone, chrome));
    }
}
