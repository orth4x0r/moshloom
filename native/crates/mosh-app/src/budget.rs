//! The design's performance budgets, measured in the app.
//!
//! The performance table in `docs/superpowers/specs/2026-11-17-native-port-design.md` is the
//! contract: render-thread CPU under 2 ms per frame and GPU under 6 ms per frame at 1920x1080.
//! The render thread samples both every frame; [`BudgetWatch`] averages them over a second and
//! returns **one log line per budget per exceed**, because the app's own output must not become
//! a per-frame cost (and a per-frame warning at 60 fps buries the first one).
//!
//! The watch is a plain state machine over `Instant`, so the policy is tested without a GPU.

use std::time::{Duration, Instant};

/// Render-thread CPU per frame, all of the app's own work in that frame.
pub const CPU_MS_PER_FRAME: f32 = 2.0;
/// GPU per frame, 1920x1080 at 60 fps, three layers plus overlay, integrated GPU on battery.
pub const GPU_MS_PER_FRAME: f32 = 6.0;
/// The averaging window the design asks for ("averaged over 1 s").
pub const WINDOW: Duration = Duration::from_secs(1);

/// Which budget a line is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Budget {
    Cpu,
    Gpu,
}

impl Budget {
    pub fn label(self) -> &'static str {
        match self {
            Budget::Cpu => "render thread CPU",
            Budget::Gpu => "GPU",
        }
    }

    pub fn limit_ms(self) -> f32 {
        match self {
            Budget::Cpu => CPU_MS_PER_FRAME,
            Budget::Gpu => GPU_MS_PER_FRAME,
        }
    }
}

/// Per-frame samples, a one-second average, and one line per budget per exceed.
#[derive(Debug)]
pub struct BudgetWatch {
    window: Duration,
    window_start: Option<Instant>,
    cpu_sum: f64,
    gpu_sum: f64,
    frames: u32,
    /// True while the budget is over: keeps an exceed to one line until it recovers.
    latched: [bool; 2],
    /// The last closed window's averages, for the stats line in the control window.
    last_cpu_ms: f32,
    last_gpu_ms: f32,
    exceeds: u64,
}

impl Default for BudgetWatch {
    fn default() -> Self {
        Self::new()
    }
}

impl BudgetWatch {
    pub fn new() -> BudgetWatch {
        BudgetWatch::with_window(WINDOW)
    }

    /// A watch with a custom averaging window (tests use a short one).
    pub fn with_window(window: Duration) -> BudgetWatch {
        BudgetWatch {
            window,
            window_start: None,
            cpu_sum: 0.0,
            gpu_sum: 0.0,
            frames: 0,
            latched: [false; 2],
            last_cpu_ms: 0.0,
            last_gpu_ms: 0.0,
            exceeds: 0,
        }
    }

    /// Average of the last closed window.
    pub fn last_cpu_ms(&self) -> f32 {
        self.last_cpu_ms
    }

    /// Average of the last closed window.
    pub fn last_gpu_ms(&self) -> f32 {
        self.last_gpu_ms
    }

    /// How many times a budget crossed its limit since the app started.
    pub fn exceeds(&self) -> u64 {
        self.exceeds
    }

    /// Record one frame. The returned lines are the ones to log *now*: empty on almost every
    /// frame, at most one per budget, and none until the average actually goes over.
    pub fn sample(&mut self, cpu_ms: f32, gpu_ms: f32, now: Instant) -> Vec<String> {
        let start = *self.window_start.get_or_insert(now);
        // A clock that went backwards (a suspended laptop) starts a new window instead of
        // producing a negative average.
        if now < start {
            self.window_start = Some(now);
            self.reset_window();
            return Vec::new();
        }
        if cpu_ms.is_finite() {
            self.cpu_sum += f64::from(cpu_ms.max(0.0));
            self.gpu_sum += f64::from(gpu_ms.max(0.0));
            self.frames += 1;
        }
        if now.saturating_duration_since(start) < self.window {
            return Vec::new();
        }

        let frames = self.frames.max(1) as f64;
        let cpu = (self.cpu_sum / frames) as f32;
        let gpu = (self.gpu_sum / frames) as f32;
        let seconds = now.saturating_duration_since(start).as_secs_f32();
        self.last_cpu_ms = cpu;
        self.last_gpu_ms = gpu;
        self.reset_window();
        self.window_start = Some(now);

        let mut lines = Vec::new();
        for (index, (budget, value)) in [(Budget::Cpu, cpu), (Budget::Gpu, gpu)].into_iter().enumerate() {
            if value > budget.limit_ms() {
                if !self.latched[index] {
                    self.latched[index] = true;
                    self.exceeds += 1;
                    lines.push(format!(
                        "budget exceeded: {} {value:.2} ms/frame over {seconds:.1} s (limit {:.2} ms)",
                        budget.label(),
                        budget.limit_ms(),
                    ));
                }
            } else {
                self.latched[index] = false;
            }
        }
        lines
    }

    fn reset_window(&mut self) {
        self.cpu_sum = 0.0;
        self.gpu_sum = 0.0;
        self.frames = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(watch: &mut BudgetWatch, start: Instant, cpu: f32, gpu: f32, frames: u32) -> Vec<String> {
        let mut lines = Vec::new();
        for i in 0..frames {
            let now = start + Duration::from_millis(u64::from(i) * 20);
            lines.extend(watch.sample(cpu, gpu, now));
        }
        lines
    }

    #[test]
    fn budget_check_logs_once() {
        let mut watch = BudgetWatch::with_window(WINDOW);
        let start = Instant::now();
        // 60 frames at 20 ms is 1.18 s: one closed window, over the CPU budget, under the GPU one.
        let lines = feed(&mut watch, start, 3.0, 1.0, 60);
        assert_eq!(lines.len(), 1, "one line for one exceeded budget: {lines:?}");
        assert!(lines[0].contains("render thread CPU"), "{}", lines[0]);
        assert!(lines[0].contains("3.00"), "{}", lines[0]);
        assert_eq!(watch.exceeds(), 1);

        // Still over budget for three more windows: no repeated lines.
        let later = start + Duration::from_secs(2);
        let lines = feed(&mut watch, later, 3.0, 1.0, 180);
        assert!(lines.is_empty(), "an exceed logs once, not per frame: {lines:?}");
        assert_eq!(watch.exceeds(), 1);

        // Under budget re-arms the watch ...
        let calm = start + Duration::from_secs(6);
        let lines = feed(&mut watch, calm, 0.5, 1.0, 60);
        assert!(lines.is_empty(), "{lines:?}");

        // ... so the next exceed is reported again.
        let over = start + Duration::from_secs(8);
        let lines = feed(&mut watch, over, 3.0, 1.0, 60);
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert_eq!(watch.exceeds(), 2);
    }

    #[test]
    fn budgets_are_checked_separately() {
        let mut watch = BudgetWatch::with_window(WINDOW);
        let start = Instant::now();
        // Under the CPU budget, over the GPU one: exactly the GPU line.
        let lines = feed(&mut watch, start, 1.0, 9.0, 60);
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(lines[0].contains("GPU 9.00"), "{}", lines[0]);

        // Under both for a window: the watch re-arms.
        let calm = start + Duration::from_secs(2);
        let lines = feed(&mut watch, calm, 0.5, 1.0, 60);
        assert!(lines.is_empty(), "{lines:?}");

        // Both over: one line each, because each budget latches on its own.
        let both = start + Duration::from_secs(4);
        let lines = feed(&mut watch, both, 5.0, 12.0, 60);
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert!(lines.iter().any(|line| line.contains("render thread CPU")));
        assert!(lines.iter().any(|line| line.contains("GPU")));
    }

    #[test]
    fn no_lines_before_the_window_closes() {
        let mut watch = BudgetWatch::with_window(WINDOW);
        let start = Instant::now();
        assert!(watch.sample(50.0, 50.0, start).is_empty());
        assert!(watch.sample(50.0, 50.0, start + Duration::from_millis(500)).is_empty());
        assert!(watch.exceeds() == 0);
    }
}
