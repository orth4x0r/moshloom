//! `FlashLimiter`: the one place a full-frame flash can be vetoed.
//!
//! The design calls this a safety rule: at least **400 ms** between full-frame flash starts
//! (the web app's 12 frames at 30 fps), enforced in the render path, and nothing — clip,
//! intensity or controller input — can bypass it. There is deliberately no opt-out argument.

use std::time::{Duration, Instant};

/// Minimum spacing between two full-frame flash starts.
pub const FLASH_INTERVAL: Duration = Duration::from_millis(400);

/// Rate limiter for full-frame flashes.
#[derive(Clone, Debug, Default)]
pub struct FlashLimiter {
    last_allowed: Option<Instant>,
}

impl FlashLimiter {
    pub fn new() -> FlashLimiter {
        FlashLimiter { last_allowed: None }
    }

    /// True when a flash may start at `now`; records `now` when it returns true.
    ///
    /// A clock that steps backwards (or a caller passing an earlier instant) does not unlock a
    /// flash: it is treated as "too soon" and left to the next call.
    pub fn allow(&mut self, now: Instant) -> bool {
        match self.last_allowed {
            Some(last) if now.saturating_duration_since(last) < FLASH_INTERVAL => false,
            _ => {
                self.last_allowed = Some(now);
                true
            }
        }
    }

    /// When a flash was last allowed, for the stats overlay.
    pub fn last_allowed(&self) -> Option<Instant> {
        self.last_allowed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_flash_is_always_allowed() {
        let mut limiter = FlashLimiter::new();
        assert!(limiter.allow(Instant::now()));
    }

    #[test]
    fn flashes_are_at_least_400ms_apart() {
        let mut limiter = FlashLimiter::new();
        let start = Instant::now();
        assert!(limiter.allow(start));
        assert!(!limiter.allow(start + Duration::from_millis(1)));
        assert!(!limiter.allow(start + Duration::from_millis(399)));
        assert!(limiter.allow(start + FLASH_INTERVAL));
        // A clock that steps back must not unlock a flash.
        assert!(!limiter.allow(start));

        // A 16 flashes per second strobe for one second, from idle: only 400 ms and 800 ms
        // start. That is "at most three flash starts per second", and nothing can opt out.
        let mut strobe = FlashLimiter::new();
        assert!(strobe.allow(start));
        let mut starts = 0;
        for step in 1..=100u64 {
            if strobe.allow(start + Duration::from_millis(step * 10)) {
                starts += 1;
            }
        }
        assert_eq!(starts, 2, "16 Hz for 1000 ms with a 400 ms floor");
    }
}
