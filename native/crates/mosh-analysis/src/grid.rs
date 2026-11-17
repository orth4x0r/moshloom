//! Ableton's beat grid: beat/bar phase, beat/bar indices, and bar boundaries.
//!
//! All musical time is ppq (quarter-note beats). A bar starts where Live says it starts
//! (`bar_start_ppq`); it is never `ppq / 4`. The time signature gives the bar length
//! (`sig_num * 4 / sig_den` quarter notes) and the beat length (`4 / sig_den`), so 6/8 is six
//! eighth-note beats in a three-quarter-note bar. A playhead that is moved (a loop, a jump, a
//! scrub) is a discontinuity: the partial bar is dropped, not reported as a complete bar.

use mosh_core::Transport;

/// How long a bar is, in ppq, for a time signature. `4/4` is 4.0, `3/4` is 3.0, `6/8` is 3.0.
pub fn bar_len_ppq(sig_num: u16, sig_den: u16) -> f64 {
    let num = sig_num.max(1) as f64;
    let den = sig_den.max(1) as f64;
    (num * 4.0 / den).max(0.25)
}

/// How long one beat is, in ppq, for a time signature. 4/4 and 3/4 are 1.0, 6/8 is 0.5.
pub fn beat_len_ppq(sig_den: u16) -> f64 {
    (4.0 / sig_den.max(1) as f64).max(0.0625)
}

/// Two ppq values this close are the same musical position (Live sends f64 beats).
const PPQ_EPS: f64 = 1e-6;

/// A bar that closed on a bar line, with the grid it closed on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompletedBar {
    pub bar_index: i64,
    pub start_ppq: f64,
    pub beats: u16,
}

/// One transport update, as the grid sees it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GridTick {
    /// 0..1 within the current beat.
    pub beat_phase: f32,
    /// 0..1 within the current bar.
    pub bar_phase: f32,
    pub beat_index: i64,
    pub bar_index: i64,
    /// This update crossed a beat line.
    pub beat_started: bool,
    /// This update crossed a bar line normally (not a jump).
    pub bar_started: bool,
    /// The bar that just closed, if this update crossed a bar line normally.
    pub completed: Option<CompletedBar>,
    /// The playhead moved discontinuously: the partial bar was dropped.
    pub jumped: bool,
    pub bpm: f32,
    pub playing: bool,
}

/// Follows Live's transport bar by bar.
pub struct BeatGrid {
    phrase_bars: u32,
    started: bool,
    bar_start: f64,
    bar_len: f64,
    beat_len: f64,
    sig_num: u16,
    beat_index: i64,
    bar_index: i64,
}

impl BeatGrid {
    pub fn new(phrase_bars: u32) -> Self {
        Self {
            phrase_bars: phrase_bars.max(1),
            started: false,
            bar_start: 0.0,
            bar_len: 4.0,
            beat_len: 1.0,
            sig_num: 4,
            beat_index: 0,
            bar_index: 0,
        }
    }

    pub fn phrase_bars(&self) -> u32 {
        self.phrase_bars
    }

    /// A phrase starts every `phrase_bars` bars. Bar 0 is the start of the show, not a phrase
    /// boundary, so nothing fires before the first eight bars have played.
    pub fn is_phrase_boundary(&self, bar_index: i64) -> bool {
        bar_index > 0 && bar_index % self.phrase_bars as i64 == 0
    }

    /// The ppq where the next bar line is expected, given where this bar started.
    pub fn next_bar_ppq(&self) -> f64 {
        self.bar_start + self.bar_len
    }

    pub fn bar_len(&self) -> f64 {
        self.bar_len
    }

    /// Feed one transport snapshot. The analysis stage calls this once per block.
    ///
    /// `bar_index` is Live's bar number counted from 0 (Live's bar 1 is index 0): it starts from
    /// the transport's `bar_start_ppq` when the stream starts or jumps, then counts bar lines, so
    /// phrases and every-N-bar rules line up with the song rather than with when the app started.
    pub fn update(&mut self, t: &Transport) -> GridTick {
        let bar_len = bar_len_ppq(t.sig_num, t.sig_den);
        let beat_len = beat_len_ppq(t.sig_den);
        let mut tick = GridTick {
            bpm: t.tempo_bpm as f32,
            playing: t.playing,
            ..GridTick::default()
        };

        if !self.started {
            // The first snapshot only says where we are; it does not close a bar.
            self.started = true;
            self.bar_start = t.bar_start_ppq;
            self.bar_len = bar_len;
            self.beat_len = beat_len;
            self.sig_num = t.sig_num;
            self.bar_index = live_bar_index(t.bar_start_ppq, bar_len);
            self.beat_index = (t.ppq / beat_len).floor() as i64;
        } else if (t.bar_start_ppq - self.bar_start).abs() <= PPQ_EPS {
            // Same bar. The signature can be changed inside it; keep the bar we are in.
            self.bar_len = bar_len;
            self.beat_len = beat_len;
            self.sig_num = t.sig_num;
        } else if (t.bar_start_ppq - self.next_bar_ppq()).abs() <= PPQ_EPS {
            // A bar line: the bar we were in is complete.
            let completed = CompletedBar {
                bar_index: self.bar_index,
                start_ppq: self.bar_start,
                beats: self.sig_num,
            };
            self.bar_index += 1;
            self.bar_start = t.bar_start_ppq;
            self.bar_len = bar_len;
            self.beat_len = beat_len;
            self.sig_num = t.sig_num;
            tick.completed = Some(completed);
            tick.bar_started = true;
        } else {
            // A jump, a loop or a scrub: whatever bar we were in is abandoned, and the count
            // restarts from where Live is now.
            self.bar_index = live_bar_index(t.bar_start_ppq, bar_len);
            self.bar_start = t.bar_start_ppq;
            self.bar_len = bar_len;
            self.beat_len = beat_len;
            self.sig_num = t.sig_num;
            tick.jumped = true;
        }

        let beat_pos = t.ppq / self.beat_len;
        let beat_index = beat_pos.floor() as i64;
        tick.beat_started = beat_index != self.beat_index;
        self.beat_index = beat_index;
        tick.beat_index = beat_index;
        tick.bar_index = self.bar_index;
        tick.beat_phase = (beat_pos - beat_index as f64).clamp(0.0, 1.0) as f32;
        tick.bar_phase = ((t.ppq - self.bar_start) / self.bar_len).clamp(0.0, 1.0) as f32;
        tick
    }
}

/// Live's bar number (0-based) for a bar starting at `bar_start_ppq`, assuming the current bar
/// length held since the start of the song. Exact for a constant meter; after a meter change it is
/// the bar count in the current meter, which keeps every-N-bar rules regular.
fn live_bar_index(bar_start_ppq: f64, bar_len: f64) -> i64 {
    if bar_len <= 0.0 {
        return 0;
    }
    (bar_start_ppq / bar_len).round() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bar_lengths() {
        assert_eq!(bar_len_ppq(4, 4), 4.0);
        assert_eq!(bar_len_ppq(3, 4), 3.0);
        assert_eq!(bar_len_ppq(6, 8), 3.0);
        assert_eq!(beat_len_ppq(4), 1.0);
        assert_eq!(beat_len_ppq(8), 0.5);
        // A nonsense signature never divides by zero.
        assert!(bar_len_ppq(0, 0) >= 0.25);
    }
}
