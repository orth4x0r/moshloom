//! Short-term loudness: K-weighted (BS.1770-style high-pass + high shelf) and averaged over a
//! sliding 3 s window, in dBFS. `BarStats::loudness_db` and `AnalysisFrame::loudness_db` come
//! from here, so a breakdown can be recognised as "below -50 dB" without depending on the mix.

/// Length of the loudness window. Short-term, meter-like, not momentary.
pub const SHORT_TERM_S: f32 = 3.0;
/// What the meter reports when it has nothing to measure.
pub const FLOOR_DB: f32 = -70.0;

/// A direct-form-II transposed biquad.
#[derive(Clone, Copy, Debug, Default)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    /// RBJ second-order high-pass at `hz`.
    fn high_pass(sr: f32, hz: f32, q: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * hz / sr;
        let cos = w0.cos();
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self {
            b0: (1.0 + cos) * 0.5 / a0,
            b1: -(1.0 + cos) / a0,
            b2: (1.0 + cos) * 0.5 / a0,
            a1: -2.0 * cos / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// RBJ high shelf, `gain_db` above `hz`.
    fn high_shelf(sr: f32, hz: f32, gain_db: f32, s: f32) -> Self {
        let a = 10f32.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f32::consts::PI * hz / sr;
        let cos = w0.cos();
        let sin = w0.sin();
        let alpha = sin / 2.0 * ((a + 1.0 / a) * (1.0 / s - 1.0) + 2.0).max(0.0).sqrt();
        let a0 = (a + 1.0) - (a - 1.0) * cos + 2.0 * a.sqrt() * alpha;
        Self {
            b0: a * ((a + 1.0) + (a - 1.0) * cos + 2.0 * a.sqrt() * alpha) / a0,
            b1: -2.0 * a * ((a - 1.0) + (a + 1.0) * cos) / a0,
            b2: a * ((a + 1.0) + (a - 1.0) * cos - 2.0 * a.sqrt() * alpha) / a0,
            a1: 2.0 * ((a - 1.0) - (a + 1.0) * cos) / a0,
            a2: ((a + 1.0) - (a - 1.0) * cos - 2.0 * a.sqrt() * alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Short-term K-weighted loudness over the trailing [`SHORT_TERM_S`] seconds.
pub struct LoudnessMeter {
    sr: u32,
    hp: Biquad,
    shelf: Biquad,
    /// Squared K-weighted samples, one window long (reallocated only when the rate changes).
    ring: Vec<f32>,
    pos: usize,
    sum: f64,
    filled: usize,
    db: f32,
}

impl LoudnessMeter {
    pub fn new() -> Self {
        Self {
            sr: 0,
            hp: Biquad::default(),
            shelf: Biquad::default(),
            ring: Vec::new(),
            pos: 0,
            sum: 0.0,
            filled: 0,
            db: FLOOR_DB,
        }
    }

    /// The last reported level, in dBFS.
    pub fn db(&self) -> f32 {
        self.db
    }

    /// Forget everything: used at a song marker or after silence.
    pub fn reset(&mut self) {
        self.ring.iter_mut().for_each(|s| *s = 0.0);
        self.pos = 0;
        self.sum = 0.0;
        self.filled = 0;
        self.db = FLOOR_DB;
        self.hp.reset();
        self.shelf.reset();
    }

    fn set_rate(&mut self, sr: u32) {
        self.sr = sr;
        let rate = sr as f32;
        // K-weighting: remove the rumble, lift the presence range a touch, like a loudness meter.
        self.hp = Biquad::high_pass(rate, 38.0, 0.5);
        self.shelf = Biquad::high_shelf(rate, 1500.0, 4.0, 1.0);
        let len = (SHORT_TERM_S * rate) as usize;
        self.ring = vec![0.0; len.max(1)];
        self.reset();
    }

    /// Filter one block and return the new short-term level in dBFS.
    pub fn push(&mut self, mono: &[f32], sample_rate: u32) -> f32 {
        if mono.is_empty() || sample_rate == 0 {
            return self.db;
        }
        if sample_rate != self.sr || self.ring.is_empty() {
            self.set_rate(sample_rate);
        }
        let cap = self.ring.len();
        for &x in mono {
            let k = self.shelf.process(self.hp.process(x));
            let sq = (k as f64) * (k as f64);
            let old = self.ring[self.pos] as f64;
            self.ring[self.pos] = sq as f32;
            self.sum += sq - old;
            self.pos += 1;
            if self.pos == cap {
                self.pos = 0;
            }
            if self.filled < cap {
                self.filled += 1;
            }
        }
        let mean = if self.filled > 0 {
            (self.sum / self.filled as f64).max(0.0)
        } else {
            0.0
        };
        self.db = if mean > 0.0 {
            ((-0.691 + 10.0 * mean.log10()) as f32).max(FLOOR_DB)
        } else {
            FLOOR_DB
        };
        self.db
    }
}

impl Default for LoudnessMeter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(frames: usize, hz: f32, amp: f32, sr: u32) -> Vec<f32> {
        (0..frames)
            .map(|i| amp * (2.0 * std::f32::consts::PI * hz * i as f32 / sr as f32).sin())
            .collect()
    }

    #[test]
    fn digital_silence_is_the_floor() {
        let mut m = LoudnessMeter::new();
        let db = m.push(&vec![0.0; 256], 48_000);
        assert_eq!(db, FLOOR_DB);
    }

    #[test]
    fn a_half_scale_sine_meters_near_minus_nine_dbfs() {
        let mut m = LoudnessMeter::new();
        let block = sine(256, 1000.0, 0.5, 48_000);
        let mut db = FLOOR_DB;
        // 1.5 s is half the loudness window: the average has converged on the tone by then.
        for _ in 0..(48_000 * 3 / 256 + 4) {
            db = m.push(&block, 48_000);
        }
        assert!(
            (-16.0..=-2.0).contains(&db),
            "0.5 amplitude 1 kHz sine metered {db} dB"
        );
    }

    #[test]
    fn louder_is_louder() {
        let mut quiet = LoudnessMeter::new();
        let mut loud = LoudnessMeter::new();
        let q = sine(256, 1000.0, 0.1, 48_000);
        let l = sine(256, 1000.0, 0.5, 48_000);
        let mut dq = 0.0;
        let mut dl = 0.0;
        for _ in 0..600 {
            dq = quiet.push(&q, 48_000);
            dl = loud.push(&l, 48_000);
        }
        assert!(
            dl > dq + 10.0,
            "5x amplitude should be >14 dB: {dq} vs {dl}"
        );
    }
}
