//! Causal per-block spectral features: FFT, low/mid/high bands, 24 mel bands, onset, brightness.
//!
//! One call per audio block, on the trailing `fft_size` samples, so the feature vector for a
//! block only ever depends on audio that has already arrived. Everything is allocated in
//! [`FeatureExtractor::new`]; [`FeatureExtractor::process`] allocates nothing once the sample
//! rate is known (the band edges are rebuilt only when the rate changes).

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// Number of log-spaced mel bands, 30 Hz .. 16 kHz.
pub const MEL_BANDS: usize = 24;
/// Band edges in Hz: low (kick/bass), mid (body/vocal), high (hats/air).
pub const LOW_HZ: f32 = 20.0;
pub const LOW_HIGH_HZ: f32 = 150.0;
pub const MID_HIGH_HZ: f32 = 2500.0;
pub const HIGH_HIGH_HZ: f32 = 16_000.0;
pub const MEL_LOW_HZ: f32 = 30.0;
pub const MEL_HIGH_HZ: f32 = 16_000.0;
/// Brightness is the spectral centroid divided by this and clamped to 0..1.
pub const BRIGHTNESS_HZ: f32 = 8000.0;
/// Log magnitudes are floored here (-80 dBFS) so a silent bin does not produce an infinite
/// spectral-flux step.
pub const LOG_FLOOR: f32 = 1e-4;
/// A block whose onset report is below this (mean dB per bin above the local mean) is not an
/// onset peak. A kick lands around 0.5, a steady tone and steady noise stay far below.
pub const ONSET_PEAK_FLOOR: f32 = 0.1;
/// Trailing window for the onset local mean, in seconds.
const ONSET_MEAN_SECONDS: f32 = 0.3;
/// Slots in the flux history; the window is `ONSET_MEAN_SECONDS` worth of blocks, at most this.
const FLUX_RING: usize = 64;

/// The features of one block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureValues {
    /// Mean linear power in 20-150, 150-2500 and 2500-16000 Hz.
    pub power: [f32; 3],
    /// 24 log-spaced bands, `10 log10(power)` mapped from -80 dB .. 0 dB to 0 .. 1.
    pub mel: [f32; MEL_BANDS],
    /// Positive spectral flux above the trailing local mean: mean dB per bin. Typical transients
    /// are 0.2 .. 1.0, steady material stays near 0.
    pub onset: f32,
    /// This block's positive spectral flux, before the local mean is removed.
    pub flux: f32,
    /// Spectral centroid / [`BRIGHTNESS_HZ`], clamped to 0 .. 1.
    pub brightness: f32,
    /// RMS of the block, linear.
    pub rms: f32,
}

impl Default for FeatureValues {
    fn default() -> Self {
        Self {
            power: [0.0; 3],
            mel: [0.0; MEL_BANDS],
            onset: 0.0,
            flux: 0.0,
            brightness: 0.0,
            rms: 0.0,
        }
    }
}

/// A windowed FFT plus the band layout, reused for every block.
pub struct FeatureExtractor {
    n: usize,
    half: usize,
    window: Vec<f32>,
    /// Trailing `n` samples in arrival order (circular).
    ring: Vec<f32>,
    ring_pos: usize,
    buf: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    fft: Arc<dyn Fft<f32>>,
    /// Previous block's log magnitudes, for spectral flux.
    prev_log: Vec<f32>,
    band_bin: Vec<u8>,
    mel_bin: Vec<u8>,
    sr: u32,
    /// Raw flux history for the trailing mean.
    flux_ring: Vec<f32>,
    flux_pos: usize,
    flux_sum: f32,
    flux_count: usize,
}

impl FeatureExtractor {
    pub fn new(fft_size: usize) -> Self {
        let n = {
            let n = fft_size.max(64);
            if n.is_multiple_of(2) { n } else { n + 1 }
        };
        let half = n / 2;
        let window: Vec<f32> = (0..n)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
            .collect();
        let fft = FftPlanner::new().plan_fft_forward(n);
        Self {
            n,
            half,
            window,
            ring: vec![0.0; n],
            ring_pos: 0,
            buf: vec![Complex::new(0.0, 0.0); n],
            scratch: vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()],
            fft,
            prev_log: vec![-80.0; half],
            band_bin: vec![u8::MAX; half],
            mel_bin: vec![u8::MAX; half],
            sr: 0,
            flux_ring: vec![0.0; FLUX_RING],
            flux_pos: 0,
            flux_sum: 0.0,
            flux_count: 0,
        }
    }

    pub fn fft_size(&self) -> usize {
        self.n
    }

    /// The band edges currently in use (low, mid, high as inclusive-exclusive bin ranges).
    #[cfg(test)]
    fn band_edges(&self) -> [(usize, usize); 3] {
        let build = |lo: f32, hi: f32| {
            let a = self.bin(lo);
            let b = self.bin(hi).max(a + 1).min(self.half);
            (a, b)
        };
        [
            build(LOW_HZ, LOW_HIGH_HZ),
            build(LOW_HIGH_HZ, MID_HIGH_HZ),
            build(MID_HIGH_HZ, HIGH_HIGH_HZ),
        ]
    }

    fn bin(&self, hz: f32) -> usize {
        let b = (hz * self.n as f32 / self.sr as f32).round() as usize;
        b.clamp(1, self.half - 1)
    }

    /// Rebuild the bin -> band maps. Only runs when the sample rate changes.
    fn set_rate(&mut self, sr: u32) {
        self.sr = sr;
        let bands = {
            let mut v = [(0usize, 1usize); 3];
            let pairs = [
                (LOW_HZ, LOW_HIGH_HZ),
                (LOW_HIGH_HZ, MID_HIGH_HZ),
                (MID_HIGH_HZ, HIGH_HIGH_HZ),
            ];
            for (i, (lo, hi)) in pairs.iter().enumerate() {
                let a = self.bin(*lo);
                let b = self.bin(*hi).max(a + 1).min(self.half);
                v[i] = (a, b);
            }
            v
        };
        for slot in self.band_bin.iter_mut() {
            *slot = u8::MAX;
        }
        for slot in self.mel_bin.iter_mut() {
            *slot = u8::MAX;
        }
        for (i, (lo, hi)) in bands.iter().enumerate() {
            for k in *lo..*hi {
                if self.band_bin[k] == u8::MAX {
                    self.band_bin[k] = i as u8;
                }
            }
        }

        // 24 log-spaced bands from 30 Hz to 16 kHz, clamped to the bins this FFT has.
        let nyquist = sr as f32 * 0.5;
        let top = MEL_HIGH_HZ.min(nyquist * 0.99);
        let ratio = (top / MEL_LOW_HZ).max(1.0);
        let mut prev = 1usize;
        for m in 0..MEL_BANDS {
            let lo = MEL_LOW_HZ * ratio.powf(m as f32 / MEL_BANDS as f32);
            let hi = MEL_LOW_HZ * ratio.powf((m + 1) as f32 / MEL_BANDS as f32);
            let a = self.bin(lo).max(prev).min(self.half - 1);
            let b = self.bin(hi).max(a + 1).min(self.half);
            for k in a..b {
                if self.mel_bin[k] == u8::MAX {
                    self.mel_bin[k] = m as u8;
                }
            }
            prev = b;
        }
    }

    fn push_samples(&mut self, mono: &[f32]) {
        let n = self.n;
        for &s in mono {
            self.ring[self.ring_pos] = s;
            self.ring_pos += 1;
            if self.ring_pos == n {
                self.ring_pos = 0;
            }
        }
    }

    /// Analyse the trailing window. `mono` is the block that was just pushed and its length is the
    /// hop the onset local mean is measured over.
    pub fn process(&mut self, mono: &[f32], sample_rate: u32) -> FeatureValues {
        if mono.is_empty() || sample_rate == 0 {
            return FeatureValues::default();
        }
        if sample_rate != self.sr {
            self.set_rate(sample_rate);
        }
        self.push_samples(mono);
        let hop = mono.len();

        let n = self.n;
        let half = self.half;
        for i in 0..n {
            let idx = (self.ring_pos + i) % n;
            self.buf[i] = Complex::new(self.ring[idx] * self.window[i], 0.0);
        }
        self.fft
            .process_with_scratch(&mut self.buf, &mut self.scratch);

        let scale = 1.0 / n as f32;
        let bin_hz = sample_rate as f32 / n as f32;
        let mut band_sum = [0.0f64; 3];
        let mut band_cnt = [0u32; 3];
        let mut mel_sum = [0.0f64; MEL_BANDS];
        let mut mel_cnt = [0u32; MEL_BANDS];
        let mut flux = 0.0f64;
        let mut centroid_num = 0.0f64;
        let mut centroid_den = 0.0f64;

        for k in 1..half {
            let re = self.buf[k].re * scale;
            let im = self.buf[k].im * scale;
            let mag = (re * re + im * im).sqrt();
            let power = mag * mag;

            let b = self.band_bin[k];
            if b != u8::MAX {
                band_sum[b as usize] += power as f64;
                band_cnt[b as usize] += 1;
            }
            let m = self.mel_bin[k];
            if m != u8::MAX {
                mel_sum[m as usize] += power as f64;
                mel_cnt[m as usize] += 1;
            }

            centroid_num += (k as f32 * bin_hz * mag) as f64;
            centroid_den += mag as f64;

            let log_mag = 20.0 * (mag + LOG_FLOOR).log10();
            let d = log_mag - self.prev_log[k];
            if d > 0.0 {
                flux += d as f64;
            }
            self.prev_log[k] = log_mag;
        }

        let mut out = FeatureValues::default();
        for b in 0..3 {
            out.power[b] = if band_cnt[b] > 0 {
                (band_sum[b] / band_cnt[b] as f64) as f32
            } else {
                0.0
            };
        }
        for m in 0..MEL_BANDS {
            let mean_power = if mel_cnt[m] > 0 {
                mel_sum[m] / mel_cnt[m] as f64
            } else {
                0.0
            };
            let db = 10.0 * (mean_power as f32 + 1e-12).log10();
            out.mel[m] = ((db + 80.0) / 80.0).clamp(0.0, 1.0);
        }
        out.brightness = if centroid_den > 0.0 {
            ((centroid_num / centroid_den) as f32 / BRIGHTNESS_HZ).clamp(0.0, 1.0)
        } else {
            0.0
        };
        out.rms = {
            let sum: f64 = mono.iter().map(|&s| (s as f64) * (s as f64)).sum();
            (sum / mono.len() as f64).sqrt() as f32
        };

        let flux_mean = (flux / (half - 1) as f64) as f32;
        out.flux = flux_mean;
        out.onset = self.onset_report(flux_mean, sample_rate, hop);
        out
    }

    /// Positive flux above its own trailing mean. The peak of a transient lands in the block that
    /// caused it or the one after (measured: clicks, hats, kicks and noise bursts at a 256-frame
    /// hop peak 1-2 blocks after the transient enters the window), so nothing is delayed and the
    /// onset envelope stays on the transport grid.
    fn onset_report(&mut self, flux_mean: f32, sample_rate: u32, hop: usize) -> f32 {
        let cap = self.flux_ring.len();
        let warm = ((ONSET_MEAN_SECONDS * sample_rate as f32) / hop as f32).round() as usize;
        let warm = warm.clamp(4, cap);

        if self.flux_count == warm {
            self.flux_sum -= self.flux_ring[self.flux_pos];
        } else {
            self.flux_count += 1;
        }
        self.flux_ring[self.flux_pos] = flux_mean;
        self.flux_sum += flux_mean;
        self.flux_pos = (self.flux_pos + 1) % warm;
        let mean = self.flux_sum / self.flux_count as f32;
        if self.flux_count >= warm {
            (flux_mean - mean).max(0.0)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_edges_are_ordered_and_inside_the_spectrum() {
        let mut fx = FeatureExtractor::new(2048);
        fx.set_rate(48_000);
        let e = fx.band_edges();
        assert!(e[0].0 < e[0].1, "{e:?}");
        assert!(e[0].1 <= e[1].0, "{e:?}");
        assert!(e[1].1 <= e[2].0, "{e:?}");
        assert!(e[2].1 <= fx.half, "{e:?}");
        // 20-150 Hz at 23.4 Hz per bin, 2.5-16 kHz over the top half of the spectrum.
        assert!((e[0].0 as f32 * 23.4 - 20.0).abs() < 30.0, "{e:?}");
        assert!(e[2].1 > fx.half / 2, "{e:?}");
    }

    #[test]
    fn mel_bands_are_monotonic() {
        let mut fx = FeatureExtractor::new(2048);
        fx.set_rate(48_000);
        let mut last = 0u8;
        let mut seen = 0;
        for k in 1..fx.half {
            let m = fx.mel_bin[k];
            if m != u8::MAX {
                assert!(m >= last, "bin {k} jumps back");
                last = m;
                seen += 1;
            }
        }
        assert_eq!(last as usize, MEL_BANDS - 1, "the top band is missing");
        assert!(seen > fx.half / 2, "most bins should carry a mel band");
    }

    #[test]
    fn fft_size_stays_even() {
        assert_eq!(FeatureExtractor::new(2047).fft_size(), 2048);
        assert_eq!(FeatureExtractor::new(1).fft_size(), 64);
    }
}
