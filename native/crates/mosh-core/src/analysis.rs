//! Output of the live analysis, consumed by the show logic and the renderer.

use bitflags::bitflags;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TransportSource {
    /// Exact position from the Moshloom Tap plugin.
    Plugin,
    /// Estimated by the tempo tracker (loopback mode).
    Tracked,
    /// No beat grid at all.
    #[default]
    None,
}

bitflags! {
    /// Events since the previous `take_frame`. OR-ed, never lost between frames.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
    pub struct EventFlags: u32 {
        const BEAT = 1;
        const BAR = 2;
        const PHRASE = 4;
        const BUILD_START = 8;
        const DROP = 16;
        const BREAKDOWN = 32;
        const LEVEL_UP = 64;
        const LEVEL_DOWN = 128;
        const SONG_START = 256;
    }
}

/// Everything the visuals react to, for one video frame.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct AnalysisFrame {
    pub time_s: f64,
    pub ppq: f64,
    /// 0..1 within the current beat.
    pub beat_phase: f32,
    /// 0..1 within the current bar.
    pub bar_phase: f32,
    pub beat_index: i64,
    pub bar_index: i64,
    pub tempo_bpm: f32,
    pub playing: bool,
    pub source: TransportSource,
    /// low / mid / high, normalised 0..1 against recent history.
    pub bands: [f32; 3],
    /// 24 mel bands, log magnitude.
    pub mel: [f32; 24],
    pub onset: f32,
    pub loudness_db: f32,
    /// Normalised 0..1 against recent history.
    pub loudness: f32,
    /// Spectral centroid, normalised 0..1.
    pub brightness: f32,
    /// Continuous intensity 0..1.
    pub intensity: f32,
    /// 1..=4, changes only on bar boundaries.
    pub level: u8,
    pub events: EventFlags,
}

/// Per-bar aggregate, produced by the feature stage and consumed by the intensity model.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct BarStats {
    pub bar_index: i64,
    pub start_ppq: f64,
    pub beats: u16,
    pub loudness_db: f32,
    pub loudness: f32,
    pub low: f32,
    pub onset_density: f32,
    pub brightness: f32,
}
