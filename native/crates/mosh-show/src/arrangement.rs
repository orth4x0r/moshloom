//! The arrangement: one timeline in ppq per show, and the lanes that place clips on it.
//!
//! Positions are beats (ppq), so tempo automation changes nothing here. Bars come from the
//! time-signature map, never from `ppq / 4`.

use mosh_core::{ArrClip, ClipId, SigChange, SongMarker};

/// Beats per bar at `ppq`, from the time-signature map (`4/4` when the map says nothing).
pub fn bar_len_at(sig_map: &[SigChange], ppq: f64) -> f64 {
    const DEFAULT: f64 = 4.0;
    let mut sig = SigChange { ppq: f64::NEG_INFINITY, num: 4, den: 4 };
    for change in sig_map {
        if change.ppq <= ppq + 1e-9 && change.ppq >= sig.ppq {
            sig = *change;
        }
    }
    if sig.den == 0 || sig.num == 0 {
        return DEFAULT;
    }
    let beats = (sig.num as f64) * 4.0 / (sig.den as f64);
    if beats > 0.0 { beats } else { DEFAULT }
}

/// The clip on `lane` at `ppq`: the half-open range `[start_ppq, end_ppq)` contains it.
///
/// When placements overlap the one that starts last wins, so the answer does not depend on the
/// order of the lane. Nothing is remembered between calls: a jump, a loop or a scrub is just
/// another position.
pub fn clip_at(lane: &[ArrClip], ppq: f64) -> Option<ClipId> {
    lane.iter()
        .filter(|c| c.start_ppq <= ppq && ppq < c.end_ppq)
        .max_by(|a, b| a.start_ppq.total_cmp(&b.start_ppq))
        .map(|c| c.clip)
}

/// The song marker the playhead is in: the last one that starts at or before `ppq`.
pub fn song_at(songs: &[SongMarker], ppq: f64) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, song) in songs.iter().enumerate() {
        if song.start_ppq > ppq {
            continue;
        }
        if best.is_none_or(|b| songs[b].start_ppq <= song.start_ppq) {
            best = Some(i);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_sig_map_is_four_four() {
        assert_eq!(bar_len_at(&[], 0.0), 4.0);
        assert_eq!(bar_len_at(&[], 123.0), 4.0);
    }

    #[test]
    fn bar_len_follows_the_sig_map() {
        let map = [
            SigChange { ppq: 0.0, num: 4, den: 4 },
            SigChange { ppq: 32.0, num: 3, den: 4 },
            SigChange { ppq: 64.0, num: 6, den: 8 },
        ];
        assert_eq!(bar_len_at(&map, 0.0), 4.0);
        assert_eq!(bar_len_at(&map, 31.9), 4.0);
        assert_eq!(bar_len_at(&map, 32.0), 3.0);
        assert_eq!(bar_len_at(&map, 70.0), 3.0);
        // Before the map's first change nothing is known, so the 4/4 default stands.
        assert_eq!(bar_len_at(&map[1..], 0.0), 4.0);
    }

    #[test]
    fn clip_at_is_half_open() {
        let lane = [
            ArrClip { clip: 1, start_ppq: 0.0, end_ppq: 8.0 },
            ArrClip { clip: 2, start_ppq: 8.0, end_ppq: 16.0 },
        ];
        assert_eq!(clip_at(&lane, 0.0), Some(1));
        assert_eq!(clip_at(&lane, 7.999), Some(1));
        assert_eq!(clip_at(&lane, 8.0), Some(2));
        assert_eq!(clip_at(&lane, 16.0), None);
        assert_eq!(clip_at(&lane, -1.0), None);
    }

    #[test]
    fn song_at_walks_the_markers() {
        let songs = [SongMarker { name: "a".into(), start_ppq: 0.0 }, SongMarker { name: "b".into(), start_ppq: 64.0 }];
        assert_eq!(song_at(&songs, -1.0), None);
        assert_eq!(song_at(&songs, 0.0), Some(0));
        assert_eq!(song_at(&songs, 63.9), Some(0));
        assert_eq!(song_at(&songs, 64.0), Some(1));
    }
}
