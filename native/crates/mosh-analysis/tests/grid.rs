//! Task 1.3 (N-features): Ableton's beat grid — phases, indices and bar boundaries.
//!
//! Musical time is ppq (quarter-note beats). Bars come from `bar_start_ppq` and the time
//! signature, never from `ppq / 4`, and a playhead jump is never mistaken for a completed bar.

use mosh_analysis::grid::BeatGrid;
use mosh_core::Transport;

fn transport(ppq: f64, bar_start_ppq: f64, num: u16, den: u16) -> Transport {
    Transport {
        tempo_bpm: 120.0,
        ppq,
        bar_start_ppq,
        sig_num: num,
        sig_den: den,
        playing: true,
        sample_pos: (ppq * 24_000.0) as i64,
    }
}

#[test]
fn beat_and_bar_phase_from_transport() {
    let mut g = BeatGrid::new(8);
    let tick = g.update(&transport(6.5, 4.0, 4, 4));
    assert!((tick.beat_phase - 0.5).abs() < 1e-6, "{tick:?}");
    assert!((tick.bar_phase - 0.625).abs() < 1e-6, "{tick:?}");
    assert_eq!(tick.beat_index, 6);
    assert_eq!(tick.bar_index, 0, "the first transport opens bar 0");
}

#[test]
fn bar_length_follows_signature_denominator() {
    let mut g = BeatGrid::new(8);
    // 6/8: the bar is three quarter notes long, the beat is an eighth note.
    let tick = g.update(&transport(1.25, 0.0, 6, 8));
    assert!(
        (tick.beat_phase - 0.5).abs() < 1e-6,
        "1.25 ppq is the middle of the second eighth note: {tick:?}"
    );
    assert!((tick.bar_phase - 1.25 / 3.0).abs() < 1e-6, "{tick:?}");
    assert_eq!(tick.beat_index, 2);
}

#[test]
fn bar_index_follows_time_signature_change() {
    let mut g = BeatGrid::new(8);
    // Eight bars of 4/4 (ppq 0 .. 32), then 3/4 (each bar three beats).
    let mut edges: Vec<(f64, u16, u16)> = (0..8).map(|b| (b as f64 * 4.0, 4, 4)).collect();
    edges.extend((0..4).map(|b| (32.0 + b as f64 * 3.0, 3, 4)));

    let first = g.update(&transport(edges[0].0, edges[0].0, edges[0].1, edges[0].2));
    assert!(!first.bar_started, "the first bar is not a boundary event");
    assert!(first.completed.is_none());

    let mut closed = Vec::new();
    for (start, num, den) in edges.iter().copied().skip(1) {
        let tick = g.update(&transport(start, start, num, den));
        assert!(!tick.jumped, "a bar line is not a jump: {tick:?}");
        assert!(tick.bar_started, "bar line at ppq {start}: {tick:?}");
        closed.push(tick.completed.expect("a bar line closes the previous bar"));
        assert_eq!(tick.bar_index, closed.len() as i64);
    }

    assert_eq!(closed.len(), 11);
    for (i, bar) in closed.iter().enumerate() {
        assert_eq!(bar.bar_index, i as i64);
        assert_eq!(
            bar.start_ppq, edges[i].0,
            "bar {i} must start where Live says it does"
        );
        let expected_beats = if i < 8 { 4 } else { 3 };
        assert_eq!(
            bar.beats,
            expected_beats,
            "bar {i} is in {} ({expected_beats} beats)",
            if i < 8 { "4/4" } else { "3/4" }
        );
    }
}

#[test]
fn ppq_jump_backward_is_not_a_bar() {
    let mut g = BeatGrid::new(8);
    g.update(&transport(60.0, 60.0, 4, 4));
    let back = g.update(&transport(16.0, 16.0, 4, 4));
    assert!(back.jumped, "a backward jump is a discontinuity: {back:?}");
    assert!(
        back.completed.is_none(),
        "the partial bar must not be reported as complete"
    );
    assert!(!back.bar_started, "a jump is not a bar line event");
    assert!(
        (back.bar_phase).abs() < 1e-6,
        "the new bar is at its start: {back:?}"
    );
}

#[test]
fn phrase_boundaries_count_bars() {
    let mut g = BeatGrid::new(8);
    g.update(&transport(0.0, 0.0, 4, 4));
    assert!(
        !g.is_phrase_boundary(0),
        "bar 0 is the start, not a phrase end"
    );
    for b in 1..8 {
        assert!(!g.is_phrase_boundary(b), "bar {b} is mid-phrase");
    }
    assert!(g.is_phrase_boundary(8), "bar 8 starts the second phrase");
    assert!(g.is_phrase_boundary(16));
    assert!(!g.is_phrase_boundary(17));

    // Eight closed bars later the counter says bar 8, which is the next phrase.
    let mut tick = g.update(&transport(4.0, 4.0, 4, 4));
    for b in 2..=8 {
        tick = g.update(&transport(b as f64 * 4.0, b as f64 * 4.0, 4, 4));
    }
    assert_eq!(tick.completed.expect("bar closes").bar_index, 7);
    assert_eq!(tick.bar_index, 8, "the bar that starts at ppq 32 is bar 8");
    assert!(g.is_phrase_boundary(tick.bar_index));
}

#[test]
fn tempo_is_reported_from_the_transport() {
    let mut g = BeatGrid::new(8);
    let mut t = transport(0.0, 0.0, 4, 4);
    t.tempo_bpm = 137.5;
    let tick = g.update(&t);
    assert!((tick.bpm - 137.5).abs() < 1e-6);
    assert!(tick.playing);
}
