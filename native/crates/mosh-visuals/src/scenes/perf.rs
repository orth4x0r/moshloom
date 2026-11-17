//! What the two hand-written scenes cost, and what they look like next to the bodies they replace.
//!
//! Scene 02 (`numogram`) and scene 15 (`swarm`) were the only two scenes that could not meet the
//! design's GPU budget (3.4-9.5 and 1.6-4.5 ms per pass at 960x540 on the integrated GPU, against
//! 0.10-0.32 ms for the other fourteen). Both now come from a hand-written body that does the same
//! maths in a cheaper arrangement, and both old generations are still compiled here.
//!
//! Two tests, both offscreen:
//!
//! * [`hand_written_scenes_keep_their_look`] renders the old and the new generation with identical
//!   uniforms at 3 intensity levels x 3 seeds and reports the mean absolute pixel difference.
//! * [`hand_written_scenes_meet_the_frame_budget`] reports GPU milliseconds per pass at 960x540
//!   for both generations, old and new measured alternately so a change of GPU clock moves both
//!   numbers together.
//!
//! # Why the timing test asserts a ratio and not a number
//!
//! The design's budget for one scene pass is under 1 ms at 960x540, and the printed numbers are
//! what the PR body quotes. The *assertion* is the ratio to the generation being replaced, taken
//! in the same loop: a shared laptop moves this GPU's clock by 2x between runs (the same web body
//! pass measured 3.30 ms and then 6.84 ms an hour later), so an absolute milliseconds assertion
//! would test the machine's power state rather than the shader. Budget measurements belong to a
//! coordinator run on a quiet reference machine (`AGENTS.md`, native lanes).

use super::*;
use mosh_render::{GpuPreference, GpuTimings};

/// The size the likeness comparison renders at: small, and enough for the glyph and the streaks.
const W: u32 = 160;
const H: u32 = 90;
/// The size the budget is measured at, from the design's numbers.
const TIME_W: u32 = 960;
const TIME_H: u32 = 540;
const SAMPLES: usize = 20;

fn gpu() -> Gpu {
    let gpu = Gpu::new(GpuPreference::Integrated)
        .or_else(|_| Gpu::new(GpuPreference::ByName(String::new())))
        .expect("a Vulkan device");
    println!(
        "adapter: {} | validation layers: {}",
        gpu.adapter_name(),
        if gpu.validation_enabled() {
            "on"
        } else {
            "unavailable"
        }
    );
    gpu
}

fn assert_no_validation(gpu: &Gpu) {
    let messages = gpu.validation_messages();
    assert!(
        messages.is_empty(),
        "{} validation message(s):\n{}",
        messages.len(),
        messages.join("\n")
    );
}

/// Three intensity levels: a quiet bar, a mid bar and a drop (the same frames `tests/scenes.rs`
/// renders the 16 scenes at).
fn frame_at(level: usize) -> AnalysisFrame {
    let (band, pulse): (f32, f32) = match level {
        0 => (0.18, 0.25),
        1 => (0.5, 0.6),
        _ => (0.9, 1.0),
    };
    AnalysisFrame {
        time_s: 12.0,
        ppq: 16.0,
        beat_phase: (1.0 - pulse).clamp(0.0, 1.0),
        bar_phase: 0.25,
        beat_index: 16,
        bar_index: 4,
        tempo_bpm: 120.0,
        playing: true,
        bands: [band, band * 0.8, band * 0.6],
        mel: std::array::from_fn(|k| band * (1.0 - k as f32 / 30.0)),
        onset: pulse,
        loudness: band,
        level: level as u8 + 1,
        intensity: band,
        ..AnalysisFrame::default()
    }
}

/// Three procedural seeds' worth of clip parameters: speed, density, hue, variant and palette.
fn params_for(seed: u64) -> ClipParams {
    let r = |k: u64| {
        ((seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(k * 1442695040888963407))
            >> 40) as f32
            / (1u64 << 24) as f32
    };
    ClipParams {
        p: [
            0.7 + r(1) * 0.8,
            0.3 + r(2) * 0.5,
            r(3),
            (r(4) * 3.0).floor(),
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        palette: match seed % 3 {
            0 => 0,
            1 => 3,
            _ => 6,
        },
        seed,
    }
}

fn pipeline(gpu: &Gpu, source: &str, name: &str, textures: u32) -> Pipeline {
    let words = compile(source, name);
    Pipeline::fullscreen(gpu, &words, LAYER_UNIFORM_BYTES, textures)
        .unwrap_or_else(|error| panic!("{name}: {error}"))
        .with_name(name)
}

/// The `palT` palette texture `Scenes` generates on its first frame.
fn palette(gpu: &Gpu) -> LayerTarget {
    let pal = LayerTarget::new(gpu, PAL_W, PAL_H);
    let words = compile(PALETTE, "palette");
    let pipeline = Pipeline::fullscreen(gpu, &words, 0, 0)
        .expect("palette pipeline")
        .with_name("palette");
    let mut recorder = FrameRecorder::new(gpu);
    recorder
        .draw(&pipeline, &pal, &[], &[])
        .expect("draw the palette");
    gpu.wait_frame(recorder.submit());
    pal
}

/// The textures every scene binds, including the palette the swarm's colour comes from.
struct Tables {
    pal: LayerTarget,
    ca: LayerTarget,
    mel: LayerTarget,
    rd: LayerTarget,
    media: LayerTarget,
}

impl Tables {
    fn new(gpu: &Gpu) -> Tables {
        Tables {
            pal: palette(gpu),
            ca: LayerTarget::new(gpu, 1, 1),
            mel: LayerTarget::new(gpu, MEL_W, MEL_H),
            rd: LayerTarget::new(gpu, RD_W, RD_H),
            media: LayerTarget::new(gpu, 1, 1),
        }
    }

    fn inputs(&self) -> Vec<&LayerTarget> {
        vec![&self.pal, &self.ca, &self.mel, &self.rd, &self.media]
    }
}

/// One generation of the swarm: the streaks, the particle table, and (second generation) the tile
/// buckets between them.
struct Swarm {
    scene: Pipeline,
    positions: Pipeline,
    buckets: Option<Pipeline>,
    table: LayerTarget,
    tiles: LayerTarget,
}

impl Swarm {
    fn old(gpu: &Gpu) -> Swarm {
        Swarm {
            scene: pipeline(gpu, &extracted_source("swarm"), "swarm v1", INPUTS + 1),
            positions: pipeline(gpu, SWARM_POS, "swarm positions v1", 0),
            buckets: None,
            table: LayerTarget::new(gpu, SWARM_PARTICLES, 1),
            tiles: LayerTarget::new(gpu, 1, 1),
        }
    }

    fn new(gpu: &Gpu) -> Swarm {
        Swarm {
            scene: pipeline(gpu, &scene_source("swarm"), "swarm v2", INPUTS + 2),
            positions: pipeline(gpu, SWARM_POS_FAST, "swarm positions v2", 1),
            buckets: Some(pipeline(
                gpu,
                SWARM_BUCKETS,
                "swarm buckets v2",
                1,
            )),
            table: LayerTarget::new(gpu, SWARM_PARTICLES, 3),
            tiles: LayerTarget::new(
                gpu,
                SWARM_TILES_X * SWARM_TILES_Y,
                1 + SWARM_BUCKET_CAPACITY,
            ),
        }
    }

    /// One swarm frame: the table, the buckets, then the streaks.
    fn draw(
        &self,
        gpu: &Gpu,
        tables: &Tables,
        dst: &LayerTarget,
        uniforms: &[u8],
    ) -> GpuTimings {
        let mut recorder = FrameRecorder::new(gpu);
        let position_inputs: &[&LayerTarget] = if self.buckets.is_some() {
            &[&tables.pal]
        } else {
            &[]
        };
        recorder
            .draw(&self.positions, &self.table, uniforms, position_inputs)
            .expect("particle table");
        if let Some(buckets) = &self.buckets {
            recorder
                .draw(buckets, &self.tiles, uniforms, &[&self.table])
                .expect("tile buckets");
        }
        let mut inputs = tables.inputs();
        inputs.push(&self.table);
        if self.buckets.is_some() {
            inputs.push(&self.tiles);
        }
        recorder
            .draw(&self.scene, dst, uniforms, &inputs)
            .expect("swarm streaks");
        gpu.wait_frame(recorder.submit())
    }
}

fn draw(
    gpu: &Gpu,
    pipeline: &Pipeline,
    dst: &LayerTarget,
    uniforms: &[u8],
    inputs: &[&LayerTarget],
) -> GpuTimings {
    let mut recorder = FrameRecorder::new(gpu);
    recorder
        .draw(pipeline, dst, uniforms, inputs)
        .unwrap_or_else(|error| panic!("{}: {error}", pipeline.name()));
    gpu.wait_frame(recorder.submit())
}

/// The GPU milliseconds of one named pass of a frame.
fn pass_ms(timings: &GpuTimings, name: &str) -> f32 {
    timings
        .per_pass_ms
        .iter()
        .find(|(pass, _)| pass == name)
        .map(|(_, ms)| *ms)
        .unwrap_or_else(|| panic!("no {name} pass: {:?}", timings.per_pass_ms))
}

/// Two warm-up frames, then `SAMPLES` frames; returns every named pass's samples as `(min,
/// median)` for each name asked for.
fn time_frames(
    mut frame: impl FnMut() -> GpuTimings,
    names: &[&str],
) -> Vec<(String, f32, f32)> {
    for _ in 0..2 {
        frame();
    }
    let mut samples: Vec<Vec<f32>> = vec![Vec::with_capacity(SAMPLES); names.len()];
    for _ in 0..SAMPLES {
        let timings = frame();
        for (index, name) in names.iter().enumerate() {
            samples[index].push(pass_ms(&timings, name));
        }
    }
    names
        .iter()
        .zip(samples.iter_mut())
        .map(|(name, samples)| {
            samples.sort_by(f32::total_cmp);
            (
                name.to_string(),
                samples[0],
                samples[samples.len() / 2].max(samples[0]),
            )
        })
        .collect()
}

/// Mean absolute per-channel difference, 0..255, over the three colour channels.
fn mean_difference(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len(), "read-backs of different sizes");
    let total: f32 = a
        .chunks(4)
        .zip(b.chunks(4))
        .map(|(left, right)| {
            (0..3)
                .map(|channel| (f32::from(left[channel]) - f32::from(right[channel])).abs())
                .sum::<f32>()
        })
        .sum();
    total / (a.len() / 4 * 3) as f32
}

/// Pixels that differ by more than `tolerance` in some colour channel.
fn differing_pixels(a: &[u8], b: &[u8], tolerance: i32) -> usize {
    assert_eq!(a.len(), b.len(), "read-backs of different sizes");
    a.chunks(4)
        .zip(b.chunks(4))
        .filter(|(left, right)| {
            (0..3).any(|channel| {
                (i32::from(left[channel]) - i32::from(right[channel])).abs() > tolerance
            })
        })
        .count()
}

#[test]
fn hand_written_scenes_keep_their_look() {
    let gpu = gpu();
    let tables = Tables::new(&gpu);
    let old_numogram = pipeline(&gpu, &extracted_source("numogram"), "numogram v1", INPUTS);
    let new_numogram = pipeline(&gpu, &scene_source("numogram"), "numogram v2", INPUTS);
    let old_swarm = Swarm::old(&gpu);
    let new_swarm = Swarm::new(&gpu);
    let old_picture = LayerTarget::new(&gpu, W, H);
    let new_picture = LayerTarget::new(&gpu, W, H);

    let mut worst = [0.0f32; 2];
    for (index, label) in ["numogram", "swarm"].into_iter().enumerate() {
        for level in 0..3 {
            for seed in 0..3u64 {
                let frame = frame_at(level);
                let params = params_for(seed);
                let t = 2.0 + seed as f32 * 0.7;
                let palette = palette_for(&params, Some(if index == 0 { 2 } else { 15 }));
                let uniforms = scene_uniforms(&frame, &params, palette, t, [W as f32, H as f32]);
                if index == 0 {
                    draw(
                        &gpu,
                        &old_numogram,
                        &old_picture,
                        uniforms.as_bytes(),
                        &tables.inputs(),
                    );
                    draw(
                        &gpu,
                        &new_numogram,
                        &new_picture,
                        uniforms.as_bytes(),
                        &tables.inputs(),
                    );
                } else {
                    old_swarm.draw(&gpu, &tables, &old_picture, uniforms.as_bytes());
                    new_swarm.draw(&gpu, &tables, &new_picture, uniforms.as_bytes());
                }
                let old_pixels = gpu.read_back(&old_picture);
                let new_pixels = gpu.read_back(&new_picture);
                let mean = mean_difference(&old_pixels, &new_pixels);
                let differing = differing_pixels(&old_pixels, &new_pixels, 2);
                worst[index] = worst[index].max(mean);
                println!(
                    "{label:<8} level {level} seed {seed}: mean |diff| {mean:.3} / 255, \
                     {differing} of {} pixels differ by more than 2",
                    W * H
                );
            }
        }
    }
    println!(
        "hand-written scenes: worst mean absolute difference, numogram {:.3} / 255, swarm {:.3} \
         / 255",
        worst[0], worst[1]
    );

    // The 160x90 comparison above has 10x10-pixel tiles, so most of its tiles are saturated and it
    // exercises the "scan them all" path. One larger frame exercises the bucket lists themselves,
    // where a pixel sees only its tile's candidates.
    let (bw, bh) = (480u32, 270u32);
    let old_big = LayerTarget::new(&gpu, bw, bh);
    let new_big = LayerTarget::new(&gpu, bw, bh);
    let frame = frame_at(2);
    let params = params_for(2);
    let uniforms = scene_uniforms(
        &frame,
        &params,
        palette_for(&params, Some(15)),
        3.4,
        [bw as f32, bh as f32],
    );
    old_swarm.draw(&gpu, &tables, &old_big, uniforms.as_bytes());
    new_swarm.draw(&gpu, &tables, &new_big, uniforms.as_bytes());
    let old_pixels = gpu.read_back(&old_big);
    let new_pixels = gpu.read_back(&new_big);
    let mean = mean_difference(&old_pixels, &new_pixels);
    let differing = differing_pixels(&old_pixels, &new_pixels, 2);
    println!(
        "swarm    {bw}x{bh} (tile lists): mean |diff| {mean:.3} / 255, {differing} of {} pixels \
         differ by more than 2",
        bw * bh
    );
    worst[1] = worst[1].max(mean);

    assert!(
        worst[0] < 2.0 && worst[1] < 2.0,
        "the hand-written bodies must keep the look: worst mean difference {worst:?} of 255"
    );
    assert_no_validation(&gpu);
}

#[test]
fn hand_written_scenes_meet_the_frame_budget() {
    let gpu = gpu();
    let tables = Tables::new(&gpu);
    let old_numogram = pipeline(&gpu, &extracted_source("numogram"), "numogram v1", INPUTS);
    let new_numogram = pipeline(&gpu, &scene_source("numogram"), "numogram v2", INPUTS);
    let old_swarm = Swarm::old(&gpu);
    let new_swarm = Swarm::new(&gpu);
    let picture = LayerTarget::new(&gpu, TIME_W, TIME_H);
    let frame = frame_at(2);
    let params = params_for(2);
    let res = [TIME_W as f32, TIME_H as f32];
    let numogram = scene_uniforms(&frame, &params, palette_for(&params, Some(2)), 4.0, res);
    let swarm = scene_uniforms(&frame, &params, palette_for(&params, Some(15)), 4.0, res);

    // Old and new alternate in one loop, so a change of GPU clock moves both numbers together.
    let mut rounds: Vec<(String, f32, f32)> = Vec::new();
    for _ in 0..2 {
        rounds.extend(time_frames(
            || {
                draw(
                    &gpu,
                    &old_numogram,
                    &picture,
                    numogram.as_bytes(),
                    &tables.inputs(),
                )
            },
            &["numogram v1"],
        ));
        rounds.extend(time_frames(
            || {
                draw(
                    &gpu,
                    &new_numogram,
                    &picture,
                    numogram.as_bytes(),
                    &tables.inputs(),
                )
            },
            &["numogram v2"],
        ));
        rounds.extend(time_frames(
            || old_swarm.draw(&gpu, &tables, &picture, swarm.as_bytes()),
            &["swarm v1"],
        ));
        rounds.extend(time_frames(
            || new_swarm.draw(&gpu, &tables, &picture, swarm.as_bytes()),
            &["swarm v2", "swarm positions v2", "swarm buckets v2"],
        ));
    }
    // The fastest round per pass: the clock only ever drops under load, so the minimum is the
    // pass's own cost.
    let mut best: Vec<(String, f32, f32)> = Vec::new();
    for (name, min, median) in rounds {
        match best.iter_mut().find(|(existing, _, _)| *existing == name) {
            Some(slot) => {
                if min < slot.1 {
                    *slot = (name, min, median);
                }
            }
            None => best.push((name, min, median)),
        }
    }
    let of = |name: &str| {
        let (_, min, median) = best
            .iter()
            .find(|(existing, _, _)| existing == name)
            .unwrap_or_else(|| panic!("no {name} timing: {best:?}"));
        (*min, *median)
    };
    let (numogram_v1_min, _) = of("numogram v1");
    let (numogram_v2_min, _) = of("numogram v2");
    let (swarm_v1_min, _) = of("swarm v1");
    let (swarm_v2_min, swarm_v2_median) = of("swarm v2");
    let (positions_min, positions_median) = of("swarm positions v2");
    let (buckets_min, buckets_median) = of("swarm buckets v2");

    println!(
        "GPU ms per scene pass at {TIME_W}x{TIME_H} on {}",
        gpu.adapter_name()
    );
    println!("  {:<22} {:>7} {:>7}", "pass", "min", "median");
    for (name, min, median) in &best {
        println!("  {name:<22} {min:>7.3} {median:>7.3}");
    }
    let swarm_new_min = swarm_v2_min + positions_min + buckets_min;
    println!(
        "  {:<22} {swarm_new_min:>7.3} {:>7.3}",
        "swarm frame (new)",
        swarm_v2_median + positions_median + buckets_median
    );
    println!(
        "  targets: one scene pass < 1.000 ms at {TIME_W}x{TIME_H}; measured min {numogram_v2_min:.3} \
         (numogram) and {swarm_new_min:.3} (swarm, all three passes)"
    );

    // The ratio is what holds on any machine; see the module docs.
    let numogram_ratio = numogram_v1_min / numogram_v2_min;
    let swarm_ratio = swarm_v1_min / swarm_new_min;
    println!(
        "  speed-up at the same clock: numogram {numogram_ratio:.2}x, swarm (all passes) \
         {swarm_ratio:.2}x"
    );
    assert!(
        numogram_ratio > 2.0,
        "the numogram must cost less than half what the web body costs, got {numogram_ratio:.2}x"
    );
    assert!(
        swarm_ratio > 2.0,
        "the swarm must cost less than half what the first generation costs, got {swarm_ratio:.2}x"
    );
    assert_no_validation(&gpu);
}
