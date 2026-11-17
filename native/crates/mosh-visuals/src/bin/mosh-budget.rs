//! mosh-budget: the whole frame the design's GPU budget is written against, offscreen.
//!
//! `native/crates/mosh-app/src/bin/mosh-stress-project.rs` writes the heaviest project the budget
//! covers (three visualizer layers with datamosh, effects and fluid, plus a video overlay). The
//! app measures it through the stage's swapchain, which needs a window and a display; this binary
//! builds the same stack the app builds â€” `Visuals` at the layers' internal size, one `LayerSlot`
//! per layer, the `Compositor` and an `OutputImage` at the output size â€” drives a synthetic
//! analysis stream through it and prints where the GPU time goes, per pass.
//!
//!     cargo run --release -p mosh-visuals --bin mosh-budget
//!     mosh-budget --frames 300 --sizes 1920x1080,2560x1600 --layers 4
//!
//! What it measures:
//!
//! * the layer frame's passes, from `GpuTimings::per_pass_ms` (scene, mosh, effects, transition,
//!   the fluid solver and helper passes such as mel and reaction-diffusion), plus that frame's
//!   `frame_ms`;
//! * the compositor's own pass, from `Compositor::timings()`;
//! * render-thread CPU per frame: the recording half only (overlay upload, `draw_layer` per layer,
//!   `FrameRecorder::submit`), which is the same segment `shell.rs` calls `cpu_ms`. The app's
//!   number also carries the analyser and the show's update; those are measured elsewhere
//!   (`README.md`, "Measured so far") and are not part of the render path.
//! * device memory the renderer holds, from `Gpu::allocated_bytes()`: per layer target, total,
//!   and per layer.
//!
//! What it does not measure: the stage's present blit into the swapchain (one more full-screen
//! pass at the output size) and anything the window compositor or the driver does outside this
//! process. No window is opened, so this is not the on-battery budget number: it is the same work
//! minus presentation, which is what makes it repeatable on a machine another agent is using.

use std::collections::BTreeMap;
use std::time::Instant;

use mosh_core::{
    ActiveTransition, AnalysisFrame, ClipParams, EffectKind, EffectSlot, EventFlags, LayerState,
    TransitionKind,
};
use mosh_render::{Compositor, FrameRecorder, Gpu, GpuPreference, OutputImage};
use mosh_visuals::{LayerSlot, Visuals};

/// The stress project's layers, exactly as `mosh-stress-project` writes them: blend mode, opacity
/// and internal scale per layer.
const LAYERS: [(mosh_core::BlendMode, f32, f32); 4] = [
    (mosh_core::BlendMode::Normal, 1.0, 0.5),
    (mosh_core::BlendMode::Add, 0.7, 0.5),
    (mosh_core::BlendMode::Screen, 0.6, 0.5),
    (mosh_core::BlendMode::LumaKey, 0.8, 0.5),
];

/// The overlay video's decoded size: the app decodes the clip's own file (1920x1080 here).
const VIDEO: (u32, u32) = (1920, 1080);

fn main() {
    let mut frames = 300usize;
    let mut warmup = 30usize;
    let mut sizes = vec![(1920u32, 1080u32), (2560, 1600)];
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--frames" => frames = parse(&mut args, "--frames"),
            "--warmup" => warmup = parse(&mut args, "--warmup"),
            "--sizes" => {
                let list = args.next().unwrap_or_default();
                sizes = list
                    .split(',')
                    .filter_map(|size| {
                        let (w, h) = size.split_once('x')?;
                        Some((w.parse().ok()?, h.parse().ok()?))
                    })
                    .collect();
                assert!(!sizes.is_empty(), "--sizes wants WxH[,WxH...]");
            }
            other => panic!("unknown argument {other:?}"),
        }
    }

    let gpu = Gpu::new(GpuPreference::Integrated)
        .or_else(|_| Gpu::new(GpuPreference::ByName(String::new())))
        .expect("a Vulkan device");
    println!(
        "adapter: {} | validation: {} | timestamps: {} ns/bit | frames {frames} (+{warmup} warm-up)",
        gpu.adapter_name(),
        if gpu.validation_enabled() {
            "on"
        } else {
            "off (release)"
        },
        gpu.timestamp_period_ns()
    );

    for (width, height) in sizes {
        let report = run(&gpu, width, height, frames, warmup);
        report.print();
    }
}

fn parse(args: &mut impl Iterator<Item = String>, flag: &str) -> usize {
    args.next()
        .unwrap_or_else(|| panic!("{flag} wants a number"))
        .parse()
        .unwrap_or_else(|_| panic!("{flag} wants a number"))
}

/// One layer's state for the stress project: the visualizer, its effects, and its blend.
struct LayerSpec {
    visualizer: &'static str,
    effects: Vec<EffectSlot>,
    blend: mosh_core::BlendMode,
    opacity: f32,
    scale: f32,
}

impl LayerSpec {
    fn stress() -> Vec<LayerSpec> {
        let slot = |kind, amount| EffectSlot {
            kind,
            amount,
            param: 0.5,
        };
        let spec = |visualizer, effects, (blend, opacity, scale)| LayerSpec {
            visualizer,
            effects,
            blend,
            opacity,
            scale,
        };
        vec![
            spec(
                "tunnel",
                vec![
                    slot(EffectKind::Mosh, 0.8),
                    slot(EffectKind::Zoom, 0.5),
                    slot(EffectKind::Echo, 0.5),
                ],
                LAYERS[0],
            ),
            spec("fluid", vec![slot(EffectKind::Mirror, 0.6)], LAYERS[1]),
            spec(
                "numogram",
                vec![slot(EffectKind::Mosh, 0.6), slot(EffectKind::Streak, 0.4)],
                LAYERS[2],
            ),
            spec("overlay", Vec::new(), LAYERS[3]),
        ]
    }

    fn state(
        &self,
        layer: usize,
        clip: Option<u32>,
        transition: Option<ActiveTransition>,
    ) -> LayerState {
        LayerState {
            layer,
            clip,
            visualizer: clip.map(|_| self.visualizer.to_string()),
            params: ClipParams {
                p: [0.5; 8],
                palette: 0,
                seed: 7,
            },
            effects: self.effects.clone(),
            transition,
            blend: self.blend,
            opacity: self.opacity,
            seed: 7,
        }
    }
}

/// The frames a run keeps, per pass name. `BTreeMap` so the table's order is stable.
struct Report {
    width: u32,
    height: u32,
    layer_size: (u32, u32),
    frames: usize,
    /// Per frame: pass name -> milliseconds summed over that frame's draws of it.
    pass_frames: Vec<BTreeMap<String, f32>>,
    totals: Vec<f32>,
    /// Frame totals split by whether a crossfade was in progress (the transition pass is a
    /// per-layer cost, so a scene change frame costs more than a steady one).
    totals_crossfade: Vec<f32>,
    totals_steady: Vec<f32>,
    layer_frames: Vec<f32>,
    composites: Vec<f32>,
    cpu: Vec<f32>,
    memory: Memory,
}

struct Memory {
    /// Per layer, the bytes its `LayerSlot` (scene + three post targets) allocated.
    layer_bytes: Vec<u64>,
    visuals: u64,
    output: u64,
    compositor: u64,
    overlay_texture: u64,
    total: u64,
    allocations: u64,
}

fn run(gpu: &Gpu, width: u32, height: u32, frames: usize, warmup: usize) -> Report {
    let baseline = gpu.allocated_bytes();
    let specs = LayerSpec::stress();
    // The app's `internal_size`: output size times the layer's scale.
    let (layer_w, layer_h) = (
        ((width as f32 * 0.5) as u32).max(16),
        ((height as f32 * 0.5) as u32).max(16),
    );

    let before_visuals = gpu.allocated_bytes();
    let mut visuals = Visuals::new(gpu, layer_w, layer_h).expect("visuals");
    let visuals_bytes = gpu.allocated_bytes() - before_visuals;

    let mut slots = Vec::with_capacity(specs.len());
    let mut layer_bytes = Vec::with_capacity(specs.len());
    for (index, spec) in specs.iter().enumerate() {
        let before = gpu.allocated_bytes();
        let (w, h) = (
            ((width as f32 * spec.scale) as u32).max(16),
            ((height as f32 * spec.scale) as u32).max(16),
        );
        slots.push(LayerSlot::new(gpu, index, w, h));
        layer_bytes.push(gpu.allocated_bytes() - before);
    }

    let before_compositor = gpu.allocated_bytes();
    // The stage composites straight into the swapchain image, whose format is 8-bit BGRA; this
    // bench writes the same format into an `OutputImage` of the output size, so the composite
    // pass here is the app's composite pass.
    let present_format = mosh_render::STAGE_FORMAT;
    let mut compositor = Compositor::new_for_format(gpu, present_format).expect("compositor");
    let compositor_bytes = gpu.allocated_bytes() - before_compositor;
    let composite_size = (width, height);
    let before_output = gpu.allocated_bytes();
    let output =
        OutputImage::new_with_format(gpu, composite_size.0, composite_size.1, present_format);
    let output_bytes = gpu.allocated_bytes() - before_output;

    // A decoded overlay frame, uploaded every frame like the app does it.
    let mut video = vec![0u8; VIDEO.0 as usize * VIDEO.1 as usize * 4];
    for (index, byte) in video.iter_mut().enumerate() {
        // Moving diagonal bands, so the cover pass and the LumaKey see a real picture.
        let x = (index / 4) % VIDEO.0 as usize;
        let y = (index / 4) / VIDEO.0 as usize;
        let band = ((x + y) / 64) % 8;
        *byte = if (index % 4) == 3 {
            255
        } else {
            (band * 32) as u8
        };
    }
    let before_overlay = gpu.allocated_bytes();
    visuals
        .overlay
        .set_frame(gpu, 3, VIDEO.0, VIDEO.1, &video)
        .expect("overlay frame");
    let overlay_bytes = gpu.allocated_bytes() - before_overlay;

    let mut report = Report {
        width,
        height,
        layer_size: (layer_w, layer_h),
        frames,
        pass_frames: Vec::with_capacity(frames),
        totals: Vec::with_capacity(frames),
        totals_crossfade: Vec::new(),
        totals_steady: Vec::new(),
        layer_frames: Vec::with_capacity(frames),
        composites: Vec::with_capacity(frames),
        cpu: Vec::with_capacity(frames),
        memory: Memory {
            layer_bytes,
            visuals: visuals_bytes,
            output: output_bytes,
            compositor: compositor_bytes,
            overlay_texture: overlay_bytes,
            total: gpu.allocated_bytes() - baseline,
            allocations: gpu.allocation_count(),
        },
    };

    // The present pass is the composite: the stage composites straight into the swapchain image
    // (`Stage::present_image`), so there is no separate blit to measure. Nothing extra is
    // allocated here.

    for frame_index in 0..(warmup + frames) {
        let frame = analysis(frame_index);
        // A clip change on frame 150 with a two-beat crossfade: what Autopilot does at a bar
        // line, and the only thing in the stress project that fills the transition pass.
        let changed = frame_index >= 150;
        let transition = if (150..210).contains(&frame_index) {
            Some(ActiveTransition {
                kind: TransitionKind::Crossfade,
                progress: (frame_index - 150) as f32 / 60.0,
                from: Some(1),
            })
        } else {
            None
        };

        let budget_start = Instant::now();
        // Overlay frames arrive per frame; uploading all of them is what the app does.
        visuals
            .overlay
            .set_frame(gpu, 3, VIDEO.0, VIDEO.1, &video)
            .expect("overlay frame");

        let mut recorder = FrameRecorder::new(gpu);
        for (index, spec) in specs.iter().enumerate() {
            let clip = if spec.visualizer == "overlay" {
                Some(if changed { 14 } else { 4 })
            } else {
                Some(if changed {
                    11 + index as u32
                } else {
                    1 + index as u32
                })
            };
            let state = spec.state(index, clip, transition);
            visuals
                .draw_layer(
                    &mut recorder,
                    &mut slots[index],
                    &state,
                    &frame,
                    frame.time_s as f32,
                )
                .expect("draw layer");
        }
        for (index, spec) in specs.iter().enumerate() {
            recorder.layer(slots[index].output(), spec.blend, spec.opacity);
        }
        let cpu_ms = budget_start.elapsed().as_secs_f32() * 1_000.0;

        let timings = gpu.wait_frame(recorder.submit());
        let layers: Vec<(&mosh_render::LayerTarget, mosh_core::BlendMode, f32)> =
            recorder_layers(&slots, &specs);
        compositor.composite(&layers, &output).expect("composite");
        let composite_ms = compositor.timings().frame_ms;

        if frame_index < warmup {
            continue;
        }

        let mut per_frame: BTreeMap<String, f32> = BTreeMap::new();
        for (name, ms) in &timings.per_pass_ms {
            *per_frame.entry(name.clone()).or_insert(0.0) += *ms;
        }
        report.composites.push(composite_ms);
        report.layer_frames.push(timings.frame_ms);
        let total = timings.frame_ms + composite_ms;
        report.totals.push(total);
        if transition.is_some() {
            report.totals_crossfade.push(total);
        } else {
            report.totals_steady.push(total);
        }
        report.cpu.push(cpu_ms);
        report.pass_frames.push(per_frame);
    }
    report
}

/// The layer stack a frame composites, in draw order.
fn recorder_layers<'a>(
    slots: &'a [LayerSlot],
    specs: &[LayerSpec],
) -> Vec<(&'a mosh_render::LayerTarget, mosh_core::BlendMode, f32)> {
    slots
        .iter()
        .zip(specs)
        .map(|(slot, spec)| (slot.output(), spec.blend, spec.opacity))
        .collect()
}

/// A synthetic analysis stream: 120 BPM at 60 fps, a beat every 30 frames, a bar every 120, an
/// onset on every beat, and an intensity level change (2 -> 4, with a drop) at frame 120.
fn analysis(index: usize) -> AnalysisFrame {
    let fps = 60.0f64;
    let bpm = 120.0f64;
    let time_s = index as f64 / fps;
    let ppq = time_s * bpm / 60.0;
    let beat_index = ppq.floor() as i64;
    let bar_index = beat_index.div_euclid(4);
    let on_beat = index.is_multiple_of(30);
    let on_bar = index.is_multiple_of(120);
    let level = if index < 120 { 2u8 } else { 4u8 };
    let mut events = EventFlags::empty();
    if on_beat {
        events |= EventFlags::BEAT;
    }
    if on_bar {
        events |= EventFlags::BAR;
    }
    if index == 120 {
        events |= EventFlags::DROP;
    }
    let pulse = (1.0 - (index % 30) as f32 / 30.0).powi(3);
    AnalysisFrame {
        time_s,
        ppq,
        beat_phase: ppq.fract() as f32,
        bar_phase: (ppq / 4.0).fract() as f32,
        beat_index,
        bar_index,
        tempo_bpm: bpm as f32,
        playing: true,
        bands: [0.35 + 0.6 * pulse, 0.4 + 0.3 * pulse, 0.3 + 0.2 * pulse],
        mel: [0.5; 24],
        onset: if on_beat { 1.0 } else { 0.15 },
        loudness_db: -18.0,
        loudness: 0.6,
        brightness: 0.5,
        intensity: (level - 1) as f32 / 3.0,
        level,
        events,
        source: mosh_core::TransportSource::Plugin,
    }
}

impl Report {
    fn print(&self) {
        println!();
        println!(
            "== {}x{} output, {}x{} layers, {} frames ==",
            self.width, self.height, self.layer_size.0, self.layer_size.1, self.frames
        );
        // Per pass name: the per-frame series (0 on frames it did not run), how many draws that
        // is per frame, and the cost of *one* draw, so a pass that happens four times on a
        // transition frame (one per layer) is not read as one expensive pass.
        let mut series: BTreeMap<&str, Vec<f32>> = BTreeMap::new();
        for frame in &self.pass_frames {
            for name in frame.keys() {
                series.entry(name.as_str()).or_default();
            }
        }
        let mut draws: BTreeMap<&str, usize> = BTreeMap::new();
        for (name, values) in series.iter_mut() {
            for frame in &self.pass_frames {
                let ms = frame.get(*name).copied().unwrap_or(0.0);
                values.push(ms);
            }
        }
        for name in series.keys() {
            draws.insert(
                name,
                self.pass_frames
                    .iter()
                    .filter(|f| f.contains_key(*name))
                    .count(),
            );
        }
        println!(
            "{:<28} {:>7} {:>9} {:>9} {:>9} {:>9}",
            "pass", "draws/f", "ms/frame", "p95/frame", "median/d", "max/d"
        );
        let total_mean: f32 = mean(&self.totals);
        let mut rows: Vec<(&str, f32, f32, f32, f32)> = series
            .iter()
            .map(|(name, values)| {
                let non_zero: Vec<f32> = values.iter().copied().filter(|v| *v > 0.0).collect();
                let per_frame_sum: f32 = values.iter().sum();
                let per_draw = if draws[*name] == 0 {
                    0.0
                } else {
                    per_frame_sum / draws[*name] as f32
                };
                (
                    *name,
                    mean(values),
                    percentile(values, 0.95),
                    per_draw,
                    non_zero.iter().copied().fold(0.0f32, f32::max),
                )
            })
            .collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut accounted = 0.0f32;
        for (name, per_frame, p95, per_draw, worst) in rows {
            accounted += per_frame;
            println!(
                "{:<28} {:>7.2} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>6.1}%",
                name,
                draws[name] as f32 / self.frames as f32,
                per_frame,
                p95,
                per_draw,
                worst,
                100.0 * per_frame / total_mean
            );
        }
        let composite = mean(&self.composites);
        accounted += composite;
        println!(
            "{:<28} {:>7.2} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>6.1}%",
            "composite + present blit",
            1.0,
            composite,
            percentile(&self.composites, 0.95),
            median(&self.composites),
            self.composites.iter().copied().fold(0.0f32, f32::max),
            100.0 * composite / total_mean
        );
        println!(
            "{:<28} {:>7} {:>9.3} {:>9.3} {:>6.1}%   (pass spans vs the frame's own span)",
            "not in any pass",
            "",
            total_mean - accounted,
            "",
            100.0 * (total_mean - accounted) / total_mean
        );
        println!(
            "{:<28} {:>7} {:>9.3} {:>9.3}",
            "layer frame (timed span)",
            "",
            mean(&self.layer_frames),
            percentile(&self.layer_frames, 0.95)
        );
        println!();
        println!(
            "TOTAL GPU   median {:.3} ms   p95 {:.3} ms   mean {:.3} ms   (n={})",
            median(&self.totals),
            percentile(&self.totals, 0.95),
            mean(&self.totals),
            self.totals.len()
        );
        // The same run split by the crossfade: `tr_mix` runs once per layer on a scene change, so
        // a change frame costs several passes more than a steady one.
        println!(
            "            steady frames {}: median {:.3}  p95 {:.3} ms",
            self.totals_steady.len(),
            median(&self.totals_steady),
            percentile(&self.totals_steady, 0.95)
        );
        println!(
            "            crossfade frames {}: median {:.3}  max {:.3} ms",
            self.totals_crossfade.len(),
            median(&self.totals_crossfade),
            self.totals_crossfade.iter().copied().fold(0.0f32, f32::max)
        );
        println!(
            "CPU record  median {:.3} ms   p95 {:.3} ms   (overlay upload, draw_layer, submit)",
            median(&self.cpu),
            percentile(&self.cpu, 0.95)
        );
        // Where the p95 comes from: the five dearest frames of the run, with their index, so a
        // transition spike and a steady frame are told apart.
        let mut worst: Vec<(usize, f32)> = self.totals.iter().copied().enumerate().collect();
        worst.sort_by(|a, b| b.1.total_cmp(&a.1));
        print!("worst frames:");
        for (index, ms) in worst.iter().take(5) {
            print!(" #{index} {ms:.2} ms");
        }
        println!();
        println!(
            "            p50 {:.3}  p90 {:.3}  p99 {:.3}  max {:.3} ms",
            percentile(&self.totals, 0.5),
            percentile(&self.totals, 0.90),
            percentile(&self.totals, 0.99),
            self.totals.iter().copied().fold(0.0f32, f32::max)
        );
        self.print_memory();
    }

    fn print_memory(&self) {
        let mb = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
        println!();
        println!(
            "GPU memory  per layer target {:>7.2} MB  ({} layers: {:.2} MB)",
            mb(self.memory.layer_bytes.first().copied().unwrap_or(0)),
            self.memory.layer_bytes.len(),
            mb(self.memory.layer_bytes.iter().sum::<u64>())
        );
        println!(
            "            visuals (fluid, mosh, mel, palettes, ...) {:>6.2} MB",
            mb(self.memory.visuals)
        );
        println!(
            "            output image {:>6.2} MB",
            mb(self.memory.output)
        );
        println!(
            "            compositor {:>6.2} MB",
            mb(self.memory.compositor)
        );
        println!(
            "            one 1920x1080 overlay texture (image + 2 staging buffers) {:>6.2} MB",
            mb(self.memory.overlay_texture)
        );
        println!(
            "            TOTAL {:>6.2} MB in {} live allocations",
            mb(self.memory.total),
            self.memory.allocations
        );
    }
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

fn median(values: &[f32]) -> f32 {
    percentile(values, 0.5)
}

/// Nearest-rank percentile: the p95 is a value a real frame measured, not an interpolation.
fn percentile(values: &[f32], fraction: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let rank = (fraction * sorted.len() as f32).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}
