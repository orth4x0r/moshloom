//! Offscreen Vulkan tests for the per-layer mosh pass (Task 2.2).
//!
//! No window is opened: each test draws a test pattern into a layer target with a WGSL shader
//! compiled here by `mosh_shaders::compile_wgsl`, runs `MoshPass::apply` for a number of frames,
//! and reads the layer targets back with `Gpu::read_back`.
//!
//! The pattern is a vertical bar (a hard 0/1 edge, so a readback is exact) and the tests measure
//! the bar's span on its middle row. Three of them are *sign* checks, which is what makes them
//! worth running:
//!
//! * `static_scene_stays_static` (no motion, amount 0.8): the feedback must be an identity, so
//!   any drift shows up as pixels that differ from the input;
//! * `moving_pattern_smears_along_motion`: a bar moving right leaves its trail on the *left*;
//! * `extra_flow_moves_pixels`: a constant flow field pushes static content in the flow's
//!   direction.
//!
//! `flow_*` and `slower_than_*` helpers exist for the numbers printed by each test: the gate
//! quotes them, not adjectives.

use mosh_core::{
    AnalysisFrame, BlendMode, ClipParams, EffectKind, EffectSlot, EventFlags, LayerState,
};
use mosh_render::{FrameRecorder, Gpu, GpuPreference, LayerTarget, Pipeline};
use mosh_shaders::{bindings, compile_wgsl, uniform_bytes};
use mosh_visuals::mosh::MoshPass;
use mosh_visuals::{LayerSlot, Stage, Visuals};

/// A vertical bar `[rect.x, rect.y)` full height, white on black.
const PATTERN_WGSL: &str = r#"
struct P {
    rect: vec4<f32>,
};

@group(0) @binding(0) var<uniform> p: P;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let inside = step(p.rect.x, pos.x) * step(pos.x, p.rect.y);
    return vec4<f32>(vec3<f32>(inside), 1.0);
}
"#;

/// A constant vector field: `v.xy` in every texel, which is the shape `FluidVisualizer::velocity`
/// produces (RG = pixels per frame).
const FLOW_WGSL: &str = r#"
struct F {
    value: vec4<f32>,
};

@group(0) @binding(0) var<uniform> f: F;

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(f.value.xy, 0.0, 1.0);
}
"#;

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

fn pipeline(gpu: &Gpu, wgsl: &str, uniform_size: usize, textures: u32) -> Pipeline {
    let words = compile_wgsl(wgsl).unwrap_or_else(|error| panic!("test shader: {error}"));
    Pipeline::fullscreen(gpu, &words, uniform_size, textures).expect("test pipeline")
}

/// `vec4` uniform, 16 bytes.
fn vec4_uniform(values: [f32; 4]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn red_row(pixels: &[u8], width: u32, y: u32) -> Vec<u8> {
    (0..width)
        .map(|x| pixels[((y * width + x) * 4) as usize])
        .collect()
}

/// First and last column of `row` at or above `threshold`.
fn span(row: &[u8], threshold: u8) -> Option<(u32, u32)> {
    let first = row.iter().position(|value| *value >= threshold)? as u32;
    let last = row.iter().rposition(|value| *value >= threshold)? as u32;
    Some((first, last))
}

/// Mean of `row[from..to]`, clamped to the row.
fn mean(row: &[u8], from: i64, to: i64) -> f32 {
    let start = from.max(0) as usize;
    let end = (to.max(0) as usize).min(row.len());
    if start >= end {
        return 0.0;
    }
    let sum: u32 = row[start..end].iter().map(|value| u32::from(*value)).sum();
    sum as f32 / (end - start) as f32
}

/// Brightness centroid of a row, in columns.
fn centroid(row: &[u8]) -> f32 {
    let mut weight = 0.0f64;
    let mut moment = 0.0f64;
    for (x, value) in row.iter().enumerate() {
        let value = f64::from(*value);
        weight += value;
        moment += value * x as f64;
    }
    if weight == 0.0 {
        return f32::NAN;
    }
    (moment / weight) as f32
}

fn assert_no_validation_messages(gpu: &Gpu) {
    let messages = gpu.validation_messages();
    assert!(
        messages.is_empty(),
        "{} validation message(s):\n{}",
        messages.len(),
        messages.join("\n")
    );
}

/// The shaders this pass drives have to agree with `mosh-render`'s fixed descriptor layout:
/// binding 0 is the uniform, then one sampled image and one sampler per input. This is a unit
/// check on the WGSL, not on pixels, and it fails before a device is even created.
#[test]
fn shaders_match_the_renderer_binding_layout() {
    for name in ["mosh_motion", "mosh_feedback"] {
        let reflected = bindings(name).unwrap_or_else(|| panic!("{name} has no SPIR-V"));
        assert_eq!(reflected[0].binding, 0, "{name}: the uniform is binding 0");
        for (index, binding) in reflected[1..].iter().enumerate() {
            let expected = 1 + index as u32;
            assert_eq!(
                binding.binding, expected,
                "{name}: bindings are dense from 1"
            );
            if index % 2 == 0 {
                assert_eq!(
                    binding.kind,
                    mosh_shaders::BindingKind::Texture,
                    "{name}: binding {expected} is a sampled image"
                );
            } else {
                assert_eq!(
                    binding.kind,
                    mosh_shaders::BindingKind::Sampler,
                    "{name}: binding {expected} is a sampler"
                );
            }
        }
    }
    // The pass binds three textures to the motion pass and four to the feedback pass.
    assert_eq!(bindings("mosh_motion").expect("motion").len(), 1 + 2 * 2);
    assert_eq!(
        bindings("mosh_feedback").expect("feedback").len(),
        1 + 2 * 4
    );
    assert_eq!(uniform_bytes("mosh_motion"), Some(16), "motion uniform");
    assert_eq!(uniform_bytes("mosh_feedback"), Some(32), "feedback uniform");
    println!("shader bindings: mosh_motion 2 inputs, mosh_feedback 4 inputs, 16/32 B uniforms");
}

/// Same input every frame at amount 0.8: the feedback must be an identity, not a decay.
#[test]
fn static_scene_stays_static() {
    let gpu = gpu();
    let (width, height) = (128, 72);
    let pattern = pipeline(&gpu, PATTERN_WGSL, 16, 0);
    let mut mosh = MoshPass::new(&gpu, width, height).expect("mosh pass");
    let src = LayerTarget::new(&gpu, width, height);
    let dst = LayerTarget::new(&gpu, width, height);
    let frame = AnalysisFrame::default();
    let bar = vec4_uniform([40.0, 88.0, 0.0, 0.0]);

    let frames = 10;
    for _ in 0..frames {
        let mut rec = FrameRecorder::new(&gpu);
        rec.draw(&pattern, &src, &bar, &[]).expect("pattern");
        let applied = mosh
            .apply(&mut rec, 0, &src, &dst, 0.8, &frame, None)
            .expect("mosh");
        assert!(applied, "amount 0.8 must write the output");
        gpu.wait_frame(rec.submit());
    }

    let input = gpu.read_back(&src);
    let output = gpu.read_back(&dst);
    let mut worst = 0i32;
    let mut differing = 0usize;
    for (a, b) in input.iter().zip(output.iter()) {
        let difference = i32::from(*a) - i32::from(*b);
        worst = worst.max(difference.abs());
        if difference.abs() > 2 {
            differing += 1;
        }
    }
    println!(
        "{width}x{height}, amount 0.8, {frames} frames: worst channel difference {worst}/255, \
         {} of {} channels beyond 2/255",
        differing,
        input.len()
    );
    assert!(
        worst <= 2,
        "a static input drifted: worst channel difference {worst}/255 after {frames} frames"
    );
    assert_no_validation_messages(&gpu);
}

/// A bar moving right: the mosh buffer trails *behind* it, so the smear lies to its left.
///
/// The analysis grid is the web app's fixed 320x180 one, so at 960x540 a 4 px/frame bar moves
/// 1.33 grid pixels per frame: block matching can only answer a whole grid pixel, so the buffer
/// is carried 3 px per frame and falls 1 px behind per frame. That lag is what the trail is made
/// of, and the sign check below is what would fail if the pass carried the buffer the wrong way
/// (ahead of the bar) or did not carry it at all (the bar would sit exactly under the input).
#[test]
fn moving_pattern_smears_along_motion() {
    let gpu = gpu();
    let (width, height) = (960, 540);
    let pattern = pipeline(&gpu, PATTERN_WGSL, 16, 0);
    let mut mosh = MoshPass::new(&gpu, width, height).expect("mosh pass");
    let src = LayerTarget::new(&gpu, width, height);
    let dst = LayerTarget::new(&gpu, width, height);
    let frame = AnalysisFrame::default();

    let bar_width = 20.0f32;
    let speed = 4.0f32;
    let start = 200.0f32;
    let frames = 8u32;
    for step in 0..frames {
        let x0 = start + speed * step as f32;
        let bar = vec4_uniform([x0, x0 + bar_width, 0.0, 0.0]);
        let mut rec = FrameRecorder::new(&gpu);
        rec.draw(&pattern, &src, &bar, &[]).expect("pattern");
        let applied = mosh
            .apply(&mut rec, 0, &src, &dst, 0.8, &frame, None)
            .expect("mosh");
        assert!(applied, "frame {step}: amount 0.8 must write the output");
        gpu.wait_frame(rec.submit());
    }

    let input = red_row(&gpu.read_back(&src), width, height / 2);
    let output = red_row(&gpu.read_back(&dst), width, height / 2);
    let (input_left, input_right) = span(&input, 200).expect("the input bar is visible");
    let (output_left, output_right) = span(&output, 200).expect("the mosh output is not empty");
    let (trail_left, _) = span(&output, 24).expect("the mosh output has no visible pixels");
    // Behind the bar: the columns it has just left.
    let behind = mean(
        &output,
        i64::from(input_left) - 10,
        i64::from(input_left) - 1,
    );
    // Ahead of it: where a right-carrying pass would put the smear.
    let ahead = mean(
        &output,
        i64::from(input_right) + 2,
        i64::from(input_right) + 11,
    );
    let input_centroid = centroid(&input);
    let output_centroid = centroid(&output);
    println!(
        "{width}x{height}, {frames} frames at {speed} px/frame, amount 0.8: input bar \
         [{input_left}, {input_right}], output's full-brightness span [{output_left}, \
         {output_right}], visible output from {trail_left}, mean behind 10 px {behind:.1}/255, \
         mean ahead 10 px {ahead:.1}/255, centroid {input_centroid:.1} -> {output_centroid:.1}"
    );
    assert!(
        i64::from(trail_left) + 3 < i64::from(input_left),
        "the trail only reaches {trail_left}, less than 3 px left of the input bar at {input_left}"
    );
    assert!(
        behind > 40.0,
        "the trail behind the bar is only {behind:.1}/255: nothing smeared"
    );
    assert!(
        ahead < 8.0,
        "the smear is {ahead:.1}/255 ahead of the bar at {input_right}, expected the trail on \
         the left"
    );
    assert!(
        output_right < input_right,
        "the mosh output is at full brightness out to {output_right}, past the input bar's \
         {input_right}: the carried picture is ahead of the motion"
    );
    assert!(
        output_centroid < input_centroid - 1.0,
        "the mosh output's brightness centroid {output_centroid:.1} is not left of the input's \
         {input_centroid:.1}"
    );
    assert_no_validation_messages(&gpu);
}

/// `amount == 0` is the "no mosh slot" case: no pass is recorded at all.
#[test]
fn amount_zero_returns_false() {
    let gpu = gpu();
    let (width, height) = (64, 36);
    let pattern = pipeline(&gpu, PATTERN_WGSL, 16, 0);
    let mut mosh = MoshPass::new(&gpu, width, height).expect("mosh pass");
    let src = LayerTarget::new(&gpu, width, height);
    let dst = LayerTarget::new(&gpu, width, height);
    let frame = AnalysisFrame::default();
    let bar = vec4_uniform([8.0, 24.0, 0.0, 0.0]);

    let mut rec = FrameRecorder::new(&gpu);
    rec.draw(&pattern, &src, &bar, &[]).expect("pattern");
    let applied = mosh
        .apply(&mut rec, 0, &src, &dst, 0.0, &frame, None)
        .expect("mosh");
    let timings = gpu.wait_frame(rec.submit());
    println!(
        "amount 0.0: applied {applied}, {} pass(es) recorded ({:?})",
        timings.per_pass_ms.len(),
        timings.per_pass_ms
    );
    assert!(!applied, "amount 0 must not claim it wrote the output");
    assert_eq!(
        timings.per_pass_ms.len(),
        1,
        "only the pattern pass may be recorded: {:?}",
        timings.per_pass_ms
    );
    assert_no_validation_messages(&gpu);
}

/// The fluid layer's velocity field, in pixels per frame, moves a static input along the flow.
#[test]
fn extra_flow_moves_pixels() {
    let gpu = gpu();
    let (width, height) = (320, 180);
    let pattern = pipeline(&gpu, PATTERN_WGSL, 16, 0);
    let flow_shader = pipeline(&gpu, FLOW_WGSL, 16, 0);
    let mut mosh = MoshPass::new(&gpu, width, height).expect("mosh pass");
    let src = LayerTarget::new(&gpu, width, height);
    let dst = LayerTarget::new(&gpu, width, height);
    let flow = LayerTarget::new(&gpu, width, height);
    let frame = AnalysisFrame::default();
    let bar = vec4_uniform([80.0, 140.0, 0.0, 0.0]);
    let velocity = 6.0f32;
    let flow_uniform = vec4_uniform([velocity, 0.0, 0.0, 0.0]);

    let frames = 5u32;
    let mut first = (0u32, 0u32);
    let mut last = (0u32, 0u32);
    for step in 0..frames {
        let mut rec = FrameRecorder::new(&gpu);
        rec.draw(&pattern, &src, &bar, &[]).expect("pattern");
        rec.draw(&flow_shader, &flow, &flow_uniform, &[])
            .expect("flow field");
        let applied = mosh
            .apply(&mut rec, 0, &src, &dst, 0.8, &frame, Some(&flow))
            .expect("mosh");
        assert!(applied, "frame {step}: amount 0.8 must write the output");
        gpu.wait_frame(rec.submit());
        let span = span(&red_row(&gpu.read_back(&dst), width, height / 2), 100)
            .unwrap_or_else(|| panic!("frame {step}: the mosh output is empty"));
        if step == 0 {
            first = span;
        }
        last = span;
    }
    let input = span(&red_row(&gpu.read_back(&src), width, height / 2), 200).expect("input bar");
    println!(
        "{width}x{height}, flow {velocity} px/frame, {frames} frames at amount 0.8: input bar \
         {input:?}, output bar [{first:?}] -> [{last:?}]"
    );
    assert!(
        last.1 > first.1 + 4,
        "the flow pushed the leading edge from {} to {} only: 4 frames of {velocity} px/frame is \
         a visible move",
        first.1,
        last.1
    );
    assert!(
        i64::from(last.0) + 1 >= i64::from(first.0),
        "the flow pushed the trailing edge from {} to {}, the wrong way",
        first.0,
        last.0
    );
    assert!(
        i64::from(last.1) > i64::from(input.1) + 4,
        "the leading edge {} did not pass the static input's {}",
        last.1,
        input.1
    );
    assert_no_validation_messages(&gpu);
}

/// The facade hands the mosh stage the scene target and `post[0]`; make sure that pairing works
/// through `Visuals::draw_layer` (the pass reads `dst`'s own previous image as its buffer).
#[test]
fn facade_chain_runs_the_pass() {
    let gpu = gpu();
    let mut visuals = Visuals::new(&gpu, 64, 36).expect("visuals");
    let mut slot = LayerSlot::new(&gpu, 0, 64, 36);
    let frame = AnalysisFrame::default();
    let state = LayerState {
        layer: 0,
        clip: Some(1),
        visualizer: Some("tunnel".to_string()),
        params: ClipParams::default(),
        effects: vec![EffectSlot {
            kind: EffectKind::Mosh,
            amount: 0.8,
            param: 0.0,
        }],
        transition: None,
        blend: BlendMode::Normal,
        opacity: 1.0,
        seed: 0,
    };
    let frames = 3;
    for _ in 0..frames {
        let mut rec = FrameRecorder::new(&gpu);
        visuals
            .draw_layer(&mut rec, &mut slot, &state, &frame, 0.0)
            .expect("draw layer");
        gpu.wait_frame(rec.submit());
    }
    println!(
        "facade, {frames} frames: output stage {:?}, {} validation message(s)",
        slot.output_stage,
        gpu.validation_messages().len()
    );
    assert_eq!(
        slot.output_stage,
        Stage::Mosh,
        "the chain kept the mosh output"
    );
    assert_no_validation_messages(&gpu);
}

/// `EventFlags::BAR` on a multiple of four bars, and `DROP`, refresh the buffer (I-frames).
#[test]
fn iframe_events_refresh_the_buffer() {
    let gpu = gpu();
    let (width, height) = (64, 36);
    let pattern = pipeline(&gpu, PATTERN_WGSL, 16, 0);
    let mut mosh = MoshPass::new(&gpu, width, height).expect("mosh pass");
    let src = LayerTarget::new(&gpu, width, height);
    let dst = LayerTarget::new(&gpu, width, height);
    let bar_a = vec4_uniform([8.0, 24.0, 0.0, 0.0]);
    let bar_b = vec4_uniform([40.0, 56.0, 0.0, 0.0]);

    // A couple of ordinary frames so the buffer holds bar A anywhere but at bar B.
    for _ in 0..3 {
        let mut rec = FrameRecorder::new(&gpu);
        rec.draw(&pattern, &src, &bar_a, &[]).expect("pattern");
        mosh.apply(
            &mut rec,
            0,
            &src,
            &dst,
            0.9,
            &AnalysisFrame::default(),
            None,
        )
        .expect("mosh");
        gpu.wait_frame(rec.submit());
    }

    let mut draw = |frame: &AnalysisFrame, bar: &[u8]| {
        let mut rec = FrameRecorder::new(&gpu);
        rec.draw(&pattern, &src, bar, &[]).expect("pattern");
        mosh.apply(&mut rec, 0, &src, &dst, 0.9, frame, None)
            .expect("mosh");
        gpu.wait_frame(rec.submit());
        red_row(&gpu.read_back(&dst), width, height / 2)
    };

    // A bar drop (a new clip) with no event: the buffer keeps the old picture, so B does not
    // appear yet.
    let quiet = AnalysisFrame {
        bar_index: 1,
        ..AnalysisFrame::default()
    };
    let stale = draw(&quiet, &bar_b);
    let stale_hits = mean(&stale, 40, 56);
    // The same frame on a DROP: an I-frame copies B in.
    let drop = AnalysisFrame {
        bar_index: 2,
        events: EventFlags::DROP,
        ..AnalysisFrame::default()
    };
    let fresh = draw(&drop, &bar_b);
    let fresh_hits = mean(&fresh, 40, 56);
    // And on bar 4 (a multiple of four) with no other event.
    let bar_four = AnalysisFrame {
        bar_index: 4,
        events: EventFlags::BAR,
        ..AnalysisFrame::default()
    };
    let bared = draw(&bar_four, &bar_b);
    let bared_hits = mean(&bared, 40, 56);
    println!(
        "amount 0.9: stale {stale_hits:.0}/255, after DROP {fresh_hits:.0}/255, on bar 4 \
         {bared_hits:.0}/255 in the new bar's columns"
    );
    assert!(
        stale_hits < 200.0,
        "a plain frame already shows the new bar ({stale_hits:.0}/255): nothing is carried"
    );
    assert!(
        fresh_hits > 200.0,
        "DROP did not refresh the buffer ({fresh_hits:.0}/255 in the new bar's columns)"
    );
    assert!(
        bared_hits > 200.0,
        "bar 4 (bar_index % 4 == 0) did not refresh the buffer ({bared_hits:.0}/255)"
    );
    assert_no_validation_messages(&gpu);
}

/// Per-pass mean milliseconds of one mosh layer at `width`x`height`, and the worst whole frame.
fn measure_apply(gpu: &Gpu, width: u32, height: u32, frames: u32) -> ([f32; 3], f32) {
    let pattern = pipeline(gpu, PATTERN_WGSL, 16, 0);
    let mut mosh = MoshPass::new(gpu, width, height).expect("mosh pass");
    let src = LayerTarget::new(gpu, width, height);
    let dst = LayerTarget::new(gpu, width, height);
    let frame = AnalysisFrame::default();

    let mut passes: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut worst = 0.0f32;
    for step in 0..frames {
        let x0 = 40.0 + (step as f32) * 3.0;
        let bar = vec4_uniform([x0, x0 + 40.0, 0.0, 0.0]);
        let mut rec = FrameRecorder::new(gpu);
        rec.draw(&pattern, &src, &bar, &[]).expect("pattern");
        mosh.apply(&mut rec, 0, &src, &dst, 0.8, &frame, None)
            .expect("mosh");
        let timings = gpu.wait_frame(rec.submit());
        // Frame 0 is an I-frame: it records one copy pass instead of motion + feedback.
        for (name, milliseconds) in &timings.per_pass_ms {
            match name.as_str() {
                "mosh motion" => passes[0].push(*milliseconds),
                "mosh feedback" => passes[1].push(*milliseconds),
                "mosh copy" => passes[2].push(*milliseconds),
                _ => {}
            }
        }
        worst = worst.max(timings.frame_ms);
    }
    let mean = |values: &[f32]| {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f32>() / values.len() as f32
        }
    };
    let minimum = |values: &[f32]| {
        if values.is_empty() {
            0.0
        } else {
            values.iter().copied().fold(f32::INFINITY, f32::min)
        }
    };
    println!(
        "{width}x{height}, amount 0.8, {frames} frames: mosh motion {:.3} ms (min {:.3}), mosh \
         feedback {:.3} ms (min {:.3}), mosh copy {:.3} ms (min {:.3}), total per apply \
         {:.3} ms, worst frame {worst:.3} ms",
        mean(&passes[0]),
        minimum(&passes[0]),
        mean(&passes[1]),
        minimum(&passes[1]),
        mean(&passes[2]),
        minimum(&passes[2]),
        mean(&passes[0]) + mean(&passes[1]) + mean(&passes[2]),
    );
    (
        [mean(&passes[0]), mean(&passes[1]), mean(&passes[2])],
        worst,
    )
}

/// The cost the plan's budget table needs: one mosh layer per frame, at the size the brief asks
/// for and at the plan's 1280x720 internal size. The motion pass runs on the web's fixed 320x180
/// grid, so only the two full-size passes grow with the layer's size.
#[test]
fn apply_cost_per_frame() {
    let gpu = gpu();
    let (small, _) = measure_apply(&gpu, 960, 540, 30);
    let (large, _) = measure_apply(&gpu, 1280, 720, 30);
    let total = |passes: [f32; 3]| passes[0] + passes[1] + passes[2];
    // A 60 fps frame has 16.7 ms; three mosh layers must leave room for the rest of the chain.
    assert!(total(small) > 0.0, "no GPU time was measured at 960x540");
    assert!(total(large) > 0.0, "no GPU time was measured at 1280x720");
    assert!(total(small).is_finite() && total(large).is_finite());
    assert_no_validation_messages(&gpu);
}

/// `invalidate` (called by the facade on a clip change) makes the next frame an I-frame.
#[test]
fn invalidate_refreshes_the_buffer() {
    let gpu = gpu();
    let (width, height) = (64, 36);
    let pattern = pipeline(&gpu, PATTERN_WGSL, 16, 0);
    let mut mosh = MoshPass::new(&gpu, width, height).expect("mosh pass");
    let src = LayerTarget::new(&gpu, width, height);
    let dst = LayerTarget::new(&gpu, width, height);
    let bar_a = vec4_uniform([8.0, 24.0, 0.0, 0.0]);
    let bar_b = vec4_uniform([40.0, 56.0, 0.0, 0.0]);
    let draw = |mosh: &mut MoshPass, bar: &[u8]| {
        let mut rec = FrameRecorder::new(&gpu);
        rec.draw(&pattern, &src, bar, &[]).expect("pattern");
        mosh.apply(&mut rec, 0, &src, &dst, 0.9, &AnalysisFrame::default(), None)
            .expect("mosh");
        gpu.wait_frame(rec.submit());
        red_row(&gpu.read_back(&dst), width, height / 2)
    };
    for _ in 0..3 {
        draw(&mut mosh, &bar_a);
    }
    let stale = mean(&draw(&mut mosh, &bar_b), 40, 56);
    mosh.invalidate(0);
    let fresh = mean(&draw(&mut mosh, &bar_b), 40, 56);
    println!("amount 0.9: stale {stale:.0}/255, after invalidate {fresh:.0}/255");
    assert!(stale < 200.0, "a plain frame already shows the new bar ({stale:.0}/255)");
    assert!(fresh > 200.0, "invalidate did not refresh the buffer ({fresh:.0}/255)");
    assert_no_validation_messages(&gpu);
}
