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
    ActiveTransition, AnalysisFrame, BlendMode, ClipParams, EffectKind, EffectSlot, EventFlags,
    LayerState, TransitionKind,
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

/// A moving picture with texture everywhere: 32 px stripes crossed by a 48 px band, scrolled by
/// `shift`. Every 8x8 analysis block then has edges at two scales, so the block matcher finds the
/// true motion across the whole frame - the moving bar above is only 20 px wide and most of its
/// blocks are flat.
const STRIPES_WGSL: &str = r#"
struct S {
    shift: vec4<f32>,
};

@group(0) @binding(0) var<uniform> s: S;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = pos.x + s.shift.x;
    let y = pos.y + s.shift.y;
    let stripes = step(0.5, fract(x / 32.0));
    let band = step(0.5, fract(y / 48.0));
    return vec4<f32>(vec3<f32>(0.2 + 0.6 * stripes + 0.2 * band), 1.0);
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

/// Mean absolute difference between two RGBA8 read-backs, per channel, in 0..255 units.
fn mean_abs_difference(a: &[u8], b: &[u8]) -> f32 {
    let mut sum = 0u64;
    let mut count = 0u64;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += u64::from(x.abs_diff(*y));
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        sum as f32 / count as f32
    }
}

/// Pixels (not channels) whose red channel differs by more than 8/255.
fn differing_pixels(a: &[u8], b: &[u8]) -> usize {
    let (a, _) = a.as_chunks::<4>();
    let (b, _) = b.as_chunks::<4>();
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x[0].abs_diff(y[0]) > 8)
        .count()
}

/// Mean of the RGB channels of a read-back, 0..255.
fn mean_luma(pixels: &[u8]) -> f32 {
    let mut sum = 0u64;
    let mut count = 0u64;
    let (pixels, _) = pixels.as_chunks::<4>();
    for pixel in pixels {
        sum += u64::from(pixel[0]) + u64::from(pixel[1]) + u64::from(pixel[2]);
        count += 3;
    }
    if count == 0 {
        0.0
    } else {
        sum as f32 / count as f32
    }
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
    // The pass binds two textures to the motion pass and five to the feedback pass (source,
    // buffer, previous source, vectors, fluid flow).
    assert_eq!(bindings("mosh_motion").expect("motion").len(), 1 + 2 * 2);
    assert_eq!(
        bindings("mosh_feedback").expect("feedback").len(),
        1 + 2 * 5
    );
    assert_eq!(uniform_bytes("mosh_motion"), Some(16), "motion uniform");
    assert_eq!(uniform_bytes("mosh_feedback"), Some(40), "feedback uniform");
    println!(
        "shader bindings: mosh_motion 2 inputs, mosh_feedback 5 inputs, 16/40 B uniforms"
    );
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

/// The residual, pinned on its own: a *still* picture that changes to a *still* picture leaves
/// `src - prv` as a full replacement, so with the residual on the new picture is there at once,
/// and with the residual skipped (`skipres`, which is what a clip entry does) the old one stays.
///
/// Neither the leak nor the motion can produce this: the source does not move, so every vector is
/// zero, and `leak = 0.004` would take hundreds of frames to show anything. This is the one
/// behaviour in the P-frame that only the residual explains.
#[test]
fn the_residual_brings_a_still_change_in_at_once() {
    let gpu = gpu();
    let (width, height) = (320, 180);
    let pattern = pipeline(&gpu, PATTERN_WGSL, 16, 0);
    let mut mosh = MoshPass::new(&gpu, width, height).expect("mosh pass");
    let src = LayerTarget::new(&gpu, width, height);
    let dst = LayerTarget::new(&gpu, width, height);
    let frame = AnalysisFrame::default();
    let old = vec4_uniform([40.0, 60.0, 0.0, 0.0]);
    let new = vec4_uniform([240.0, 300.0, 0.0, 0.0]);

    // Warm the buffer on the old still, then hand the pass the new one.
    let mut entry = |carry: bool| -> f32 {
        for _ in 0..3 {
            let mut rec = FrameRecorder::new(&gpu);
            rec.draw(&pattern, &src, &old, &[]).expect("pattern");
            mosh.apply(&mut rec, 0, &src, &dst, 0.8, &frame, None)
                .expect("mosh");
            gpu.wait_frame(rec.submit());
        }
        if carry {
            mosh.carry_over(0);
        }
        let mut rec = FrameRecorder::new(&gpu);
        rec.draw(&pattern, &src, &new, &[]).expect("pattern");
        mosh.apply(&mut rec, 0, &src, &dst, 0.8, &frame, None)
            .expect("mosh");
        gpu.wait_frame(rec.submit());
        mean_abs_difference(&gpu.read_back(&dst), &gpu.read_back(&src))
    };

    let with_residual = entry(false);
    let with_skipres = entry(true);
    println!(
        "{width}x{height}, amount 0.8, a still picture replaced by another still: mean |output - \
         fresh| {with_residual:.2}/255 with the residual, {with_skipres:.2}/255 with `skipres`"
    );
    assert!(
        with_residual < 1.0,
        "the residual did not bring the new still in ({with_residual:.2}/255): it is not being \
         applied"
    );
    assert!(
        with_skipres > 10.0,
        "`skipres` did not keep the old still ({with_skipres:.2}/255): the two paths are the same"
    );
    assert_no_validation_messages(&gpu);
}

/// A bar moving right: the mosh buffer is carried *ahead* of it, because the web app scales the
/// displacement by `gain = 1 + 2.2 * amount` (`decide()`), 2.76x at amount 0.8. The sign check
/// catches the two failure modes that matter: a pass that does not carry the buffer at all (the
/// bar would sit exactly under the input) and one that carries it backwards (the smear would be
/// on the trailing side).
///
/// The analysis grid is the web app's fixed 320x180 one, so at 960x540 a 4 px/frame bar moves
/// 1.33 grid pixels per frame and block matching answers a whole grid pixel: 3 px at 960 wide,
/// times 2.76 is 8 px of carry per frame, against the bar's 4.
#[test]
fn moving_pattern_is_carried_ahead_by_the_gain() {
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
    let (_, trail_right) = span(&output, 24).expect("the mosh output has no visible pixels");
    // Behind the bar: the columns it has just left.
    let behind = mean(
        &output,
        i64::from(input_left) - 10,
        i64::from(input_left) - 1,
    );
    // Ahead of it: where the gain carries the picture.
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
         {output_right}], visible output out to {trail_right}, mean behind 10 px {behind:.1}/255, \
         mean ahead 10 px {ahead:.1}/255, centroid {input_centroid:.1} -> {output_centroid:.1}"
    );
    assert!(
        i64::from(trail_right) > i64::from(input_right) + 3,
        "the carried picture only reaches {trail_right}, not past the input bar's {input_right}: \
         the gain is not carrying it"
    );
    assert!(
        ahead > 20.0,
        "the smear ahead of the bar is only {ahead:.1}/255: nothing was carried forward"
    );
    assert!(
        behind < 8.0,
        "the smear behind the bar is {behind:.1}/255 at {input_left}: the buffer is carried \
         backwards"
    );
    assert!(
        output_centroid > input_centroid + 1.0,
        "the mosh output's brightness centroid {output_centroid:.1} is not right of the input's \
         {input_centroid:.1}"
    );
    assert_no_validation_messages(&gpu);
}

/// The pass **carries** the picture: after N P-frames the mosh output still holds a picture the
/// fresh frame does not have. The metric is the mean absolute channel difference between the
/// output and the frame the pass was given, in 0..255 units, over a striped picture scrolling
/// 8 px/frame at amount 0.8.
///
/// What this pins is the *carry*: `leak = (1 - amount)^5` floored at 0.004 and
/// `gain = 1 + 2.2 * amount`, so the picture is kept and pushed ahead of its own motion. A
/// crossfade (`out = mix(fresh, carried, amount)`) fails it: it measures 3.55/255 here, against
/// 46/255 for the port. The residual is a different behaviour and is pinned on its own, sharper,
/// by `the_residual_brings_a_still_change_in_at_once` - removing it from this scene's pass would
/// not move this number much, because the carried smear is what this measures.
#[test]
fn p_frames_keep_the_carried_picture() {
    let gpu = gpu();
    let (width, height) = (640, 360);
    let stripes = pipeline(&gpu, STRIPES_WGSL, 16, 0);
    let mut mosh = MoshPass::new(&gpu, width, height).expect("mosh pass");
    let src = LayerTarget::new(&gpu, width, height);
    let dst = LayerTarget::new(&gpu, width, height);
    let frame = AnalysisFrame::default();
    let speed = 8.0f32;
    let frames = 12u32;
    let mut series = Vec::new();
    let mut last = (0.0f32, 0usize, Vec::new());

    for step in 0..frames {
        let shift = vec4_uniform([speed * step as f32, 0.0, 0.0, 0.0]);
        let mut rec = FrameRecorder::new(&gpu);
        rec.draw(&stripes, &src, &shift, &[]).expect("stripes");
        let applied = mosh
            .apply(&mut rec, 0, &src, &dst, 0.8, &frame, None)
            .expect("mosh");
        assert!(applied, "frame {step}: amount 0.8 must write the output");
        gpu.wait_frame(rec.submit());

        let input = gpu.read_back(&src);
        let output = gpu.read_back(&dst);
        let difference = mean_abs_difference(&output, &input);
        series.push(difference);
        last = (difference, differing_pixels(&output, &input), red_row(&output, width, height / 2));
    }

    println!(
        "{width}x{height}, stripes at {speed} px/frame, amount 0.8: mean |output - fresh| per \
         channel {:.2}/255 at frame 1, {:.2}/255 at frame {frames} (per frame: {series:?}); {} of \
         {} pixels differ by more than 8/255; row spans {:#?}",
        series[0],
        last.0,
        last.1,
        width * height,
        span(&last.2, 200),
    );
    assert!(
        last.0 > 20.0,
        "after {frames} P-frames at amount 0.8 the output is only {:.2}/255 away from the fresh \
         frame: the carried picture is not being kept",
        last.0
    );
    assert!(
        last.1 * 4 >= (width * height) as usize,
        "only {} of {} pixels differ from the fresh frame",
        last.1,
        width * height
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

/// The fluid layer's velocity field, in pixels per frame, moves the carried picture *against* the
/// field: the web app adds its velocity to the sample offset (`d += vel; sample(buf, uv + d)`,
/// `WG_FB`), so a positive velocity pulls the picture in the negative direction.
#[test]
fn extra_flow_moves_pixels_against_the_field() {
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
        let span = span(&red_row(&gpu.read_back(&dst), width, height / 2), 24)
            .unwrap_or_else(|| panic!("frame {step}: the mosh output is empty"));
        if step == 0 {
            first = span;
        }
        last = span;
    }
    let input = span(&red_row(&gpu.read_back(&src), width, height / 2), 200).expect("input bar");
    println!(
        "{width}x{height}, flow +{velocity} px/frame, {frames} frames at amount 0.8: input bar \
         {input:?}, output bar [{first:?}] -> [{last:?}]"
    );
    assert!(
        i64::from(last.0) + 4 < i64::from(first.0),
        "the flow +{velocity} px/frame moved the picture from {} to {}: it went with the field, \
         not against it",
        first.0,
        last.0
    );
    assert!(
        last.1 < first.1,
        "the picture's leading edge moved from {} to {}, the wrong way",
        first.1,
        last.1
    );
    assert!(
        last.1 + 4 < input.1,
        "the leading edge {} did not move left of the static input's {}",
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

/// (c) `amount = 1.0` must not freeze a layer forever: `leak` is floored at 0.004, so even at full
/// strength the fresh picture keeps being mixed in.
///
/// A static source holds `src - prv` at zero, so the residual cannot help here. A fluid field
/// first displaces the buffer away from the source; then the field stops. From that point nothing
/// moves the picture but the leak, so the difference from the fresh frame must decay as
/// `(1 - 0.004)^frames` - with a leak of zero it would sit at the displaced picture for ever.
#[test]
fn amount_one_still_leaks_the_new_frame_in() {
    let gpu = gpu();
    let (width, height) = (320, 180);
    let pattern = pipeline(&gpu, PATTERN_WGSL, 16, 0);
    let flow_shader = pipeline(&gpu, FLOW_WGSL, 16, 0);
    let mut mosh = MoshPass::new(&gpu, width, height).expect("mosh pass");
    let src = LayerTarget::new(&gpu, width, height);
    let dst = LayerTarget::new(&gpu, width, height);
    let flow = LayerTarget::new(&gpu, width, height);
    let frame = AnalysisFrame::default();
    let bar = vec4_uniform([120.0, 200.0, 0.0, 0.0]);
    let velocity = 12.0f32;
    let flow_uniform = vec4_uniform([velocity, 0.0, 0.0, 0.0]);

    // `push`: run with the fluid field, which displaces the buffer while the source stands still.
    let mut draw = |push: bool| -> Vec<u8> {
        let mut rec = FrameRecorder::new(&gpu);
        rec.draw(&pattern, &src, &bar, &[]).expect("pattern");
        rec.draw(&flow_shader, &flow, &flow_uniform, &[])
            .expect("flow field");
        let field = if push { Some(&flow) } else { None };
        mosh.apply(&mut rec, 0, &src, &dst, 1.0, &frame, field)
            .expect("mosh");
        gpu.wait_frame(rec.submit());
        gpu.read_back(&dst)
    };

    // Push the picture a long way off the source, then let the field go.
    for _ in 0..10 {
        draw(true);
    }
    let early = mean_abs_difference(&draw(false), &gpu.read_back(&src));
    let frames = 400u32;
    let mut late = 0.0;
    for _ in 0..frames {
        late = mean_abs_difference(&draw(false), &gpu.read_back(&src));
    }
    // (1 - 0.004)^400 is about 0.20, so four fifths of the fresh picture is back in.
    println!(
        "{width}x{height}, amount 1.0, {frames} frames after the field stopped: mean |output - \
         fresh| {early:.2}/255 -> {late:.2}/255 (leak floor 0.004 per frame: expect about a fifth)"
    );
    assert!(
        early > 4.0,
        "the field never moved the picture ({early:.2}/255 from the fresh frame): nothing to leak"
    );
    assert!(
        late < early * 0.5,
        "amount 1.0 froze the layer: {late:.2}/255 from the fresh frame after {frames} frames, \
         from {early:.2}/255"
    );
    assert_no_validation_messages(&gpu);
}

/// (d) The I-frame rules: a first frame, a playback jump, and `amount = 0` refresh the buffer;
/// a bar boundary and a DROP do not (`decide()` in engine.c has no such rule, and an I-frame -
/// an exact copy of the source - is the opposite of a mosh). The observable is the mean absolute
/// difference from the fresh frame: zero on an I-frame, non-zero while the buffer is carried.
#[test]
fn iframes_come_from_a_first_frame_a_jump_or_amount_zero() {
    let gpu = gpu();
    let (width, height) = (320, 180);
    let pattern = pipeline(&gpu, PATTERN_WGSL, 16, 0);
    let mut mosh = MoshPass::new(&gpu, width, height).expect("mosh pass");
    let src = LayerTarget::new(&gpu, width, height);
    let dst = LayerTarget::new(&gpu, width, height);
    let plain = AnalysisFrame::default();
    let bar_width = 20.0f32;
    let speed = 4.0f32;

    let mut step = 0u32;
    let mut draw = |frame: &AnalysisFrame, amount: f32| -> (bool, f32) {
        let x0 = 120.0 + speed * step as f32;
        let bar = vec4_uniform([x0, x0 + bar_width, 0.0, 0.0]);
        step += 1;
        let mut rec = FrameRecorder::new(&gpu);
        rec.draw(&pattern, &src, &bar, &[]).expect("pattern");
        let applied = mosh
            .apply(&mut rec, 0, &src, &dst, amount, frame, None)
            .expect("mosh");
        gpu.wait_frame(rec.submit());
        (applied, mean_abs_difference(&gpu.read_back(&dst), &gpu.read_back(&src)))
    };

    // The first frame is an I-frame: the output *is* the source.
    let (applied, first) = draw(&plain, 0.9);
    assert!(applied);
    println!("first frame: mean |output - fresh| {first:.2}/255");
    assert!(
        first < 1.0,
        "the first frame is not a copy of the source ({first:.2}/255)"
    );

    // Four moving frames: the buffer is carried, so the output is not the source.
    let mut carried = first;
    for _ in 0..4 {
        carried = draw(&plain, 0.9).1;
    }
    assert!(
        carried > 1.0,
        "the buffer is not being carried at all ({carried:.2}/255 from the fresh frame)"
    );

    // A DROP on a bar boundary is an ordinary P-frame.
    let drop = AnalysisFrame {
        bar_index: 8,
        events: EventFlags::DROP | EventFlags::BAR,
        ..AnalysisFrame::default()
    };
    let (applied, dropped) = draw(&drop, 0.9);
    assert!(applied);
    println!("after 4 P-frames {carried:.2}/255, on DROP/bar 8 {dropped:.2}/255");
    assert!(
        dropped > 1.0,
        "DROP on bar 8 refreshed the buffer ({dropped:.2}/255 from the fresh frame): an I-frame \
         reached a musical event"
    );

    // A playback jump backwards is an I-frame.
    let jump = AnalysisFrame {
        time_s: -1.0,
        ..AnalysisFrame::default()
    };
    let (applied, jumped) = draw(&jump, 0.9);
    assert!(applied);
    println!("on a backwards time jump {jumped:.2}/255");
    assert!(
        jumped < 1.0,
        "a backwards time jump did not refresh the buffer ({jumped:.2}/255)"
    );

    // `amount = 0` writes nothing and forgets the buffer, so the next non-zero amount is an
    // I-frame even with no time jump (the web app's `if (m <= 0.001) iframe = 1`).
    let (applied, _) = draw(&plain, 0.0);
    assert!(!applied, "amount 0 must not write the output");
    let (applied, after_zero) = draw(&plain, 0.9);
    assert!(applied);
    println!("first frame after amount 0 {after_zero:.2}/255");
    assert!(
        after_zero < 1.0,
        "after amount 0 the next mosh frame was not an I-frame ({after_zero:.2}/255)"
    );
    assert_no_validation_messages(&gpu);
}

/// `decide()`'s `entered` rule, through the facade, on pictures this test owns: only Cut, Strobe
/// and the first clip take an I-frame. A **Crossfade** entry is an ordinary P-frame with the
/// residual on (the transition blends the two pictures itself); **Datamosh** and **Melt** keep the
/// buffer and skip the residual, so the old clip's picture is still there.
///
/// The *entry frame itself* cannot tell Cut from Crossfade apart: the residual (`buf + s - prv`)
/// reconstructs a changed source exactly, which is the whole point of it. So this measures the
/// frame *after* the entry, once the new clip is still and its vectors are zero:
///
/// * `ghost`, the difference from the picture the layer was given, is 0 only after an I-frame;
/// * `lit`, the mean over the new picture's own bar, is ~0 when the old clip is still carried.
#[test]
fn clip_entry_rules_follow_decide() {
    let gpu = gpu();
    let (width, height) = (320u32, 180u32);
    // A 20 px bright bar on black, in `OverlayLayer::set_frame`'s BGRA order.
    let bar = |x0: u32| -> Vec<u8> {
        let mut bytes = Vec::with_capacity((width * height * 4) as usize);
        for _y in 0..height {
            for x in 0..width {
                let lit = if (x0..x0 + 20).contains(&x) { 255 } else { 0 };
                bytes.extend_from_slice(&[lit, lit, lit, 255]);
            }
        }
        bytes
    };

    let state = |clip: u32, kind: Option<TransitionKind>| LayerState {
        layer: 0,
        clip: Some(clip),
        visualizer: Some("overlay".to_string()),
        params: ClipParams::default(),
        effects: vec![EffectSlot {
            kind: EffectKind::Mosh,
            amount: 0.9,
            param: 0.0,
        }],
        transition: kind.map(|kind| ActiveTransition {
            kind,
            progress: 1.0,
            from: Some(1),
        }),
        blend: BlendMode::Normal,
        opacity: 1.0,
        seed: 0,
    };

    // Four frames of clip 1 (the bar slides 4 px a frame, so the buffer holds a real smear), then
    // the entry frame of clip 2 with `kind`. `x0` is where clip 2's bar is: 76 is the last warm
    // frame's position (a picture that does not change), 240 is somewhere else entirely.
    let entry_frame = |kind: Option<TransitionKind>, x0: u32| -> (f32, f32) {
        let mut visuals = Visuals::new(&gpu, width, height).expect("visuals");
        let mut slot = LayerSlot::new(&gpu, 0, width, height);
        let mut frame = AnalysisFrame::default();
        let draw = |visuals: &mut Visuals,
                        slot: &mut LayerSlot,
                        frame: &AnalysisFrame,
                        clip: u32,
                        kind: Option<TransitionKind>,
                        x0: u32|
         -> (f32, f32) {
            let picture = bar(x0);
            visuals
                .overlay
                .set_frame(&gpu, 0, width, height, &picture)
                .expect("overlay frame");
            let mut rec = FrameRecorder::new(&gpu);
            visuals
                .draw_layer(&mut rec, slot, &state(clip, kind), frame, 0.0)
                .expect("draw layer");
            gpu.wait_frame(rec.submit());
            let output = gpu.read_back(slot.output());
            let row = red_row(&output, width, height / 2);
            (mean(&row, 240, 260), mean_abs_difference(&output, &picture))
        };
        for step in 0..4u32 {
            draw(&mut visuals, &mut slot, &frame, 1, None, 60 + 4 * step);
            frame.time_s += 1.0 / 60.0;
        }
        draw(&mut visuals, &mut slot, &frame, 2, kind, x0)
    };

    // A picture that does not change across the change: an I-frame is an exact copy, a P-frame
    // with the residual is not (the residual is zero, so the carried smear comes through whole).
    let cut_same = entry_frame(None, 76);
    let crossfade_same = entry_frame(Some(TransitionKind::Crossfade), 76);
    // A picture that does change: the residual brings the new one straight in, `skipres` keeps
    // the old one.
    let cut_new = entry_frame(None, 240);
    let crossfade_new = entry_frame(Some(TransitionKind::Crossfade), 240);
    let datamosh_new = entry_frame(Some(TransitionKind::Datamosh), 240);
    let melt_new = entry_frame(Some(TransitionKind::Melt), 240);
    println!(
        "{width}x{height}, mosh 0.9, the entry frame: (new bar's columns, distance from the given \
         picture), 0..255: Cut/same {cut_same:?}, Crossfade/same {crossfade_same:?}, Cut/new \
         {cut_new:?}, Crossfade/new {crossfade_new:?}, Datamosh/new {datamosh_new:?}, Melt/new \
         {melt_new:?}"
    );
    assert!(
        cut_same.1 < 1.0,
        "a Cut entry is not an exact copy of an unchanged picture ({cut_same:?})"
    );
    assert!(
        crossfade_same.1 > 1.0,
        "a Crossfade entry took an I-frame ({crossfade_same:?}): the buffer should have been kept, \
         so the carried smear should still be there"
    );
    assert!(
        cut_new.0 > 200.0,
        "a Cut entry did not show the new picture ({cut_new:?})"
    );
    assert!(
        crossfade_new.0 > 200.0,
        "a Crossfade entry lost the new picture ({crossfade_new:?}): its residual was skipped"
    );
    assert!(
        datamosh_new.0 < 60.0 && melt_new.0 < 60.0,
        "a Datamosh ({datamosh_new:?}) or Melt ({melt_new:?}) entry did not carry the old picture"
    );
    assert_no_validation_messages(&gpu);
}

/// (b) A Datamosh clip change carries the old clip's picture into the new clip for several frames;
/// a Cut does not. Driven through `Visuals::draw_layer`, because that is what decides between
/// `carry_over` and `invalidate`.
///
/// The old clip is a bright scene with a mosh slot; the new clip is black (an id that is not a
/// visualizer draws black) and has **no** mosh slot at all, so any picture after the change comes
/// from the carry-over and from the forced mosh amount.
#[test]
fn datamosh_clip_change_carries_the_old_picture_and_a_cut_does_not() {
    let gpu = gpu();
    let (width, height) = (128, 72);

    let state = |clip: u32, visualizer: &str, mosh: Option<f32>, kind: Option<TransitionKind>| {
        LayerState {
            layer: 0,
            clip: Some(clip),
            visualizer: Some(visualizer.to_string()),
            params: ClipParams::default(),
            effects: mosh
                .map(|amount| {
                    vec![EffectSlot {
                        kind: EffectKind::Mosh,
                        amount,
                        param: 0.0,
                    }]
                })
                .unwrap_or_default(),
            transition: kind.map(|kind| ActiveTransition {
                kind,
                progress: 0.2,
                from: Some(1),
            }),
            blend: BlendMode::Normal,
            opacity: 1.0,
            seed: 0,
        }
    };

    // The old clip: a bright scene with mosh 0.8, so the layer's buffer holds a full picture
    // before the change.
    let bright = state(1, "tunnel", Some(0.8), None);

    let run = |kind: TransitionKind| -> (f32, Vec<f32>) {
        let mut visuals = Visuals::new(&gpu, width, height).expect("visuals");
        let mut slot = LayerSlot::new(&gpu, 0, width, height);
        let mut frame = AnalysisFrame::default();
        // Six frames of the old clip, animating so the block matcher has motion to find. The
        // scene is drawn by the facade, so the picture is a real scene, not the bar.
        for step in 0..6 {
            let mut rec = FrameRecorder::new(&gpu);
            visuals
                .draw_layer(&mut rec, &mut slot, &bright, &frame, step as f32 * 0.1)
                .expect("old clip");
            gpu.wait_frame(rec.submit());
            frame.time_s += 1.0 / 60.0;
        }
        let old = mean_luma(&gpu.read_back(slot.output()));
        // The new clip: black, no mosh slot, entering on `kind`.
        let new_clip = state(2, "not-a-visualizer", None, Some(kind));
        let mut series = Vec::new();
        for progress in [0.2f32, 0.5, 0.8, 1.0] {
            let mut rec = FrameRecorder::new(&gpu);
            let entering = LayerState {
                transition: Some(ActiveTransition {
                    kind,
                    progress,
                    from: Some(1),
                }),
                ..new_clip.clone()
            };
            visuals
                .draw_layer(&mut rec, &mut slot, &entering, &frame, 1.0)
                .expect("new clip");
            gpu.wait_frame(rec.submit());
            series.push(mean_luma(&gpu.read_back(slot.output())));
            frame.time_s += 1.0 / 60.0;
        }
        (old, series)
    };

    let (old, carried) = run(TransitionKind::Datamosh);
    let (_, cut) = run(TransitionKind::Cut);
    println!(
        "{width}x{height}: old clip mean luma {old:.1}/255; after the clip change, a Datamosh \
         entry holds {carried:?} and a Cut holds {cut:?}"
    );
    assert!(old > 4.0, "the old clip drew nothing ({old:.1}/255)");
    assert!(
        carried[0] > old * 0.5,
        "a Datamosh entry lost the old picture on its first frame ({:.1} of {old:.1}/255)",
        carried[0]
    );
    assert!(
        carried[2] > old * 0.25,
        "a Datamosh entry lost the old picture by progress 0.8 ({:.1} of {old:.1}/255)",
        carried[2]
    );
    assert!(
        cut[0] < old * 0.1,
        "a Cut kept the old picture ({:.1} of {old:.1}/255)",
        cut[0]
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

/// The 4x4 subsample `mosh_motion_fast` compares instead of the full 8x8 block (V-budget): the
/// same exhaustive search, a quarter of the samples. This test drives both shaders over the same
/// moving pictures and reports how far the vectors move.
///
/// `mosh_motion.wgsl` stays in the tree as the reference (and as the definition of the search);
/// `MoshPass` uses `mosh_motion_fast`.
#[test]
fn motion_subsample_matches_the_full_search() {
    let gpu = gpu();
    let (width, height) = (960u32, 540u32);
    let grid = (320u32, 180u32);
    let pattern = pipeline(&gpu, PATTERN_WGSL, 16, 0);
    let motion_uniform_bytes = vec4_uniform([grid.0 as f32, grid.1 as f32, 0.0, 0.0]);
    let full = Pipeline::fullscreen(
        &gpu,
        mosh_shaders::spirv("mosh_motion").expect("mosh_motion"),
        16,
        2,
    )
    .expect("full search pipeline");
    let fast = Pipeline::fullscreen(
        &gpu,
        mosh_shaders::spirv("mosh_motion_fast").expect("mosh_motion_fast"),
        16,
        2,
    )
    .expect("subsampled search pipeline");
    let current = LayerTarget::new(&gpu, width, height);
    let previous = LayerTarget::new(&gpu, width, height);
    let full_vectors = LayerTarget::new(&gpu, 40, 23);
    let fast_vectors = LayerTarget::new(&gpu, 40, 23);

    let mut identical = 0usize;
    let mut compared = 0usize;
    let mut worst = 0.0f32;
    let mut moved = 0usize;
    // A bar moving 4 px a frame, and a second run moving diagonal stripes so the match is not
    // only ever a single horizontal edge.
    for step in 0..6u32 {
        let x0 = 200.0 + 4.0 * step as f32;
        let bar = vec4_uniform([x0, x0 + 20.0, 0.0, 0.0]);
        let mut rec = FrameRecorder::new(&gpu);
        rec.draw(&pattern, &current, &bar, &[]).expect("pattern");
        rec.draw(
            &full,
            &full_vectors,
            &motion_uniform_bytes,
            &[&current, &previous],
        )
        .expect("full motion");
        rec.draw(
            &fast,
            &fast_vectors,
            &motion_uniform_bytes,
            &[&current, &previous],
        )
        .expect("fast motion");
        rec.draw(&pattern, &previous, &bar, &[]).expect("previous");
        gpu.wait_frame(rec.submit());

        // Read one row of vectors from the middle of the field: the bar crosses it.
        let full_row = red_row(&gpu.read_back(&full_vectors), 40, 11);
        let fast_row = red_row(&gpu.read_back(&fast_vectors), 40, 11);
        for (a, b) in full_row.iter().zip(&fast_row) {
            compared += 1;
            let difference = f32::from(*a) - f32::from(*b);
            if difference.abs() < 0.5 {
                identical += 1;
            }
            worst = worst.max(difference.abs());
            if *a != 0 || *b != 0 {
                moved += 1;
            }
        }
    }
    println!(
        "motion vectors, {compared} compared (row 11 of a 40x23 field over 6 frames): {identical} \
         identical, {} differ, {moved} non-zero, worst readback difference {worst:.0}/255",
        compared - identical
    );
    assert!(
        compared > 0 && moved > 0,
        "the bar has to move a vector for this to compare anything"
    );
    // The vector is stored as `clamp(-offset)` per channel, so the readback cannot tell one
    // negative offset from another: a "difference" here is either a one-grid-pixel move or the
    // sign of a zero crossing, never the 8-pixel miss a broken search would show.
    assert!(
        identical * 100 >= compared * 95,
        "{} of {compared} vectors differ between the full and the subsampled search",
        compared - identical
    );
    assert_no_validation_messages(&gpu);
}

/// `mosh_motion_fast` has to fit `mosh-render`'s layout and uniform size like the shader it
/// replaces, or `MoshPass` would bind the wrong things.
#[test]
fn the_subsampled_motion_shader_matches_the_layout() {
    let reflected = bindings("mosh_motion_fast").expect("mosh_motion_fast");
    assert_eq!(reflected.len(), 1 + 2 * 2, "uniform plus two (image, sampler) pairs");
    assert_eq!(uniform_bytes("mosh_motion_fast"), Some(16));
    assert_eq!(uniform_bytes("mosh_grid"), Some(16), "the grid pass");
    assert_eq!(bindings("mosh_grid").expect("mosh_grid").len(), 1 + 2);
}

/// `invalidate` (called by the facade on a clip change) makes the next frame an I-frame: the
/// output becomes an exact copy of the source instead of a carried picture.
///
/// The observable cannot be "the new picture appears": the residual (`src - prv`) carries a hard
/// source change in by itself on the next P-frame, exactly as it does in the web app. What an
/// I-frame changes is that the carried picture is *dropped*, so the measure here is the mean
/// absolute difference from the source on a moving picture.
#[test]
fn invalidate_refreshes_the_buffer() {
    let gpu = gpu();
    let (width, height) = (320, 180);
    let pattern = pipeline(&gpu, PATTERN_WGSL, 16, 0);
    let mut mosh = MoshPass::new(&gpu, width, height).expect("mosh pass");
    let src = LayerTarget::new(&gpu, width, height);
    let dst = LayerTarget::new(&gpu, width, height);
    let mut step = 0u32;
    let mut draw = |mosh: &mut MoshPass| {
        let x0 = 100.0 + 4.0 * step as f32;
        step += 1;
        let bar = vec4_uniform([x0, x0 + 20.0, 0.0, 0.0]);
        let mut rec = FrameRecorder::new(&gpu);
        rec.draw(&pattern, &src, &bar, &[]).expect("pattern");
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
        mean_abs_difference(&gpu.read_back(&dst), &gpu.read_back(&src))
    };
    let mut carried = 0.0;
    for _ in 0..5 {
        carried = draw(&mut mosh);
    }
    mosh.invalidate(0);
    let fresh = draw(&mut mosh);
    println!("amount 0.9: carried {carried:.2}/255 from the fresh frame, after invalidate {fresh:.2}/255");
    assert!(
        carried > 1.0,
        "nothing is being carried ({carried:.2}/255 from the fresh frame)"
    );
    assert!(
        fresh < 0.01,
        "invalidate did not refresh the buffer ({fresh:.2}/255 from the fresh frame)"
    );
    assert_no_validation_messages(&gpu);
}

/// A ping-pong `src` (like `LayerSlot::scene`) keeps last frame's picture in its back image so
/// `MoshPass::apply` samples it directly with no `mosh copy` pass on P-frames; a single-image
/// `src` falls back to copying into `prev_src`. Both paths must produce bit-identical output.
#[test]
fn ping_pong_and_single_image_sources_produce_identical_mosh() {
    let gpu = gpu();
    let (width, height) = (320, 180);
    let pattern = pipeline(&gpu, PATTERN_WGSL, 16, 0);
    let mut mosh_pp = MoshPass::new(&gpu, width, height).expect("mosh pass ping-pong");
    let mut mosh_single = MoshPass::new(&gpu, width, height).expect("mosh pass single");
    let src_pp = LayerTarget::new(&gpu, width, height);
    let dst_pp = LayerTarget::new(&gpu, width, height);
    let src_single = LayerTarget::new_single(&gpu, width, height);
    let dst_single = LayerTarget::new(&gpu, width, height);

    let mut last_pp_passes: Vec<String> = Vec::new();
    let mut last_single_passes: Vec<String> = Vec::new();
    for step in 0..6u32 {
        let x0 = 60.0 + 6.0 * step as f32;
        let bar = vec4_uniform([x0, x0 + 24.0, 0.0, 0.0]);
        let frame = AnalysisFrame {
            time_s: step as f64 / 60.0,
            ..Default::default()
        };

        let mut rec = FrameRecorder::new(&gpu);
        rec.draw(&pattern, &src_pp, &bar, &[]).expect("pattern pp");
        mosh_pp
            .apply(&mut rec, 0, &src_pp, &dst_pp, 0.8, &frame, None)
            .expect("mosh pp");
        let timings_pp = gpu.wait_frame(rec.submit());

        let mut rec = FrameRecorder::new(&gpu);
        rec.draw(&pattern, &src_single, &bar, &[])
            .expect("pattern single");
        mosh_single
            .apply(&mut rec, 0, &src_single, &dst_single, 0.8, &frame, None)
            .expect("mosh single");
        let timings_single = gpu.wait_frame(rec.submit());

        if step > 0 {
            last_pp_passes = timings_pp
                .per_pass_ms
                .iter()
                .map(|(n, _)| (*n).to_string())
                .collect();
            last_single_passes = timings_single
                .per_pass_ms
                .iter()
                .map(|(n, _)| (*n).to_string())
                .collect();
            assert!(
                !last_pp_passes.iter().any(|n| n == "mosh copy"),
                "frame {step}: ping-pong src must not record mosh copy on P-frames: {last_pp_passes:?}"
            );
            assert!(
                last_single_passes.iter().any(|n| n == "mosh copy"),
                "frame {step}: single-image src must record mosh copy fallback: {last_single_passes:?}"
            );
        }
    }

    let out_pp = gpu.read_back(&dst_pp);
    let out_single = gpu.read_back(&dst_single);
    assert_eq!(
        out_pp, out_single,
        "ping-pong zero-copy output differed from single-image fallback"
    );
    println!(
        "ping-pong P-frame passes: {last_pp_passes:?} | single-image passes: {last_single_passes:?}"
    );
    assert_no_validation_messages(&gpu);
}

#[test]
fn freeze_bloom_grows_mv_scale_skips_motion_and_resets_on_unfreeze() {
    const TEXTURE_WGSL: &str = r#"
struct S {
    shift: vec4<f32>,
};
@group(0) @binding(0) var<uniform> s: S;
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    if (s.shift.z > 0.5) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
    let x = pos.x + s.shift.x;
    let y = pos.y + s.shift.y;
    let r = 0.5 + 0.45 * sin(x * 0.37) * cos(y * 0.23);
    let g = 0.5 + 0.45 * sin((x + y) * 0.29 + 1.1);
    let b = 0.5 + 0.45 * cos((x - y * 0.7) * 0.41 + 2.3);
    return vec4<f32>(r, g, b, 1.0);
}
"#;
    let gpu = gpu();
    let (width, height) = (320u32, 180u32);
    let texture = pipeline(&gpu, TEXTURE_WGSL, 16, 0);
    let src = LayerTarget::new(&gpu, width, height);
    let dst = LayerTarget::new(&gpu, width, height);
    let mut mosh = MoshPass::new(&gpu, width, height).expect("mosh pass");

    // 3 warm frames: I-frame + 2 P-frames of the 2D texture sliding right 4 px/frame so every 8x8
    // analysis block holds a real non-zero motion vector before the freeze.
    for step in 0..3u32 {
        let shift = -4.0 * step as f32;
        let frame = AnalysisFrame {
            time_s: step as f64 / 60.0,
            ..Default::default()
        };
        let mut rec = FrameRecorder::new(&gpu);
        rec.draw(&texture, &src, &vec4_uniform([shift, 0.0, 0.0, 0.0]), &[])
            .expect("texture");
        mosh.apply(&mut rec, 0, &src, &dst, 0.85, &frame, None)
            .expect("mosh");
        gpu.wait_frame(rec.submit());
        assert_eq!(mosh.mv_scale(0), 1.0, "unfrozen frame {step} keeps mv_scale 1.0");
    }
    let before_freeze = gpu.read_back(&dst);

    // Freeze for 10 frames while the source is completely STILL (and on frame 10 turns solid white
    // to prove `skipres` suppresses residual & leak during freeze):
    // * motion estimation must be skipped (`mosh motion` absent from passes);
    // * mv_scale must multiply by 1.02 each frame;
    // * the carried picture must keep advecting every frame along the frozen vectors.
    let mut prev_out = before_freeze.clone();
    let mut frozen_step_deltas = Vec::new();
    for k in 1..=10u32 {
        let frame = AnalysisFrame {
            time_s: (2 + k) as f64 / 60.0,
            ..Default::default()
        };
        let solid_white = if k == 10 { 1.0 } else { 0.0 };
        let mut rec = FrameRecorder::new(&gpu);
        rec.draw(
            &texture,
            &src,
            &vec4_uniform([-8.0, 0.0, solid_white, 0.0]),
            &[],
        )
        .expect("static texture");
        mosh.set_freeze(0, true);
        mosh.apply(&mut rec, 0, &src, &dst, 0.85, &frame, None)
            .expect("frozen mosh");
        let timings = gpu.wait_frame(rec.submit());
        let expected = 1.02f32.powi(k as i32);
        assert!(
            (mosh.mv_scale(0) - expected).abs() < 1e-5,
            "frozen frame {k}: mv_scale {} != expected {expected}",
            mosh.mv_scale(0)
        );
        assert!(
            !timings.per_pass_ms.iter().any(|(name, _)| *name == "mosh motion"),
            "frozen frame {k}: motion pass must be skipped: {:?}",
            timings.per_pass_ms
        );
        let curr_out = gpu.read_back(&dst);
        frozen_step_deltas.push(mean_abs_difference(&prev_out, &curr_out));
        prev_out = curr_out;
    }
    let total_drift = mean_abs_difference(&before_freeze, &prev_out);
    let mean_luma_after_white = mean(&red_row(&prev_out, width, height / 2), 0, width.into());
    println!(
        "freeze bloom (10 frames, static source): mv_scale {:.4}, total drift {:.1}/255, mean red after white src {:.1}/255, per-frame deltas {:?}",
        mosh.mv_scale(0),
        total_drift,
        mean_luma_after_white,
        frozen_step_deltas
    );
    assert!(
        total_drift > 25.0 && frozen_step_deltas.iter().all(|&d| d > 15.0),
        "frozen motion vectors did not keep carrying/blooming the picture (total {total_drift:.1}, steps {frozen_step_deltas:?})"
    );
    assert!(
        mean_luma_after_white < 180.0,
        "frozen frame leaked solid-white source despite skipres (mean red {mean_luma_after_white:.1}/255)"
    );

    // Unfreeze on the next frame: motion estimation resumes and mv_scale resets to 1.0.
    let frame = AnalysisFrame {
        time_s: 13.0 / 60.0,
        ..Default::default()
    };
    let mut rec = FrameRecorder::new(&gpu);
    rec.draw(&texture, &src, &vec4_uniform([-8.0, 0.0, 0.0, 0.0]), &[])
        .expect("static texture");
    mosh.set_freeze(0, false);
    mosh.apply(&mut rec, 0, &src, &dst, 0.85, &frame, None)
        .expect("unfrozen mosh");
    let timings = gpu.wait_frame(rec.submit());
    assert_eq!(mosh.mv_scale(0), 1.0, "unfreeze must reset mv_scale to 1.0");
    assert!(
        timings.per_pass_ms.iter().any(|(name, _)| *name == "mosh motion"),
        "unfreeze must resume motion estimation: {:?}",
        timings.per_pass_ms
    );
    assert_no_validation_messages(&gpu);
}

#[test]
fn visuals_triggers_freeze_bloom_on_melt_and_bar_end_bloom() {
    let gpu = gpu();
    let (width, height) = (160u32, 90u32);
    let mut visuals = Visuals::new(&gpu, width, height).expect("visuals");
    let mut slot = LayerSlot::new(&gpu, 0, width, height);

    let make_state = |clip: u32, bloom: f32, transition: Option<ActiveTransition>| LayerState {
        layer: 0,
        clip: Some(clip),
        visualizer: Some("tunnel".to_string()),
        params: ClipParams {
            p: [1.0, 0.5, 0.2, 0.5, 0.0, 0.0, 0.0, 0.0],
            palette: 1,
            seed: 1,
        },
        effects: vec![EffectSlot {
            kind: EffectKind::Mosh,
            amount: 0.85,
            param: bloom,
        }],
        transition,
        blend: BlendMode::Normal,
        opacity: 1.0,
        seed: 1,
    };

    let mut step = 0u32;
    let mut draw_step = |visuals: &mut Visuals,
                         slot: &mut LayerSlot,
                         state: &LayerState,
                         bar_phase: f32| {
        let frame = AnalysisFrame {
            time_s: step as f64 / 60.0,
            bar_phase,
            playing: true,
            ..Default::default()
        };
        let mut rec = FrameRecorder::new(&gpu);
        visuals
            .draw_layer(&mut rec, slot, state, &frame, step as f32 / 60.0)
            .expect("draw layer");
        gpu.wait_frame(rec.submit());
        step += 1;
    };

    // Bloom 0.5 -> freeze when bar_phase > 0.75.
    let bloom_state = make_state(1, 0.5, None);
    draw_step(&mut visuals, &mut slot, &bloom_state, 0.20); // I-frame
    draw_step(&mut visuals, &mut slot, &bloom_state, 0.50); // P-frame, unfrozen
    assert_eq!(visuals.mosh.mv_scale(0), 1.0);

    for (k, phase) in [0.78f32, 0.84, 0.91].into_iter().enumerate() {
        draw_step(&mut visuals, &mut slot, &bloom_state, phase);
        let expected = 1.02f32.powi((k + 1) as i32);
        assert!(
            (visuals.mosh.mv_scale(0) - expected).abs() < 1e-5,
            "bloom phase {phase}: mv_scale {} != {expected}",
            visuals.mosh.mv_scale(0)
        );
    }

    // Next bar downbeat: unfreezes back to 1.0.
    draw_step(&mut visuals, &mut slot, &bloom_state, 0.05);
    assert_eq!(visuals.mosh.mv_scale(0), 1.0);

    // Melt transition (even with bloom = 0.0) freezes and grows mv_scale while progress < 1.0.
    for (k, progress) in [0.15f32, 0.45, 0.75].into_iter().enumerate() {
        let melt_state = make_state(
            2,
            0.0,
            Some(ActiveTransition {
                kind: TransitionKind::Melt,
                progress,
                from: Some(1),
            }),
        );
        draw_step(&mut visuals, &mut slot, &melt_state, 0.25);
        let expected = 1.02f32.powi((k + 1) as i32);
        assert!(
            (visuals.mosh.mv_scale(0) - expected).abs() < 1e-5,
            "melt progress {progress}: mv_scale {} != {expected}",
            visuals.mosh.mv_scale(0)
        );
    }

    // Once Melt finishes, mv_scale resets to 1.0.
    draw_step(&mut visuals, &mut slot, &make_state(2, 0.0, None), 0.35);
    assert_eq!(visuals.mosh.mv_scale(0), 1.0);
    assert_no_validation_messages(&gpu);
}


