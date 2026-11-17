//! L-stack gate: Adjustment and Mask layers, offscreen.
//!
//! Every test composes a whole layer stack through `mosh_visuals::stack::Stack` and reads the
//! result back; the plain-stack test compares against the compositor itself, which is the
//! reference (a stack with no Adjustment and no Mask must be that pass, unchanged).
//!
//! Pictures are drawn with the shared fixtures in `tests/common`: `pattern` for a picture that
//! moves under every effect, `solid` for a flat colour, plus one local shader whose luminance is
//! black on the left half and white on the right half, which is what a Mask layer is for.

mod common;

use common::*;
use mosh_core::{
    AnalysisFrame, BlendMode, ClipParams, EffectKind, EffectSlot, LayerKind, LayerState,
};
use mosh_render::{CompositeTarget, Compositor, FrameRecorder, LayerTarget, OutputImage};
use mosh_visuals::stack::{Stack, StackLayer};
use mosh_visuals::LayerSlot;

/// Small on purpose: this lane's tests run beside other lanes' on one machine.
const W: u32 = 96;
const H: u32 = 64;

/// A vertical split: black left of `split`, white right of it, so the mask's luminance is 0 on
/// the left half and 1 on the right half.
const SPLIT_WGSL: &str = r#"
struct Params {
    res: vec2<f32>,
    split: f32,
    pad: f32,
};

@group(0) @binding(0) var<uniform> params: Params;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = pos.xy / params.res;
    let value = select(0.0, 1.0, uv.x >= params.split);
    return vec4<f32>(value, value, value, 1.0);
}
"#;

fn spirv(source: &str) -> Vec<u32> {
    let module = naga::front::wgsl::parse_str(source).expect("test shader parses");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator.validate(&module).expect("test shader validates");
    let options = naga::back::spv::Options {
        lang_version: (1, 3),
        ..Default::default()
    };
    naga::back::spv::write_vec(&module, &info, &options, None).expect("test shader compiles")
}

fn float_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Draws a black/white split into `target`.
fn draw_split(gpu: &mosh_render::Gpu, target: &LayerTarget) {
    let pipeline =
        mosh_render::Pipeline::fullscreen(gpu, &spirv(SPLIT_WGSL), 16, 0).expect("split pipeline");
    let uniforms = float_bytes(&[target.width() as f32, target.height() as f32, 0.5, 0.0]);
    let mut recorder = FrameRecorder::new(gpu);
    recorder
        .draw(&pipeline, target, &uniforms, &[])
        .expect("draw split");
    gpu.wait_frame(recorder.submit());
}

fn effect(kind: EffectKind, amount: f32, param: f32) -> EffectSlot {
    EffectSlot {
        kind,
        amount,
        param,
    }
}

/// A layer's clip state: only the effects matter to the stack (they are what an Adjustment layer
/// applies); the picture was drawn into the slot already.
fn state(layer: usize, effects: Vec<EffectSlot>) -> LayerState {
    LayerState {
        layer,
        clip: Some(layer as u32 + 1),
        visualizer: Some("tunnel".to_string()),
        params: ClipParams::default(),
        effects,
        transition: None,
        blend: BlendMode::Normal,
        opacity: 1.0,
        seed: 7,
    }
}

fn entry<'a>(
    slot: &'a LayerSlot,
    kind: LayerKind,
    blend: BlendMode,
    opacity: f32,
    state: &'a LayerState,
) -> StackLayer<'a> {
    StackLayer::new(slot, kind, blend, opacity, state)
}

/// A picture layer at the test size, drawn with the pattern.
fn picture_layer(gpu: &mosh_render::Gpu, index: usize) -> LayerSlot {
    let slot = LayerSlot::new(gpu, index, W, H);
    draw_pattern(gpu, &pattern_pipeline(gpu), &slot.scene);
    slot
}

/// A solid-colour picture layer.
fn solid_layer(gpu: &mosh_render::Gpu, index: usize, color: [f32; 4]) -> LayerSlot {
    let slot = LayerSlot::new(gpu, index, W, H);
    draw_solid(gpu, &solid_pipeline(gpu), &slot.scene, color);
    slot
}

/// An Adjustment or Mask layer draws nothing of its own, so its slot is only there to keep the
/// layer list aligned with the show's layer indices.
fn blank_layer(gpu: &mosh_render::Gpu, index: usize) -> LayerSlot {
    LayerSlot::new(gpu, index, W, H)
}

/// Composes `layers` and reads the result back.
fn compose(gpu: &mosh_render::Gpu, stack: &mut Stack, layers: &[StackLayer<'_>]) -> Vec<u8> {
    let out = OutputImage::new(gpu, W, H);
    stack
        .compose(
            FrameRecorder::new(gpu),
            None,
            layers,
            CompositeTarget::Output(&out),
            &AnalysisFrame::default(),
            0.0,
        )
        .expect("compose the stack");
    gpu.read_back_output(&out)
}

/// What today's compositor produces for the same layers: the reference a plain stack is measured
/// against.
fn reference(gpu: &mosh_render::Gpu, layers: &[(&LayerTarget, BlendMode, f32)]) -> Vec<u8> {
    let out = OutputImage::new(gpu, W, H);
    let mut compositor = Compositor::new(gpu).expect("compositor");
    compositor.composite(layers, &out).expect("composite");
    gpu.read_back_output(&out)
}

fn pixel(pixels: &[u8], x: u32, y: u32) -> [i32; 4] {
    let at = ((y * W + x) * 4) as usize;
    [
        i32::from(pixels[at]),
        i32::from(pixels[at + 1]),
        i32::from(pixels[at + 2]),
        i32::from(pixels[at + 3]),
    ]
}

fn assert_rgb(pixels: &[u8], x: u32, y: u32, expected: [i32; 3], tolerance: i32, what: &str) {
    let got = pixel(pixels, x, y);
    for channel in 0..3 {
        assert!(
            (got[channel] - expected[channel]).abs() <= tolerance,
            "{what} at ({x},{y}): channel {channel} is {}, expected {} +-{tolerance} (pixel {got:?})",
            got[channel],
            expected[channel]
        );
    }
}

#[test]
fn a_stack_without_adjustment_or_mask_is_todays_compositor() {
    let gpu = gpu();
    let slots = [
        solid_layer(&gpu, 0, [0.9, 0.1, 0.2, 1.0]),
        solid_layer(&gpu, 1, [0.2, 0.8, 0.3, 1.0]),
        picture_layer(&gpu, 2),
    ];
    let states = [
        state(0, Vec::new()),
        state(1, Vec::new()),
        state(2, Vec::new()),
    ];
    let plan = [
        entry(
            &slots[0],
            LayerKind::Visualizer,
            BlendMode::Normal,
            1.0,
            &states[0],
        ),
        entry(
            &slots[1],
            LayerKind::Visualizer,
            BlendMode::Add,
            0.7,
            &states[1],
        ),
        entry(
            &slots[2],
            LayerKind::Visualizer,
            BlendMode::Screen,
            0.5,
            &states[2],
        ),
    ];

    let reference = reference(
        &gpu,
        &[
            (slots[0].output(), BlendMode::Normal, 1.0),
            (slots[1].output(), BlendMode::Add, 0.7),
            (slots[2].output(), BlendMode::Screen, 0.5),
        ],
    );
    let mut stack = Stack::new();
    let composed = compose(&gpu, &mut stack, &plan);

    let differing = differing_pixels(&reference, &composed, 0);
    println!(
        "plain stack: {differing} of {} pixels differ from the compositor (mean {:.4})",
        W * H,
        mean_difference(&reference, &composed)
    );
    assert_eq!(
        differing, 0,
        "a stack with no Adjustment and no Mask must be today's composite pass"
    );
    assert_no_validation(&gpu);
}

#[test]
fn a_negative_adjustment_inverts_every_layer_below_it() {
    let gpu = gpu();
    let slots = [
        solid_layer(&gpu, 0, [1.0, 0.0, 0.0, 1.0]),
        solid_layer(&gpu, 1, [0.0, 0.0, 0.8, 1.0]),
        blank_layer(&gpu, 2),
    ];
    let below = [state(0, Vec::new()), state(1, Vec::new())];
    let adjustment = state(2, vec![effect(EffectKind::Negative, 1.0, 0.0)]);
    let plan = [
        entry(
            &slots[0],
            LayerKind::Visualizer,
            BlendMode::Normal,
            1.0,
            &below[0],
        ),
        entry(
            &slots[1],
            LayerKind::Visualizer,
            BlendMode::Normal,
            1.0,
            &below[1],
        ),
        entry(
            &slots[2],
            LayerKind::Adjustment,
            BlendMode::Normal,
            1.0,
            &adjustment,
        ),
    ];

    let plain = reference(
        &gpu,
        &[
            (slots[0].output(), BlendMode::Normal, 1.0),
            (slots[1].output(), BlendMode::Normal, 1.0),
        ],
    );
    let mut stack = Stack::new();
    let inverted = compose(&gpu, &mut stack, &plan);

    // Both layers are in the adjustment's group, so both are inverted in one pass.
    let mut worst = 0;
    for y in (0..H).step_by(7) {
        for x in (0..W).step_by(5) {
            let before = pixel(&plain, x, y);
            let after = pixel(&inverted, x, y);
            for channel in 0..3 {
                let expected = 255 - before[channel];
                worst = worst.max((after[channel] - expected).abs());
            }
        }
    }
    println!(
        "negative adjustment over two layers: worst channel error {worst}, first pixel {:?} -> {:?}",
        pixel(&plain, 0, 0),
        pixel(&inverted, 0, 0)
    );
    assert!(
        worst <= 3,
        "a Negative adjustment at 1.0 must invert the composite of every layer below it (worst {worst})"
    );
    assert_no_validation(&gpu);
}

#[test]
fn adjustment_opacity_half_is_the_average_of_effected_and_not() {
    let gpu = gpu();
    let slots = [
        picture_layer(&gpu, 0),
        picture_layer(&gpu, 1),
        blank_layer(&gpu, 2),
    ];
    let states = [state(0, Vec::new()), state(1, Vec::new())];
    // Fold at 1.0 moves the picture, so the average is a real per-pixel mix rather than 0.5.
    let folded = state(2, vec![effect(EffectKind::Fold, 1.0, 0.4)]);
    let unchanged = state(2, Vec::new());

    let mut stack = Stack::new();
    let untouched = compose(
        &gpu,
        &mut stack,
        &[
            entry(
                &slots[0],
                LayerKind::Visualizer,
                BlendMode::Normal,
                1.0,
                &states[0],
            ),
            entry(
                &slots[1],
                LayerKind::Visualizer,
                BlendMode::Normal,
                1.0,
                &states[1],
            ),
            entry(
                &slots[2],
                LayerKind::Adjustment,
                BlendMode::Normal,
                0.0,
                &unchanged,
            ),
        ],
    );
    let full = compose(
        &gpu,
        &mut stack,
        &[
            entry(
                &slots[0],
                LayerKind::Visualizer,
                BlendMode::Normal,
                1.0,
                &states[0],
            ),
            entry(
                &slots[1],
                LayerKind::Visualizer,
                BlendMode::Normal,
                1.0,
                &states[1],
            ),
            entry(
                &slots[2],
                LayerKind::Adjustment,
                BlendMode::Normal,
                1.0,
                &folded,
            ),
        ],
    );
    let half = compose(
        &gpu,
        &mut stack,
        &[
            entry(
                &slots[0],
                LayerKind::Visualizer,
                BlendMode::Normal,
                1.0,
                &states[0],
            ),
            entry(
                &slots[1],
                LayerKind::Visualizer,
                BlendMode::Normal,
                1.0,
                &states[1],
            ),
            entry(
                &slots[2],
                LayerKind::Adjustment,
                BlendMode::Normal,
                0.5,
                &folded,
            ),
        ],
    );

    let mut worst = 0;
    let mut moved = 0;
    for index in 0..(W * H) as usize {
        for channel in 0..3 {
            let at = index * 4 + channel;
            let expected = (i32::from(untouched[at]) + i32::from(full[at])) as f32 / 2.0;
            let error = (f32::from(half[at]) - expected).abs();
            worst = worst.max(error.round() as i32);
            if (i32::from(full[at]) - i32::from(untouched[at])).abs() > 8 {
                moved += 1;
            }
        }
    }
    println!(
        "adjustment at 0.5: worst error against the average {worst}, {moved} channel values moved by the fold (of {})",
        W * H * 3
    );
    assert!(
        moved > 500,
        "the fold must actually move the picture, or the average proves nothing ({moved})"
    );
    assert!(
        worst <= 3,
        "an adjustment at 0.5 must be the average of the effected and un-effected composite (worst {worst})"
    );
    assert_no_validation(&gpu);
}

#[test]
fn an_adjustment_at_the_bottom_affects_nothing() {
    let gpu = gpu();
    let slots = [
        picture_layer(&gpu, 0),
        picture_layer(&gpu, 1),
        blank_layer(&gpu, 2),
    ];
    let states = [state(0, Vec::new()), state(1, Vec::new())];
    let adjustment = state(2, vec![effect(EffectKind::Negative, 1.0, 0.0)]);

    let mut stack = Stack::new();
    let without = compose(
        &gpu,
        &mut stack,
        &[
            entry(
                &slots[0],
                LayerKind::Visualizer,
                BlendMode::Normal,
                1.0,
                &states[0],
            ),
            entry(
                &slots[1],
                LayerKind::Visualizer,
                BlendMode::Normal,
                1.0,
                &states[1],
            ),
        ],
    );
    let with = compose(
        &gpu,
        &mut stack,
        &[
            entry(
                &slots[2],
                LayerKind::Adjustment,
                BlendMode::Normal,
                1.0,
                &adjustment,
            ),
            entry(
                &slots[0],
                LayerKind::Visualizer,
                BlendMode::Normal,
                1.0,
                &states[0],
            ),
            entry(
                &slots[1],
                LayerKind::Visualizer,
                BlendMode::Normal,
                1.0,
                &states[1],
            ),
        ],
    );

    let differing = differing_pixels(&without, &with, 0);
    println!(
        "adjustment at the bottom: {differing} of {} pixels differ",
        W * H
    );
    assert_eq!(
        differing, 0,
        "an Adjustment layer with nothing below it has nothing to adjust"
    );
    assert_no_validation(&gpu);
}

#[test]
fn a_mask_is_the_opacity_of_the_layer_above_it() {
    let gpu = gpu();
    // Solid colours on both sides of the mask, so the assertion is about the mask's luminance and
    // not about the orientation of a pattern.
    let slots = [
        solid_layer(&gpu, 0, [1.0, 0.0, 0.0, 1.0]),
        blank_layer(&gpu, 1),
        solid_layer(&gpu, 2, [0.1, 0.8, 0.3, 1.0]),
    ];
    draw_split(&gpu, &slots[1].scene);

    let states = [
        state(0, Vec::new()),
        state(1, Vec::new()),
        state(2, Vec::new()),
    ];
    let plan = [
        entry(
            &slots[0],
            LayerKind::Visualizer,
            BlendMode::Normal,
            1.0,
            &states[0],
        ),
        entry(
            &slots[1],
            LayerKind::Mask,
            BlendMode::Normal,
            1.0,
            &states[1],
        ),
        entry(
            &slots[2],
            LayerKind::Visualizer,
            BlendMode::Normal,
            1.0,
            &states[2],
        ),
    ];
    let mut stack = Stack::new();
    let composed = compose(&gpu, &mut stack, &plan);

    // Black half of the mask: the layer above contributes nothing, so the red shows.
    assert_rgb(&composed, W / 4, H / 2, [255, 0, 0], 3, "masked-out half");
    // White half: the layer above is fully opaque there and hides the red.
    assert_rgb(
        &composed,
        3 * W / 4,
        H / 2,
        [26, 204, 77],
        3,
        "unmasked half",
    );
    println!(
        "mask: black half {:?}, white half {:?}",
        pixel(&composed, W / 4, H / 2),
        pixel(&composed, 3 * W / 4, H / 2)
    );
    assert_no_validation(&gpu);
}

#[test]
fn a_mask_over_the_top_layer_does_nothing() {
    let gpu = gpu();
    let slots = [
        picture_layer(&gpu, 0),
        picture_layer(&gpu, 1),
        blank_layer(&gpu, 2),
    ];
    draw_split(&gpu, &slots[2].scene);
    let states = [
        state(0, Vec::new()),
        state(1, Vec::new()),
        state(2, Vec::new()),
    ];

    let mut stack = Stack::new();
    let without = compose(
        &gpu,
        &mut stack,
        &[
            entry(
                &slots[0],
                LayerKind::Visualizer,
                BlendMode::Normal,
                1.0,
                &states[0],
            ),
            entry(
                &slots[1],
                LayerKind::Visualizer,
                BlendMode::Normal,
                1.0,
                &states[1],
            ),
        ],
    );
    let with = compose(
        &gpu,
        &mut stack,
        &[
            entry(
                &slots[0],
                LayerKind::Visualizer,
                BlendMode::Normal,
                1.0,
                &states[0],
            ),
            entry(
                &slots[1],
                LayerKind::Visualizer,
                BlendMode::Normal,
                1.0,
                &states[1],
            ),
            entry(
                &slots[2],
                LayerKind::Mask,
                BlendMode::Normal,
                1.0,
                &states[2],
            ),
        ],
    );

    let differing = differing_pixels(&without, &with, 0);
    println!("mask on top: {differing} of {} pixels differ", W * H);
    assert_eq!(
        differing, 0,
        "a Mask layer with no layer above it masks nothing and is not drawn"
    );
    assert_no_validation(&gpu);
}

#[test]
fn a_mosh_adjustment_runs_at_the_output_size() {
    let gpu = gpu();
    let slots = [
        picture_layer(&gpu, 0),
        picture_layer(&gpu, 1),
        blank_layer(&gpu, 2),
    ];
    let states = [state(0, Vec::new()), state(1, Vec::new())];
    let adjustment = state(2, vec![effect(EffectKind::Mosh, 1.0, 0.5)]);

    let mut stack = Stack::new();
    let without = compose(
        &gpu,
        &mut stack,
        &[
            entry(
                &slots[0],
                LayerKind::Visualizer,
                BlendMode::Normal,
                1.0,
                &states[0],
            ),
            entry(
                &slots[1],
                LayerKind::Visualizer,
                BlendMode::Normal,
                1.0,
                &states[1],
            ),
        ],
    );
    // The first mosh frame for an index is an I-frame: a copy of the composite below it, so the
    // picture must come through unchanged (through one more half-float target).
    let mosh = compose(
        &gpu,
        &mut stack,
        &[
            entry(
                &slots[0],
                LayerKind::Visualizer,
                BlendMode::Normal,
                1.0,
                &states[0],
            ),
            entry(
                &slots[1],
                LayerKind::Visualizer,
                BlendMode::Normal,
                1.0,
                &states[1],
            ),
            entry(
                &slots[2],
                LayerKind::Adjustment,
                BlendMode::Normal,
                1.0,
                &adjustment,
            ),
        ],
    );

    let differing = differing_pixels(&without, &mosh, 2);
    println!(
        "mosh adjustment, first frame: {differing} of {} pixels differ from the plain stack (mean {:.4})",
        W * H,
        mean_difference(&without, &mosh)
    );
    assert_eq!(
        differing, 0,
        "the first mosh frame on a composite is a copy of it"
    );
    assert_no_validation(&gpu);
}
