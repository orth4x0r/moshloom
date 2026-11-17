//! The camera layer's parameter math: the source rectangle `zoom`/`pan` select, whether `mirror`
//! flips it, and the grade `exposure`/`contrast`/`tint`/`posterize` apply.
//!
//! All of it is pure and lives here, so the shader stays a handful of lines and every number can
//! be checked without a GPU or a camera. [`PARAMS`](super::PARAMS) names the eight slots and the
//! clip editor shows those names.
//!
//! # Neutral at zero
//!
//! `ClipParams::default()` is all zeros, and a fresh camera clip must show the whole frame. So
//! every slot is defined to be the identity at 0:
//!
//! | slot | neutral | runs to |
//! | --- | --- | --- |
//! | `p0` zoom | 1x | 4x (`MAX_ZOOM`) |
//! | `p1` pan x | centred | the edge of the frame |
//! | `p2` pan y | centred | the edge of the frame |
//! | `p3` mirror | not mirrored | mirrored at 0.5 (`mirror`) |
//! | `p4` exposure | 1x | 4x / 0.25x (`EXPOSURE_STOPS` = 2 stops) |
//! | `p5` contrast | 1x | 2x / 0x |
//! | `p6` tint | neutral | magenta (+) / green (-) |
//! | `p7` posterize | off | 64 levels (`MAX_POSTERIZE_LEVELS`) |
//!
//! A non-finite parameter is treated as its neutral value: a NaN out of a project file must not
//! blank the layer.

use mosh_core::ClipParams;

/// `p0` runs 1x..4x, so a clip can punch in by four.
pub(crate) const MAX_ZOOM: f32 = 4.0;

/// `p4` (exposure) runs this many stops either way: 0.25x..4x around 1x.
pub(crate) const EXPOSURE_STOPS: f32 = 2.0;

/// `p7` (posterize) maps to 2..64 levels.
pub(crate) const MIN_POSTERIZE_LEVELS: f32 = 2.0;
pub(crate) const MAX_POSTERIZE_LEVELS: f32 = 64.0;

/// `p3` at or above this mirrors.
pub(crate) const MIRROR_THRESHOLD: f32 = 0.5;

/// Bytes of the camera pass's uniform block: `dst` and `src` as two `vec2<f32>`s, then `view` and
/// `grade` as two `vec4<f32>`s, which WGSL packs as 8 + 8 + 16 + 16 = 48 bytes with no padding.
/// Keep in sync with `src/camera/overlay_camera.wgsl`; a test reflects the shader and fails when
/// they drift.
pub(crate) const UNIFORM_BYTES: usize = 48;

/// The part of a captured frame a layer target shows, in 0..1 source coordinates.
///
/// A negative `u_half` is the horizontal mirror: the shader multiplies the target's u offset by it,
/// so a mirror is the same rectangle traversed backwards.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SourceRect {
    pub u_center: f32,
    pub v_center: f32,
    pub u_half: f32,
    pub v_half: f32,
}

impl SourceRect {
    /// The source u range, low to high, so a test can prove it stays inside the frame whatever the
    /// sign of `u_half`.
    #[cfg(test)]
    pub(crate) fn u_range(&self) -> (f32, f32) {
        let half = self.u_half.abs();
        (self.u_center - half, self.u_center + half)
    }

    #[cfg(test)]
    pub(crate) fn v_range(&self) -> (f32, f32) {
        let half = self.v_half.abs();
        (self.v_center - half, self.v_center + half)
    }
}

/// The grade, in the order the shader applies it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Grade {
    pub exposure: f32,
    pub contrast: f32,
    pub tint: f32,
    pub posterize_levels: f32,
}

/// `p0` as a zoom factor: 1x at 0, `MAX_ZOOM` at 1.
pub(crate) fn zoom(p0: f32) -> f32 {
    1.0 + (MAX_ZOOM - 1.0) * unit(p0)
}

/// `p3` as a boolean: mirrored at and above [`MIRROR_THRESHOLD`].
pub(crate) fn mirror(p3: f32) -> bool {
    if p3.is_finite() {
        p3 >= MIRROR_THRESHOLD
    } else {
        false
    }
}

/// `p7` as a number of posterize levels: 0 means off (the shader skips the step below 1.5),
/// otherwise 2..64.
pub(crate) fn posterize_levels(p7: f32) -> f32 {
    let t = unit(p7);
    if t <= 0.0 {
        0.0
    } else {
        (MIN_POSTERIZE_LEVELS + (MAX_POSTERIZE_LEVELS - MIN_POSTERIZE_LEVELS) * t).round()
    }
}

/// The source rectangle of `src` a `dst`-sized target shows, with the clip's zoom, pan and mirror.
///
/// Cover scaling sets the base crop (the larger of the two axis scales fills the target, the other
/// axis is cropped evenly), zoom shrinks the rectangle about its centre, and pan slides it within
/// the margin that leaves; a full-frame rectangle therefore cannot be panned out of the picture.
pub(crate) fn source_rect(params: &ClipParams, src: (u32, u32), dst: (u32, u32)) -> SourceRect {
    let (visible_u, visible_v) = cover_visible_fraction(src, dst);
    let zoom = zoom(params.p[0]);
    let u_half = visible_u / zoom / 2.0;
    let v_half = visible_v / zoom / 2.0;
    // `0.5 - half` is the margin a centred crop leaves on that axis, so pan 1 lands on the edge.
    let u_center = 0.5 + signed_unit(params.p[1]) * (0.5 - u_half);
    let v_center = 0.5 + signed_unit(params.p[2]) * (0.5 - v_half);
    SourceRect {
        u_center,
        v_center,
        u_half: if mirror(params.p[3]) { -u_half } else { u_half },
        v_half,
    }
}

/// The grade the shader applies.
pub(crate) fn grade(params: &ClipParams) -> Grade {
    Grade {
        exposure: (EXPOSURE_STOPS * signed_unit(params.p[4])).exp2(),
        contrast: 1.0 + signed_unit(params.p[5]),
        tint: signed_unit(params.p[6]),
        posterize_levels: posterize_levels(params.p[7]),
    }
}

/// The uniform block `overlay_camera.wgsl` reads: `dst`, `src`, `view`, `grade`, each four
/// little-endian `f32`s. `view` is written as (u centre, v centre, u fraction, v fraction): the
/// fraction of the source the target shows, so the shader maps its -0.5..0.5 offset by it, and the
/// sign of the u fraction is the mirror. A test can parse the bytes back and compare them with
/// [`source_rect`] and [`grade`].
pub(crate) fn uniforms(
    params: &ClipParams,
    src: (u32, u32),
    dst: (u32, u32),
) -> [u8; UNIFORM_BYTES] {
    let rect = source_rect(params, src, dst);
    let grade = grade(params);
    let values = [
        dst.0 as f32,
        dst.1 as f32,
        src.0 as f32,
        src.1 as f32,
        rect.u_center,
        rect.v_center,
        rect.u_half * 2.0,
        rect.v_half * 2.0,
        grade.exposure,
        grade.contrast,
        grade.tint,
        grade.posterize_levels,
    ];
    let mut block = [0u8; UNIFORM_BYTES];
    for (index, value) in values.iter().enumerate() {
        block[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    block
}

/// The fraction of the source each axis shows under cover scaling: one axis is 1, the other is
/// cropped. A zero-sized input has nothing to crop, so both are 1 and the shader sees the whole
/// texture.
fn cover_visible_fraction(src: (u32, u32), dst: (u32, u32)) -> (f32, f32) {
    let (src_w, src_h) = (f64::from(src.0), f64::from(src.1));
    let (dst_w, dst_h) = (f64::from(dst.0), f64::from(dst.1));
    if src_w <= 0.0 || src_h <= 0.0 || dst_w <= 0.0 || dst_h <= 0.0 {
        return (1.0, 1.0);
    }
    let scale = (dst_w / src_w).max(dst_h / src_h);
    let visible_u = (dst_w / (src_w * scale)).min(1.0) as f32;
    let visible_v = (dst_h / (src_h * scale)).min(1.0) as f32;
    (visible_u, visible_v)
}

/// A parameter in 0..1, with a non-finite value taken as 0.
fn unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// A parameter in -1..1, with a non-finite value taken as 0.
fn signed_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `p[0]` is zoom, `p[1]` pan x, `p[2]` pan y, `p[3]` mirror, `p[4]` exposure, `p[5]` contrast,
    /// `p[6]` tint, `p[7]` posterize: a clip with exactly those values.
    fn params(p: [f32; 8]) -> ClipParams {
        ClipParams {
            p,
            ..ClipParams::default()
        }
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn default_is_the_whole_frame() {
        let rect = source_rect(&params([0.0; 8]), (160, 90), (64, 36));
        assert!(close(rect.u_half, 0.5), "u_half {}", rect.u_half);
        assert!(close(rect.v_half, 0.5), "v_half {}", rect.v_half);
        assert!(close(rect.u_center, 0.5));
        assert!(close(rect.v_center, 0.5));
        assert!(!mirror(0.0));
        let grade = grade(&params([0.0; 8]));
        assert!(close(grade.exposure, 1.0));
        assert!(close(grade.contrast, 1.0));
        assert!(close(grade.tint, 0.0));
        assert!(close(grade.posterize_levels, 0.0), "posterize is off at 0");
        println!("default: full frame, grade neutral, posterize off");
    }

    #[test]
    fn a_target_wider_than_the_camera_crops_the_top_and_bottom() {
        // 160x90 into 96x36: scale 0.6 fills the width, so the visible height is 36/54 = 2/3.
        let rect = source_rect(&params([0.0; 8]), (160, 90), (96, 36));
        assert!(close(rect.u_half, 0.5), "u_half {}", rect.u_half);
        assert!(close(rect.v_half, 1.0 / 3.0), "v_half {}", rect.v_half);
        assert!(close(rect.v_center, 0.5));
        println!(
            "cover 160x90 -> 96x36: u {:?} v {:?}",
            rect.u_range(),
            rect.v_range()
        );
    }

    #[test]
    fn a_target_taller_than_the_camera_crops_the_sides() {
        // 160x90 into 36x36: the height fills, so the visible width is 36/64 = 0.5625.
        let rect = source_rect(&params([0.0; 8]), (160, 90), (36, 36));
        assert!(close(rect.v_half, 0.5), "v_half {}", rect.v_half);
        assert!(close(rect.u_half, 0.28125), "u_half {}", rect.u_half);
        assert!(close(rect.u_center, 0.5));
    }

    #[test]
    fn zoom_shrinks_the_rectangle_about_its_centre() {
        let wide = source_rect(&params([0.0; 8]), (160, 90), (160, 90));
        for (p0, expected) in [(0.0, 1.0), (1.0 / 3.0, 2.0), (1.0, 4.0)] {
            let rect = source_rect(
                &params([p0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                (160, 90),
                (160, 90),
            );
            let expected_half = wide.u_half / expected;
            assert!(
                close(rect.u_half, expected_half),
                "p0 {p0}: u_half {} expected {expected_half}",
                rect.u_half
            );
            assert!(close(rect.u_center, 0.5), "zoom does not move the centre");
        }
        assert!(close(zoom(0.0), 1.0));
        assert!(close(zoom(1.0), MAX_ZOOM));
        assert!(close(zoom(5.0), MAX_ZOOM), "out of range clamps");
        assert!(close(zoom(-1.0), 1.0));
        println!("zoom: 0 -> 1x, 1/3 -> 2x, 1 -> 4x, out of range clamped");
    }

    #[test]
    fn pan_stays_inside_the_frame_and_does_nothing_at_full_frame() {
        // At 1x the whole frame is already visible: there is no margin to pan into.
        let full = source_rect(
            &params([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            (160, 90),
            (160, 90),
        );
        assert!(close(full.u_center, 0.5));
        assert!(close(full.v_center, 0.5));

        // At 2x the visible half is 0.25, so the largest centre is 0.75 and the edge is 1.0.
        let zoomed = 1.0 / 3.0;
        let right = source_rect(
            &params([zoomed, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            (160, 90),
            (160, 90),
        );
        assert!(close(right.u_center, 0.75), "u_center {}", right.u_center);
        assert!(close(right.v_center, 0.75), "v_center {}", right.v_center);
        assert!(
            close(right.u_range().1, 1.0),
            "the right edge lands on the frame's edge"
        );

        let left = source_rect(
            &params([zoomed, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            (160, 90),
            (160, 90),
        );
        assert!(close(left.u_center, 0.25));
        assert!(close(left.u_range().0, 0.0));

        // Out-of-range pan is clamped to the same edges.
        let beyond = source_rect(
            &params([zoomed, 9.0, -9.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            (160, 90),
            (160, 90),
        );
        assert!(close(beyond.u_center, 0.75));
        assert!(close(beyond.v_center, 0.25));
        println!(
            "pan at 2x: left {:?} right {:?} (clamped at both edges)",
            left.u_range(),
            right.u_range()
        );
    }

    #[test]
    fn mirror_flips_the_u_axis_only() {
        let plain = source_rect(&params([0.0; 8]), (160, 90), (64, 36));
        let flipped = source_rect(
            &params([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            (160, 90),
            (64, 36),
        );
        assert!(mirror(1.0));
        assert!(!mirror(0.49));
        assert!(close(flipped.u_half, -plain.u_half), "u half is negated");
        assert!(close(flipped.v_half, plain.v_half), "v half is untouched");
        assert!(close(flipped.u_center, plain.u_center));
        assert!(close(flipped.v_center, plain.v_center));

        // Mirror is a threshold, not a blend: 0.5 is mirrored, 0.499 is not.
        assert!(mirror(MIRROR_THRESHOLD));
        assert!(!mirror(MIRROR_THRESHOLD - 0.001));
        println!("mirror: u_half {} -> {}", plain.u_half, flipped.u_half);
    }

    #[test]
    fn exposure_and_contrast_are_neutral_at_zero() {
        let stops = |p4: f32| grade(&params([0.0, 0.0, 0.0, 0.0, p4, 0.0, 0.0, 0.0])).exposure;
        assert!(close(stops(0.0), 1.0));
        assert!(close(stops(1.0), 4.0), "+{} stops is 4x", EXPOSURE_STOPS);
        assert!(close(stops(-1.0), 0.25));
        assert!(close(stops(4.0), 4.0), "out of range clamps");

        let contrast = |p5: f32| grade(&params([0.0, 0.0, 0.0, 0.0, 0.0, p5, 0.0, 0.0])).contrast;
        assert!(close(contrast(0.0), 1.0));
        assert!(close(contrast(0.5), 1.5));
        assert!(close(contrast(1.0), 2.0));
        assert!(
            close(contrast(-1.0), 0.0),
            "contrast bottoms out at flat grey"
        );
        assert!(close(contrast(-9.0), 0.0));
        println!("grade: exposure {:.2}..{:.2}x, contrast 0..2x", 0.25, 4.0);
    }

    #[test]
    fn tint_is_the_green_magenta_axis_and_clamps() {
        let tint = |p6: f32| grade(&params([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p6, 0.0])).tint;
        assert!(close(tint(0.0), 0.0));
        assert!(close(tint(1.0), 1.0));
        assert!(close(tint(-1.0), -1.0));
        assert!(close(tint(7.0), 1.0));
        assert!(close(tint(-7.0), -1.0));
        println!("tint: -1 (green) .. 0 (neutral) .. 1 (magenta)");
    }

    #[test]
    fn posterize_maps_to_two_through_sixty_four_levels() {
        assert!(
            close(posterize_levels(0.0), 0.0),
            "0 is off, not two levels"
        );
        assert!(close(posterize_levels(0.001), 2.0));
        assert!(close(posterize_levels(0.5), 33.0));
        assert!(close(posterize_levels(1.0), 64.0));
        assert!(close(posterize_levels(9.0), 64.0), "out of range clamps");
        let levels = posterize_levels(0.5);
        // The shader rounds to the nearest of `levels` steps over 0..1.
        let quantize =
            |value: f32| ((value.clamp(0.0, 1.0) * (levels - 1.0) + 0.5).floor()) / (levels - 1.0);
        assert!(close(quantize(0.0), 0.0));
        assert!(close(quantize(1.0), 1.0));
        assert!(close(quantize(0.51), 0.5));
        println!(
            "posterize: p7 0 -> off, 0.5 -> {levels} levels, 1 -> {} levels",
            posterize_levels(1.0)
        );
    }

    #[test]
    fn non_finite_parameters_are_neutral() {
        let nan = f32::NAN;
        let rect = source_rect(&params([nan; 8]), (160, 90), (64, 36));
        assert!(close(rect.u_half, 0.5));
        assert!(close(rect.u_center, 0.5));
        let grade = grade(&params([nan; 8]));
        assert!(close(grade.exposure, 1.0));
        assert!(close(grade.contrast, 1.0));
        assert!(close(grade.tint, 0.0));
        assert!(close(grade.posterize_levels, 0.0));
        assert!(!mirror(nan));
        println!("NaN parameters: neutral, the layer keeps its picture");
    }

    #[test]
    fn every_rectangle_stays_inside_the_frame() {
        let mut checked = 0;
        for src in [(160u32, 90u32), (1280, 720), (720, 1280), (1, 1)] {
            for dst in [(64u32, 36u32), (36, 64), (1920, 1080), (1, 1)] {
                for step in 0..=8 {
                    let t = step as f32 / 8.0;
                    for pan in [-1.0f32, 0.0, 1.0] {
                        let rect =
                            source_rect(&params([t, pan, -pan, t, 0.0, 0.0, 0.0, 0.0]), src, dst);
                        let (u0, u1) = rect.u_range();
                        let (v0, v1) = rect.v_range();
                        assert!(
                            u0 >= -1e-6 && u1 <= 1.0 + 1e-6,
                            "src {src:?} dst {dst:?} t {t} pan {pan}: u {u0}..{u1} leaves the frame"
                        );
                        assert!(
                            v0 >= -1e-6 && v1 <= 1.0 + 1e-6,
                            "src {src:?} dst {dst:?} t {t} pan {pan}: v {v0}..{v1} leaves the frame"
                        );
                        checked += 1;
                    }
                }
            }
        }
        println!("{checked} zoom/pan/aspect combinations stay inside the captured frame");
    }

    #[test]
    fn the_uniform_block_is_the_shaders_bytes() {
        let clip = params([0.5, 0.25, -0.25, 1.0, 0.5, 0.0, -0.5, 0.75]);
        let src = (1280u32, 720u32);
        let dst = (960u32, 540u32);
        let block = uniforms(&clip, src, dst);
        assert_eq!(block.len(), UNIFORM_BYTES);
        assert_eq!(
            UNIFORM_BYTES, 48,
            "two vec2s and two vec4s pack into 48 bytes"
        );

        let float = |index: usize| {
            f32::from_le_bytes([
                block[index * 4],
                block[index * 4 + 1],
                block[index * 4 + 2],
                block[index * 4 + 3],
            ])
        };
        assert!(close(float(0), 960.0), "dst width");
        assert!(close(float(1), 540.0), "dst height");
        assert!(close(float(2), 1280.0), "src width");
        assert!(close(float(3), 720.0), "src height");

        let rect = source_rect(&clip, src, dst);
        assert!(close(float(4), rect.u_center));
        assert!(close(float(5), rect.v_center));
        assert!(
            close(float(6), rect.u_half * 2.0),
            "the shader takes the fraction, not the half"
        );
        assert!(close(float(7), rect.v_half * 2.0));

        let grade = grade(&clip);
        assert!(close(float(8), grade.exposure));
        assert!(close(float(9), grade.contrast));
        assert!(close(float(10), grade.tint));
        assert!(close(float(11), grade.posterize_levels));
        println!(
            "uniforms: {} bytes, view ({:.3}, {:.3}, {:.3}, {:.3}), grade ({:.2}, {:.2}, {:.2}, {:.0})",
            block.len(),
            rect.u_center,
            rect.v_center,
            rect.u_half,
            rect.v_half,
            grade.exposure,
            grade.contrast,
            grade.tint,
            grade.posterize_levels
        );
    }
}
