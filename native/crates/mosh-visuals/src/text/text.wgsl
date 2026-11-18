// One full-screen pass that draws a baked text sheet (see `mosh-visuals/src/text/sheet.rs`).
//
// The sheet holds the laid-out block as ONE signed distance field plus, in the colour channels, a
// baked surface direction and bevel depth (see `text/sheet.rs`). There are no per-glyph cells and
// no per-glyph resampling here: a pixel's distance is the distance to the nearest edge of the
// whole line, so a stroke a neighbour's outline, glow or bevel reaches is never cut at a cell
// border.
//
// `p0..p7` (see `text::PARAMS`):
//   p0 size      p1 font (three looks)   p2 spacing   p3 jitter
//   p4 glitch    p5 outline              p6 beat pop  p7 scroll
//
// Three contemporary looks, one face each:
//   chrome  Unbounded      liquid chrome: environment reflection, iridescent rim, soft specular
//   sigil   Grenze Gotisch blackletter in polished silver, barbed accents, a blood-red glow
//   clean   Space Grotesk  crisp minimal, soft shadow, thin cool rim
// Nothing here flashes the whole frame: every bright thing is inside the glyphs.
//
// Uniform block (96 bytes, matches `text::UNIFORM_BYTES`):
//   0  dst: vec2f          8  sheet: vec2f
//   16 inset_min: vec2f    24 inset_max: vec2f
//   32 p0: vec4f (p0..p3)  48 p1: vec4f (p4..p7)
//   64 audio: vec4f (low, mid, high, intensity)
//   80 time: f32  84 beat: f32  88 loud: f32  92 pad: f32

struct TextUniforms {
    dst: vec2<f32>,
    sheet: vec2<f32>,
    inset_min: vec2<f32>,
    inset_max: vec2<f32>,
    p0: vec4<f32>,
    p1: vec4<f32>,
    audio: vec4<f32>,
    time: f32,
    beat: f32,
    loud: f32,
    pad: f32,
};

@group(0) @binding(0) var<uniform> U: TextUniforms;
@group(0) @binding(1) var sheetT: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;

// Keep in sync with `text/layout.rs::SDF_RANGE`: the sheet's 0..1 range covers this many baked
// pixels on each side of an edge.
const SDF_RANGE: f32 = 12.0;

// Keep in sync with 	ext/layout.rs::BEVEL_PX: this many baked pixels of the field is the
// bevel's height, which is what turns the flat distance into a rounded stroke.
const BEVEL_PX: f32 = 9.0;

fn hash11(n: f32) -> f32 {
    return fract(sin(n * 127.1) * 43758.5453);
}

fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn texel(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(sheetT, lin, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
}

// Filmic roll-off: a specular highlight gets bright without ever clipping to flat white.
fn tonemap(c: vec3<f32>) -> vec3<f32> {
    let nn = c * (2.51 * c + 0.03);
    let dd = c * (2.43 * c + 0.59) + 0.14;
    return clamp(nn / dd, vec3<f32>(0.0), vec3<f32>(1.0));
}

// x^2n by squaring: the same highlight lobe as `pow` without the transcendental.
fn soft_power(x: f32, doublings: u32) -> f32 {
    var value = clamp(x, 0.0, 1.0);
    var index = 0u;
    loop {
        if (index >= doublings) {
            break;
        }
        value = value * value;
        index = index + 1u;
    }
    return value;
}

// A thin-film rim: the colour of the outer millimetre of a polished surface, from the viewing
// angle. Never a rainbow paint job, just an iridescent edge. A triangle wave, not three cosines.
fn iridescence(t: f32) -> vec3<f32> {
    let phase = fract(t * 0.8 + 0.12);
    let a = abs(phase * 2.0 - 1.0);
    let b = abs(fract(phase + 0.3333) * 2.0 - 1.0);
    let c = abs(fract(phase + 0.6667) * 2.0 - 1.0);
    return vec3<f32>(0.5) + 0.5 * vec3<f32>(1.0 - 2.0 * a, 1.0 - 2.0 * b, 1.0 - 2.0 * c);
}

// A studio environment for the chrome reflection: sky above, a dark horizon, a lit floor below,
// and two soft bands that read as reflected softboxes.
fn environment(dir: vec3<f32>) -> vec3<f32> {
    let up = clamp(dir.y, -1.0, 1.0);
    let sky = mix(
        vec3<f32>(0.44, 0.52, 0.66),
        vec3<f32>(0.99, 0.995, 1.0),
        smoothstep(-0.04, 0.80, up),
    );
    let ground = mix(
        vec3<f32>(0.035, 0.040, 0.058),
        vec3<f32>(0.58, 0.59, 0.64),
        smoothstep(0.0, -0.70, up),
    );
    var c = mix(ground, sky, smoothstep(-0.025, 0.025, up));
    // One thin horizon line, then the floor's soft reflection bands.
    c = c * (1.0 - 0.42 * exp(-abs(up) * 32.0));
    let azimuth = dir.x / (abs(dir.z) + 0.35);
    let band = pow(0.5 + 0.5 * sin(azimuth * 2.0 + up * 5.0), 3.0);
    return c + vec3<f32>(0.15, 0.19, 0.27) * band * (1.0 - abs(up));
}

// Premultiplied over: `src` in front of `base`.
fn over(base: vec4<f32>, src: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(src.rgb + base.rgb * (1.0 - src.a), src.a + base.a * (1.0 - src.a));
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let res = U.dst;
    // A layer target is V-mirrored relative to the picture an audience sees. There is exactly one
    // flip: the compositor samples every layer with `fullscreen.vert`'s uv, which is 0 at the
    // bottom of the output, so a layer's row 0 lands at the bottom of the frame (the present pass
    // adds no second flip). `the_word_is_upright_on_the_stage` measures this through the real
    // compositor. So the word is drawn mirrored in the layer here and reads upright on the stage:
    // `screen` is the audience's coordinate, y down from the top of the picture.
    let screen = vec2<f32>(position.x, res.y - position.y);
    let look = clamp(U.p0.y, 0.0, 0.999) * 3.0;
    let size = clamp(U.p0.x, 0.0, 1.0);
    let spacing = 0.92 + clamp(U.p0.z, 0.0, 1.0) * 0.55;
    let jitter = clamp(U.p0.w, 0.0, 1.0);
    let glitch = clamp(U.p1.x + U.audio.w * 0.20, 0.0, 1.2);
    let outline = clamp(U.p1.y, 0.0, 1.0);
    let pop_amount = clamp(U.p1.z, 0.0, 1.0);
    let scroll = clamp(U.p1.w, 0.0, 1.0);
    let intensity = clamp(U.audio.w, 0.0, 1.0);
    let low = clamp(U.audio.x, 0.0, 1.0);
    let high = clamp(U.audio.z, 0.0, 1.0);

    // The block's aspect comes from what was baked, so the glyphs keep their shape at any size.
    let span = U.inset_max - U.inset_min;
    let block_aspect = (span.x * U.sheet.x) / max(span.y * U.sheet.y, 1.0);

    // The beat: one envelope per beat, plus a slow breath from the low band.
    let beat_env = exp(-fract(U.beat) * 6.0) * pop_amount;
    var block_h = res.y * (0.14 + 0.44 * size) * (1.0 + beat_env * 0.08);
    // A wide word stops growing when it would leave the frame, so `size` never crops the letters;
    // spacing widens the block, so it counts against the same fit.
    let fit_h = res.x * 0.86 / max(block_aspect * spacing, 0.0001);
    block_h = min(block_h, fit_h);

    // Scroll rolls the block from below the frame to above it; at 0 it sits still in the centre.
    let cycle = fract(U.time * scroll * 0.07);
    let centre_y = mix(0.5, 1.5 - 2.0 * cycle, step(0.001, scroll));

    // Jitter wobbles the whole block on a 24 Hz clock. A per-glyph shift would tear the line at
    // every glyph boundary, which is the same defect as a clipped cell, so the wobble is coherent.
    let clock = floor(U.time * 24.0);
    let wobble = vec2<f32>(
        hash11(clock * 0.73 + 1.0) * 2.0 - 1.0,
        hash11(clock * 1.31 + 7.0) * 2.0 - 1.0,
    );
    let angle = (hash11(clock * 2.17 + 3.0) - 0.5) * 0.03 * jitter;

    var offset = (screen - vec2<f32>(0.5 * res.x, centre_y * res.y)) / block_h;
    let cs = cos(angle);
    let sn = sin(angle);
    offset = vec2<f32>(cs * offset.x - sn * offset.y, sn * offset.x + cs * offset.y);
    offset = offset + wobble * (0.008 * jitter);

    var block_uv = vec2<f32>(offset.x / max(block_aspect, 0.0001) + 0.5, offset.y + 0.5);
    // `spacing` widens the block: a screen position maps closer to the block's centre.
    block_uv.x = 0.5 + (block_uv.x - 0.5) / spacing;

    // Every pixel outside the sheet (block plus its margin, in block units) has nothing to sample:
    // the pass costs a compare there.
    let margin_u = U.inset_min.x / max(span.x, 0.000001);
    let margin_v = U.inset_min.y / max(span.y, 0.000001);
    if (abs(block_uv.x - 0.5) > 0.5 + margin_u || abs(block_uv.y - 0.5) > 0.5 + margin_v) {
        return vec4<f32>(0.0);
    }

    var sheet_uv = mix(U.inset_min, U.inset_max, block_uv);

    // Glitch: whole rows of the sheet slide sideways on a fast clock.
    let slice_row = floor(sheet_uv.y * 42.0);
    let slice_clock = floor(U.time * 11.0);
    let slice_on = step(1.0 - glitch * 0.38, hash21(vec2<f32>(slice_row, slice_clock)));
    let slice_shift = (hash21(vec2<f32>(slice_row * 3.7, slice_clock)) - 0.5) * 0.05 * glitch;
    sheet_uv = sheet_uv + vec2<f32>(slice_shift * slice_on * span.x, 0.0);

    let sample = texel(sheet_uv);
    // Past the field's range every term below is zero: the glow fades out well before it, and the
    // body, ring and barbs are all measured from the distance. Most of the block is that empty
    // space between letters, so this one tap is what the pass costs there.
    if (sample.a <= 0.0) {
        return vec4<f32>(0.0);
    }
    let distance_baked = (0.5 - sample.a) * (2.0 * SDF_RANGE);
    // Screen pixels per baked pixel: the distance, the outline and the glow all live in screen
    // space so they stay the same weight at every `size`.
    let scale = block_h / max(span.y * U.sheet.y, 1.0);
    let d0 = distance_baked * scale;
    // The distance is only meaningful out to the sheet's range; every effect measured from it
    // (barbs, glow) must finish inside this, or it would paint the sheet's margin.
    let field_reach = SDF_RANGE * scale;

    // Baked surface: the outward direction across the stroke comes from the sheet, the height
    // from the distance. The bevel is a few *screen* pixels, never a fraction of the stroke: at
    // 540p a wide word is drawn small, and a baked-width bevel would swallow the whole glyph.
    let facing = sample.rg * 2.0 - vec2<f32>(1.0);
    let bevel = max(0.008 * block_h, 1.0);
    let depth = clamp(-d0 / bevel, 0.0, 1.0);

    // The stroke's weight, before anything is added to the silhouette.
    // The dilation is what gives the hairline default instances a display weight, and it also
    // closes any gap narrower than twice itself: a pair like "at" must stay two letters, so it is
    // a fraction of the stroke and never more than a couple of percent of the block.
    var bold = 0.008 * block_h;
    if (look < 1.0) {
        bold = (0.009 + 0.003 * low) * block_h;
    } else if (look < 2.0) {
        bold = 0.012 * block_h;
    }

    // Barbs (sigil only): sharp accents that grow out of the strokes. They are geometry in the
    // field, not a wobble: a narrow periodic tooth shifted outward along the distance, so the
    // silhouette gets points while the stroke itself stays crisp. Every part of the pattern is
    // measured from the *body's* edge (`d0 - bold`), so it can never hatch the letter itself.
    var shape = d0;
    var barb_tip = 0.0;
    if (look >= 1.0 && look < 2.0) {
        // A thorn: a tapered spike growing out of the silhouette. The lattice runs along the
        // edge, one cell per thorn, with two incommensurate hashes so the spacing never lines up
        // like hatching; each thorn bends its own way as it rises, is thick at the stroke and
        // comes to a point, and most cells grow nothing at all.
        let barb_len = min((0.075 + 0.020 * glitch) * block_h, 0.45 * field_reach);
        let along = dot(offset * block_h, vec2<f32>(0.94, 0.34)) / max(barb_len, 0.001);
        let cell = floor(along);
        let bend_hash = hash11(cell * 7.13 + 3.7);
        let grow_hash = hash11(cell * 3.71 + 9.1);
        let phase = fract(along) - 0.5;
        let beyond = d0 - bold;
        let rise = clamp(beyond / max(barb_len, 0.001), 0.0, 1.0);
        // The spike leans as it rises, so it is a curve, not a comb tooth.
        let curved = phase + (bend_hash - 0.5) * 1.2 * rise;
        // Thick where it leaves the stroke, a point at the tip.
        let width = mix(0.30, 0.01, rise);
        let pointed = 1.0 - smoothstep(0.0, width, abs(curved));
        // Few cells grow a thorn, at their own length: sparse and asymmetric, not a fringe.
        let sparse = smoothstep(0.74, 0.88, grow_hash);
        let length = 0.6 + 0.4 * bend_hash;
        let present = smoothstep(-0.02, 0.08, beyond) * (1.0 - smoothstep(0.9, 1.0, rise));
        let spike = pointed * sparse * present * (1.0 - rise * rise);
        shape = d0 - spike * length * barb_len;
        barb_tip = spike * (0.35 + 0.65 * rise);
    }

    // The antialiasing width in distance units: the field is in screen pixels and its gradient
    // across an edge is ~1, so one pixel of coverage is one unit. A constant also keeps the
    // derivative math out of the early-out branch, where those derivatives are undefined.
    let aa = 0.9;
    let body = smoothstep(aa, -aa, shape - bold);

    // The rim just outside the body: one number for its weight, one for where it starts.
    var ring_width = 0.002 * block_h + outline * 0.030 * block_h;
    if (look >= 1.0 && look < 2.0) {
        ring_width = 0.002 * block_h + outline * 0.022 * block_h;
    }
    let ring_outer = smoothstep(aa, -aa, shape - bold - ring_width);
    let ring = clamp(ring_outer - body, 0.0, 1.0);

    // Glow, in screen pixels. Its width is a fraction of the block, and it is faded out before the
    // field's range ends so the sheet's margin can never show as a soft rectangle.
    var glow_width = 0.020 * block_h;
    var glow_colour = vec3<f32>(0.62, 0.72, 0.88);
    var glow_strength = 0.055 + 0.05 * high + 0.03 * intensity;
    if (look < 1.0) {
        glow_width = 0.030 * block_h;
        glow_colour = vec3<f32>(0.36, 0.72, 1.0);
        glow_strength = 0.20 + 0.20 * high + 0.14 * intensity;
    } else if (look < 2.0) {
        glow_width = 0.032 * block_h;
        glow_colour = vec3<f32>(0.92, 0.09, 0.14);
        glow_strength = 0.36 + 0.26 * low + 0.16 * intensity;
    }
    let reach_fade = 1.0 - smoothstep(0.30 * field_reach, 0.65 * field_reach, shape - bold);
    // A rational falloff instead of `exp`: same soft halo, no transcendental per pixel.
    let falloff = max(shape - bold, 0.0) / max(glow_width, 0.001);
    let glow = (1.0 / (1.0 + falloff * falloff * falloff)) * glow_strength * reach_fade;

    // Per-look shading. Each look computes only the lighting it uses: a crisp fill has no business
    // paying for a chrome reflection.
    var body_colour = vec3<f32>(1.0);
    var ring_colour = vec3<f32>(0.02, 0.03, 0.06);
    var ring_strength = 0.55;
    let view = vec3<f32>(0.0, 0.0, 1.0);
    let light = normalize(vec3<f32>(-0.42, 0.55, 0.72));

    if (look < 1.0) {
        // Chrome: liquid metal mirroring the studio environment, a thin iridescent rim, one soft
        // specular. The reflection is warped a little so the surface reads as a liquid.
        let shoulder = sqrt(max(1.0 - depth * depth, 0.0));
        let normal = normalize(vec3<f32>(facing * shoulder, 0.16 + 0.84 * depth));
        let ndl = clamp(dot(normal, light), 0.0, 1.0);
        let halfway = normalize(light + view);
        let specular = soft_power(clamp(dot(normal, halfway), 0.0, 1.0), 6);
        let fresnel = soft_power(1.0 - clamp(normal.z, 0.0, 1.0), 3);
        let film = mix(
            vec3<f32>(0.72, 0.80, 1.0),
            iridescence(normal.z * 0.6 + normal.y * 0.4),
            0.45,
        );
        let warp = vec2<f32>(
            sin(block_uv.x * 9.0 + block_uv.y * 5.0),
            cos(block_uv.y * 8.0 - block_uv.x * 4.0),
        ) * 0.05;
        let reflected = reflect(-view, normal);
        var c = environment(reflected + vec3<f32>(warp.x, warp.y, 0.0))
            + vec3<f32>(0.03, 0.035, 0.05);
        c = c * (0.82 + 0.34 * ndl);
        c = c + film * fresnel * 0.55;
        c = c + vec3<f32>(1.0) * specular * 0.40;
        body_colour = tonemap(c * 1.12);
        ring_colour = mix(vec3<f32>(0.03, 0.04, 0.09), film, 0.30);
        ring_strength = 0.50;
    } else if (look < 2.0) {
        // Sigil: polished silver, brushed along the stroke, an iridescent edge and a red glow.
        let shoulder = sqrt(max(1.0 - depth * depth, 0.0));
        let normal = normalize(vec3<f32>(facing * shoulder, 0.16 + 0.84 * depth));
        let halfway = normalize(light + view);
        let specular = soft_power(clamp(dot(normal, halfway), 0.0, 1.0), 6);
        let fresnel = soft_power(1.0 - clamp(normal.z, 0.0, 1.0), 3);
        let film = mix(
            vec3<f32>(0.78, 0.84, 1.0),
            iridescence(normal.z * 0.6 + normal.y * 0.4),
            0.35,
        );
        let silver = mix(
            vec3<f32>(0.30, 0.31, 0.35),
            vec3<f32>(0.99, 0.995, 1.0),
            smoothstep(0.0, 0.34, depth),
        );
        let brush = 0.5 + 0.5 * sin((offset.y * 7.0 + offset.x * 1.5));
        // A darker rim and a bright core: polished metal needs the falloff, not a flat plate.
        var c = silver * (0.86 + 0.18 * brush) * mix(0.55, 1.05, smoothstep(0.0, 0.40, depth));
        c = c + film * fresnel * 0.45;
        c = c + vec3<f32>(1.0) * specular * 1.10;
        c = c + vec3<f32>(0.70, 0.05, 0.10) * barb_tip * 0.9;
        body_colour = tonemap(c * 1.05);
        ring_colour = mix(vec3<f32>(0.045, 0.028, 0.032), film, 0.14);
        ring_strength = 0.30;
    } else {
        // Clean: near-white paper, a cool edge, one soft shadow, no chrome.
        body_colour = mix(
            vec3<f32>(0.86, 0.89, 0.96),
            vec3<f32>(0.995, 0.998, 1.0),
            smoothstep(0.0, 0.22, depth),
        );
        body_colour = body_colour * 1.0;
        ring_colour = vec3<f32>(0.07, 0.09, 0.14);
        ring_strength = 0.16;
    }

    var acc = vec4<f32>(0.0);

    // Glow first, behind everything.
    acc = over(
        acc,
        vec4<f32>(glow_colour * glow, clamp(glow, 0.0, 1.0)),
    );

    // Clean gets a soft shadow; the other looks are lit from inside.
    if (look >= 2.0) {
        let shadow_delta = vec2<f32>(0.012 / max(block_aspect, 0.0001), 0.014);
        let shadow_sample = texel(sheet_uv - shadow_delta);
        let shadow_distance = (0.5 - shadow_sample.a) * (2.0 * SDF_RANGE) * scale;
        let shadow_fade = 1.0 - smoothstep(0.35 * SDF_RANGE * scale, 0.95 * SDF_RANGE * scale, shadow_distance);
        let shadow = (1.0 - smoothstep(-0.04 * block_h, 0.10 * block_h, shadow_distance))
            * shadow_fade
            * (1.0 - body)
            * 0.42;
        acc = over(acc, vec4<f32>(vec3<f32>(0.015, 0.02, 0.04) * shadow, shadow));
    }

    // The rim.
    acc = over(
        acc,
        vec4<f32>(ring_colour * ring * ring_strength, ring * ring_strength),
    );

    // The body, with the beat lifting its fill rather than clipping it.
    var fill = body_colour * (1.0 + beat_env * 0.14);
    if (look >= 1.0 && look < 2.0) {
        fill = fill + vec3<f32>(1.0, 0.92, 0.94) * barb_tip * 0.18;
    }
    acc = over(acc, vec4<f32>(fill * body, body));

    // Chromatic split: two more taps of the field, pushed apart by glitch and the high band. The
    // taps are skipped when there is no glitch to show, which is most frames.
    if (glitch > 0.06) {
        let split_px = (0.5 + 3.0 * glitch) * (0.4 + high);
        let split_uv = vec2<f32>(split_px / max(scale, 0.0001) / U.sheet.x, 0.0);
        let field_r = (0.5 - texel(sheet_uv + split_uv).a) * (2.0 * SDF_RANGE) * scale;
        let field_b = (0.5 - texel(sheet_uv - split_uv).a) * (2.0 * SDF_RANGE) * scale;
        let body_r = smoothstep(aa, -aa, field_r - bold);
        let body_b = smoothstep(aa, -aa, field_b - bold);
        acc = acc + vec4<f32>(
            vec3<f32>(0.95, 0.12, 0.18) * clamp(body_r - body, 0.0, 1.0) * (0.30 + glitch)
                + vec3<f32>(0.10, 0.36, 1.0) * clamp(body_b - body, 0.0, 1.0) * (0.30 + glitch),
            0.0,
        );
    }

    // Scramble: at heavy glitch, square holes open and close on the beat clock.
    if (glitch > 0.25) {
        let blocks = floor(screen / (6.0 + 16.0 * glitch));
        let roll = hash21(blocks + vec2<f32>(floor(U.time * 9.0), glitch * 13.0));
        let cut = step(roll, 0.10 * glitch);
        acc = acc * (1.0 - cut);
    }

    return vec4<f32>(clamp(acc.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), clamp(acc.a, 0.0, 1.0));
}
