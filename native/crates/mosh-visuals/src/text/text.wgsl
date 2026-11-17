// One full-screen pass that draws a baked text sheet (see `mosh-visuals/src/text/sheet.rs`).
//
// The sheet holds a laid-out word as a signed distance field plus, in the colour channels, each
// glyph's own cell centre and noise value. That is what lets a single pass work per glyph: a pixel
// looks up which glyph's cell it belongs to, then resamples the field around that cell's centre for
// the beat pop, jitter and rotation. Nothing is re-laid out per frame.
//
// `p0..p7` (see `text::PARAMS`):
//   p0 size      p1 font (three looks)   p2 spacing   p3 jitter
//   p4 glitch    p5 outline              p6 beat pop  p7 scroll
//
// The three looks share one sheet: chrome dilates the field into a bubble and shades it with a
// bevel normal from the field's own gradient; thorny warps the edge with noise and adds thorns;
// clean is the field thresholded crisply. There is no full-frame flash anywhere in here: every
// bright thing is inside the glyphs.
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

// Keep in sync with `text::SDF_RANGE`: the sheet's 0..1 range covers this many baked pixels on
// each side of an edge.
const SDF_RANGE: f32 = 6.0;

fn hash11(n: f32) -> f32 {
    return fract(sin(n * 127.1) * 43758.5453);
}

fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    let frac = fract(p);
    let weight = frac * frac * (3.0 - 2.0 * frac);
    let a = hash21(cell);
    let b = hash21(cell + vec2<f32>(1.0, 0.0));
    let c = hash21(cell + vec2<f32>(0.0, 1.0));
    let d = hash21(cell + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, weight.x), mix(c, d, weight.x), weight.y);
}

fn texel(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(sheetT, lin, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let res = U.dst;
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
    let block_aspect = ((U.inset_max.x - U.inset_min.x) * U.sheet.x)
        / max((U.inset_max.y - U.inset_min.y) * U.sheet.y, 1.0);

    // The beat: one envelope per beat, plus a slow breath from the low band.
    let beat_env = exp(-fract(U.beat) * 6.0) * pop_amount;
    var block_h = res.y * (0.14 + 0.44 * size) * (1.0 + beat_env * 0.10);
    // A wide word stops growing when it would leave the frame, so `size` never crops the letters;
    // spacing widens the block, so it counts against the same fit.
    let fit_h = res.x * 0.86 / max(block_aspect * spacing, 0.0001);
    block_h = min(block_h, fit_h);

    // Scroll rolls the block from below the frame to above it; at 0 it sits still in the centre.
    let cycle = fract(U.time * scroll * 0.07);
    let centre_y = mix(0.5, 1.5 - 2.0 * cycle, step(0.001, scroll));

    let offset = (position.xy - vec2<f32>(0.5 * res.x, centre_y * res.y)) / block_h;
    var block_uv = vec2<f32>(offset.x / max(block_aspect, 0.0001) + 0.5, offset.y + 0.5);
    // `spacing` widens the block: a screen position maps closer to the block's centre.
    block_uv.x = 0.5 + (block_uv.x - 0.5) / spacing;

    // Most of the frame is empty: outside the block (plus room for the outline and the glow) there
    // is nothing to sample, so the pass costs a compare per pixel there.
    if (any(abs(block_uv - vec2<f32>(0.5)) > vec2<f32>(0.58))) {
        return vec4<f32>(0.0);
    }

    var sheet_uv = mix(U.inset_min, U.inset_max, block_uv);

    // Glitch: whole rows of the sheet slide sideways on a fast clock.
    let inset = U.inset_max - U.inset_min;
    let slice_row = floor(sheet_uv.y * 42.0);
    let slice_clock = floor(U.time * 11.0);
    let slice_on = step(1.0 - glitch * 0.38, hash21(vec2<f32>(slice_row, slice_clock)));
    let slice_shift = (hash21(vec2<f32>(slice_row * 3.7, slice_clock)) - 0.5) * 0.05 * glitch;
    sheet_uv = sheet_uv + vec2<f32>(slice_shift * slice_on * inset.x, 0.0);

    // The cell this pixel belongs to, before any per-glyph transform.
    let base = texel(sheet_uv);
    let cell = base.rg;
    let noise = base.b;
    let present = step(0.0005, max(max(base.r, base.g), max(base.b, base.a)));

    // Per-glyph jitter and rotation, in sheet pixels about the cell centre.
    let clock = floor(U.time * 24.0);
    let jitter_px = vec2<f32>(
        hash11(noise * 131.0 + clock) - 0.5,
        hash11(noise * 197.0 + clock * 1.7) - 0.5,
    ) * jitter * (0.03 + 0.09 * glitch) * (0.4 + high) * block_h;
    let local_px = (sheet_uv - cell) * U.sheet;
    // A glyph may only be moved a little about its own cell centre: a pixel whose own cell has no
    // ink has no glyph to be driven by, so a large per-glyph turn would tear the word at the
    // borders between cells. The thorny look's crookedness comes from the baked face and the
    // noise-warped edge below, not from turning glyphs here.
    let angle = (hash11(noise * 71.0 + clock) - 0.5) * 0.05 * jitter * (0.3 + glitch);
    let cs = cos(angle);
    let sn = sin(angle);
    let rotated = vec2<f32>(
        cs * local_px.x - sn * local_px.y,
        sn * local_px.x + cs * local_px.y,
    );
    // Per-glyph beat pop: each letter grows around its own cell centre, inside its PAD.
    let glyph_pop = 1.0 + beat_env * 0.12 + low * pop_amount * 0.05;
    let glyph_uv = cell + (rotated + jitter_px) / max(glyph_pop, 0.2) / U.sheet;

    // Distance in screen pixels: positive outside the ink, negative inside.
    let inset_h_px = (U.inset_max.y - U.inset_min.y) * U.sheet.y;
    let scale = block_h / max(inset_h_px, 1.0);
    // The base tap already carries the field; only a transformed glyph needs a second one.
    var field = base.a;
    if (any(abs(glyph_uv - sheet_uv) > vec2<f32>(0.0002))) {
        field = texel(glyph_uv).a;
    }
    var edge = (0.5 - field) * (2.0 * SDF_RANGE) * scale;

    // Bold: the chrome look is a bubble, clean is a medium weight, thorny keeps its thin cuts.
    var bold = 0.008 * block_h;
    if (look < 1.0) {
        bold = (0.042 + 0.012 * low) * block_h;
    } else if (look >= 2.0) {
        bold = 0.007 * block_h;
    } else {
        bold = 0.004 * block_h;
    }

    // Thorny: noise-warped edge plus spikes radiating from the cell centre.
    if (look >= 1.0 && look < 2.0) {
        let warp = (value_noise(local_px * 0.11 + vec2<f32>(U.time * 1.6, 0.0)) * 2.0 - 1.0)
            + (value_noise(local_px * 0.29 - vec2<f32>(U.time * 0.9, 3.0)) * 2.0 - 1.0) * 0.6;
        edge = edge - warp * (0.016 + 0.03 * jitter + 0.02 * glitch) * block_h;
        let radius = length(local_px);
        let reach = smoothstep(0.10 * block_h, 0.55 * block_h, radius);
        let thorn = pow(abs(sin(atan2(local_px.y, local_px.x) * 7.0 + noise * 6.283)), 14.0);
        edge = edge - thorn * reach * (0.010 + 0.020 * glitch) * block_h;
    }

    let aa = max(fwidth(edge), 0.0015) * 1.1;
    let body = smoothstep(aa, -aa, edge - bold);
    let outline_px = outline * 0.045 * block_h + 0.002 * block_h;
    let ring = clamp(smoothstep(aa, -aa, edge - bold + outline_px) - body, 0.0, 1.0);
    let glow = exp(-max(edge - bold, 0.0) / max(0.035 * block_h, 0.001))
        * (0.10 + 0.30 * high + 0.20 * intensity + 0.15 * clamp(U.loud, 0.0, 1.0));
    let dome = smoothstep(0.0, 0.20 * block_h, -edge);

    // Where the ink ends, inside: the bevel, the acid rim, the clean edge line. The band is a
    // fixed slice of the block's height, so a thin stroke keeps a dark core instead of being
    // swallowed by its own rim.
    let rim_width = max(0.8, 0.012 * block_h);
    let edge_band = clamp(1.0 - smoothstep(0.0, rim_width, -edge), 0.0, 1.0);

    // Wet look: the bevel is lit from the field's own depth and the glyph's vertical position.
    // The field's screen-space gradient is not used here: an 8-bit field's derivative is noisy
    // enough to dither the fill.
    let vy = clamp(0.5 - (sheet_uv.y - cell.y) * U.sheet.y / max(inset_h_px, 1.0), 0.0, 1.0);
    let up = clamp(vy * 0.90 + (dome - 0.5) * 0.25, 0.0, 1.0);

    var fill = vec3<f32>(1.0);
    var top_colour = vec3<f32>(1.0);
    var ring_colour = vec3<f32>(0.02, 0.03, 0.06);
    var glow_colour = vec3<f32>(0.5, 0.7, 1.0);
    var ring_strength = 0.9;
    var inner_amount = 0.35;
    var glow_amount = 1.0;

    if (look < 1.0) {
        // Y2K liquid chrome: sky above the horizon, a dark horizon line, the floor's reflection
        // below it, then a silver base. Four bands, because that is what reads as metal.
        fill = vec3<f32>(0.10, 0.14, 0.26);
        fill = mix(fill, vec3<f32>(0.95, 0.98, 1.0), smoothstep(0.28, 0.44, up));
        fill = mix(fill, vec3<f32>(0.02, 0.03, 0.07), smoothstep(0.44, 0.50, up));
        fill = mix(fill, vec3<f32>(0.52, 0.66, 0.92), smoothstep(0.50, 0.64, up));
        fill = mix(fill, vec3<f32>(1.0, 1.0, 1.0), smoothstep(0.70, 0.94, up));
        fill = fill + vec3<f32>(1.0) * pow(dome, 20.0) * 0.30;
        fill = fill + vec3<f32>(0.20, 0.55, 0.95) * (1.0 - up) * dome * 0.25;
        top_colour = vec3<f32>(0.95, 0.99, 1.0);
        ring_colour = vec3<f32>(0.55, 0.74, 1.0);
        ring_strength = 0.60;
        glow_colour = vec3<f32>(0.35, 0.75, 1.0);
        inner_amount = 0.85;
    } else if (look < 2.0) {
        // Cybersigil: near-black ink, an acid contour, coarse magenta abrasion, thorns.
        fill = mix(vec3<f32>(0.010, 0.014, 0.010), vec3<f32>(0.06, 0.11, 0.03), dome * 0.9);
        let abrasion = smoothstep(
            0.62,
            0.98,
            value_noise(vec2<f32>(local_px.x * 0.09, local_px.y * 0.30) + vec2<f32>(0.0, U.time * 3.0)),
        );
        fill = mix(fill, vec3<f32>(0.42, 0.02, 0.24), abrasion * 0.55);
        top_colour = vec3<f32>(0.55, 1.0, 0.25);
        ring_colour = vec3<f32>(0.45, 1.0, 0.20);
        ring_strength = 1.0;
        glow_colour = vec3<f32>(0.35, 0.90, 0.20);
        inner_amount = 0.70;
        glow_amount = 0.45;
    } else {
        // Clean: a bright grotesk with a thin edge line.
        fill = mix(vec3<f32>(1.0), vec3<f32>(0.66, 0.78, 1.0), clamp(vy, 0.0, 1.0) * 0.9);
        fill = fill * (0.80 + 0.30 * dome);
        top_colour = vec3<f32>(1.0, 1.0, 1.0);
        ring_colour = vec3<f32>(0.62, 0.76, 1.0);
        ring_strength = 0.5;
        glow_colour = vec3<f32>(0.50, 0.70, 1.0);
        inner_amount = 0.25;
    }

    var colour = fill * body;
    colour = mix(colour, top_colour, clamp(edge_band * inner_amount, 0.0, 1.0) * body);

    // Chromatic split: two more taps of the field, pushed apart by glitch and the high band. The
    // taps are skipped when there is no glitch to show, which is most frames.
    if (glitch > 0.06) {
        let split_px = (0.4 + 3.0 * glitch) * (0.4 + high);
        let split_uv = vec2<f32>(split_px / max(scale, 0.0001) / U.sheet.x, 0.0);
        let field_r = texel(glyph_uv + split_uv).a;
        let field_b = texel(glyph_uv - split_uv).a;
        let edge_r = (0.5 - field_r) * (2.0 * SDF_RANGE) * scale;
        let edge_b = (0.5 - field_b) * (2.0 * SDF_RANGE) * scale;
        let body_r = smoothstep(aa, -aa, edge_r - bold);
        let body_b = smoothstep(aa, -aa, edge_b - bold);
        colour = colour + vec3<f32>(0.95, 0.12, 0.18) * clamp(body_r - body, 0.0, 1.0) * (0.35 + glitch);
        colour = colour + vec3<f32>(0.12, 0.40, 1.0) * clamp(body_b - body, 0.0, 1.0) * (0.35 + glitch);
    }

    colour = colour + ring_colour * ring * ring_strength;
    colour = colour + glow_colour * glow * 0.6 * glow_amount;

    var alpha = clamp(body + ring * ring_strength * 0.9 + glow * 0.35 * glow_amount, 0.0, 1.0);

    // Scramble: at heavy glitch, square holes open and close on the beat clock.
    if (glitch > 0.25) {
        let blocks = floor(position.xy / (6.0 + 16.0 * glitch));
        let roll = hash21(blocks + vec2<f32>(floor(U.time * 9.0), glitch * 13.0));
        let cut = step(roll, 0.10 * glitch);
        colour = colour * (1.0 - cut);
        alpha = alpha * (1.0 - cut);
    }

    // Pixels outside every glyph cell are the sheet's margin: nothing to draw there.
    return vec4<f32>(colour * present, alpha * present);
}
