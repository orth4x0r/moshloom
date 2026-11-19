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
//   sigil   Grenze Gotisch blackletter in polished silver, long whip thorns and mirrored sigil
//                          marks grown at bake time, a thin blood-red glow
//   clean   Space Grotesk  crisp minimal, soft shadow, thin cool rim
// Nothing here flashes the whole frame: every bright thing is inside the glyphs.
//
// The sigil look's linework is geometry, not a pattern: `text/thorn.rs` grows it at bake time —
// thorns on the block's outer contour, a bramble mark beside each end of the word — and the shader
// draws the strokes in the same silver as the letters. Nothing is ever drawn across a glyph's face,
// and no thorn crosses a space into the next word.
//
// Uniform block (1008 bytes, matches `text::UNIFORM_BYTES`; mosh-render caps a per-draw block at
// 1024, so this is sixteen bytes — half a stroke — under that cap):
//   0  dst: vec2f          8  sheet: vec2f
//   16 inset_min: vec2f    24 inset_max: vec2f
//   32 p0: vec4f (p0..p3)  48 p1: vec4f (p4..p7)
//   64 audio: vec4f (low, mid, high, intensity)
//   80 time: f32  84 beat: f32  88 loud: f32  92 thorn_count: f32
//   96 marks: vec4f (symbol_count, 0, 0, 0)
//   112 thorns: array<Stroke, 20>, 32 bytes each: a, b, c, half-width, mirror
//   752 symbols: array<Stroke, 8>
//      (the counts and the strides are `thorn::MAX_THORNS`/`MAX_SYMBOLS`; the reflection test
//      pins every offset together with `text.rs`)

// One stroke of the sigil look's linework: two tapered capsules `a -> b -> c`, wide at `a` and a
// needle point at `c`. `mirror` is 1 for a symbol stroke, which is drawn again mirrored about the
// block's vertical centre, so the pair of marks is symmetric.
struct Stroke {
    a: vec2<f32>,
    b: vec2<f32>,
    c: vec2<f32>,
    width: f32,
    mirror: f32,
};

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
    thorn_count: f32,
    marks: vec4<f32>,
    thorns: array<Stroke, 20>,
    symbols: array<Stroke, 8>,
};

@group(0) @binding(0) var<uniform> U: TextUniforms;
@group(0) @binding(1) var sheetT: texture_2d<f32>;
@group(0) @binding(2) var lin: sampler;

// Keep in sync with `text/layout.rs::SDF_RANGE`: the sheet's 0..1 range covers this many baked
// pixels on each side of an edge.
const SDF_RANGE: f32 = 12.0;

// Keep in sync with `text/layout.rs::BEVEL_PX`: this many baked pixels of the field is the
// bevel's height, which is what turns the flat distance into a rounded stroke.
const BEVEL_PX: f32 = 9.0;

// How far the sigil look's linework reaches past the block's box, in block heights: `text/thorn.rs`
// grows a thorn at most 0.60 of a line's height plus its base half-width, and puts the flanking
// marks within 0.40 of a line of the box; a line is never taller than the block.
const LINEWORK_REACH: f32 = 0.72;

// The stroke's taper, as fractions of its base half-width: the second capsule of a stroke starts at
// `BEND_WIDTH` of the first's radius and ends at `TIP_WIDTH` of it, which is a needle point.
// `text/thorn.rs` rasters the same two capsules for its tests, and
// `the_thorn_geometry_constants_match_the_shader` compares these literals with that module's.
const BEND_WIDTH: f32 = 0.7;
const TIP_WIDTH: f32 = 0.0;

fn hash11(n: f32) -> f32 {
    return fract(sin(n * 127.1) * 43758.5453);
}

fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn texel(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(sheetT, lin, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
}

// One tapered segment of a stroke: the signed distance to its edge (negative inside), the unit
// vector across it — which is the stroke's own surface direction as it turns — and the radius
// there. Distances are in block heights, the space the stroke arrays carry.
fn stroke_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, ra: f32, rb: f32) -> vec4<f32> {
    let ab = b - a;
    let t = clamp(dot(p - a, ab) / max(dot(ab, ab), 0.000001), 0.0, 1.0);
    let delta = p - (a + ab * t);
    let run = length(delta);
    let radius = mix(ra, rb, t);
    let across = select(vec2<f32>(0.0, 1.0), delta / max(run, 0.000001), run > 0.000001);
    return vec4<f32>(run - radius, across, radius);
}

// A whole stroke: the nearer of its two tapered capsules. Same return as `stroke_segment`.
fn stroke_sdf(p: vec2<f32>, stroke: Stroke) -> vec4<f32> {
    let first = stroke_segment(p, stroke.a, stroke.b, stroke.width, stroke.width * BEND_WIDTH);
    let second = stroke_segment(
        p, stroke.b, stroke.c, stroke.width * BEND_WIDTH, stroke.width * TIP_WIDTH,
    );
    return select(first, second, second.x < first.x);
}

// The slack `out_of_reach` adds to a stroke's radius, in block heights: a fraction of a baked
// pixel. `text/thorn.rs::reach_radius` mirrors this formula, and `the_thorn_geometry_constants_
// match_the_shader` compares this literal with that module's constant.
const REACH_SLACK: f32 = 0.01;

// The cheapest way to know a stroke cannot reach `p`: one bounding circle around its whole body.
// The radius is the stroke's **path length** — `|a-b| + |b-c|` — plus its base half-width twice,
// never the chord `|c-a|`: a bent stroke's chord is shorter than its path, and a bound built from
// it cuts pixels off the bend. That defect lost two or three baked pixels of a mark barb on `W`
// and `THE LOOM IS US`, because the early-out rejected them before the capsule maths ran.
fn out_of_reach(p: vec2<f32>, stroke: Stroke) -> bool {
    let radius = length(stroke.b - stroke.a) + length(stroke.c - stroke.b)
        + stroke.width * 2.0 + REACH_SLACK;
    return distance(p, stroke.a) > radius;
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

    // The sigil look is the only one whose ink leaves the block's box: its thorns reach past the
    // silhouette and its marks flank the word, so that look's pass covers a wider rectangle and
    // every other look covers exactly the sheet's margin.
    let thorny = look >= 1.0 && look < 2.0;

    // Every pixel outside the sheet (block plus its margin, in block units) has nothing to sample:
    // the pass costs a compare there.
    let margin_u = U.inset_min.x / max(span.x, 0.000001);
    let margin_v = U.inset_min.y / max(span.y, 0.000001);
    var reach_u = 0.0;
    var reach_v = 0.0;
    if (thorny) {
        reach_v = LINEWORK_REACH;
        reach_u = LINEWORK_REACH / max(block_aspect, 0.0001);
    }
    if (abs(block_uv.x - 0.5) > 0.5 + margin_u + reach_u
        || abs(block_uv.y - 0.5) > 0.5 + margin_v + reach_v) {
        return vec4<f32>(0.0);
    }

    var sheet_uv = mix(U.inset_min, U.inset_max, block_uv);

    // Glitch: whole rows of the sheet slide sideways on a fast clock.
    let slice_row = floor(sheet_uv.y * 42.0);
    let slice_clock = floor(U.time * 11.0);
    let slice_on = step(1.0 - glitch * 0.38, hash21(vec2<f32>(slice_row, slice_clock)));
    let slice_shift = (hash21(vec2<f32>(slice_row * 3.7, slice_clock)) - 0.5) * 0.05 * glitch;
    sheet_uv = sheet_uv + vec2<f32>(slice_shift * slice_on * span.x, 0.0);

    // The sigil look's linework (sigil only): thorns grown on the block's outer contour and the
    // marks that flank the word, both baked geometry and never inside the ink — a spike that would
    // touch a glyph, lean into a counter or spear the next word is dropped at bake time, not here.
    // `local` is the block's own space (one unit per block height, x by the aspect), which is the
    // space the stroke arrays are written in; it slides with the same glitch the sheet takes.
    var line_distance = 1.0e9;
    var line_across = vec2<f32>(0.0, 1.0);
    var line_radius = 1.0;
    let local = vec2<f32>(
        block_uv.x * block_aspect + slice_shift * slice_on * block_aspect,
        block_uv.y,
    );
    if (thorny) {
        var index = 0u;
        loop {
            if (index >= u32(U.thorn_count)) {
                break;
            }
            let stroke = U.thorns[index];
            index = index + 1u;
            if (out_of_reach(local, stroke)) {
                continue;
            }
            let hit = stroke_sdf(local, stroke);
            if (hit.x < line_distance) {
                line_distance = hit.x;
                line_across = hit.yz;
                line_radius = hit.w;
            }
        }
        // A mark is authored once, on the right of the block, and drawn again mirrored for the
        // left: the mirror of a block-space x is the block's width in its own units minus x. Both
        // copies sit outside the block's box, so there is nothing to evaluate over the letters.
        if (local.x < 0.0 || local.x > block_aspect) {
            // Which side of the block this pixel is on, and so which copy of an authored mark it
            // can see. `stroke.mirror` is the CPU's bit for the second copy (`text/thorn.rs`
            // writes 1 for every mark stroke): a stroke without it is drawn once, on the right.
            let mirrored = local.x < 0.5 * block_aspect;
            let probe = select(
                vec2<f32>(block_aspect - local.x, local.y),
                local,
                local.x > block_aspect,
            );
            var mark = 0u;
            loop {
                if (mark >= u32(U.marks.x)) {
                    break;
                }
                let stroke = U.symbols[mark];
                mark = mark + 1u;
                if (mirrored && stroke.mirror < 0.5) {
                    continue;
                }
                if (out_of_reach(probe, stroke)) {
                    continue;
                }
                let hit = stroke_sdf(probe, stroke);
                if (hit.x < line_distance) {
                    line_distance = hit.x;
                    line_across = hit.yz;
                    line_radius = hit.w;
                }
            }
        }
    }

    let sample = texel(sheet_uv);
    // Past the field's range every term below is zero: the glow fades out well before it, and the
    // body and ring are both measured from the distance. Most of the block is that empty space
    // between letters, so this one tap is what the pass costs there. The linework's ink is the one
    // thing out there that still draws, so it is measured in screen pixels first.
    let line_sdf = line_distance * block_h;
    if (sample.a <= 0.0 && line_sdf >= 0.0) {
        return vec4<f32>(0.0);
    }
    let distance_baked = (0.5 - sample.a) * (2.0 * SDF_RANGE);
    // Screen pixels per baked pixel: the distance, the outline and the glow all live in screen
    // space so they stay the same weight at every `size`.
    let scale = block_h / max(span.y * U.sheet.y, 1.0);
    let d0 = distance_baked * scale;
    // The distance is only meaningful out to the sheet's range; the glow must finish inside this,
    // or it would paint the sheet's margin.
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

    // The linework is part of the silhouette — the glow wraps it like the letters — so its distance
    // joins the glyph's. The stroke dilation is the letters' (`bold` gives a hairline face its
    // display weight); a stroke carries its own taper, so the dilation is added back to its
    // distance and the spike keeps the needle point the bake drew instead of swelling into a rod.
    // Nothing is ever added *inside* the ink: `min` only moves the edge outward, and the geometry
    // is outside the ink to begin with.
    let shape = select(d0, min(d0, line_sdf + bold), thorny);
    // Across a stroke the surface turns like a thin metal rod, not like a letter's bevel: the
    // stroke's own width is its depth. Only the nearer edge owns the surface, and only outside the
    // body's edge (`d0 - bold`): inside a letter the stroke's root would otherwise shade a dark
    // notch in the face, which is the same defect as a barb drawn across it.
    var surface = facing;
    var into = depth;
    if (thorny && line_sdf < d0 - bold && d0 - bold > 0.0) {
        surface = line_across;
        into = clamp(-line_sdf / max(line_radius * block_h * 0.75, 1.0), 0.0, 1.0);
    }

    // The antialiasing width in distance units: the field is in screen pixels and its gradient
    // across an edge is ~1, so one pixel of coverage is one unit. A constant also keeps the
    // derivative math out of the early-out branch, where those derivatives are undefined.
    let aa = 0.9;
    let body = smoothstep(aa, -aa, shape - bold);

    // The rim just outside the body: one number for its weight, one for where it starts. It is
    // the *letters'* rim, measured on the glyph's own distance, because a thorn is thinner than
    // the rim: giving a spike this outline would turn a silver thorn into a dark scratch.
    var ring_width = 0.002 * block_h + outline * 0.030 * block_h;
    if (look >= 1.0 && look < 2.0) {
        ring_width = 0.002 * block_h + outline * 0.022 * block_h;
    }
    let glyph_body = smoothstep(aa, -aa, d0 - bold);
    let ring_outer = smoothstep(aa, -aa, d0 - bold - ring_width);
    let ring = clamp(ring_outer - glyph_body, 0.0, 1.0);

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
        // A thin, dark red halo: it should read as a shadow of blood behind silver, not a neon
        // tube. Thorns carry it too, at their base where the field still reaches.
        glow_width = 0.020 * block_h;
        glow_colour = vec3<f32>(0.72, 0.06, 0.10);
        glow_strength = 0.30 + 0.18 * low + 0.10 * intensity;
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
        // Sigil: polished silver, brushed along the stroke, an iridescent edge and a thin red
        // glow. Thorns shade with `surface`/`into`, which is their own rod-like cross-section.
        let shoulder = sqrt(max(1.0 - into * into, 0.0));
        let normal = normalize(vec3<f32>(surface * shoulder, 0.16 + 0.84 * into));
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
            smoothstep(0.0, 0.34, into),
        );
        let brush = 0.5 + 0.5 * sin((offset.y * 7.0 + offset.x * 1.5));
        // A darker rim and a bright core: polished metal needs the falloff, not a flat plate.
        var c = silver * (0.86 + 0.18 * brush) * mix(0.55, 1.05, smoothstep(0.0, 0.40, into));
        c = c + film * fresnel * 0.45;
        c = c + vec3<f32>(1.0) * specular * 1.10;
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
    let fill = body_colour * (1.0 + beat_env * 0.14);
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
