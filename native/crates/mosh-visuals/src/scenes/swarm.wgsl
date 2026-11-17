// Scene 15 (`swarm`) — first generation, kept for the perf test in `scenes.rs`.
//
// `scenes/swarm_fast.wgsl` replaces this body (and `scenes/swarm_pos_fast.wgsl` replaces its
// particle table): same streaks, same widths, same colours, same falloff, but the per-particle
// work moved into the table and each tile carries the list of particles a pixel has to look at,
// which took the pass from 1.6-4.5 ms to 0.15-0.24 ms at 960x540. Nothing draws this file any
// more; it is compiled only so `hand_written_scenes_keep_their_look` can compare the two.
//
// gpu.js draws the swarm as N instanced quads with additive blending: `p.draw(6, round(150 + 1300
// * dens))`. `mosh-render` records exactly one 3-vertex, one-instance draw per pass and its
// pipelines have blending off, so the streaks are accumulated inside the fragment shader instead,
// over a bounded particle count (64..176 rather than 150..1430). The particle paths come from the
// per-frame table `swarm_pos.wgsl` renders (one texel per particle), which is what instancing
// would have given the vertex stage. Paths, widths, colours and the tapered falloff are the web
// scene's; the swarm is sparser and one draw instead of one instanced draw.
//
// Each streak is kept at least ~3 pixels wide: a layer renders at a lower internal resolution than
// the web app's canvas, and a 0.006-unit half-width is sub-pixel at 160x90, which would make the
// scene vanish instead of shrink.
const SWARM_MIN: i32 = 64;
const SWARM_MAX: i32 = 176;

@group(0) @binding(11) var posT: texture_2d<f32>;

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let asp = U.res.x / U.res.y;
  let count = SWARM_MIN + i32(f32(SWARM_MAX - SWARM_MIN) * clamp(U.dens, 0., 1.));
  let min_width = 3. / (U.res.y * 0.5);
  var c = vec3f(0.);
  for (var k = 0; k < count; k = k + 1) {
    let table = textureLoad(posT, vec2i(k, 0), 0);
    let i = f32(k);
    // The web vertex stage negates y, so a screen-space point is (p.x / asp, -p.y).
    let a = vec2f(table.x / asp, -table.y);
    let b = vec2f(table.z / asp, -table.w);
    var dir = a - b;
    let len = length(dir);
    if (len < 1e-5) { continue; }
    dir = dir / len;
    let w = max(0.006 + 0.012 * h1(i + 5.) + 0.01 * U.bass, min_width);
    let rel = q - b;
    let along = dot(rel, dir);
    if (along < -w || along > len + w) { continue; }
    let across = dot(rel, vec2f(-dir.y, dir.x));
    let d = abs(across) / w;
    let cover = 1. - smoothstep(0.55, 1., d);
    if (cover <= 0.) { continue; }
    let u = clamp(along / len, 0., 1.);
    let falloff = max(0., (1. - d * d) * (0.25 + 0.75 * u));
    let col = (lut(h1(i + 9.) * 0.5 + U.hue) + 0.15) * (0.5 + mel(k % 24));
    c += col * falloff * falloff * cover;
  }
  return fin(c);
}
