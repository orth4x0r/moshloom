// The per-particle table, three rows per particle (`scenes/swarm_pos_fast.wgsl`'s trick, applied
// to this layer's state):
//
//   row 0: the particle's screen-space box (min xy, max xy): the capsule's ends, padded by its
//          radius. It is the cheap reject that decides which tiles list the particle.
//   row 1: the capsule's head (xy) and tail (zw), in screen units.
//   row 2: radius, energy, hue, and whether the particle takes the second palette hue.
//
// Everything that is per particle rather than per pixel happens here — the radius with its hash
// and its motion swell, the life fade, the twinkle and the glitter flicker — so the fullscreen
// gather pays only for the pixels a particle actually covers.
//
// The capsule's tail is where the particle was `1.6 * dt` ago. A spark moves 20-50 pixels a frame
// in a busy clip, so a round point would leave a row of dots instead of a streak; the tail fills
// the gap between frames and the trail buffer carries the rest.

@group(0) @binding(1) var st: texture_2d<f32>;

/// A box far outside every tile, so a dead particle is never a candidate.
const DEAD: vec4f = vec4f(1e3, 1e3, 1e3, 1e3);

struct Particle {
  pos: vec2f,
  vel: vec2f,
  life: f32,
  age: f32,
  hue: f32,
  phase: f32,
};

fn read_state(i: i32) -> Particle {
  let a = textureLoad(st, vec2i(i, 0), 0);
  let b = textureLoad(st, vec2i(i, 1), 0);
  return Particle(a.xy, a.zw, safe(b.x, 0.0), safe(b.y, 0.0), b.z, b.w);
}

@fragment fn fs_main(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
  let index = i32(floor(pixel.x));
  let row = i32(floor(pixel.y));
  if (index >= MAX_COUNT) {
    return vec4f(0.0);
  }
  let p = read_state(index);
  if (!(p.life > 0.0)) {
    if (row == 0) {
      return DEAD;
    }
    return vec4f(0.0);
  }

  let fi = f32(index);
  let s = seed_of(fi);
  let t = U.clock.y;
  let speed = length(p.vel);

  // Radius: `p1` sets it, the particle's hash varies it, the high band and the particle's speed
  // swell it (a faster spark reads as a smeared point).
  var radius = U.gains.x * (0.55 + 1.15 * s.size) * (1.0 + 0.45 * U.bands.z);
  radius = radius * (1.0 + 0.30 * min(speed, 2.0));
  radius = clamp(radius, 0.0022, 0.085);

  let tail = p.pos - p.vel * (U.clock.x * 1.6);

  // Life: a short fade in, then out over its last third, so a spark never pops.
  let fade = smoothstep(0.0, 0.12, p.age) * smoothstep(0.0, 0.34, p.life);
  let twinkle = 0.60 + 0.40 * sin(t * (1.4 + 2.6 * s.radius) + p.phase * TAU);
  // Glitter: a 12 Hz per-particle flicker, scaled by the high band.
  let glitter = h1(fi * 7.3 + floor(t * U.motion.z));
  let flicker = 1.0 + U.grade.y * glitter * U.bands.z;
  var energy = (0.55 + 0.45 * U.state.x) * fade * twinkle * flicker;
  energy = energy * (1.0 + 0.5 * U.kick.w);
  energy = energy * (1.0 + 0.45 * U.bands.z + 0.25 * U.bands.y);
  // A bigger particle spreads the same light over an area that grows with the square of its
  // radius, so a large-glow clip reads as soft dust instead of a white disc. `motion.w` carries
  // the CPU's flux normalisation on top, so a dense field stays in the same light range as a
  // sparse one.
  let spread = U.gains.x / max(radius, 1e-4);
  energy = energy * spread * spread * U.motion.w;
  energy = clamp(energy, 0.0, 6.0);

  // Two hues per clip: most particles take the base, the ones whose hash is high take the second,
  // which is what makes a Y2K sparkle read as more than one colour.
  let hue = U.look.y + (s.hue - 0.5) * 0.22 + 0.12 * U.bands.z;
  let second = select(0.0, 1.0, s.hue > 0.78);

  if (row == 0) {
    return vec4f(min(p.pos, tail) - radius, max(p.pos, tail) + radius);
  }
  if (row == 1) {
    return vec4f(p.pos, tail);
  }
  return vec4f(radius, energy, fract(hue), second);
}
