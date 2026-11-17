// The particle step. One feedback pass over the state target's two rows:
//
//   row 0: position (screen units, centre 0, y down) and velocity (screen units per second)
//   row 1: life (1 -> 0), age (seconds), hue phase, spawn phase
//
// Both rows recompute the same deterministic spawn from (index, time), so either half can be
// written without waiting for the other's output. A dead particle (life <= 0, which is also what a
// freshly cleared target holds) spawns at its home disk.

@group(0) @binding(1) var st: texture_2d<f32>;

const VEL_LIMIT: f32 = 3.0;

struct Particle {
  pos: vec2f,
  vel: vec2f,
  life: f32,
  age: f32,
  hue: f32,
  phase: f32,
};

/// A fresh particle: home near the middle, a small outward push, a full life. `tick` makes a
/// respawn look new while keeping both state rows in agreement.
fn spawn(i: f32, t: f32) -> Particle {
  let tick = floor(safe(t, 0.0) * 1.7);
  let s = seed_of(i);
  let angle = s.angle + 0.9 * h1(i * 3.1 + tick * 0.37);
  let radius = 0.02 + 0.20 * h1(i * 5.3 + tick * 1.91);
  let dir = vec2f(cos(angle), sin(angle));
  let speed = (0.05 + 0.30 * U.gains.y) * (0.35 + 0.65 * s.size);
  return Particle(
    dir * radius,
    dir * speed,
    1.0,
    0.0,
    s.hue,
    h1(i * 7.7 + tick * 2.71),
  );
}

fn step(i: f32, t: f32, prev: Particle) -> Particle {
  if (!(prev.life > 0.0)) {
    return spawn(i, t);
  }
  let dt = U.clock.x;
  let aspect = U.motion.y;
  let s = seed_of(i);
  let q = prev.pos;
  // Screen units, centre at the origin: isotropic, so the vortex and the shockwaves stay round
  // whatever the layer's aspect is.
  let dl = max(length(q), 0.12);
  let out = q / dl;
  var acc = (home_of(s) - q) * U.forces.x;
  acc = acc + vec2f(0.0, U.look.z);
  // Swirl: a vortex around the centre, strongest on the downbeat (`U.gains.w` already carries
  // the beat phase from the CPU).
  acc = acc + vec2f(-out.y, out.x) * (U.gains.w * 2.4);
  // The bands steer: low pushes outward, mid twists, high lifts.
  acc = acc + out * (U.bands.x * U.forces.z);
  acc = acc + vec2f(-out.y, out.x) * (U.bands.y * 1.6);
  acc = acc + vec2f(0.0, -1.0) * (U.bands.z * U.forces.w);
  // A kick's shockwave from the point the CPU hashed out of the beat index: a velocity impulse,
  // not an acceleration, so one frame of kick visibly throws the field outward. The falloff has a
  // floor: a kick that lands at the screen's edge still moves the field it can't sit in the middle
  // of, which is what a kick's "pop" is.
  let kd = q - U.kick.xy;
  let kl = max(length(kd), 0.09);
  let shock = 0.35 + 0.65 * exp(-dot(kd, kd) * 1.4);
  let impulse = (kd / kl) * (U.kick.z * 5.0 * shock);

  var vel = prev.vel + impulse + acc * dt;
  vel = vec2f(safe(vel.x, 0.0), safe(vel.y, 0.0)) * exp(-U.forces.y * dt);
  vel = clamp(vel, vec2f(-VEL_LIMIT), vec2f(VEL_LIMIT));
  let pos = prev.pos + vel * dt;
  let life = prev.life - dt / max(s.life, 0.5);
  let age = prev.age + dt;
  let escaped = abs(pos.x) > aspect * 0.6 + 0.12 || abs(pos.y) > 0.62;
  if (!(life > 0.0) || escaped || !(pos.x == pos.x) || !(pos.y == pos.y)) {
    return spawn(i, t);
  }
  return Particle(pos, vel, life, age, prev.hue, prev.phase);
}

@fragment fn fs_main(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
  let i = floor(pixel.x);
  let row = floor(pixel.y);
  let a = textureLoad(st, vec2i(i32(i), 0), 0);
  let b = textureLoad(st, vec2i(i32(i), 1), 0);
  let prev = Particle(a.xy, a.zw, safe(b.x, 0.0), safe(b.y, 0.0), b.z, b.w);
  let next = step(i, U.clock.y, prev);
  if (row < 0.5) {
    return vec4f(next.pos, next.vel);
  }
  return vec4f(next.life, next.age, next.hue, next.phase);
}
