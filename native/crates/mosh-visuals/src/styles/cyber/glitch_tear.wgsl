// Glitch Tear: horizontal tearing, corrupt macroblocks and RGB splits over a generated test card.
//
// The frame is cut into bands a few pixels tall; each band is displaced sideways by a hash that is
// re-rolled several times a second, and a slow sweep line drags the rows it passes much further, so
// the picture tears along one moving seam. The card behind it (rings and spokes) is sampled three
// times, once per colour channel, with a split that opens on the beat. Blocks of the displaced
// image are then rewritten: channel rotation, invert, posterise or a black dropout.
//
// Audio: `U.high` widens the tear and re-rolls it faster, `U.bass` pushes the sweep line's drag,
// `U.pulse` opens the channel split and lights the seam. The sweep runs at 0.3 Hz, well inside the
// 400 ms flash limit; nothing here flashes the frame on its own.
//
// P(0) clock rate, P(1) band height, P(2) hue, P(3) tear, P(4) macroblocks, P(5) RGB split,
// P(6) contrast, P(7) scanlines.

const RED = vec3f(0.82, 0.05, 0.12);
const CYAN = vec3f(0.10, 0.76, 0.92);

/// The test card behind the tearing: a ring target, a spoke wheel and a horizon band.
fn card(p: vec2f, t: f32) -> vec3f {
  let c = p - vec2f(0.0, 0.04);
  let r = length(c);
  let a = atan2(c.y, c.x);
  let rings = 0.5 + 0.5 * sin(r * 7.0 - t * 0.9);
  let spokes = smoothstep(0.42, 0.48, fract(a * 5.0 / TAU + t * 0.05));
  let horizon = smoothstep(0.012, 0.0, abs(p.y + 0.52));
  let v = clamp(0.30 + 0.52 * rings * (0.55 + 0.45 * spokes) - 0.34 * smoothstep(0.95, 0.15, r), 0.0, 1.0);
  var col = lut(0.10 + 0.80 * v + 0.30 * p.y) * (0.30 + 1.15 * v);
  col += RED * horizon * 0.7;
  col += CYAN * smoothstep(0.55, 0.0, r) * 0.18 * (0.4 + 0.6 * U.pulse);
  return col;
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let px = 2.0 / U.res.y;
  let q = XY(pos);
  let t = U.T * (0.5 + 0.5 * P(0));
  let energy = clamp(U.high, 0.0, 1.0);

  // --- the bands: a displaced row is a tear, and the seam moves slower than a beat.
  let bh = 3.0 + 26.0 * clamp(P(1), 0.0, 1.0) * (0.5 + 0.8 * clamp(U.dens, 0.0, 1.0));
  let band = floor(pos.y / bh);
  let roll = floor(t * (3.0 + 7.0 * energy));
  let seed = h2(vec2f(band, roll));
  let amp = (10.0 + 52.0 * P(3) + 30.0 * energy + 26.0 * U.pulse) * px * 55.0;
  let seam_y = fract(t * 0.30);
  let seam = exp(-abs(pos.y / U.res.y - seam_y) * 26.0);
  let disp = (seed - 0.5) * amp + seam * (seed - 0.35) * 520.0 * px + U.pulse * 40.0 * px * (seed - 0.5);

  // --- the card, with a split that grows with the beat.
  let split = (1.0 + 6.0 * P(5)) * (0.35 + 1.1 * U.pulse) + 2.5 * energy;
  let pr = XY(vec4f(pos.x + disp + split, pos.y, 0.0, 0.0));
  let pg = XY(vec4f(pos.x + disp, pos.y, 0.0, 0.0));
  let pb = XY(vec4f(pos.x + disp - split, pos.y, 0.0, 0.0));
  // Two evaluations, not three: red reads the far sample, green and blue the near one.
  let cr = card(pr, t);
  let cg = card(pg, t);
  var c = vec3f(cr.r, cg.g, cg.b * 0.96);

  // --- corrupt macroblocks: channel rotation, invert, posterise, or a black dropout.
  let block = floor(vec2f((pos.x + disp) / 16.0, pos.y / 16.0));
  let rnd = h2(block + floor(t * 2.0));
  let gate = P(4) * 0.08 + 0.03 * energy;
  if (rnd > 1.0 - gate) {
    let kind = fract(rnd * 37.0);
    if (kind < 0.34) {
      c = vec3f(c.b, c.r, c.g) * 1.20;
    } else if (kind < 0.62) {
      c = vec3f(0.04, 0.02, 0.03) + RED * 0.10;
    } else if (kind < 0.86) {
      c = floor(c * 3.0) / 3.0 + 0.06;
    } else {
      c = vec3f(1.0) - c;
    }
  }

  // --- contrast, the seam's glow, scanlines and the capture vignette.
  let contrast = 0.85 + 0.65 * P(6);
  c = (c - 0.42) * contrast + 0.42;
  c += RED * seam * (0.10 + 0.55 * U.pulse);
  c += CYAN * 0.10 * exp(-abs(pos.y / U.res.y - seam_y) * 90.0) * (0.3 + 0.7 * energy);
  c *= 1.0 - (0.10 + 0.16 * P(7)) * (0.5 + 0.5 * cos(pos.y * PI * 0.75));
  c *= 0.62 + 0.38 * smoothstep(1.45, 0.20, length(vec2f(q.x * 0.60, q.y * 0.95)));
  c += vec3f(0.006, 0.008, 0.012);
  // Hue: a channel rotation on the finished frame, so P(2) shifts the capture's colour cast.
  c *= mix(vec3f(1.0), vec3f(0.88, 0.62, 1.24), U.hue);
  return fin(c);
}
