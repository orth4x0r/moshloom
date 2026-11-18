// Smear · Sorted Streaks: pixel sorting as painting, not as a broken screen.
//
// The picture underneath is a melt: three soft colour pools and one flowing gradient, dragged by two
// slow sines so the whole field is already in motion. Then the sort: each row is cut into runs, and a
// run above the threshold is rewritten from the sample at its own head, so a run becomes one long
// luminous streak instead of a texture of noise. The streak keeps a gradient because it is mixed back
// toward the pixel's own value along the run, and the head, where the value crosses the threshold,
// gets a hot core — the comet look real pixel sorting has. Above the head the sort drags upward by
// `Drag`, which bends the streaks into the long curved smears the style is about.
//
// There are no scanlines, no block corruption and no channel rotation: what reads as a screen is what
// this scene must not be. The dispersion control is a soft chromatic offset inside a streak (glass,
// not cathode ray), the colours are a full-chroma palette walk, and a shoulder keeps the hottest
// heads from clipping to white.
//
// Audio: `U.mid` lowers the threshold so more of the frame sorts on loud parts, the mel bands set each
// row's run length, `U.pulse` lengthens the drag and brightens the heads. Nothing strobes.
//
// P(0) clock rate, P(1) run length, P(2) hue, P(3) threshold, P(4) drag, P(5) dispersion,
// P(6) melt, P(7) grain.

/// The melt: colour pools and a flowing band, dragged by two slow sines. `.a` is the value the sort
/// measures, so a streak's head is the same field the colours come from. One shared warp, three
/// sines, two rational falloffs.
fn melt(q: vec2f, t: f32, hue: f32, warp: f32, swirl: f32) -> vec4f {
  let w = vec2f(
    sin(q.y * 1.30 + t * 0.21 + swirl) * warp,
    sin(q.x * 1.10 - t * 0.17 + swirl * 1.7) * warp
  );
  let p = q + w;
  let pool = falloff(length(p * vec2f(0.80, 1.30) - vec2f(0.34, -0.12)), 0.85)
    + 0.75 * falloff(length(p * vec2f(1.15, 0.95) + vec2f(0.55, 0.30)), 1.30);
  let band = 0.5 + 0.5 * sin(p.x * 2.10 + p.y * 0.70 + t * 0.31);
  let v = clamp(0.14 + 0.30 * pool + 0.32 * band - 0.10 * p.y, 0.0, 1.0);
  return vec4f(lut(0.04 + 0.90 * v + hue * 0.10) * (0.26 + 1.30 * v), v);
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let q = XY(pos);
  let t = U.T * (0.30 + 0.55 * P(0));
  let hue = P(2);
  let dens = clamp(U.dens, 0.0, 1.0);
  let warp = 0.10 + 0.34 * P(6);
  let own = melt(q, t, hue, warp, 0.0);

  // --- the runs: the frame is cut into long runs whose boundaries curve with the picture, so a streak
  // is a ribbon of light rather than a cut band. Run length follows the spectrum smoothly.
  let rowf = pos.y / (1.0 + 2.0 * dens);
  let band = mel(i32(4.0 + 8.0 * (0.5 + 0.5 * sin(pos.y * 0.010 + t * 0.20))));
  let run = mix(210.0, 34.0, clamp(0.85 * P(1) + 0.20 * dens, 0.0, 1.0)) * (0.60 + 0.80 * band);
  let off = run * (0.5 + 0.5 * sin(rowf * 0.055 + t * 0.25 + 1.3));
  let head_x = floor((pos.x - off) / run) * run + off;
  // How close this pixel is to its streak's leading edge: the edge is where the sorted light is hottest.
  let lane = (pos.x - off) / run;
  let edge = glow(min(fract(lane), 1.0 - fract(lane)) * run, 0.30);

  // --- the head: the sample the whole run is rewritten from. `Drag` lifts it, which is what curves a
  // streak instead of leaving it level.
  let drag = (P(4) - 0.5) * 70.0 + 20.0 * U.pulse;
  let headq = XY(vec4f(head_x, pos.y + drag, 0.0, 0.0));
  let head = melt(headq, t, hue, warp, 1.1);

  // --- the threshold, the sorted share and the streak's own rank along its run.
  let thr = mix(0.60, 0.14, clamp(P(3) + 0.45 * clamp(U.mid, 0.0, 1.0), 0.0, 1.0));
  let sorted = smoothstep(thr - 0.07, thr + 0.07, own.a);
  let s = clamp((pos.x - (head_x - run * 0.5)) / max(run, 1.0), 0.0, 1.0);
  let v = mix(head.a, own.a, 0.16 + 0.60 * s);

  // The streak: the palette walked in rank order, hot at the head and cooler along the tail.
  var streak = lut(0.04 + 0.90 * v + hue * 0.10) * (0.30 + 1.45 * v * v);
  // Dispersion: the streak's own colours split a hair, the way light does through the melt's glass.
  // A uniform branch, so a clip with no dispersion pays nothing.
  if (P(5) > 0.02) {
    let d = 0.05 + 0.16 * P(5);
    let r = lut(0.04 + 0.90 * clamp(v + d, 0.0, 1.0) + hue * 0.10).r;
    let b = lut(0.04 + 0.90 * clamp(v - d, 0.0, 1.0) + hue * 0.10).b;
    streak = vec3f(mix(streak.r, r, 0.65), streak.g, mix(streak.b, b, 0.65));
  }
  var c = mix(own.rgb, streak, sorted);

  // The head's core: a hot luminous line where the value crosses the threshold, which is what makes
  // the streak read as sorted light and not as a stretched photograph.
  let core = sorted * (1.0 - smoothstep(0.0, 0.10 + 0.10 * s, abs(own.a - thr)));
  c += lut(0.10 + 0.80 * clamp(v + 0.30, 0.0, 1.0)) * core * (0.35 + 0.55 * U.pulse);
  // The streak's leading edge: a thin line of light along each ribbon, bright where it is loud.
  c += lut(0.08 + 0.84 * v + hue * 0.10) * edge * (0.20 + 0.35 * clamp(U.mid, 0.0, 1.0));

  // The melt's own bloom: the pools glow through where nothing is sorted.
  c += own.rgb * (0.18 + 0.30 * P(6)) * (1.0 - sorted);
  c *= 0.88 + 0.30 * smoothstep(1.5, 0.2, length(vec2f(q.x * 0.62, q.y * 0.95)));
  c *= grain(pos.xy, 0.10 + 0.20 * P(7));
  return fin(shoulder(c));
}
