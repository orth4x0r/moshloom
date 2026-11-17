// Glitch Sort: a generative photograph smeared into pixel-sorted bands.
//
// Pixel sorting's signature is a run of pixels reordered by brightness, so the frame is cut into
// runs (per 3-pixel row, each row offset by a hash) and a run above the threshold is rewritten in
// the palette's brightness order: the run's own value sets where along the ramp it lands, and the
// row above decides how far the streak is dragged, which is where the long bands come from. Below
// the threshold the photograph is untouched, and that boundary is what reads as a tear.
//
// Audio: `U.mid` lowers the threshold so the whole frame sorts on loud parts, the mel bands set
// each row's run length, `U.pulse` opens the RGB split and brightens the sorted bands. Corrupt
// 8x8 blocks and the scanline gate are the databending.
//
// P(0) clock rate, P(1) run length, P(2) hue, P(3) threshold, P(4) smear, P(5) RGB split,
// P(6) corrupt blocks, P(7) scanlines.

const RED = vec3f(0.80, 0.06, 0.10);
const CYAN = vec3f(0.10, 0.74, 0.88);

/// The generative photograph: a lit form with a bright core, a dark side, hard edges and grain.
/// `.a` is its value.
fn photo(p: vec2f, t: f32) -> vec4f {
  let a = sin(p.x * 1.7 + t * 0.21) * cos(p.y * 1.25 - t * 0.13);
  let b = exp(-length(p * vec2f(0.95, 1.45) - vec2f(0.32, -0.18)) * 2.1);
  let c = exp(-length(p * vec2f(0.70, 1.00) + vec2f(0.48, 0.26)) * 1.7);
  let ridge = smoothstep(0.02, 0.00, abs(p.y + 0.35 + 0.25 * sin(p.x * 2.3 + t * 0.2)));
  let grain = h2(floor(p * vec2f(220.0, 260.0))) - 0.5;
  let v = clamp(0.34 + 0.34 * a + 0.34 * b - 0.34 * c + 0.14 * ridge + 0.09 * grain + 0.10 * sin(p.x * 17.0 + p.y * 9.0) - 0.10 * p.y, 0.0, 1.0);
  let col = lut(0.08 + 0.82 * v) * (0.34 + 1.20 * v);
  return vec4f(col, v);
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = XY(pos);
  let t = U.T * (0.35 + 0.45 * P(0));
  let density = clamp(U.dens, 0.0, 1.0);
  let px = 2.0 / U.res.y;

  // --- the runs: one per 3-pixel row, offset by a hash, length from the row's band.
  let row = floor(pos.y / 3.0);
  let band = mel(i32(fract(row * 0.0131) * 24.0));
  let run = mix(150.0, 14.0, clamp(P(1) + 0.35 * density, 0.0, 1.0)) * (0.55 + 0.9 * band);
  let offset = h1(row * 1.71 + 3.0) * run;
  let x0 = (floor((pos.x - offset) / run) + 0.5) * run + offset;

  // --- the photograph, the threshold, and the streak the sorted part becomes.
  let own = photo(p, t);
  let thresh = mix(0.60, 0.14, clamp(P(3) + 0.45 * clamp(U.mid, 0.0, 1.0), 0.0, 1.0));
  let sorted = smoothstep(thresh - 0.05, thresh + 0.05, own.a);
  let rank = clamp((own.a - thresh) / max(1.0 - thresh, 0.05), 0.0, 1.0);
  // The run's own value: a run is rewritten at one brightness, and the drag from the row above is
  // approximated by the streak's own offset, so this is two evaluation of the photograph instead of
  // four, which is what the pass costs.
  let smear = 6.0 + 30.0 * P(4) + 12.0 * U.pulse;
  let start = photo(vec2f(x0, pos.y - smear * 0.5), t).a;
  let band_rank = clamp((start - thresh) / max(1.0 - thresh, 0.05), 0.0, 1.0);
  // Local contrast: this pixel against the run it lands in. A run of flat pixels stays photographic;
  // a run with range stretches into a full band.
  let local = clamp((own.a - start) / (abs(start - own.a) + 0.10), 0.0, 1.0);
  let r = mix(rank, mix(band_rank, local, 0.45), 0.70);

  // The sorted band: the palette walked in brightness order, hot at the top of the run.
  var streak = lut(0.10 + 0.80 * r) * (0.45 + 1.15 * r);
  // A slight channel offset: the capture card's own colour error, taken from the run's own value.
  let split_b = lut(0.10 + 0.80 * clamp(r + 0.18, 0.0, 1.0)).b * (0.45 + 1.15 * r);
  streak.b = mix(streak.b, split_b, 0.65 * (0.3 + 0.9 * P(5)));

  var c = mix(own.rgb, streak, sorted * 0.88);

  // --- the tear line: where a sorted band meets untouched pixels, one hot accent scanline.
  let edge = sorted * (1.0 - smoothstep(0.0, 0.18, abs(own.a - thresh)));
  c += mix(RED, CYAN, fract(row * 0.37)) * edge * (0.30 + 0.70 * U.pulse);

  // --- corrupted 8x8 blocks: channel rotation, a black hole, or a bright wash.
  let block = floor(pos.xy / 8.0);
  let rnd = h2(block + floor(t * 3.0));
  let gate = P(6) * 0.010 + 0.006 * clamp(U.high, 0.0, 1.0);
  if (rnd > 1.0 - gate) {
    c = mix(c, vec3f(c.b, c.r, c.g) * 1.15, 0.6);
  } else if (rnd > 1.0 - gate * 0.55) {
    c = mix(c, vec3f(0.02, 0.0, 0.02), 0.85);
  }

  // --- the gate of a capture card: scanlines and a cold vignette.
  let scan = 1.0 - (0.14 + 0.20 * P(7)) * (0.5 + 0.5 * cos(pos.y * PI * 0.75));
  c *= scan;
  c *= 0.66 + 0.34 * smoothstep(1.40, 0.25, length(vec2f(p.x * 0.60, p.y * 0.95)));
  c += vec3f(0.006, 0.008, 0.012);
  // Hue: a channel rotation on the finished frame, so P(2) shifts the capture's colour cast.
  c *= mix(vec3f(1.0), vec3f(0.88, 0.62, 1.24), U.hue);
  return fin(c);
}
