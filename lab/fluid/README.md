# lab/fluid — stable fluids on WebGPU

Standalone WebGPU fluid solver used as a reference for **Task F1**. One HTML file, no build
step, no network requests. Open it directly:

```
file:///C:/Users/ilanf/Desktop/moshloom/lab/fluid/index.html
```

Needs a browser with WebGPU (Chrome/Edge 113+, `chrome://gpu` should show WebGPU enabled).
There is no server, no bundler and nothing is fetched at runtime.

```
lab/fluid/index.html   the page: solver + page-side display/HUD/beat
lab/fluid/verify.mjs   headless check (Playwright + SwiftShader)
```

## What it does

- Grid **256 x 144**, every simulation texture is `rgba16float`:
  velocity ×2 (ping-pong), pressure ×2 (ping-pong), divergence, curl.
- Pass order per step: **advect** (semi-Lagrangian, linear sampling) → **add forces**
  (Gaussian splats) → **curl** → **vorticity confinement** → **divergence** →
  **20 × Jacobi pressure** → **subtract pressure gradient**. 26 dispatches per step.
- Velocity is shown as colour (hue = direction, brightness = speed).
- The checkbox switches to the same field **advecting a test image**: an abstract pattern is
  drawn into a 2D canvas, uploaded as an `rgba8unorm` texture, and the display shader samples it
  at `uv - v * warp * dt / grid` — one semi-Lagrangian backtrace step along the velocity field.
- Synthetic beat: an impulse every **0.5 s of simulated time**, at a position and angle hashed
  from the beat number with the brief's hash style —
  `frac(sin(b * 12.9898 + seed) * 43758.5453)`. No `Math.random` anywhere.
- Mouse drag adds force (pointer events, force ∝ drag velocity).
- HUD shows ms/step averaged over the last 30 steps; the same number is logged once to the
  console after the first 30 steps: `[fluid] ms/step avg30 = 24.03 (adapter: …)`.

Nothing drawn inside the simulation picture is text or a glyph — the HUD and the checkbox are
page UI layered over the canvas.

## Verification

```
node lab/fluid/verify.mjs
```

Launches headless Chromium with the same flags as `gpu_test.mjs`
(`--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=swiftshader
--use-webgpu-adapter=swiftshader --ignore-gpu-blocklist`), opens the page over `file://`,
runs 200 steps, collects `uncapturederror`, then reads the velocity texture back with
`copyTextureToBuffer` + `mapAsync`.

Last run:

```
init ok | adapter: google / swiftshader | grid 256x144
200 steps, 29.45 ms/step wall
beat: 12 impulses in 5.84 s of sim time
velocity readback: 36864/36864 cells non-zero (100.0%)
  max 75.0 cells/s, mean 11.47 cells/s, non-finite 0
velocity display: mean 39.7/255, lit 100.0%
image-advect mode: 30.07 ms/step
on-page readout: "30.03"
PASS - no WebGPU validation errors
```

**Measured cost on SwiftShader: ~25–31 ms per step** (varies 24–33 between runs; the number is
wall time per step including submit + `queue.onSubmittedWorkDone()`, so it is CPU + GPU for one
full step: 26 dispatches at 256×144 plus the display pass). This is the software rasteriser —
a real GPU should be roughly two orders of magnitude faster. Headless runs set
`window.__FLUID_OFFSCREEN` so the frame is rendered to an offscreen `rgba8unorm` texture
instead of the swap chain (presenting to a canvas in headless SwiftShader is unreliable);
the page renders to the canvas normally otherwise. A 1500-step soak test stays bounded
(max ≈ 75 cells/s, mean ≈ 17 cells/s), so the solver does not pump energy.

## Solver surface

`Fluid` is a single object literal; it never touches the DOM, canvas, camera or UI.

| member | meaning |
| --- | --- |
| `Fluid.W`, `Fluid.H` | grid size, defaults 256 × 144. Set before `init` if you need another size (shaders are generated from these). |
| `Fluid.init(device)` | **async**. Creates textures, sampler, uniform buffers, 7 pipelines and all bind groups; calls `reset()`; returns the object. |
| `Fluid.step(encoder, forces)` | **sync**. Encodes one full step into `encoder`. Caller submits. |
| `Fluid.velocityView` | getter → `GPUTextureView` of the current velocity texture (sampleable, linear-filterable). Identity is stable per parity, so it is safe as a cache key. |
| `Fluid.velocityTexture` | getter → the underlying `GPUTexture` (has `COPY_SRC`), for readback/debug. |
| `Fluid.pressureView` | getter → current pressure view. |
| `Fluid.reset()` | clears velocity, pressure, divergence, curl and resets both ping-pong indices. |
| `Fluid.pressureIterations` | default 20. |
| `Fluid.vorticity` | confinement strength, default 6 (higher values slowly pump energy — 20 diverges). |
| `Fluid.velocityDissipation` | per-step multiplier, default 0.999. |
| `Fluid.maxSpeed` | clamp baked into the shaders (cells/s), also maps NaN/Inf to 0. |

### `step(encoder, forces)`

```js
Fluid.step(encoder, {
  dt: 1 / 60,                       // seconds; clamped to [1/240, 0.05]
  splats: [                         // optional, max 8 per call
    { x: 0.5, y: 0.5,               // position in [0,1], y up, origin bottom-left
      fx: 0, fy: 6000,              // force, in sim cells per second (= velocity delta per second)
      radius: 0.07 },               // Gaussian radius as a fraction of grid height
  ],
});
```

- Velocities live in **cells per second** with +y up, so 256 cells/s ≈ one full width per second.
  Splat forces are the same unit; the shader does `v += f * dt * exp(-d²/r²)`.
- Splat `radius` is normalised to grid height; the shader applies the grid aspect ratio so splats
  stay circular.
- Uniforms are written with `queue.writeBuffer` inside `step`, before your submit. Therefore:
  **one step per encoder/submit**. Submitting several steps from one encoder would make every
  step read the last uniform values.

### Uniform layout (per pass, std140-ish)

| pass | buffer |
| --- | --- |
| advect | `vec4(dt, dissipation, 0, 0)` |
| splat | `vec4(dt, count, aspect, 0)` + `array<vec4<f32>, 16>`; splat `i` uses `[2i] = (x, y, fx, fy)` and `[2i+1] = (radius, 0, 0, 0)` |
| vorticity | `vec4(dt, epsilon, 0, 0)` |
| jacobi | `vec4(alpha = 0.25, 0, 0, 0)` |
| curl / divergence / gradient | no uniform (their bindings are texture-only; adding an unused uniform would be dropped by `layout: 'auto'`) |

All bind groups for both ping-pong parities are built once in `init`; textures are never
recreated, so nothing needs rebuilding per frame.

## What Task F1 has to change when lifting it

1. **Copy** the `Fluid` object plus its WGSL block (`SH_HEADER`, `WGSL_ADVECT`, `WGSL_SPLAT`,
   `WGSL_CURL`, `WGSL_VORT`, `WGSL_DIV`, `WGSL_JACOBI`, `WGSL_GRAD`). That is the whole solver;
   everything from `const OFFSCREEN = …` down is demo scaffolding.
2. **Forces come from the app, not the beat.** Delete the page's `forcesFor()` and pass splats
   built from whatever drives the visual (audio band energy → `fx/fy`, position from a pointer,
   camera or beat detector). One call per frame, ≤ 8 splats; call `step` once more if you need
   more. Keep `dt` in seconds; the solver clamps it.
3. **Replace the display.** The page's `DISPLAY_WGSL` is a demo: it samples `Fluid.velocityView`
   and either colour-maps it or warps the test image. F1 should sample the same view in its own
   composite pass and drop the test-image branch. Note the view flips between two textures each
   step — read `Fluid.velocityView` after `step()` and rebuild/cache your bind group by view
   identity.
4. **Coordinate spaces.** Sim UVs are `[0,1]` with y up regardless of canvas aspect; multiply
   force magnitudes in cells/s by `Fluid.H` (not `W`) to convert from "heights per second".
   If F1 feeds camera-space or pixel-space vectors, divide by the grid size first.
5. **Uniform layout** only changes if you expose more tunables: `dt`, dissipation, splat count,
   aspect and the splat array are the whole dynamic surface. If you drive dissipation from audio,
   write it in the advect uniform each step rather than once at init.
6. **Grid size**: set `Fluid.W`/`Fluid.H` before `init(device)` if 256×144 is wrong for the
   composition. The shaders bake both dimensions in, so re-`init` after changing them.
   `reset()` on scene changes; it is cheap (six clears).
7. **Do not** present to the swap chain from inside the solver: it owns no canvas. Let the host
   app call `getCurrentTexture()` once per frame and run `Fluid.step` in its own encoder.
