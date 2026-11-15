# Moshloom

Audio-reactive datamosh visualizer with a block timeline.

- **wasm (C)** is the brain: audio analysis, tempo, the timeline, and every per-frame decision
  (which block, I-frame or not, freeze, strobe, glitch ops). `plan(f)` returns that as floats.
- **WebGPU (WGSL)** draws the pixels: 16 scene shaders, block-matching motion estimation,
  a stable-fluids solver that can stir the mosh, the mosh feedback pass, and the output pass.
- If WebGPU is missing or the device is lost, the app falls back to the CPU renderer in the
  same wasm (`render(f)`), which has the first 8 visuals only.

Files
- `engine.c`        engine (freestanding C, no libc); `build.sh` / `build.bat` build `engine.wasm`
- `gpu.js`          WGSL shaders and the WebGPU frame loop
- `index.src.html`  UI; `pack.py` inlines gpu.js and the base64 wasm into `index.html`
- `test.mjs`        CPU harness: synthetic features, full render, determinism replay
- `gpu_test.mjs`    headless-Chromium harness: renders through WebGPU to an offscreen texture
- `scripts/make-fixtures.mjs`  regenerates the committed test fixtures (not part of `npm test`)

## Setup

```
npm install
npx playwright install chromium
```

`npm install` brings the two muxers that `pack.py` inlines plus Playwright. Building
`engine.wasm` needs `clang` and `wasm-ld` (LLVM): on Windows `winget install LLVM.LLVM`,
then use `build.bat`; on a POSIX shell use `./build.sh`. A prebuilt `engine.wasm` is
committed, so the app and the tests run before the toolchain is installed.

Build and pack: `npm run build` (finds `clang` on PATH, in `%ProgramFiles%\LLVM\bin`, or
via `LLVM_BIN`; then runs `pack.py` with `python`). `build.sh` and `build.bat` are the same
command by hand. Use `python`, not `python3` — on Windows `python3` may be a Store stub.

## Tests

```
npm test
```

- `pretest` re-packs `index.html` so the tests always run against the current sources.
- `test.mjs` renders 3,840 frames of the CPU path at 640x360, reports ms/frame, then replays
  from a reset and fails on any frame that changed between runs, or on a mostly black run.
- `gpu_test.mjs` launches Chromium with SwiftShader WebGPU, renders through the
  `window.__MOSH_OFFSCREEN` hook (headless loses the device when presenting to a canvas),
  and exercises media clips, a custom shader (good and deliberately broken), all nine
  effects, the fluid solver (including a seek-replay determinism check) and thumbnails. It
  exits non-zero on any WebGPU validation error, page error or blank frame. Output lands in
  `out/`, and the scene/effect frames are compared against `fixtures/golden/` at tolerance 2
  and 0.2% of pixels (`--no-golden` skips that, `node scripts/imgtest.mjs` runs it alone).
  The fluid frame is deliberately not a golden: a chaotic solver diverges across engine
  rebuilds, so its determinism is covered by the in-page replay instead.
  The suite also exports 45 frames from the *visible* canvas (the offscreen hook is dropped
  for that step) and decodes the result, failing if the frames are black. Software adapters
  such as SwiftShader cannot encode from a presented WebGPU canvas, so that step reports
  "skipped here" on them; on hardware it is the regression test for the capture-ordering bug.
  A lost device during export is a skip too, and an encoder that produces no chunks must fail
  instead of reporting `export done · 0.0 MB`.
  A second part opens a fresh main page and a fresh worker page, runs a worker smoke test
  (init, frames, readback) and compares one readback hash and plan against main. Where the
  machine cannot run worker + GPU it prints a skip with the reason. The same section switches
  modes through the header control, boots `#cpu`, and checks that `rwCall` rejects clearly
  when there is no worker. In worker mode it also exports 45 frames, exports a video clip
  twice and requires the same decoded clip-frame hash, checks worker thumbnails, and resizes
  the worker through the Size control. Every long-lived-page section starts by pinning 120 BPM
  and a known project, so a tempo or arrangement left by an earlier section cannot change a
  later result.

### Real-GPU export check

`npm test` uses SwiftShader, which cannot encode from a presented WebGPU canvas, so the export
step always skips there. Before a milestone, run:

```
npm run test:gpu:hw
```

That launches **headed Chrome with the real adapter** (no SwiftShader flags) and runs the same
suite. Export is not allowed to skip in this mode: a lost device or an encoder that refuses the
presented canvas is a failure. The export step now really turns the offscreen hook off, so it
captures the presented canvas. This is the run that covers the real-hardware export path where
the capture-ordering bug was found.

The committed goldens were recorded on SwiftShader and are adapter-specific, so the hardware
run skips the golden comparison by default. Pass `--hardware-golden` only if you have baselines
from that adapter and want to compare against them.

Record sizes are not hardcoded: `block_floats()` and `fx_floats()` are exported by the
engine and read by JS and both harnesses, so a change to `Blk` or `Fx` can't silently
desync the writers.

Fixtures (`fixtures/clip.png`, `fixtures/clip.mp4`) are committed. Regenerate with
`node scripts/make-fixtures.mjs`; it must load a `file://` page because WebCodecs needs a
secure context.

## Worker probe (P1)

`node lab/probe/run.mjs` asks three questions in two contexts and asserts the answers rather than
trusting "no error": did a blob worker start, did `navigator.gpu` exist inside it, and did a frame
reach the transferred OffscreenCanvas (checked by reading pixels back).

| context | blob worker | navigator.gpu in worker | frame on the transferred canvas |
| --- | --- | --- | --- |
| `http://127.0.0.1` | yes | yes | yes (readback luminance 0.112) |
| `file://` | yes | yes | yes (0.112) |
| published sandboxed page (Chrome, Windows, AMD RDNA-2) | yes | yes | yes (0.112) |

All three contexts pass, including the published sandboxed page: blob workers, `navigator.gpu`
in a worker and a frame on the transferred canvas all work there, so **worker + GPU is confirmed**
and T1 targets mode 1. The falling-back modes stay in the plan as the recovery path, not as the
expected default.

To re-run the sandbox half, publish `lab/probe/index.html` and read the four values it prints
(`blobWorker`, `gpu`, `drew`, `lum`).

### Render-mode probe (T1b2)

The app can ask whether a worker can present a WebGPU canvas, on a **throwaway canvas**:
`transferControlToOffscreen()` is one-shot, so the real canvas is never spent on a capability
test. That probe is lazy: it runs only when worker mode is requested (`?mode=worker`, `#worker`
or the header control) or when recovery has to decide whether a worker mode is available. Main
mode never starts it at boot, because a cosmetic label must not create a second GPU device or be
able to destroy the main one. The header shows no worker suffix until the probe has answered;
then it reads `WebGPU · worker ready` or `WebGPU · no worker`.

The probe clears a small texture and reads it back, so "no worker" means the worker really failed
rather than a copy being silently dropped. Two lessons are baked into it: `bytesPerRow` must be a
multiple of 256 (an 8-pixel copy at 32 bytes per row is a validation error that leaves the buffer
zeroed), and it installs `uncapturederror` and reports the first message.

The two blockers for dispatching frames are now gone: the renderer takes its wasm instance as a
parameter (`gpuWasm(X)`, so a worker can pass its own) and it owns the project writer
(`gpuSetProject`, which keeps `project` messages non-resetting and is covered by a record-level
test). Moving the writer changed no pixels: all 13 goldens stayed identical.

`?mode=worker` runs the renderer in a worker: its own wasm instance, its own GPU device, a
transferred OffscreenCanvas and the same protocol as the analysis worker. `gpu_test.mjs` opens
the app twice — once main, once worker — runs a worker smoke test, renders the same project at
the same frame indices and compares the readback and the plan:

    main   ?mode=main                     1280x720  hash 2576632446  lum 0.3203
    worker ?mode=worker&offscreen=1       1280x720  hash 2576632446  lum 0.3203   IDENTICAL

Two bugs only appear across that boundary, and both were caught by the comparison rather than by a
missing-error check: the worker never called `X.init()`, so the palette and CA tables were zeros and
every `lut()` returned black (the plan matched bit for bit, which is what localised it to the GPU
side), and `gpuMel` read the page's `nf` global, now `WASM.get_nf()`.

All four run modes can be requested and each renders:

| request | run mode | evidence |
| --- | --- | --- |
| `?mode=main`, header `main`, `#main` (default) | main + GPU | readback luminance 0.325 |
| `?mode=cpu`, header `CPU`, `#cpu` | main + CPU | 2D canvas luminance 0.104 |
| `?mode=worker`, header `worker`, `#worker` | worker + GPU | 0.325, hash identical to main |
| `?mode=worker-cpu` (test hook) | worker + CPU | 0.329 from the worker's own frame buffer |

The published page gets no query string, so the header **Engine** control (`main` / `worker` /
`CPU`) restarts the renderer through the recovery path. A plain fragment also works:
`index.html#worker` or `index.html#cpu`. `?mode=` stays for the automated tests.

`window.__runMode` reports the active one. The worker+CPU mode renders in the worker's wasm and
writes through a 2D OffscreenCanvas, so the CPU fallback keeps working after the split.

### Recovery

`window.__recover(why[, target])` replaces the canvas — a transfer cannot be undone, so recovery
always starts with a **new element** — replays `features`, `project`, `seek` and the current
`frame`, then re-renders. With no target it steps down the mode list; with a target (the header
Engine control) it restarts into that mode. If the target is worker + GPU and the probe has not
answered yet, this is where the probe runs:

    before: worker+gpu
    after : main+gpu  canvas swapped, G.ok, frame rendered, no page errors

The worker's own `device-lost` message and the worker `onerror` call the same function. A silent
`terminate()` cannot be made to look like a crash, so the mechanism is tested by inviting it
explicitly; the spontaneous triggers are wired but not simulated in the suite.

### T1c: export, thumbnails and media in the worker

In worker mode, export runs **in the render worker**. The worker switches to an offscreen
texture, reads each frame back, and encodes it; it owns the muxers, so the main thread only
supplies audio PCM and video frames. Thumbnails also render in the worker (`thumb` message),
and the Size control sends a `resize` to the worker instead of touching the transferred canvas.

Video clips are requested per export frame. The worker computes the clip time from the same
grid the plan uses, quantizes it to the export frame rate, and asks the main thread for that
frame. The main thread seeks the video, draws it to its 2D canvas and transfers an `ImageBitmap`.
Two exports of the same clip project decode to the same frame hash. The worker wasm, muxers and
OffscreenCanvas are all in the worker body; `pack.py` inlines the muxers once and the worker
source includes that text.

### Builds, goldens and what "same pixels" means

`engine.wasm` is committed and is the reference for every golden image. `npm test` does not
rebuild it (`pretest` only packs), and `scripts/build.mjs` writes `engine.build.json` with the
hash and the compiler that produced it.

The goldens in `fixtures/golden/` belong to that binary **and to the SwiftShader adapter**
that `npm test` uses. Given the same `engine.wasm` under SwiftShader, renders are bit-identical
across machines and Chrome versions (checked on two machines). A real GPU adapter has its own
precision and colour path and does not match these images, which is why `npm run test:gpu:hw`
skips the golden comparison unless a hardware baseline is supplied. Given the same `engine.c`
rebuilt by a different clang, the mosh frames differ substantially - up to 19% of pixels -
because the feedback amplifies rounding differences into different block matches.
"A different compiler gives the same picture" is false for anything with Mosh above 0.

`gpu_test.mjs` checks the hash first. If `engine.wasm` is not the binary the goldens were recorded
with, it stops with `FAIL: engine.wasm is not the build these goldens were recorded with.` and
prints no image table, because the table would be misleading. To accept a new build, re-record
deliberately with `npm run golden:record` (it prints which images changed and by how much) and
say why in the commit. A missing baseline is a failure; nothing is skipped silently.

The committed binary was built with clang 23.1.5 and the flags in `build.sh` / `build.bat`.

## Stats overlay

Press `` ` `` while playing to toggle a panel with: the run mode and canvas size, frame time, per-pass
milliseconds, the last `analyze()` time, and an estimate of live memory (wasm linear memory, the
decoded audio buffer, the feature arrays, and GPU textures by size and format).

Per-pass numbers come from GPU timestamp queries. They need `timestamp-query`, and in current
Chrome writing a stamp inside a render pass additionally needs
`chromium-experimental-timestamp-query-inside-passes`, which the app requests when the adapter
offers it. Where either is refused — including some Chromium builds — the overlay falls back to
wall-clock time around `onSubmittedWorkDone()` and shows the frame total only. Timestamps are only
recorded while the panel is open.

## Adding a visual

Append a fragment shader to `WG_SCENES` in gpu.js (uniforms in `SU`: T, hue, bass,
mid, high, pulse, vari, dens, beat, mel[24], line colours l0..l2; helpers `lut()`,
`mel()`, `melh()`, `ca()`), add its name/colour to `SCENES` in index.src.html, and a
default palette to `DEFPAL` in engine.c (raise the scene limit there and in `pack_scene`).

## Added in this version

- Offline export (`exportVideo` in index.src.html): renders frame by frame, encodes with
  WebCodecs, muxes with mp4-muxer / webm-muxer.
- Undo/redo, named projects in localStorage, tap tempo.
- Downbeat phase and section detection, used by Auto-arrange.
- Block thumbnails.
- Modulation: 3 slots per block, applied in wasm (`eff()` in engine.c). Bloom = periodic
  P-frame freeze.
- Effects lane: Fold, Mirror, Streak, Crush, Zoom, Spin, Echo, Negative, Fluid. Types are
  `0..15`: plan amounts at `PL[480..495]`, their parameters at `PL[496..511]`, and `fxa..fxd`
  / `fpa..fpd` in `FU`.
- Fluid (type 8) is a stable-fluids solver on a fixed 256x144 grid: advect, forces, curl,
  vorticity confinement, divergence, 20 Jacobi iterations, gradient subtract — 27 small
  passes, all inside the frame's single command encoder, so it is still one submit. Its
  velocity field is added to the feedback displacement, so it needs Mosh above 0. Measured
  cost at 1280x720 on an AMD RDNA-2 GPU: **+0.17 ms/frame** (6.52 -> 6.69 ms).
- Custom WGSL visuals (scene ids 17+) and Media blocks for images/video (scene id 16).
- Portable test suite, engine-driven record sizes, `build.bat` for Windows.
- Stats overlay (`` ` ``): frame and per-pass GPU timings, last `analyze()` time, and a memory
  estimate (wasm + audio + features + textures). Falls back to wall clock where the browser
  will not write timestamp queries.
- Strobe rate limit in the engine (12 frames between flashes; `strobe` 3 opts out).
- Export: falls back to WebM/VP9+Opus when the browser has no AAC encoder, and throws on an
  empty result (zero chunks or zero bytes) instead of reporting `export done · 0.0 MB`.
- Fixed: export captured the canvas in the same task as its submit, so every exported frame
  was black on a real GPU. The capture now happens after `onSubmittedWorkDone()`; verified on
  hardware (2.28 MB MP4 at 1280x720, decoded frame luminance 0.13 and 0.29).
- Fixed: the schema constant had been swallowed by a comment, which took out project save, load
  and the Project dialog. Restored, and covered by a new "projects" test section (round trip,
  both stores, migrate, dialog).
- Strobe: a flash is invert then tint again; flash starts are at least 12 frames apart; the
  Strobe transition alternates on the beat (every beat above 75 BPM) unless the incoming block's
  Strobe is set to Fast. Existing projects with a Strobe transition look different above 75 BPM,
  by design.
- Golden images are hash-pinned to `engine.wasm` (`fixtures/golden/meta.json`), recorded with
  `npm run golden:record`, and `fx_fluid` now has a baseline.
- T1b1: analysis runs in a worker with its own wasm instance. The protocol is defined once, in a
  `<script id="proto">` block that is executed by the page and injected verbatim into the worker;
  the worker body lives in `<script type="text/plain" id="analysis-worker">`. Features cross back as
  transferred arrays and are loaded with `load_features(nf)`, then `set_grid(bpm, offs)` derives
  `fPulse`/`fTT`/`fHue` — in that order, or `U.T` stays zero and the picture freezes. If workers are
  refused the app falls back to analysing locally and says so in a toast.
- T1c: export, thumbnails and video clips in the worker. Worker export encodes from an offscreen
  texture with the muxers inside the worker; clip frames are requested per export frame, quantized
  to the export rate, and two exports of the same clip project land on the same decoded frame.
  Worker thumbnails and worker-mode resize are covered by `npm test`. Main-mode export still uses
  the visible canvas and is the SwiftShader skip.
- T1b2: the render worker (its own wasm, GPU device and OffscreenCanvas) with the same protocol as
  the analysis worker. Worker mode is chosen by `?mode=`, `#worker` or the header Engine control and
  restarts through recovery. The worker probe is lazy (never in main mode at boot), a worker smoke
  test and a main-vs-worker hash/plan comparison are part of `npm test`, and `rwCall` rejects
  clearly when no worker is running.
- T1a: `gpu.js` touches no `document` or `window`. `gpuAttach({doc, offscreen})` supplies both, and
  the renderer's surface is exposed as `Renderer` (init, frame, resize, thumb, addMedia,
  compileShader, probe, reset, dispose). `setProject`/`loadFeatures` arrive with T1b, when the
  renderer owns its wasm instance. The 13 goldens were unchanged by this refactor.
- Not built: live audio input, MIDI, desktop packaging.
