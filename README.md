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
- `scripts/gate.mjs`  reduces an `npm test` log to the gate line; `scripts/zip.mjs` builds the reviewer zip

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
node scripts/gate.mjs --run      # npm test, logged to out/gate.log (UTF-8), reduced to one line
```

There is no hosted CI: GitHub Actions is off and every gate runs on the self-hosted machine.
`gate.mjs --run` prints the line AGENTS.md rule 3 asks for and exits 0 PASS, 1 FAIL, 2 truncated:

    gate: PASS · goldens 13/13, 0 differing px · CPU 13.73 ms/frame · page errors 0 · validation errors 0 · skips: none

It FAILs on any differing pixel, page or validation error, `FAIL` line, non-zero `npm test`
exit, any skip at all, and on a committed `index.html` that
`pack.py` would not reproduce (pretest re-packs, so a working-tree diff afterwards is a stale or
hand-edited file). `node scripts/gate.mjs <log>` gates a saved log; PowerShell-redirected logs are
UTF-16 and are decoded. `node scripts/gate.test.mjs` pins the parser to eight real logs in
`scripts/testdata/`.

**One GPU run at a time, enforced.** `gpu_test.mjs` takes a lock in `~/.moshloom` (shared by every
checkout and worktree of the user) before it launches a browser: `--hardware` takes the exclusive
`gpu.lock`, every other run its own `sw.<pid>.lock`. A run that would overlap prints
`gpu lock: refused ...` and exits 3 without starting Chromium; a lock whose pid is dead is removed.
`node gpu_test.mjs --lock-self-test` checks this without a browser (10 cases).

**No wait is unbounded.** Every `page.evaluate`, `waitForFunction` and `goto` in `gpu_test.mjs`
has a budget (60 s to boot or evaluate, 120 s to render or export, about 5 times the measured
duration with a 60 s floor). A timeout names the section and what it waited for, closes that page,
and fails the run: it never hangs and never becomes a skip. The run ends with a `bounded waits:`
line listing the slowest wait of each kind against its budget.

The suite also checks the timing grid right after the first feature load, on a fresh main page and
a fresh worker page, against constants (`grid order, main, f30 scene A clock 1.638, ...`), because
a main-vs-worker comparison cannot see a fault both sides share. It checks the stats overlay in
both modes (`stats main: ...`, `stats worker: ...`, including 0 worker messages after the panel
closes).

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
  The suite also exports 45 frames in main mode and decodes the result, failing if the frames
  are black. Export renders into the offscreen texture and reads every frame back, never the
  presented canvas, so it runs on SwiftShader too (`export: 1.02 MB video/mp4 640x360 1.5s,
  decoded luminance 0.273`). A lost device during export is a skip, and an encoder that produces no chunks must fail
  instead of reporting `export done · 0.0 MB`.
  A second part opens a fresh main page and a fresh worker page, runs a worker smoke test
  (init, frames, readback) and compares one readback hash and plan against main. Where the
  machine cannot run worker + GPU it prints a skip with the reason. The same section switches
  modes through the header control, boots `#cpu`, and checks that `rwCall` rejects clearly
  when there is no worker. In worker mode it also exports 45 frames, exports a video clip
  twice and requires the same decoded clip-frame hash, checks worker thumbnails, and resizes
  the worker through the Size control. Thumbnail checks wait for the real condition (every
  block has one, 60 s budget) instead of a fixed sleep: about 0.5-0.8 s instead of 14 s. Two
  fresh worker pages deliver the worker's `device-lost` message and its `onerror` through the
  page's own handlers and require `worker+gpu -> main+gpu`, a new canvas and a lit frame.
  Every long-lived-page section starts by pinning 120 BPM
  and a known project, so a tempo or arrangement left by an earlier section cannot change a
  later result.

### Real-GPU export check

`npm test` uses SwiftShader. Before a milestone, also run the suite on the real adapter:

```
npm run test:gpu:hw
```

That launches **headed Chrome with the real adapter** (no SwiftShader flags) and runs the same
suite. Export is not allowed to skip in this mode. Leave the machine alone while it runs: the
window is headed, and while the machine was in use the results changed between runs with no code
change (see T1c below).

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
able to destroy the main one. The header shows no worker suffix until the probe starts, then
`worker: checking` (and `worker: checking (retry)`), and only an answer gives `WebGPU · worker
ready` or `WebGPU · no worker`.

A timeout is not an answer (P1). The probe worker reports each step it reaches (start, adapter,
device, submit, readback), and progress re-arms its stall timer. An attempt that goes 5 s without
a message is retried once on a fresh worker with 10 s, both under a 15 s cap. Idle, the probe takes
40-190 ms. If both attempts stall, the header says `worker: no answer yet`, `window.__workerGpu`
stays null, and the next Engine-control choice or recovery probes again. Before P1, one slow probe
was cached as `no: probe timed out` and disabled worker mode for the session. Real failures (no
`navigator.gpu` in the worker, worker or transfer refused, an unlit readback, a worker error) are
still cached as `no: <reason>`. `window.__workerProbe` holds `{state, attempt, ms, why}`.

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
`terminate()` cannot be made to look like a crash, so `npm test` delivers both triggers through the
handlers the page installs on `rw`, each on a fresh worker page:

    recovery device-lost: worker+gpu -> main+gpu, new canvas, lum 0.6111 (G.off texture), page errors 0, validation errors 0, recovered in 434 ms (budget 120000 ms)
    recovery onerror: worker+gpu -> main+gpu, new canvas, lum 0.6696 (G.off texture), page errors 0, validation errors 0, recovered in 443 ms (budget 120000 ms)

The wait for recovery uses the 120 s render budget. A recovery that does not finish in time
fails as `recovery <trigger>: __runMode is still worker+gpu after 120000 ms`, not as a wrong
step-down. Before H8 a 45 s wait was swallowed, and a slow recovery on a busy machine was reported
as `stepped down to worker+gpu`. No wait a later assertion depends on swallows its timeout (H8).
A real device loss or a real worker crash is not simulated.

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

On SwiftShader (`npm test`, main at 6bcd8dd): worker export 1.94 MB 1280x720, decoded luminance
0.111; the clip export lands on hash 1500507020 twice; worker thumbnails 39/39.

**Hardware gate: green at dd5a5ec, two back-to-back runs, identical numbers.**

| line | run 1 (19:49-20:02) | run 2 (20:02-20:11) |
| --- | --- | --- |
| main export | 1.02 MB 640x360, decoded luminance 0.273 | same |
| worker export | 1.96 MB 1280x720, decoded luminance 0.11 | same |
| worker clip export | hash 195076583 twice, luminance 0.022 | same |
| main vs worker | hash 3336782791, lum 0.3203, IDENTICAL | same |
| recovery device-lost / onerror | worker+gpu -> main+gpu, lum 0.5665 / 0.6217 | 0.5593 / 0.6166 |
| profile | 4.23 ms total, fluid 0.21 | 4.93 ms, fluid 0.27 |
| gate | `PASS · goldens skipped (hardware) · page errors 0 · validation errors 0 · skips: none` | same |

Idle state: no input from 19:52:30 to 20:03:30 (run 1's last 10 minutes); run 2 saw input at about
20:03 and 20:08. The clip hash differs from SwiftShader's 1500507020 because the real adapter
renders the clip differently; it is repeatable within and across runs. Goldens are SwiftShader
references and are not compared on hardware.

History. `npm run test:gpu:hw` on the T1c merge (f4517c3) failed:
the main-mode export (visible canvas) decoded black, luminance 0, and the worker clip export was
not repeatable (3606441599 vs 3996078533); worker export itself was lit (1.96 MB, 0.11). Main-mode
export has decoded black in every hardware run since T1c work began; the last lit one was 0.101.
The same suite on a candidate fix (PR #4) also failed, including a worker export that had been
lit, so on this machine the hardware result currently changes between runs without a code change.
Two changes removed the dependence on the window: since X2 (#7) every export, main and worker,
renders into the offscreen texture and reads it back, and since H4 (#10) the harness decodes
exported files in a separate headless Chromium (`--use-angle=swiftshader`, so the decoded numbers
match the SwiftShader run) instead of inside the headed window.

### Milestone zip

    npm run zip -- --sha <rev> [--out <path>] [--check-pack] <log> [<log> ...]

Builds `out/moshloom-<shortsha>.zip` for the external reviewer: the tree at `<rev>` from
`git archive`, the given logs under `logs/`, and `MANIFEST.txt` (full SHA, commit subject, each
log's size and sha256). It reads the finished zip back, fails on a missing or empty log, on any
`node_modules` or `out` path segment, and leaves no zip behind on failure. `--check-pack` rebuilds
`index.html` in a temporary worktree of `<rev>` and requires it byte-identical.
`node scripts/zip.mjs --self-test` checks the entry-name rules.

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

The numbers always come from the renderer that is drawing. In worker mode the page's own `G` draws
nothing, so the page sends `stats-on` / `stats-off` to the render worker, which sets its own
`G.prof` and, while the panel is open, posts `stats` about 4 times a second (timings, texture bytes,
fluid, error count, size). Closing the panel stops both the recording and the messages. Texture
memory is one formula, `Renderer.texBytes(G)` in gpu.js, used on both sides. The first line names
the run mode (`worker+gpu 1280x720`, `main+gpu ...`). `window.__statsSnapshot()` returns the panel's
numbers for tests. Measured on SwiftShader in worker mode: before, `frame -` and ~70.3 MB (no
textures); after, `frame 145.76 ms (gpu timestamps)` with per-pass lines and ~126.8 MB, the same
texture total as main mode.

The frame time is GPU timestamps whenever the device has `timestamp-query` (A2). Timestamps are
resolved into two readback slots, so a frame never falls back to a wall clock just because the
previous frame's readback is still mapping. Before A2 that fallback timed the whole queue
draining: 200-56,000 ms in worker mode on a real GPU. A device without timestamps gets a per-frame
wall clock and says why (`frame 12.34 ms (wall clock: <reason>)`). If the experimental
inside-passes feature is refused, device creation asks for plain `timestamp-query` alone.
`__statsSnapshot()` carries `tsQ`, `tsWhy`, `frameMsSource` and `frameMsWhy`.

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
  Worker thumbnails and worker-mode resize are covered by `npm test`. Main-mode export later moved
  to offscreen readback as well (X2, #7).
- T1b2: the render worker (its own wasm, GPU device and OffscreenCanvas) with the same protocol as
  the analysis worker. Worker mode is chosen by `?mode=`, `#worker` or the header Engine control and
  restarts through recovery. The worker probe is lazy (never in main mode at boot), a worker smoke
  test and a main-vs-worker hash/plan comparison are part of `npm test`, and `rwCall` rejects
  clearly when no worker is running.
- T1a: `gpu.js` touches no `document` or `window`. `gpuAttach({doc, offscreen})` supplies both, and
  the renderer's surface is exposed as `Renderer` (init, frame, resize, thumb, addMedia,
  compileShader, probe, reset, dispose). `setProject` and `loadFeatures` were added in R1: the
  feature-loading sequence (arrays, mel, then `set_grid`) exists once, in `gpuLoadFeatures`, used by the
  page and the render worker. The 13 goldens were unchanged by the T1a refactor and by R1.
- Not built: live audio input, MIDI, desktop packaging.
