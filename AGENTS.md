# AGENTS.md — working contract

Moshloom is a single-page app with two test harnesses. Several agents work in this repo at
the same time, so this file is the contract. If a task prompt conflicts with it, this file
wins and the conflict gets reported, never silently resolved.

## The shape of the code

    engine.c   --build-->  engine.wasm     freestanding C: analysis, tempo, timeline, per-frame plan
    gpu.js                                 WGSL shaders + WebGPU frame loop; no document/window,
                                           exposed as `Renderer` (init, frame, resize, thumb, ...)
    index.src.html                         UI, worker protocol, export, thumbnails
    pack.py    --------->  index.html      inlines gpu.js + base64 wasm + the two muxers
    test.mjs                               CPU harness (node only)
    gpu_test.mjs                           headless Chromium + WebGPU harness

## Hard rules

1. **Never edit `index.html` by hand.** It is generated: edit `index.src.html` / `gpu.js`, then
   `python pack.py`. `npm test` re-packs in `pretest`, so a hand edit is silently lost.
2. **Goldens are pinned to `engine.wasm`.** `fixtures/golden/` was recorded under SwiftShader
   with the committed binary, and `fixtures/golden/meta.json` records which. Differing pixels
   are a finding, not something to accept. Re-record only with `npm run golden:record`, only
   when the pixel change is the point of the task, and say why in the commit body.
3. **`npm test` is the gate, and green is numbers, not a sentence:** goldens *N/13, 0 differing
   pixels*, CPU ms/frame, plus every step that skipped and the reason it skipped.
4. **Do not rebuild `engine.wasm`** unless the task is about the engine. A different clang moves
   mosh frames by up to 19% of pixels and invalidates every golden.
5. **`python`, not `python3`** (the Windows Store stub). The shell is PowerShell.
6. **One headed-Chrome run at a time.** `npm run test:gpu:hw` takes the real GPU. Only the
   coordinator runs it, only on the primary checkout, never alongside another run.

## Fan-out lanes

Up to **four** workers at once, one lane each, never two workers in the same lane.

| lane | owns | queue, in order |
| --- | --- | --- |
| **A · UI / worker** | `index.src.html`, `gpu.js`, `pack.py` | T1d (parked, see State) |
| **B · harness** | `gpu_test.mjs`, `test.mjs` | (empty) |
| **C · gates** | `scripts/gate*`, `scripts/testdata/**` | (empty) |
| **D · tooling** | `scripts/zip.mjs`, `package.json`, `package-lock.json` | (empty) |
| **E · docs** | `README.md`, `lab/*/README.md`, `.github/workflows/ci.yml` | coordinator |
| **frozen** | `AGENTS.md`, `fixtures/**`, `engine.c`, `engine.wasm`, `build.*`, `lab/`, `index.html` | coordinator only |

Lane A is the bottleneck: `index.src.html` carries the UI, the worker protocol and export in one
~90 KB file, so lane A runs one task at a time and other lanes merge around it. Splitting that
file is the only change that adds lanes (see State).

Two rules that make this survivable:

- **`index.html` is generated.** When a merge conflicts on it, resolve by running `python pack.py`
  and re-adding, never by hand-merging. Same for `engine.wasm` — if it conflicts, the branch
  rebuilt it, which rule 4 already forbids.
- **A lane that needs the GPU waits.** While `npm run test:gpu:hw` is running, no worker starts
  Chromium at all. Software-adapter runs of `npm test` in four worktrees is the outer edge of
  what the fixed timeouts in `gpu_test.mjs` tolerate.

## Native lanes (`native/`, the Rust port)

Design: `docs/superpowers/specs/2026-11-17-native-port-design.md`. Plan, with every lane's
interfaces and tests: `docs/superpowers/plans/2026-11-17-native-port.md`. The human asked for as
many parallel workers as possible here, so the four-worker cap above applies to the web lanes only.
Native lanes are one crate each, one worker each.

| lane | owns | task |
| --- | --- | --- |
| **N-core** | `native/Cargo.toml`, `native/rust-toolchain.toml`, `native/crates/mosh-core/**` | coordinator only (the contract) |
| **N-ipc** | `native/crates/mosh-ipc/**` | 1.1 |
| **N-tap** | `native/crates/mosh-tap/**` | 1.2 |
| **N-features** | `mosh-analysis`: `lib.rs`, `features.rs`, `grid.rs`, `loudness.rs`, their tests | 1.3 |
| **N-intensity** | `mosh-analysis`: `intensity.rs`, `tests/intensity.rs` | 1.4 |
| **N-harness** | `mosh-analysis`: `src/bin/**`, `offline.rs`, `tests/offline.rs` | 1.5 |
| **N-capture** | `native/crates/mosh-capture/**` | 1.6 |
| **N-render** | `native/crates/mosh-render/**` | 1.7 |
| **N-show** | `native/crates/mosh-show/**` | 1.8 |
| **N-als** | `native/crates/mosh-als/**` | 1.9 |
| **N-midi** | `native/crates/mosh-midi/**` | 1.10 |
| **N-video** | `native/crates/mosh-video/**` | 1.11 |
| **N-shaders** | `native/crates/mosh-shaders/**`, `scripts/extract_wgsl.mjs` | 1.12 |
| **N-app** | `native/crates/mosh-app/**` | 2.6, 2.7 |
| **N-tempo** | `mosh-analysis`: `src/tempo.rs`, `tests/tempo.rs` (+ the loopback hook in `Analyzer::push`) | 2.5 |
| **V-scenes** | `native/crates/mosh-visuals/src/{visuals.rs,scenes.rs}`, `wgsl/scene_*` additions in mosh-shaders | 2.1 |
| **V-mosh** | `native/crates/mosh-visuals/src/mosh.rs` (+ `src/mosh/`) | 2.2 |
| **V-fx** | `native/crates/mosh-visuals/src/{effects.rs,transitions.rs}` (+ `src/effects/`, `src/transitions/`) | 2.3 |
| **V-fluid** | `native/crates/mosh-visuals/src/fluid.rs` (+ `src/fluid/`) | 2.4 |
| **V-overlay** | `native/crates/mosh-visuals/src/overlay.rs`, a new `mosh-render/src/upload.rs` | 2.8 |

Rules for native lanes:

- `mosh-core` is the contract. A lane that needs a change to it stops and reports; it never edits it.
- The public signatures in `native/crates/mosh-visuals/src/*.rs` are a second contract (Phase 2):
  a V-lane fills in bodies and private items, never changes a public signature.
- Shader files a V-lane adds to `mosh-shaders/wgsl/` use its own prefix (`scene_`, `mosh_`, `fx_`,
  `tr_`, `fluidvis_`, `overlay_`); only V-scenes edits `mosh-shaders/src/lib.rs`.
- **`native/Cargo.lock` is generated.** On a merge conflict, take either side and run
  `cargo generate-lockfile` then `cargo test --workspace`; never hand-merge it.
- The gate for a native lane is `cargo test -p <crate>` and `cargo clippy -p <crate> -- -D warnings`
  in `native/`, plus `cargo test --workspace` after merge. The web gate (`npm test`) still runs on
  every merge, because one repo has one main.
- Workers may run short offscreen Vulkan tests. Windowed runs, Ableton, budget measurements and
  battery runs are coordinator-only, on the primary checkout.
- Audio for tuning lives in `C:\Users\ilanf\moshloom-tracks` and is never committed.
- `index.src.html`, `gpu.js`, `engine.c`, `fixtures/**` are read-only for native lanes.


- one task = one branch = one PR = one worker
- PR body: what changed, the test numbers, and what is *not* verified
- `npm test` green on the branch's own checkout; goldens unchanged whenever the change touches
  the render path
- no new page errors, no new WebGPU validation errors
- `README.md` updated when behaviour or evidence changed
- "should work" is not evidence. A claim without a number is a TODO.

## Test facts that bite

- **Exports never read the presented canvas.** Main and worker export render into the offscreen
  texture and read it back (`gpuCapFrame` in gpu.js), because a canvas capture is blank whenever
  the window is not composited. So export runs on SwiftShader too: `npm test` has **no allowed
  skip**, and the gate FAILs on any skip, including the old `export: software adapter cannot
  encode from the presented canvas`.
- **Hardware runs need an untouched machine with the display held on.** `npm run test:gpu:hw`
  drives headed windows. In worker mode, a second window showing a live WebGPU page stalls GPU
  readbacks while the display is on, and with the display off everything passes, so a run's result
  depends on display state as well as on the code. Since H9 the harness keeps one live app window
  (`windows: ... max live GPU windows 1`). Hold the display on, and quote the display and idle state
  with any hardware number.
- **The worker + GPU comparison needs a machine that can present from a worker.** Elsewhere it
  prints a skip with its reason. A skip is not a pass — quote it verbatim.
- **Sections share one long-lived page.** Every section starts with `resetSection()` (120 BPM,
  a known project) or a fresh page. A tempo left behind by an earlier section has already broken
  one comparison once.
- **Goldens are SwiftShader references, not adapter-independent.** `--hardware` skips the golden
  comparison unless `--hardware-golden` is passed.

## Coordinator protocol

The coordinator owns the integration checkout and the merges. Workers get their own worktree,
one branch per task, and never touch another worker's area.

- split by the lane table; never two workers in one lane
- merge serially, then re-run `npm test` on the integration checkout after each merge
- a worker's test claim is unverified until it is re-run after merge
- a worker that cannot run a step says so; the coordinator does not paper over the gap
- workers report numbers, not adjectives

## State

- In: M0, T1a, T1b1, T1b2. Worker mode exists and is **off by default on purpose**, until it has
  been used for real through the header Engine control.
- In: T1c (thumbnails, export and repeatable clip exports in the worker), with X2 (exports read
  back from the texture), H1-H8, Z1, C2, R1, U1, P1, A2, A3 (a stalled export readback fails after
  15 s) and H9 (one live window). Hardware gate green at b0eb768, display held on, two back-to-back
  runs (README, T1c section). The earlier dd5a5ec green was display-off luck.
- Known limit for T1d: worker mode stalls readbacks when another window shows a live WebGPU page
  with the display on; A3 turns the export hang into an error. Tabs in one window are fine.
- Next: T1d, still parked until the human has used worker mode through the header Engine control.
  The reviewer zip is no longer part of the flow (human decision, 2026-11-15).
- Ceiling: lane A is one file. `T1e` (split `index.src.html` into UI / worker body / export, with
  `pack.py` inlining modules) is the only task that widens the fan-out beyond four. It must land
  with 13/13 goldens untouched, and it does not go before a milestone gate.
- Milestone gate: `npm run test:gpu:hw` green twice back to back with the display held on.
