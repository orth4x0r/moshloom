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
| **A · UI / worker** | `index.src.html`, `gpu.js`, `pack.py` | R1 → U1 → T1d |
| **B · harness** | `gpu_test.mjs`, `test.mjs` | H2 |
| **C · gates** | `scripts/gate*`, `scripts/testdata/**`, `.github/workflows/pack.yml` | H1 |
| **D · tooling** | `scripts/zip.mjs`, `package.json`, `package-lock.json` | Z1 |
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

## Definition of done

- one task = one branch = one PR = one worker
- PR body: what changed, the test numbers, and what is *not* verified
- `npm test` green on the branch's own checkout; goldens unchanged whenever the change touches
  the render path
- no new page errors, no new WebGPU validation errors
- `README.md` updated when behaviour or evidence changed
- "should work" is not evidence. A claim without a number is a TODO.

## Test facts that bite

- **Export skips on SwiftShader.** A software adapter cannot encode from a presented WebGPU
  canvas. That skip is expected there; on `--hardware` the export step must fail instead.
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
- In flight: T1c — thumbnails and export in the worker, repeatable video-clip exports.
- Ceiling: lane A is one file. `T1e` (split `index.src.html` into UI / worker body / export, with
  `pack.py` inlining modules) is the only task that widens the fan-out beyond four. It must land
  with 13/13 goldens untouched, and it does not go before a milestone gate.
- Milestone gate: `npm run test:gpu:hw` green, then a zip to the reviewer, who reviews the whole
  thread split.
