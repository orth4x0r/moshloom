# Moshloom Native: design

Status: draft for approval, 2026-11-17. Lives next to the web app in the same repo, under `native/`.

## Intent

Moshloom becomes a **live performance visualizer that runs on the same machine as Ableton Live**
during a set. The web app stays as the reference implementation; the native app is a port plus new
live features. Because Ableton needs the machine, **CPU and GPU cost are the first constraint**,
ahead of features.

Two ways to play, modelled directly on Ableton:

- **Arrangement mode**: a per-song timeline of visual clips that follows Ableton's playhead bar for
  bar.
- **Session mode (improv)**: a clip grid launched live from a hardware controller, quantized to the
  beat, with random and procedural options.

In both, the music's **intensity** decides *when* and *how hard* the visuals react; the clips decide
*what* is shown.

**How the human plays (shapes everything below):** they press play on backing tracks in Live and sing
over them. Hands are not free during a set: no controller, no mouse. So the show must run
**hands-free**: Arrangement mode follows Live's playhead, and Session mode can run on autopilot
(follow actions, random and procedural launches driven by intensity). The controller (an Akai APC
mini) is for building and rehearsing looks, and for an optional manual takeover.

## Goals

1. Audio and Ableton's transport (tempo, bar/beat position, play/stop) reach the app from a VST3
   plugin on the master, through shared memory. Windows loopback capture is the fallback.
2. A 2D layer stack: 2-3 visualizer layers plus an overlay/video layer on top, each layer with its
   own effects, composited with blend modes and opacity.
3. The web app's visuals ported: the 16 scenes, the mosh feedback pass with motion estimation,
   the effects lane (Fold, Mirror, Streak, Crush, Zoom, Spin, Echo, Negative), custom WGSL scenes.
4. **Fluid as its own visualizer** (dye and force injected by the music), not only an effect. It can
   still stir a mosh layer.
5. Live, causal audio analysis aligned to Ableton's beat grid, producing an intensity level and
   build/drop/breakdown events.
6. Session mode: clip grid (columns = layers, rows = scenes), quantized launch, scene launch, follow
   actions (next / random), procedural parameter generation with seeds, presets, live parameter
   control from controller knobs.
7. Arrangement mode: timeline of clips per song following Ableton's playhead; a Session launch can
   override the arrangement and "back to arrangement" returns to it.
8. Controller support for the **Akai APC mini** (8x8 pad grid with LED feedback, 9 faders) via MIDI,
   for building and rehearsing; generic MIDI learn for anything else. Ableton MIDI clips driving the
   app is built in parallel as a second input path.
9. Rendering on Vulkan, close to the metal (`ash`), Windows first.

## Non-goals (for this design)

- 3D scenes, 3D geometry, camera work.
- macOS/Linux builds (Vulkan keeps MoltenVK open for later; nothing Windows-only beyond capture,
  shared memory and video decode, each behind one module).
- CLAP or AU plugins (Ableton on Windows loads VST3; CLAP is not supported by Live).
- Porting the web app's CPU fallback renderer.
- Bit-exact parity with the web app's pixels. The web goldens stay with the web app.

## Anti-goals (failure even if it works)

- **Audio dropouts in Ableton caused by the plugin.** The plugin's audio callback never allocates,
  locks, or does analysis.
- **The app competing with Ableton for CPU** so the set needs a bigger buffer.
- **Flicker**: intensity or scene changes that jitter frame to frame instead of landing on beats,
  bars or phrases.
- **Anything that needs hands during a set.** Once Live is playing, the show runs on its own until
  the set ends; a crash or stall recovers by itself (the stage output never shows a dialog).

## Performance budget (proposed numbers, to confirm)

Reference machine: the laptop's **integrated AMD Radeon GPU, on battery** (human: the set is played
off battery, so the discrete RX 6700S is never used live). Live 12 Suite, Focusrite/UMC ASIO. The
integrated GPU shares memory bandwidth and power with the CPU that runs Ableton, so GPU cost is also
CPU and battery cost.

| where | budget |
| --- | --- |
| plugin audio callback | < 20 us per 256-sample block, zero allocations, zero locks |
| app CPU, all threads | < 1 core-equivalent average, < 2 ms on the render thread per frame |
| app GPU on the integrated GPU, on battery, 1920x1080 output at 60 fps, 3 layers + overlay | < 6 ms per frame (layers render at a lower internal resolution and are upscaled) |
| app with the stage output idle (paused, black) | < 1% GPU, render thread asleep |
| audio-to-photon latency | < 50 ms, measured |
| app memory | < 500 MB with 4 videos loaded |

Every budget has a measurement in the app (stats overlay, like the web app's) and a test that fails
when it is exceeded on the reference machine.

## Approaches considered

**Audio and transport into the app**

| approach | trade-off | decision |
| --- | --- | --- |
| VST3 tap plugin on the master, shared memory to the app | works with ASIO; Ableton's exact tempo and bar position per block; microsecond cost; two things to install | **chosen** |
| WASAPI loopback of the output device | no Ableton setup; captures nothing when Live uses ASIO, picks up system sounds, +10-30 ms, no beat position | **fallback** |
| virtual audio cable | clumsy with ASIO (Live uses one audio device on Windows) | rejected |
| Max for Live device | needs Suite, streaming audio out is awkward | rejected |
| Ableton Link for tempo | gives tempo/phase but no audio; the plugin already gives both | rejected |

**Rendering**

| approach | trade-off | decision |
| --- | --- | --- |
| raw Vulkan via `ash`, WGSL compiled to SPIR-V by `naga` at build time | full control over passes, memory, frame pacing; more code | **chosen (human: Vulkan, close to metal)** |
| `wgpu` (uses Vulkan on Windows) | less code; adds a tracking/validation layer and less control over pacing | rejected |
| rewrite shaders in GLSL/HLSL | no reuse of the web app's WGSL | rejected; WGSL stays the source |

**Arrangement**

| approach | trade-off | decision |
| --- | --- | --- |
| A. hardware launcher (Launchpad/APC/Push style) + the app's own timeline | the human's meaning of "MIDI clip controller" | **chosen** |
| B. Ableton MIDI clips drive the app (notes launch clips, CCs move parameters) | arrangement lives in the Live set; needs a second plugin on a MIDI track | **built in parallel** (human decision) |

## How it works

    Ableton master --> [mosh-tap VST3] --shared memory--> [app]
    Ableton MIDI track --> [mosh-control VST3] --shared memory--> [app]
    hardware controller --MIDI--> [app]
    (fallback) WASAPI loopback --> [app]

    app:  input -> analysis -> show (session/arrangement, intensity mapping) -> render (Vulkan)

### Workspace layout (`native/`, one Cargo workspace)

| crate | kind | owns |
| --- | --- | --- |
| `mosh-ipc` | lib | shared-memory layout and the lock-free single-producer/single-consumer rings: audio blocks, transport snapshots, MIDI events; versioned header; Windows named file mapping |
| `mosh-tap` | VST3 (nih-plug) | two plugins from one crate: **Moshloom Tap** (audio effect for the master, passes audio through unchanged, writes audio + transport) and **Moshloom Control** (instrument on a MIDI track, outputs silence, writes notes/CCs + transport) |
| `mosh-capture` | lib | WASAPI loopback fallback, producing the same stream type as the ipc reader |
| `mosh-analysis` | lib | causal live analysis and the intensity model (below); offline evaluation harness on WAV files |
| `mosh-show` | lib | the data model and its logic: layers, clips, presets, parameters, session grid, arrangement, launch quantization, follow actions, procedural generation, serialization |
| `mosh-shaders` | lib + build.rs | WGSL sources (extracted from `gpu.js`, new ones added), compiled to SPIR-V by naga at build time, plus the uniform/push-constant layouts as `#[repr(C)]` Rust types |
| `mosh-render` | lib | Vulkan via ash: instance, device selection, swapchain, frame pacing, layer render targets, compositor, mosh/motion pipelines, fluid, video textures, GPU timing |
| `mosh-video` | lib | video overlay decoding (Media Foundation, hardware decode) to GPU-uploadable frames |
| `mosh-midi` | lib | controller I/O via `midir`: generic MIDI learn mapping, grid controller profiles with pad LED feedback |
| `mosh-app` | bin | window (winit), threads, wiring, performance UI, stats overlay, config |

### Contracts (frozen before the fan-out; changes go through the coordinator)

1. **`mosh-ipc` layout**: header `{magic, version, sample_rate, channels, block_capacity}`; an audio
   ring of fixed-size blocks; each block carries a transport snapshot
   `{tempo_bpm, ppq_position, bar_start_ppq, time_sig_num, time_sig_den, playing, sample_pos}`;
   a MIDI event ring `{ppq, status, data1, data2, channel}`. Writer never blocks; when the reader is
   slow the writer overwrites and counts drops.
2. **Analysis output** (`mosh-analysis` -> `mosh-show` / `mosh-render`), one struct per video frame:
   beat phase, bar phase, beat/bar counters, band levels (low/mid/high + 24 mel), onset strength,
   loudness, brightness, intensity level (1-4) and a continuous intensity value, and events
   (beat, bar, phrase, build-start, drop, breakdown).
3. **Shader parameter ABI** (`mosh-shaders`): every visualizer gets the same per-layer uniform
   block (time, beat/bar phase, bands, intensity, 8 generic float params `p0..p7` with documented
   meaning per visualizer, palette) so a clip's parameters are just values for `p0..p7` plus
   palette. Effects get the same treatment.
4. **Show model** (`mosh-show`): `Layer`, `Clip {visualizer, params, effects, intensity_ranges}`,
   `Preset`, `SessionGrid {columns: layers, rows: scenes}`, `Arrangement {clips placed in bars}`,
   `LaunchQuantize {none, beat, bar, phrase}`, `FollowAction {none, next, random, random_procedural}`.
   Serialized to a project file (RON or JSON, builder's choice) with a schema version.

### Intensity model (`mosh-analysis`)

- Features per block (2048-point FFT at 48 kHz, hop matched to the plugin's block size): loudness
  (short-term, meter-like), low-end weight (kick/bass band), onset density (spectral flux peaks per
  beat), brightness (spectral centroid), 24 mel bands.
- Aggregated **per beat and per bar** on Ableton's grid, and compared with a rolling window of the
  last 8-32 bars (causal normalisation, replacing the web engine's whole-track percentiles).
- **Intensity level 1-4** with hysteresis, allowed to change only on bar boundaries; level jumps on
  phrase boundaries (every 4/8/16 bars) count as section changes.
- **Events**: build-start (intensity rising across consecutive bars), drop (low end returns on a
  downbeat after a build or a gap), breakdown (low end and onset density fall for a full bar).
- Loopback mode has no transport: tempo and beat phase come from causal tempo tracking (the web
  engine's comb-filter idea, run on a rolling window).
- Tuned against an offline harness: WAV files with hand-labelled sections/drops, reporting
  detection timing error in beats.

### How intensity drives the picture (`mosh-show`)

- The clip decides what. Intensity decides when and how hard.
- Each clip holds parameter **ranges per intensity level**; the current level picks the range, and
  rhythm (beat/bar phase, onsets) moves values within it. Higher levels: deeper beat pulse, more
  glitch/mosh/strobe, more layers visible, faster cuts.
- Level jumps and drops can fire follow actions at the next downbeat (Session), and can be
  disabled per clip. A build can hold a change back so it lands on the drop.
- Manual always wins: a controller "lock" holds the current clip; any launch overrides automation.

### Session and arrangement

- Session grid: columns are the visualizer layers plus the overlay layer, rows are scenes; launch a
  clip, a row, or stop a column; quantize to beat/bar/phrase; follow actions next/random/random
  procedural (fresh seeded parameters within the clip's ranges; seed shown so a look can be saved
  as a preset).
- Arrangement: **one timeline per show, in beats**, mirroring Live's arrangement (human: one long Live
  arrangement holding every song, with tempo automation). Positions are in ppq (beats), so tempo
  automation changes nothing in the mapping; time-signature changes come from the plugin's
  `bar_start_ppq` and `time_sig`. Song markers split the timeline into songs; they can be placed by
  hand or **imported from the Live set's locators** (the `.als` file is gzipped XML), along with the
  time-signature map. The app follows jumps, loops and scrubbing of Live's playhead (it reads the
  position every block; it never assumes continuity). Clips sit in lanes per layer; a Session launch
  takes that layer over until "back to arrangement".
- When a set is not one long arrangement, Session mode is the way to play it.
- Path B: Moshloom Control turns notes into clip launches (note number -> grid cell) and CCs into
  parameters, so an arrangement can also live as MIDI clips in the Live set.

### Rendering (`mosh-render`)

- One device on the **integrated GPU**, chosen explicitly (it drives the laptop panel and the
  external output, so no copy between GPUs). The discrete GPU may be selectable for development but
  is never the reference.
- **Two windows on that one device**: the stage output, fullscreen on the secondary monitor (no UI on
  it, ever), and the control/editor window on the laptop screen. The control window redraws only
  when something changes, at most 30 fps, and can be hidden during a set.
- Each layer renders into its own target at a configurable internal resolution (render small,
  upscale), then the compositor blends layers and overlays into the swapchain image. Mosh feedback,
  motion estimation and fluid run per layer that uses them.
- Frame pacing: FIFO/mailbox chosen by measurement; never busy-wait; the render thread sleeps.
- GPU timestamps per pass feed the stats overlay; the budget table above is checked there.

## Constraints

- Windows 11, Rust (stable, MSVC), Vulkan 1.3+ (the machine reports 1.4), Live 12 Suite.
- The Vulkan SDK (validation layers) is installed for development; release builds run without it.
- The plugin and the app are separate processes; the app must start, stop and restart without
  affecting Live, and the plugin must work (pass audio through) with no app running.
- Same repo as the web app, under `native/`; AGENTS.md gets a native lane table.

## Left to the builder

- Exact crate versions; RON vs JSON; the UI toolkit for the editor screens (immediate-mode is
  expected; it must not run while the performance screen is up if it costs frame time).
- How the WGSL is extracted from `gpu.js` (script or by hand), as long as the web app is untouched.
- Shared-memory naming and discovery details within the contract above.
- Test strategy per crate (unit tests everywhere; offline WAV harness for analysis; offscreen
  render tests for the renderer).

## Calls made by the coordinator (correct any of these)

- **Transitions carry over** (Cut, Crossfade, Datamosh, Melt, Strobe, with the I-frame/freeze logic
  from `decide()` in engine.c). A clip has a default transition-in; a launch may override it; the
  transition runs per layer, since each layer owns its feedback buffers. Added to contract 4.
- **Custom WGSL scenes compile at runtime** with naga, with a broken shader falling back to the last
  good one (never a black layer mid-set). The per-layer uniform block is a **superset of the web
  layout** (`T`, `hue`, `vari`, `dens`, `aux0..2`, `l0..l2`, mel bands, `palT`/`caT`/`melT`/`rdT`)
  so web custom shaders run unchanged; `p0..p7` are added on top. Changes contract 3.
- **Several Taps are allowed by the IPC contract** (each instance has a role: master, drums, bass,
  other, and its own region). Version 1 uses the master only; stem taps are a later improvement to
  drop detection.
- **The flash limit is a safety rule**: at least 12 frames between full-frame flashes, enforced in
  one limiter in the render path that clips, intensity and controller input cannot bypass.
- **Overlay video v1** uses Media Foundation hardware decode (H.264/HEVC) with add/screen/multiply/
  luma-key blending; true alpha (HAP, ProRes 4444, VP9 alpha) is a later lane.
- **Authoring** happens in the native app (timeline + clip editor), since the human chose a hardware
  launcher plus the app's own timeline. Importing web-app projects is a later, optional item.
- **Loopback fallback** ships in v1 with a simple causal tempo tracker; it is the last lane to finish.
- **Own rig first**: device and controller profiles are explicit and configurable, no installer or
  code signing in v1.
- **Fan-out**: the native workspace gets its own lane table in AGENTS.md, separate from the web
  lanes' four-worker cap (human: "as many workers as possible"). Workers may run short offscreen
  Vulkan tests; windowed runs, budget measurements and anything with Ableton running are
  coordinator-only on the reference machine.

## Open questions for the human

1. ~~Set layout~~ answered: one long arrangement with all songs and tempo automation (see Session
   and arrangement).
2. ~~Output~~ answered: fullscreen on a secondary monitor; integrated GPU on battery.
3. ~~Controller~~ answered: Akai APC mini, used for building and rehearsal, not during a set.
4. Tuning tracks: 15 of the human's tracks (13 WAV 44.1 kHz 16-bit, 1 WAV 48 kHz float, 1 MP3 48 kHz), stored outside the repo in
   `C:\Users\ilanf\moshloom-tracks`. Labels: proposed by the coordinator from an offline analysis,
   corrected by the human.
