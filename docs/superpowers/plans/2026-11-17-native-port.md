# Moshloom Native Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A native Windows app plus two VST3 plugins that turn Ableton Live's master output and transport into a hands-free, intensity-driven 2D visual show on the integrated GPU, at a cost Ableton does not notice.

**Architecture:** One Cargo workspace under `native/`. Shared types live in `mosh-core` and are frozen before the fan-out (Phase 0, Appendix A). Each other crate is one lane, owned by one worker, and talks to the others only through `mosh-core` types and the public functions named in its task. The app crate wires the lanes together in Phase 2.

**Tech Stack:** Rust 1.99 stable (MSVC), edition 2024; `ash` (Vulkan 1.3), `naga` (WGSL -> SPIR-V), `nih-plug` (VST3), `midir`, `windows` crate (shared memory, Media Foundation), `wasapi`, `winit`, `rustfft`, `hound` + `symphonia` (offline audio), `serde` + `ron`, `flate2` + `quick-xml` (.als), `bytemuck`.

**Spec:** `docs/superpowers/specs/2026-11-17-native-port-design.md`

## Global Constraints

- Windows 11 only; every Windows-specific call lives in one module per crate (`win.rs`).
- Reference GPU: the integrated AMD Radeon, on battery. Never select the discrete GPU by default.
- Plugin audio callback: < 20 us per 256-sample block, zero allocations, zero locks, no syscalls except the shared-memory write.
- App: < 1 core-equivalent average CPU, < 2 ms render-thread CPU per frame; GPU < 6 ms per frame at 1920x1080, 60 fps, 3 layers + overlay; < 1% GPU when the stage output is paused; audio-to-photon < 50 ms; < 500 MB with 4 videos.
- All musical time is in ppq (beats, f64). Bars come from `bar_start_ppq` and the time signature, never from `ppq / 4`.
- Flash limit: at least **400 ms** between full-frame flash starts (the web app's 12 frames at 30 fps), enforced in one place in `mosh-render`, with no opt-out.
- Hands-free: once Live plays, nothing needs input; no dialog ever appears on the stage output; every failure recovers by itself.
- Shared types change only through the coordinator (Appendix A is the contract).
- Tests: `cargo test -p <crate>` for every lane; `cargo clippy -p <crate> -- -D warnings` clean.
- Tuning audio stays outside the repo: `C:\Users\ilanf\moshloom-tracks` (15 tracks). Never commit audio.
- The web app (`index.src.html`, `gpu.js`, `engine.c`, `fixtures/**`) is read-only for native lanes.

## Review Focus

1. **Start order and restarts:** the app starts before Live, Live starts before the app, the plugin is removed mid-set, or the app restarts mid-song. Expected: the reader attaches at any time; with no writer, transport is `None` and visuals continue at the last tempo; nothing crashes or blocks. Test: `ipc::reader_attaches_after_writer_and_survives_writer_exit` (Task 1.1).
2. **Playhead discontinuities:** Live jumps back, loops a region, or is scrubbed. Expected: arrangement shows the clip at the new position on the next frame; no stale clip; intensity history is not polluted by the jump. Tests: `show::arrangement_follows_backward_jump` (Task 1.8), `analysis::ppq_jump_resets_bar_accumulator` (Task 1.3).
3. **Tempo automation and meter changes:** ramps, and 4/4 -> 3/4 mid-song. Expected: bar and phrase counters follow `bar_start_ppq`. Test: `analysis::bar_index_follows_time_signature_change` (Task 1.3).
4. **Quiet intros and gaps between songs** in one long arrangement. Expected: no jump to level 4 on the noise floor; history restarts at song markers and after > 2 bars of silence. Test: `intensity::silence_never_raises_level` and `intensity::history_resets_after_silence` (Task 1.4).
5. **Second monitor missing or unplugged mid-set.** Expected: the stage output moves to a borderless window on the laptop screen or hides, without a crash, and returns when the monitor comes back. Test: `render::surface_lost_recreates_swapchain` (Task 1.7) and the app test in Task 2.6.

---

## Phase 0: contracts (coordinator, serial, before any worker starts)

### Task 0.1: Workspace, `mosh-core`, crate stubs, lane table

**Files:**
- Create: `native/Cargo.toml` (workspace; members are every crate below; `resolver = "3"`)
- Create: `native/rust-toolchain.toml` (channel `1.99`, target `x86_64-pc-windows-msvc`)
- Create: `native/crates/mosh-core/src/{lib.rs,transport.rs,ipc_layout.rs,analysis.rs,show.rs,control.rs}` with exactly the types in Appendix A
- Create: one stub crate per lane: `mosh-ipc`, `mosh-tap`, `mosh-capture`, `mosh-analysis`, `mosh-show`, `mosh-shaders`, `mosh-render`, `mosh-video`, `mosh-midi`, `mosh-als`, `mosh-app` (each `lib.rs` or `main.rs` compiles and is empty)
- Modify: `AGENTS.md` (native lane table, Cargo.lock rule), `.gitignore` (`native/target/`)
- Test: `native/crates/mosh-core/tests/layout.rs`

- [ ] **Step 1: Write `layout.rs`**: `size_of::<IpcHeader>() == 128`, `size_of::<IpcBlock>() == 8320`, `size_of::<IpcMidiEvent>() == 16`, `align_of::<IpcBlock>() == 64`; `ShowRole::Master as u32 == 0`; `Project::default().schema == 1`; a RON round trip of `Project::default()` is equal.
- [ ] **Step 2: Run** `cargo test -p mosh-core` from `native/`: FAIL (types missing).
- [ ] **Step 3: Add Appendix A types**, with serde derives on show types and `#[repr(C)]` + explicit padding on ipc types.
- [ ] **Step 4: Run** `cargo test --workspace` and `cargo clippy --workspace -- -D warnings`: PASS.
- [ ] **Step 5: Install the Vulkan SDK** (`winget install KhronosGroup.VulkanSDK`) and record `vulkaninfo --summary` device names in the commit body.
- [ ] **Step 6: Commit and push** `native: workspace, mosh-core contracts, lane stubs (N0)`.

---

## Phase 1: lanes in parallel (one worker each)

Every Phase 1 task depends only on `mosh-core`. Where a task needs another lane's function, it is named in Interfaces and the worker codes against that signature; the coordinator merges in dependency order.

### Task 1.1: `mosh-ipc`, shared memory rings

**Files:** `native/crates/mosh-ipc/src/{lib.rs,ring.rs,win.rs}`, `tests/ipc.rs`

**Interfaces:**
- Consumes: `IpcHeader`, `IpcBlock`, `IpcMidiEvent`, `ShowRole`, `Transport`, `AudioBlock`, `MidiEvent` (Appendix A).
- Produces:
  - `IpcWriter::create(role: ShowRole, sample_rate: u32, channels: u16) -> Result<IpcWriter, IpcError>`
  - `IpcWriter::push_block(&mut self, t: Option<&Transport>, interleaved: &[f32], frames: u32)`: never allocates, never blocks; overwrites the oldest block and increments `dropped` when full.
  - `IpcWriter::push_midi(&mut self, e: &MidiEvent)`
  - `IpcReader::open(role: ShowRole) -> Result<IpcReader, IpcError>`
  - `IpcReader::poll(&mut self, out: &mut Vec<AudioBlock>, midi: &mut Vec<MidiEvent>) -> PollStats { new_blocks: u32, dropped: u64, writer_alive: bool }`
- Mapping name: `Local\moshloom.<role>.v1` (role in lowercase). Writer heartbeat: nanoseconds, written every block; `writer_alive` = heartbeat younger than 500 ms.

- [ ] Step 1: tests: `roundtrip_one_block` (samples and transport equal); `overwrite_counts_drops` (push 300 blocks into the 256-slot ring with no reads: `dropped == 44`, the reader gets the newest 256 in order); `torn_read_is_skipped` (a seqlock odd sequence is never returned); `reader_attaches_after_writer_and_survives_writer_exit` (reader opens late, writer dropped, `writer_alive` goes false within 600 ms, no panic); `push_block_does_not_allocate` (a counting global allocator shows 0 allocations over 10 000 pushes).
- [ ] Step 2: run `cargo test -p mosh-ipc`: FAIL.
- [ ] Step 3: implement. Each block slot uses a per-slot seqlock (`seq` odd while writing). Use `CreateFileMappingW`/`MapViewOfFile` in `win.rs`.
- [ ] Step 4: run the tests: PASS; plus `cargo bench`-free timing test `push_block_256_frames_under_5us` (median of 10 000).
- [ ] Step 5: commit, push branch `n-ipc`, open the PR.

### Task 1.2: `mosh-tap`, the two VST3 plugins

**Files:** `native/crates/mosh-tap/src/{lib.rs,tap.rs,control.rs}`, `bundler.toml`, `tests/process.rs`

**Interfaces:**
- Consumes: `IpcWriter` (Task 1.1), `Transport`, `MidiEvent`.
- Produces: `Moshloom Tap.vst3` (audio effect, stereo in/out, audio passes through bit-exact; parameter `role` = master/drums/bass/vocal/other) and `Moshloom Control.vst3` (instrument, MIDI in, silent stereo out, writes notes/CCs to role `control`). Transport comes from nih-plug's `ProcessContext::transport()`: `tempo`, `pos_beats`, `bar_start_pos_beats`, `time_sig_numerator/denominator`, `playing`, `pos_samples`.

- [ ] Step 1: tests with `nih_plug` test-free unit functions: `tap_passes_audio_bit_exact`; `tap_writes_transport_fields` (fake context -> the reader sees the same ppq, tempo, bar start, signature); `tap_works_without_reader` (no mapping consumer, process still returns `Normal`); `control_maps_note_on_and_cc` (note 36 vel 100 and CC 48 value 64 arrive as `MidiEvent` with ppq).
- [ ] Step 2: run: FAIL. Step 3: implement; the writer is created in `initialize()`, never in `process()`. Step 4: run: PASS; `cargo xtask bundle mosh-tap --release` produces both `.vst3` bundles.
- [ ] Step 5: commit, push `n-tap`, PR. (Loading in Live is a coordinator check.)

### Task 1.3: `mosh-analysis` features and beat-grid aggregation

**Files:** `native/crates/mosh-analysis/src/{lib.rs,features.rs,grid.rs,loudness.rs}`, `tests/features.rs`, `tests/grid.rs`

**Interfaces:**
- Consumes: `AudioBlock`, `Transport`, `AnalysisFrame`, `BarStats`, `TransportSource`, `EventFlags`.
- Produces:
  - `Analyzer::new(cfg: AnalysisConfig) -> Analyzer` (`AnalysisConfig { fft_size: 2048, phrase_bars: 8, history_bars: 16 }`)
  - `Analyzer::push(&mut self, b: &AudioBlock)`
  - `Analyzer::take_frame(&mut self) -> AnalysisFrame` (latest values; `events` OR-ed since the previous take)
  - `Analyzer::drain_bars(&mut self) -> Vec<BarStats>` (bars completed since the previous call; consumed by Task 1.4)
- Features: mono downmix; hop = block size; 24 mel bands 30 Hz-16 kHz; low/mid/high = 20-150 / 150-2500 / 2500-16000 Hz; onset = positive spectral flux of log magnitudes; loudness = short-term (3 s) K-weighted level in dB; brightness = spectral centroid / 8000 Hz clamped to 0..1. Normalisation of `bands`, `loudness` and `brightness` to 0..1 uses a rolling window of the last `history_bars` bars (5th-95th percentile), never the future.

- [ ] Step 1: tests: `sine_1khz_lands_in_mid_band`; `kick_train_120bpm_onsets_on_beats` (synthetic 50 Hz bursts every 0.5 s: onset peaks within 1 hop of each beat); `beat_and_bar_phase_from_transport` (ppq 6.5 in 4/4 with bar start 4.0: `beat_phase == 0.5`, `bar_phase == 0.625`); `bar_index_follows_time_signature_change` (4/4 for 8 bars then 3/4: bar 9 lasts 3 beats); `ppq_jump_resets_bar_accumulator` (a jump back from ppq 64 to 16 emits no `BarStats` for the partial bar); `push_256_frames_under_200us`.
- [ ] Step 2: run: FAIL. Step 3: implement (`rustfft`, a preallocated plan and buffers). Step 4: run: PASS.
- [ ] Step 5: commit, push `n-features`, PR.

### Task 1.4: `mosh-analysis` intensity model and events

**Files:** `native/crates/mosh-analysis/src/intensity.rs`, `tests/intensity.rs` (same crate as 1.3; disjoint files; 1.3 owns `lib.rs` and re-exports `intensity`)

**Interfaces:**
- Consumes: `BarStats`, `EventFlags`.
- Produces: `IntensityModel::new(phrase_bars: u32, history_bars: u32)`; `IntensityModel::push_bar(&mut self, b: &BarStats) -> IntensityUpdate { intensity: f32, level: u8, events: EventFlags }`; `IntensityModel::reset(&mut self)` (called at song markers).
- Rules: `intensity` = weighted mean of normalised loudness 0.35, low 0.30, onset density 0.20, brightness 0.15, against the rolling history; `level` 1..=4 from `intensity` with thresholds 0.25/0.5/0.75 and hysteresis 0.08, changing only per bar; `BUILD_START` when intensity rises over 4 consecutive bars by > 0.15 total; `DROP` when the low band returns (> 0.6) on a bar after a `BUILD_START` or after >= 1 bar with low < 0.2; `BREAKDOWN` when low and onset density both fall below 0.25 for a full bar after level >= 3; `PHRASE` every `phrase_bars`; silence = loudness below -50 dB.

- [ ] Step 1: tests: `steady_loop_settles_to_one_level` (32 identical bars: level constant after bar 8); `build_then_drop_detected` (bars 1-8 low, 9-12 rising brightness/onsets with no low, bar 13 full: `BUILD_START` at 12 or earlier, `DROP` at 13); `breakdown_detected`; `silence_never_raises_level`; `history_resets_after_silence` (> 2 silent bars, then a quiet bar is not level 4); `level_changes_only_on_bars`.
- [ ] Step 2: run: FAIL. Step 3: implement. Step 4: PASS. Step 5: commit, push `n-intensity`, PR.

### Task 1.5: `mosh-analysis` offline harness and label proposals

**Files:** `native/crates/mosh-analysis/src/bin/mosh-analyze.rs`, `src/offline.rs`, `tests/offline.rs`

**Interfaces:**
- Consumes: `Analyzer`, `IntensityModel` (Tasks 1.3, 1.4), `hound`, `symphonia`.
- Produces: CLI `mosh-analyze <file> --bpm <f64> [--offset-s <f64>] [--labels <file.toml>] [--propose]`. It plays a file through `Analyzer` in 256-frame blocks with a synthetic `Transport` (constant tempo, 4/4). `--propose` writes `<name>.labels.toml` with `[[mark]] bar = N, kind = "build"|"drop"|"breakdown"|"section"`. `--labels` prints, per kind, the hits, misses and mean timing error in bars against the labels.
- Tempo hints: from the file name when present (`93`, `140`, `87`, `137`), else `--bpm`.

- [ ] Step 1: tests: `reads_wav_16bit_and_float32` (fixtures generated in the test; float covers the IEEE-float WAV in the tuning set); `reads_mp3`; `propose_writes_labels_file`; `score_counts_hits_within_one_bar`.
- [ ] Step 2: FAIL. Step 3: implement. Step 4: PASS; run `--propose` on all 15 tracks in `C:\Users\ilanf\moshloom-tracks` and write the label files there (not in the repo).
- [ ] Step 5: commit, push `n-harness`, PR with the per-track proposal counts in the body.

### Task 1.6: `mosh-capture`, WASAPI loopback fallback

**Files:** `native/crates/mosh-capture/src/{lib.rs,win.rs}`, `tests/capture.rs`

**Interfaces:**
- Produces: `Loopback::start(device: Option<&str>) -> Result<Loopback, CaptureError>` (default render endpoint when `None`); `Loopback::poll(&mut self, out: &mut Vec<AudioBlock>)` (blocks with `transport: None`, 256-frame blocks at the device rate); `list_render_devices() -> Vec<String>`.

- [ ] Step 1: tests: `lists_at_least_one_device`; `captures_silence_as_zeros_not_error` (no playback: blocks keep arriving with zeros for 1 s); `device_removed_returns_error_not_panic` (`#[ignore]`, manual).
- [ ] Step 2: FAIL. Step 3: implement with the `wasapi` crate, event-driven, on its own thread. Step 4: PASS. Step 5: commit, push `n-capture`, PR.

### Task 1.7: `mosh-render` core (Vulkan via ash)

**Files:** `native/crates/mosh-render/src/{lib.rs,device.rs,target.rs,pipeline.rs,compositor.rs,swapchain.rs,timing.rs,flash.rs}`, `tests/offscreen.rs`

**Interfaces:**
- Consumes: SPIR-V words (`&[u32]`) and uniform bytes (`&[u8]`). It has no dependency on `mosh-shaders`; the app wires them together.
- Produces:
  - `Gpu::new(pref: GpuPreference /* Integrated | Discrete | ByName(String) */) -> Result<Gpu, RenderError>`
  - `Gpu::adapter_name(&self) -> &str`
  - `LayerTarget::new(&Gpu, w: u32, h: u32) -> LayerTarget` (RGBA16F, ping-pong pair for feedback)
  - `Pipeline::fullscreen(&Gpu, frag_spirv: &[u32], uniform_size: usize, textures: u32) -> Result<Pipeline, RenderError>`
  - `FrameRecorder` with `draw(&Pipeline, target: &LayerTarget, uniforms: &[u8], inputs: &[&LayerTarget])`
  - `Compositor::composite(&mut self, layers: &[(&LayerTarget, BlendMode, f32)], out: &OutputImage)`
  - `Stage::new(&Gpu, window_handle, fullscreen_monitor: Option<MonitorId>) -> Stage`; `Stage::present(&mut self, recorder) -> PresentResult { surface_lost: bool }`
  - `Gpu::read_back(&LayerTarget) -> Vec<u8>` (tests and stats)
  - `GpuTimings { per_pass_ms: Vec<(String, f32)>, frame_ms: f32 }`
  - `FlashLimiter::allow(&mut self, now: Instant) -> bool` (400 ms)

- [ ] Step 1: tests (offscreen, no window): `picks_integrated_gpu_by_default` (name contains "Radeon(TM) Graphics" on the reference machine; `#[ignore]` elsewhere); `solid_color_shader_reads_back` (a fragment shader writing (0.25, 0.5, 0.75, 1) gives those bytes +-1); `feedback_pingpong_accumulates` (a shader that adds 0.1 to its input, 5 frames: 0.5 +-0.01); `composite_add_and_normal`; `flash_limiter_400ms`; `timestamps_reported`; `surface_lost_recreates_swapchain` (simulated `ERROR_OUT_OF_DATE_KHR` path is handled without panic).
- [ ] Step 2: FAIL. Step 3: implement (dynamic rendering, one command buffer per frame, 2 frames in flight, timeline semaphore; validation layers on in debug builds). Step 4: PASS with zero validation messages. Step 5: commit, push `n-render`, PR.

### Task 1.8: `mosh-show`, session, arrangement and intensity mapping

**Files:** `native/crates/mosh-show/src/{lib.rs,engine.rs,session.rs,arrangement.rs,procedural.rs,mapping.rs,io.rs}`, `tests/*.rs`

**Interfaces:**
- Consumes: `Project`, `Clip`, `SessionGrid`, `Arrangement`, `AnalysisFrame`, `ControlEvent`, `RenderPlan`, `LayerState` (Appendix A).
- Produces: `Show::new(p: Project) -> Show`; `Show::update(&mut self, f: &AnalysisFrame, input: &[ControlEvent]) -> RenderPlan`; `Show::mode(&self) -> ShowMode`; `io::load(path) -> Result<Project, ShowError>`; `io::save(&Project, path)`.
- Rules: launches wait for the quantize boundary from the frame's ppq; follow actions fire on `follow_after_bars` or, when `react_to_events`, on `LEVEL_UP`/`DROP` at the next downbeat; a `BUILD_START` holds follow actions until `DROP` or 8 bars; `RandomProcedural` draws each `p[i]` uniformly from the clip's range for the current level with a `u64` seed (from `rand_pcg::Pcg64`) recorded in `LayerState.seed`; rhythm modulation: `p[i] = lerp(range.lo, range.hi, base + pulse_depth[level] * beat_pulse)`; `Lock` freezes automation; manual launches always win; arrangement picks the clip whose `[start, end)` contains the ppq every frame (no continuity assumption) unless the layer is overridden until `BackToArrangement`.

- [ ] Step 1: tests: `launch_waits_for_next_bar`; `scene_launch_sets_all_layers`; `follow_random_never_repeats_same_clip_twice` (with >= 2 candidates); `procedural_is_reproducible_from_seed`; `build_holds_follow_until_drop`; `lock_blocks_automation_not_manual`; `arrangement_follows_backward_jump`; `arrangement_override_then_back`; `level_selects_param_range`; `autopilot_launches_on_drop_without_input` (`ShowMode::Autopilot`, no `ControlEvent`s, a `DROP` frame: a new clip starts on the next downbeat); `project_roundtrip_ron`.
- [ ] Step 2: FAIL. Step 3: implement. Step 4: PASS. Step 5: commit, push `n-show`, PR.

### Task 1.9: `mosh-als`, Live set import

**Files:** `native/crates/mosh-als/src/lib.rs`, `tests/als.rs`, `tests/data/minimal.als` (generated by the test from an XML string, gzip)

**Interfaces:** Produces `read_als(path) -> Result<AlsInfo, AlsError>`, `AlsInfo { locators: Vec<SongMarker>, time_signatures: Vec<SigChange>, tempo_points: Vec<(f64 /*ppq*/, f64 /*bpm*/)> }`.

- [ ] Step 1: tests: `reads_locators_in_beats`; `reads_time_signature_changes`; `reads_tempo_automation_points`; `not_gzip_returns_error`. Step 2: FAIL. Step 3: implement (`flate2` + `quick-xml`, streaming). Step 4: PASS. Step 5: commit, push `n-als`, PR.

### Task 1.10: `mosh-midi`, APC mini and MIDI learn

**Files:** `native/crates/mosh-midi/src/{lib.rs,apc_mini.rs,learn.rs}`, `tests/apc.rs`

**Interfaces:**
- Produces: `Controller::open_apc_mini() -> Result<Controller, MidiError>` (finds ports whose names contain "APC MINI"); `Controller::poll(&mut self) -> Vec<ControlEvent>`; `Controller::set_pad(&mut self, x: u8, y: u8, c: PadColor)`; `decode_apc_mini(msg: &[u8]) -> Option<ControlEvent>` (pure, tested); `MidiLearn` mapping `(status, data1) -> ControlEvent` template, serialised in the project.
- APC mini (mk1) map: pads are notes 0-63 (row 0 = bottom, note = row*8 + col) -> `LaunchClip { layer: col, scene: 7 - row }`; scene buttons notes 82-89 -> `LaunchScene`; track buttons 64-71 -> `StopLayer`; faders CC 48-55 -> `SetOpacity { layer }`, CC 56 -> master intensity bias; LED velocities 0 off, 1 green, 3 red, 5 yellow (+1 = blink). Also accept the mk2 port name and its note map (pads 0-63 top-left origin, RGB velocities) behind `ApcModel`.

- Also produces `decode_control_note(e: &MidiEvent) -> Option<ControlEvent>` for the Moshloom Control plugin path: notes 0-63 -> `LaunchClip { layer: n % 8, scene: n / 8 }`, notes 64-71 -> `LaunchScene(n - 64)`, CC 1-8 -> `SetParam { layer: 0, index: cc - 1 }`, program change -> `SelectSong`.

- [ ] Step 1: tests: `pad_note_maps_to_launch`; `control_note_maps_to_launch`; `scene_button_maps_to_scene`; `fader_maps_to_opacity`; `led_message_bytes`; `mk2_pad_origin_is_top_left`. Step 2: FAIL. Step 3: implement with `midir`. Step 4: PASS. Step 5: commit, push `n-midi`, PR.

### Task 1.11: `mosh-video`, overlay decoding

**Files:** `native/crates/mosh-video/src/{lib.rs,win.rs}`, `tests/video.rs`

**Interfaces:** Produces `VideoSource::open(path) -> Result<VideoSource, VideoError>`; `VideoSource::info() -> VideoInfo { w, h, fps, duration_s }`; `VideoSource::frame_at(&mut self, t_s: f64) -> Option<&VideoFrame>` (`VideoFrame { w, h, bgra: Vec<u8>, pts_s }`, loops at the end, hardware decode via Media Foundation source reader with `MF_SOURCE_READER_ENABLE_HARDWARE_TRANSFORMS`).

- [ ] Step 1: tests use the repo's `fixtures/clip.mp4`: `opens_fixture_and_reports_size` (160x90); `frame_at_is_monotonic_and_loops`; `missing_file_is_error`. Step 2: FAIL. Step 3: implement. Step 4: PASS. Step 5: commit, push `n-video`, PR.

### Task 1.12: `mosh-shaders`, WGSL sources and the uniform ABI

**Files:** `native/crates/mosh-shaders/{build.rs,src/lib.rs,src/abi.rs,src/runtime.rs}`, `native/crates/mosh-shaders/wgsl/*.wgsl`, `scripts/extract_wgsl.mjs` (reads `gpu.js`, writes `wgsl/`; run by hand, output committed), `tests/compile.rs`

**Interfaces:**
- Consumes: `AnalysisFrame`, `ClipParams`.
- Produces: `fn spirv(name: &str) -> Option<&'static [u32]>` (build-time SPIR-V for every `wgsl/*.wgsl`); `fn compile_wgsl(src: &str) -> Result<Vec<u32>, ShaderError>` (runtime, for custom scenes; the error carries line and message); `#[repr(C)] struct LayerUniforms` (`bytemuck::Pod`, std140-safe), built by `fn layer_uniforms(f: &AnalysisFrame, p: &ClipParams, t: f32, res: [f32; 2]) -> LayerUniforms`; `fn web_prelude() -> &'static str` (declares the web app's uniform struct and bindings so web custom shaders compile unchanged on top of `LayerUniforms`); `VISUALIZERS: &[VisualizerInfo { id, name, params: [&str; 8] }]` for the 16 scenes plus `fluid`.

- [ ] Step 1: tests: `every_wgsl_file_compiles`; `web_template_compiles_unchanged` (the `WG_TEMPLATE` text from `gpu.js`); `broken_shader_reports_line` (`nope` -> line 3, like the web test); `uniforms_size_multiple_of_16`; `visualizer_list_has_17_entries`.
- [ ] Step 2: FAIL. Step 3: extract and implement (naga `wgsl-in` + `spv-out`). Step 4: PASS. Step 5: commit, push `n-shaders`, PR.

---

## Phase 2: visuals and the app (after the Phase 1 merges they depend on)

### Task 2.1: scene visualizers (needs 1.7, 1.12)
Port the 16 web scenes as `mosh-render` layer programs with `p0..p7` meanings listed in `VISUALIZERS`. Test: each scene renders a non-black frame (mean luminance > 0.02) at 3 levels and 3 seeds; per-scene GPU time on the reference GPU in the PR body.

### Task 2.2: mosh feedback and motion estimation per layer (needs 1.7, 1.12)
Port the block-matching motion pass and the feedback pass from `gpu.js`; mosh is an `EffectKind::Mosh` slot on a layer. Tests: a static scene stays static under mosh 0; a moving test pattern smears along its motion with mosh 0.8 (displacement sign check); GPU time at 1280x720 internal resolution in the PR body.

### Task 2.3: effects lane, transitions and the flash path (needs 1.7, 1.12)
Fold, Mirror, Streak, Crush, Zoom, Spin, Echo, Negative as effect passes; Cut, Crossfade, Datamosh, Melt, Strobe as per-layer transitions driven by `ActiveTransition.progress`; every full-frame flash goes through `FlashLimiter`. Tests: each effect at amount 0 is identity (read-back equal); Strobe at 16 flashes per second yields at most 3 flash starts per second.

### Task 2.4: fluid visualizer (needs 1.7, 1.12)
Stable-fluids on a 256x144 grid as its own visualizer: kicks (onset with low band > 0.5) inject force bursts, the three bands inject dye at three positions, beat phase sets swirl; its velocity field is exposed to a mosh layer below. Tests: dye mass rises after an onset frame; the field is stable (no NaN) over 10 000 frames of noise input; GPU time in the PR body.

### Task 2.5: tempo tracker for loopback mode (needs 1.3)
A causal comb-filter tempo and phase tracker on a 12 s rolling onset window, producing `Transport`-like phase with `TransportSource::Tracked`. Test: on synthetic 93, 120 and 140 BPM click tracks it locks within 8 s to +-0.5 BPM; on the 4 tuning tracks with tempo in the file name it reports that tempo (or double/half) within 1 BPM.

### Task 2.6: `mosh-app`, wiring, two windows, stats, recovery (needs everything)
Threads: ipc/capture reader -> analysis (audio-rate) -> show + render (vsync); control window on the laptop screen (redraws on change, <= 30 fps); stage window fullscreen on the secondary monitor, borderless fallback on the laptop screen when the monitor is missing; stats overlay with the budget table; app restarts mid-song pick up from the current ppq; overlay layers upload `VideoFrame`s into their layer texture; crossing a song marker calls `IntensityModel::reset` and sets `SONG_START`. Tests: `starts_with_no_plugin_and_no_monitor`; `monitor_unplug_moves_stage_window` (simulated); the budget check in the stats module fails loudly when a budget is exceeded.

### Task 2.7: control window and editors (needs 2.6)
Session grid, arrangement timeline (lanes per layer, song markers, `.als` import button), clip editor (visualizer, params, ranges per level, effects, transition, follow action), preset save from a seed. Tests are UI-model tests (no pixels): editing a clip updates the project; importing an `.als` adds markers.

### Task 2.8: overlay video layers (needs 1.7, 1.11)
Upload `mosh-video` BGRA frames into a GPU texture each frame (one staging buffer per layer, reused) and draw them into a layer with aspect-correct cover scaling, through `mosh_visuals::overlay::OverlayLayer`. Tests: a 160x90 checkerboard uploaded and drawn into a 64x36 target reads back with the expected cover crop; 600 uploads allocate no new GPU memory after the first; upload time per 1080p frame in the PR body.

**Phase 2 contract:** the coordinator froze the public signatures of `native/crates/mosh-visuals` (the facade `Visuals::draw_layer` and one stub module per lane) before Phase 2 started; see that crate's `lib.rs` for the chain and the owners.

## Phase 3: coordinator integration on the reference machine
- Load both plugins in Live 12 with the long arrangement; measure plugin callback time, app CPU and GPU on battery, latency; tune the intensity rules against the corrected labels of the 15 tracks; record every number in `native/README.md`.

---

## Appendix A: `mosh-core` contract (frozen in Phase 0)

```rust
// transport.rs
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Transport { pub tempo_bpm: f64, pub ppq: f64, pub bar_start_ppq: f64,
  pub sig_num: u16, pub sig_den: u16, pub playing: bool, pub sample_pos: i64 }
pub struct AudioBlock { pub transport: Option<Transport>, pub sample_rate: u32, pub channels: u16,
  pub samples: Vec<f32> /* interleaved */ }
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MidiEvent { pub ppq: f64, pub status: u8, pub data1: u8, pub data2: u8 }

// ipc_layout.rs  (all #[repr(C)], sizes asserted in tests)
pub const IPC_MAGIC: u32 = 0x4D4F_5348; pub const IPC_VERSION: u32 = 1;
pub const RING_BLOCKS: usize = 256; pub const BLOCK_MAX_FRAMES: usize = 1024; pub const MIDI_RING: usize = 1024;
#[repr(u32)] pub enum ShowRole { Master = 0, Drums = 1, Bass = 2, Vocal = 3, Other = 4, Control = 16 }
#[repr(C)] pub struct TransportRaw { tempo_bpm: f64, ppq: f64, bar_start_ppq: f64, sample_pos: i64,
  sig_num: u16, sig_den: u16, playing: u8, valid: u8, _pad: [u8; 26] }          // 64 bytes
#[repr(C)] pub struct IpcHeader { magic: u32, version: u32, role: u32, sample_rate: u32, channels: u32,
  block_max_frames: u32, ring_blocks: u32, midi_ring: u32, write_index: AtomicU64, midi_write_index: AtomicU64,
  dropped: AtomicU64, heartbeat_ns: AtomicU64, _pad: [u8; 64] }                  // 128 bytes
#[repr(C, align(64))] pub struct IpcBlock { seq: AtomicU64, frames: u32, channels: u32, _pad: [u8; 48],
  transport: TransportRaw, samples: [f32; 2048] }                               // 8320 bytes
#[repr(C)] pub struct IpcMidiEvent { ppq: f64, status: u8, data1: u8, data2: u8, _pad: [u8; 5] } // 16 bytes
// region = IpcHeader + [IpcBlock; RING_BLOCKS] + [IpcMidiEvent; MIDI_RING]

// analysis.rs
pub enum TransportSource { Plugin, Tracked, None }
bitflags! { pub struct EventFlags: u32 { const BEAT = 1; const BAR = 2; const PHRASE = 4;
  const BUILD_START = 8; const DROP = 16; const BREAKDOWN = 32; const LEVEL_UP = 64; const LEVEL_DOWN = 128;
  const SONG_START = 256; } }
pub struct AnalysisFrame { pub time_s: f64, pub ppq: f64, pub beat_phase: f32, pub bar_phase: f32,
  pub beat_index: i64, pub bar_index: i64, pub tempo_bpm: f32, pub playing: bool, pub source: TransportSource,
  pub bands: [f32; 3], pub mel: [f32; 24], pub onset: f32, pub loudness_db: f32, pub loudness: f32,
  pub brightness: f32, pub intensity: f32, pub level: u8, pub events: EventFlags }
pub struct BarStats { pub bar_index: i64, pub start_ppq: f64, pub beats: u16, pub loudness_db: f32,
  pub loudness: f32, pub low: f32, pub onset_density: f32, pub brightness: f32 }

// show.rs  (serde Serialize/Deserialize on all)
pub type ClipId = u32; pub type VisualizerId = String;          // "tunnel", "fluid", "custom:<name>"
pub struct ClipParams { pub p: [f32; 8], pub palette: u16, pub seed: u64 }
pub struct ParamRange { pub lo: f32, pub hi: f32 }
pub enum EffectKind { Fold, Mirror, Streak, Crush, Zoom, Spin, Echo, Negative, Mosh }
pub struct EffectSlot { pub kind: EffectKind, pub amount: f32, pub param: f32 }
pub enum TransitionKind { Cut, Crossfade, Datamosh, Melt, Strobe }
pub struct Transition { pub kind: TransitionKind, pub beats: f32 }
pub enum LaunchQuantize { None, Beat, Bar, Phrase }
pub enum FollowAction { None, Next, Random, RandomProcedural }
pub struct Clip { pub id: ClipId, pub name: String, pub visualizer: VisualizerId, pub params: ClipParams,
  pub ranges: [[ParamRange; 8]; 4], pub pulse_depth: [f32; 4], pub effects: Vec<EffectSlot>,
  pub transition_in: Transition, pub follow: FollowAction, pub follow_after_bars: u16,
  pub react_to_events: bool, pub video: Option<String> }
pub enum BlendMode { Normal, Add, Screen, Multiply, LumaKey }
pub enum LayerKind { Visualizer, Overlay }
pub struct Layer { pub kind: LayerKind, pub blend: BlendMode, pub opacity: f32, pub internal_scale: f32 }
pub struct SceneRow { pub name: String, pub cells: Vec<Option<ClipId>> }
pub struct SessionGrid { pub layers: Vec<Layer>, pub scenes: Vec<SceneRow>, pub quantize: LaunchQuantize }
pub struct SongMarker { pub name: String, pub start_ppq: f64 }
pub struct SigChange { pub ppq: f64, pub num: u16, pub den: u16 }
pub struct ArrClip { pub clip: ClipId, pub start_ppq: f64, pub end_ppq: f64 }
pub struct Arrangement { pub songs: Vec<SongMarker>, pub lanes: Vec<Vec<ArrClip>>, pub sig_map: Vec<SigChange> }
pub struct Preset { pub name: String, pub visualizer: VisualizerId, pub params: ClipParams }
pub enum ShowMode { Arrangement, Session, Autopilot }
pub struct Project { pub schema: u32 /* 1 */, pub mode: ShowMode, pub clips: Vec<Clip>, pub presets: Vec<Preset>,
  pub session: SessionGrid, pub arrangement: Arrangement }
pub struct ActiveTransition { pub kind: TransitionKind, pub progress: f32, pub from: Option<ClipId> }
pub struct LayerState { pub layer: usize, pub clip: Option<ClipId>, pub visualizer: Option<VisualizerId>,
  pub params: ClipParams, pub effects: Vec<EffectSlot>, pub transition: Option<ActiveTransition>,
  pub blend: BlendMode, pub opacity: f32, pub seed: u64 }
pub struct RenderPlan { pub layers: Vec<LayerState>, pub flash_requested: bool }

// control.rs
pub enum ControlEvent { LaunchClip { layer: usize, scene: usize }, LaunchScene(usize), StopLayer(usize),
  Lock(bool), BackToArrangement, SetParam { layer: usize, index: usize, value: f32 },
  SetOpacity { layer: usize, value: f32 }, IntensityBias(f32), SelectSong(usize) }
```
