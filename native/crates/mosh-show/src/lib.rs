//! mosh-show: the show model and its logic.
//!
//! Layers, clips, the session grid, the arrangement, launch quantization, follow actions,
//! procedural looks and the mapping from intensity to parameters.
//! See `docs/superpowers/plans/2026-11-17-native-port.md`, Task 1.8, and the design's
//! "How intensity drives the picture" and "Session and arrangement" sections.
//!
//! Pure logic: no I/O except [`io`], no threads, deterministic given the seed passed to
//! [`Show::with_seed`]. The design leaves the details to the builder, so these are the choices
//! this crate fixes — each one is a decision a reviewer can object to in one place:
//!
//! - **`p[i]` is a position inside the level's range.** The clip's authored `params.p[i]` (or the
//!   position a procedural draw produced) is a 0..1 position, and the level's
//!   [`ParamRange`](mosh_core::ParamRange) turns it into a value:
//!   `p[i] = lerp(lo, hi, position + pulse_depth[level] * beat_pulse)`. A clip that wants an exact
//!   value sets `lo == hi`. Intensity therefore never changes *which* parameter, only the range it
//!   moves in, which is what the design asks for.
//! - **The level comes from the analysis.** `AnalysisFrame::level` owns the 1..=4 level, with the
//!   hysteresis that lane implements. `IntensityBias` only picks a level when the analysis has
//!   none (`level == 0`), and otherwise scales the beat pulse by up to ±50%, so a controller fader
//!   cannot fight the hysteresis or make the picture jitter.
//! - **Quantization.** Manual launches use `SessionGrid::quantize`. Follow actions timed by
//!   `follow_after_bars` use it too, because they are already on a bar line anyway. Launches driven
//!   by `LEVEL_UP`/`DROP` always wait for the **next downbeat**, because the rule for them is
//!   exact and a phrase-long wait after a drop would land the change in the wrong section.
//! - **Followers are clips, not scenes.** A follow action moves its column to another cell; a
//!   scene launch moves every layer that has a cell in that scene and leaves the others playing,
//!   the way Live's scene launch behaves.
//! - **Builds hold automation, never the human.** `BUILD_START` holds automated launches (follow
//!   actions, autopilot) until `DROP` or [`engine::BUILD_HOLD_BARS`] bars; manual launches apply
//!   during a build. `Lock` does the same to *all* automation, including arrangement changes, and
//!   leaves manual launches and manual parameter moves working.
//! - **The arrangement never assumes continuity.** Every frame each lane is asked which clip
//!   contains the playhead in `[start, end)`; a backward jump, a loop or a scrub shows the right
//!   clip on that frame with no history. A manual launch takes its layer over until
//!   `BackToArrangement`.
//! - **Autopilot is hands-free.** On `DROP` or `LEVEL_UP` it advances every layer to the next scene
//!   that has content, on the next downbeat, and with nothing playing it starts the first scene.

pub mod arrangement;
pub mod engine;
pub mod io;
pub mod mapping;
pub mod procedural;
pub mod session;

pub use engine::{BUILD_HOLD_BARS, Show};
pub use io::ShowError;
