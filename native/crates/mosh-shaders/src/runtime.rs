//! WGSL -> SPIR-V at runtime, for custom scenes.
//!
//! A custom scene is compiled on the render thread when the human edits it, never at startup and
//! never mid-frame: the caller keeps the last good module and swaps it in when a new one compiles
//! (the design's "never a black layer mid-set" rule). The error carries a line number in the text
//! the human typed, so the editor can point at it.

use std::fmt;

/// A shader that did not compile. `line`/`column` are 1-based and refer to the source that was
/// passed in; they are `None` when the error has no single place (a SPIR-V writer failure).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderError {
    pub line: Option<u32>,
    pub column: Option<u32>,
    pub message: String,
}

impl fmt::Display for ShaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.line, self.column) {
            (Some(line), Some(column)) => write!(f, "line {line}:{column}: {}", self.message),
            (Some(line), None) => write!(f, "line {line}: {}", self.message),
            _ => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for ShaderError {}

/// Compiles a complete WGSL module to SPIR-V words, with the same naga options the build step uses
/// for `wgsl/*.wgsl`.
pub fn compile_wgsl(src: &str) -> Result<Vec<u32>, ShaderError> {
    compile(src)
}

/// Compiles a custom scene body on top of [`web_prelude()`](crate::web_prelude), the way gpu.js
/// compiles `WG_HEAD + WG_VS + code`.
///
/// Lines in the error are rebased into `body` (the same `Math.max(1, n - prelude_lines)` gpu.js
/// applies), so `nope` on line 3 of the body reports line 3, not line 40.
pub fn compile_custom(body: &str) -> Result<Vec<u32>, ShaderError> {
    let prelude = crate::web_prelude();
    let src = format!("{prelude}{body}");
    compile(&src).map_err(|e| {
        let prelude_newlines = prelude.matches('\n').count() as u32;
        ShaderError { line: e.line.map(|l| l.saturating_sub(prelude_newlines).max(1)), ..e }
    })
}

fn compile(src: &str) -> Result<Vec<u32>, ShaderError> {
    let module = naga::front::wgsl::parse_str(src).map_err(|e| {
        let at = e.location(src);
        ShaderError {
            line: at.map(|l| l.line_number),
            column: at.map(|l| l.line_position),
            message: e.message().to_string(),
        }
    })?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    );
    let info = validator.validate(&module).map_err(|e| {
        let at = e.location(src);
        ShaderError {
            line: at.map(|l| l.line_number),
            column: at.map(|l| l.line_position),
            message: e.to_string(),
        }
    })?;
    naga::back::spv::write_vec(&module, &info, &options(), None).map_err(|e| ShaderError {
        line: None,
        column: None,
        message: format!("SPIR-V: {e}"),
    })
}

/// Keep in sync with `build.rs::options`: identical flags, so build-time and runtime SPIR-V agree
/// word for word (asserted in `tests/compile.rs`).
fn options() -> naga::back::spv::Options<'static> {
    naga::back::spv::Options {
        lang_version: (1, 3),
        flags: naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE
            | naga::back::spv::WriterFlags::LABEL_VARYINGS
            | naga::back::spv::WriterFlags::CLAMP_FRAG_DEPTH,
        // No remapping: the WGSL `@group`/`@binding` numbers are the descriptor set and binding
        // the renderer uses, which is why `bindings()` hands them straight to `mosh-render`.
        fake_missing_bindings: true,
        ..Default::default()
    }
}
