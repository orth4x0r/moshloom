//! Project files: RON, with the schema version checked on load.

use std::fs;
use std::path::Path;

use mosh_core::{Project, PROJECT_SCHEMA};

/// Everything loading or saving a project can fail with.
#[derive(Debug)]
pub enum ShowError {
    Io(std::io::Error),
    Ron(ron::Error),
    /// The file was written by a schema this build does not understand.
    Schema { found: u32, expected: u32 },
}

impl std::fmt::Display for ShowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShowError::Io(e) => write!(f, "project file: {e}"),
            ShowError::Ron(e) => write!(f, "project file is not valid RON: {e}"),
            ShowError::Schema { found, expected } => {
                write!(f, "project schema {found}, this build reads schema {expected}")
            }
        }
    }
}

impl std::error::Error for ShowError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ShowError::Io(e) => Some(e),
            ShowError::Ron(e) => Some(e),
            ShowError::Schema { .. } => None,
        }
    }
}

impl From<std::io::Error> for ShowError {
    fn from(e: std::io::Error) -> Self {
        ShowError::Io(e)
    }
}

impl From<ron::Error> for ShowError {
    fn from(e: ron::Error) -> Self {
        ShowError::Ron(e)
    }
}

impl From<ron::error::SpannedError> for ShowError {
    fn from(e: ron::error::SpannedError) -> Self {
        ShowError::Ron(e.into())
    }
}

/// Read a project from `path`, refusing a schema this build does not understand.
///
/// A project file that fails to parse is an error, never a default: a half-read show is worse
/// than none on stage.
pub fn load(path: impl AsRef<Path>) -> Result<Project, ShowError> {
    let text = fs::read_to_string(path)?;
    let project: Project = ron::from_str(&text)?;
    if project.schema != PROJECT_SCHEMA {
        return Err(ShowError::Schema { found: project.schema, expected: PROJECT_SCHEMA });
    }
    Ok(project)
}

/// Write `project` to `path` as pretty RON, so a diff of two looks is readable.
pub fn save(project: &Project, path: impl AsRef<Path>) -> Result<(), ShowError> {
    let text = ron::ser::to_string_pretty(project, ron::ser::PrettyConfig::default())?;
    fs::write(path, text)?;
    Ok(())
}
