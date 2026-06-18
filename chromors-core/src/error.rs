/// Unified error type for all engine failures.
///
/// Each variant carries a human-readable description string. The enum is
/// `Clone` so errors can be propagated through the lazy DAG without losing
/// information.
#[derive(Debug, Clone)]
pub enum Error {
    /// Generic backend error (GPU compilation/dispatch failures, etc.).
    Backend(String),
    /// A Kind mismatch — e.g. an op expected `ImageKind` but got `HistogramKind`.
    TypeMismatch(String),
    /// The passed WorkUnit shape doesn't match the operation's expected shape.
    InvalidWorkUnit(String),
    /// I/O error reading a source or writing a target.
    Io(String),
    /// libvips operation error (parsed from `vips_error_buffer()`).
    Vips(String),
    /// LibRaw decode error.
    Raw(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Backend(msg) => write!(f, "backend error: {msg}"),
            Error::TypeMismatch(msg) => write!(f, "type mismatch: {msg}"),
            Error::InvalidWorkUnit(msg) => write!(f, "invalid work unit: {msg}"),
            Error::Io(msg) => write!(f, "io error: {msg}"),
            Error::Vips(msg) => write!(f, "vips error: {msg}"),
            Error::Raw(msg) => write!(f, "raw error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}
