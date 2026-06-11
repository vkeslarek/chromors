#[derive(Debug, Clone)]
pub enum Error {
    Backend(String),
    TypeMismatch(String),
    InvalidWorkUnit(String),
    Io(String),
    Vips(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Backend(msg) => write!(f, "backend error: {msg}"),
            Error::TypeMismatch(msg) => write!(f, "type mismatch: {msg}"),
            Error::InvalidWorkUnit(msg) => write!(f, "invalid work unit: {msg}"),
            Error::Io(msg) => write!(f, "io error: {msg}"),
            Error::Vips(msg) => write!(f, "vips error: {msg}"),
        }
    }
}

impl std::error::Error for Error {}
