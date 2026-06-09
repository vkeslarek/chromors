#[derive(Debug)]
pub enum Error {
    Vips(String),
    Gpu(String),
    Raw(String),
    Render(String),
    NullPtr,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Vips(msg) => write!(f, "vips: {msg}"),
            Error::Gpu(msg) => write!(f, "gpu: {msg}"),
            Error::Raw(msg) => write!(f, "raw: {msg}"),
            Error::Render(msg) => write!(f, "render: {msg}"),
            Error::NullPtr => write!(f, "null pointer"),
        }
    }
}

impl std::error::Error for Error {}
