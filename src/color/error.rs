/// Error type for color science failures (singular matrices, invalid params, etc.).
#[derive(Debug)]
pub enum ColorError {
    /// An internal color processing error with a description message.
    Internal(String),
}

impl ColorError {
    /// Creates a new `ColorError::Internal` with the given message.
    pub(crate) fn internal(message: impl Into<String>) -> ColorError {
        ColorError::Internal(message.into())
    }
}
