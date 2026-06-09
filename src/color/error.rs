#[derive(Debug)]
pub enum ColorError {
    Internal(String),
}

impl ColorError {
    pub(crate) fn internal(message: impl Into<String>) -> ColorError {
        ColorError::Internal(message.into())
    }
}
