use just_common::error::{PreparedRequestError, TransportError};
use thiserror::Error;

/// OpenAI-compatible provider errors.
#[derive(Debug, Error)]
pub enum Error {
    /// Transport-layer error from the shared HTTP/SSE layer.
    #[error(transparent)]
    Transport(#[from] TransportError),

    /// The request shape was invalid for the selected client method.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// Failed to serialize the request body.
    #[error("serialization failed: {0}")]
    Serialize(#[from] serde_json::Error),
}

impl From<PreparedRequestError> for Error {
    fn from(e: PreparedRequestError) -> Self {
        Error::InvalidRequest(e.to_string())
    }
}
