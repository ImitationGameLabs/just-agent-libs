use just_common::error::TransportError;
use thiserror::Error;

/// OpenAI-compatible provider errors.
#[derive(Debug, Error)]
pub enum Error {
    /// Transport-layer error from the shared HTTP/SSE layer.
    #[error(transparent)]
    Transport(#[from] TransportError),

    /// The request shape was invalid for the selected client method.
    #[error("invalid request: {0}")]
    InvalidRequest(&'static str),
}
