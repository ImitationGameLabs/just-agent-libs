//! Shared error types for the transport layer and provider clients.

use std::string::FromUtf8Error;
use thiserror::Error;

use reqwest::StatusCode;

/// Errors produced by the shared HTTP/SSE transport layer.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TransportError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),

    #[error("failed to build http client: {0}")]
    BuildClient(#[source] reqwest::Error),

    #[error("request failed: {0}")]
    Transport(#[source] reqwest::Error),

    #[error("api returned {status}")]
    HttpStatus { status: StatusCode, body: String },

    /// A streamed response chunk could not be deserialized.
    ///
    /// Produced only by the SSE event parser, one event at a time. When a
    /// `TransportError` is lifted into a [`ProviderError`] via `From`,
    /// this surfaces as `ProviderError::Transport(TransportError::Deserialize)` —
    /// **not** as `ProviderError::Deserialize`, which is reserved for full-body
    /// failures produced by `parse_json`. Consumers matching for deserialization
    /// failures across both paths must account for both variants.
    #[error("failed to deserialize response body: {source}")]
    Deserialize {
        #[source]
        source: serde_json::Error,
        body: String,
    },

    #[error("failed to decode streamed response as UTF-8: {0}")]
    Utf8(#[source] FromUtf8Error),

    #[error("invalid response: {0}")]
    InvalidResponse(String),
}

/// Generic error type for OpenAI-compatible API provider clients.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProviderError {
    /// Transport-layer error from the shared HTTP/SSE layer.
    ///
    /// This is also where streaming/chunk deserialization failures land: an SSE
    /// event that fails to parse originates as `TransportError::Deserialize` and
    /// is wrapped here, **not** as the [`Deserialize`](Self::Deserialize) variant
    /// below, which is reserved for full response-body failures from `parse_json`.
    #[error(transparent)]
    Transport(#[from] TransportError),

    /// The request shape was invalid for the selected client method.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// Failed to serialize the request body.
    #[error("serialization failed: {0}")]
    Serialize(#[from] serde_json::Error),

    /// Failed to deserialize a full response body.
    ///
    /// Produced only by `parse_json`, which deserializes the entire HTTP response
    /// body. Streaming/chunk deserialization failures are a separate concern: they
    /// originate as `TransportError::Deserialize` and surface on
    /// [`Transport`](Self::Transport) (via `From<TransportError>`), not here.
    #[error("failed to deserialize response body: {source}")]
    Deserialize {
        #[source]
        source: serde_json::Error,
        body: String,
    },
}
