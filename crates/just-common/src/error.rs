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
    #[error(transparent)]
    Transport(#[from] TransportError),

    /// The request shape was invalid for the selected client method.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// Failed to serialize the request body.
    #[error("serialization failed: {0}")]
    Serialize(#[from] serde_json::Error),

    /// Failed to deserialize the response body.
    #[error("failed to deserialize response body: {source}")]
    Deserialize {
        #[source]
        source: serde_json::Error,
        body: String,
    },
}
