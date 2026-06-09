//! Shared error types for transport and prepared-request validation.

use std::string::FromUtf8Error;
use thiserror::Error;

use reqwest::StatusCode;

/// Errors produced by the shared HTTP/SSE transport layer.
#[derive(Debug, Error)]
pub enum TransportError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),

    #[error("failed to build http client: {0}")]
    BuildClient(#[source] reqwest::Error),

    #[error("request failed: {0}")]
    Transport(#[source] reqwest::Error),

    #[error("api returned {status}")]
    HttpStatus { status: StatusCode, body: String },

    #[error("failed to serialize request body: {0}")]
    Serialize(#[source] serde_json::Error),

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

/// Errors produced when constructing or validating a [`PreparedChatRequest`](crate::prepared::PreparedChatRequest).
#[derive(Debug, Error)]
pub enum PreparedRequestError {
    /// The request body is not a JSON object.
    #[error("prepared request body must be a JSON object")]
    NotJsonObject,

    /// The request body is missing a string `model` field.
    #[error("prepared request body must include a string model field")]
    MissingModelField,

    /// The request body is missing a `messages` array.
    #[error("prepared request body must include a messages array")]
    MissingMessagesArray,
}
