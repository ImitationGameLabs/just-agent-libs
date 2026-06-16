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

    /// A non-success HTTP status, with the full response body captured as diagnostic text.
    ///
    /// `body` is read under the shared size cap (`MAX_BODY_BYTES`). Bodies that fit are captured
    /// in full; an oversized error body instead surfaces as [`BodyTooLarge`](Self::BodyTooLarge)
    /// and this variant is not produced. So when this variant is present, `body` is complete —
    /// never a truncated prefix.
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

    /// A non-streaming response body exceeded the shared size cap and was not fully buffered.
    ///
    /// Produced by the capped body reader (`read_body_text`). Distinct from
    /// [`InvalidResponse`](Self::InvalidResponse), which reports *content* problems (empty body,
    /// malformed structure) rather than a size limit. The reader stops before the offending chunk
    /// is appended, so no body text is carried here. When the overflow occurs while reading an
    /// *error* body, the HTTP status is not carried here either — it is visible only to callers
    /// that read status off the raw response before consuming the body (the `prepare`/`send` path).
    #[error("response body exceeded {limit}-byte limit")]
    BodyTooLarge { limit: usize },

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
    #[error("transport error: {0}")]
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as StdError;

    /// After dropping `#[error(transparent)]` from `ProviderError::Transport`, the wrapped
    /// `TransportError` must be reachable by walking `source()` and downcasting — so a consumer
    /// can recover, e.g. a 429 status from the error object. This contract would have failed
    /// before the change: `transparent` flattened `TransportError` out of the source chain.
    #[test]
    fn transport_error_reachable_via_source_chain() {
        let te = TransportError::HttpStatus {
            status: StatusCode::TOO_MANY_REQUESTS,
            body: "rate limited".into(),
        };
        let pe: ProviderError = te.into(); // ProviderError::Transport(te)

        let mut cur: Option<&(dyn StdError + 'static)> = Some(&pe);
        let mut found = None;
        while let Some(e) = cur {
            if let Some(t) = e.downcast_ref::<TransportError>() {
                found = Some(t);
                break;
            }
            cur = e.source();
        }

        let found = found.expect("TransportError must be reachable via the source chain");
        assert!(
            matches!(
                found,
                TransportError::HttpStatus { status, .. } if *status == StatusCode::TOO_MANY_REQUESTS
            ),
            "expected HttpStatus 429, got {found:?}"
        );
    }
}
