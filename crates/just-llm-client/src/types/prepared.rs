//! Adapter-layer prepared request wrapping [`just_common::prepared::PreparedChatRequest`].
//!
//! This type adds backend identity to the common prepared request, enabling cross-backend
//! guard checks at the adapter layer.

use reqwest::header::HeaderMap;
use serde_json::Value;

use crate::error::LlmError;

/// Alias for the common (identity-free) prepared request type.
///
/// Used internally to keep the distinction between the two `PreparedChatRequest` types clear.
type CommonPrepared = just_common::prepared::PreparedChatRequest;

/// Adapter-layer prepared request carrying backend identity.
///
/// Wraps a [`just_common::prepared::PreparedChatRequest`] and adds a `backend_id` so the
/// adapter layer can reject requests prepared by one backend when sent through another.
#[derive(Clone, Debug)]
pub struct PreparedChatRequest {
    backend_id: String,
    inner: CommonPrepared,
}

impl PreparedChatRequest {
    /// Convenience constructor: validates JSON and wraps with backend identity.
    pub fn from_request_body(
        backend_id: impl Into<String>,
        request_body: Value,
    ) -> Result<Self, just_common::error::PreparedRequestError> {
        Ok(Self {
            backend_id: backend_id.into(),
            inner: CommonPrepared::from_request_body(request_body)?,
        })
    }

    /// Bridge constructor: wraps an already-validated common prepared request.
    ///
    /// Used by adapters after the provider SDK has prepared the request.
    pub(crate) fn from_common(backend_id: impl Into<String>, inner: CommonPrepared) -> Self {
        Self {
            backend_id: backend_id.into(),
            inner,
        }
    }

    /// Returns the backend identifier that prepared this request.
    pub fn backend_id(&self) -> &str {
        &self.backend_id
    }

    /// Returns a reference to the inner common prepared request.
    ///
    /// Used by adapters to pass the identity-free request to the provider SDK.
    pub fn inner(&self) -> &CommonPrepared {
        &self.inner
    }

    /// Verifies that this request is executed by the backend that prepared it.
    pub fn ensure_backend(&self, backend_id: &str) -> Result<(), LlmError> {
        if self.backend_id == backend_id {
            return Ok(());
        }

        Err(LlmError::invalid_request(format!(
            "prepared request for backend '{}' cannot be used by '{}'",
            self.backend_id, backend_id,
        )))
    }

    // -- Delegated accessors --

    /// Returns the serialized `model` value when present and valid.
    pub fn model(&self) -> Option<&str> {
        self.inner.model()
    }

    /// Returns the number of serialized messages carried by this prepared request.
    pub fn message_count(&self) -> usize {
        self.inner.message_count()
    }

    /// Returns whether the serialized payload enables streaming.
    pub fn has_stream(&self) -> bool {
        self.inner.has_stream()
    }

    /// Returns the canonical request body for execution by a backend.
    pub fn request_body(&self) -> &Value {
        self.inner.request_body()
    }

    /// Serializes the canonical request body as compact JSON text for diagnostics.
    pub fn request_body_text(&self) -> String {
        self.inner.request_body_text()
    }

    /// Returns the extra HTTP headers attached to this request.
    pub fn headers(&self) -> &HeaderMap {
        self.inner.headers()
    }

    /// Returns a new prepared request with the given extra HTTP headers, replacing any
    /// previously set extra headers.
    pub fn with_headers(self, headers: HeaderMap) -> Self {
        Self {
            backend_id: self.backend_id,
            inner: self.inner.with_headers(headers),
        }
    }
}
