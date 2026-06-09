//! Prepared chat request types.
//!
//! [`PreparedChatRequest`] is a backend-agnostic prepared request: it holds a serialized JSON body
//! and optional extra HTTP headers. It has no identity concept — backend binding is the
//! responsibility of the adapter layer.

use reqwest::header::HeaderMap;
use serde_json::Value;

use crate::error::PreparedRequestError;

/// Backend-agnostic prepared chat request.
///
/// The canonical source of truth is the serialized request body.
///
/// Extra HTTP headers can be attached via [`with_headers`](Self::with_headers) and will be
/// merged on top of the backend client's default headers when the request is sent.
#[derive(Clone, Debug)]
pub struct PreparedChatRequest {
    request_body: Value,
    extra_headers: HeaderMap,
}

impl PreparedChatRequest {
    // Keeping the canonical payload as JSON preserves provider-specific request shape without
    // forcing the client layer to understand every backend extension.

    /// Creates a prepared request from a pre-serialized JSON body.
    ///
    /// Validates that the body is a JSON object with a `model` string field and a `messages`
    /// array field.
    pub fn from_request_body(request_body: Value) -> Result<Self, PreparedRequestError> {
        let request = Self {
            request_body,
            extra_headers: HeaderMap::new(),
        };
        request.validate()?;
        Ok(request)
    }

    /// Returns the serialized `model` value when present and valid.
    pub fn model(&self) -> Option<&str> {
        self.request_body.get("model").and_then(Value::as_str)
    }

    /// Returns the number of serialized messages carried by this prepared request.
    pub fn message_count(&self) -> usize {
        self.messages_value().map_or(0, Vec::len)
    }

    /// Returns whether the serialized payload enables streaming.
    pub fn has_stream(&self) -> bool {
        self.request_body
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    /// Returns the canonical request body for execution by a backend.
    pub fn request_body(&self) -> &Value {
        &self.request_body
    }

    /// Serializes the canonical request body as compact JSON text for diagnostics.
    ///
    /// This is intentionally a text snapshot rather than a structured `serde_json::Value`
    /// accessor. The prepared request stays opaque by default so callers do not accidentally
    /// couple themselves to the internal canonical representation.
    pub fn request_body_text(&self) -> String {
        serde_json::to_string(&self.request_body)
            .expect("serializing serde_json::Value is infallible")
    }

    /// Returns the extra HTTP headers attached to this request.
    pub fn headers(&self) -> &HeaderMap {
        &self.extra_headers
    }

    /// Returns a new prepared request with the given extra HTTP headers, replacing any
    /// previously set extra headers.
    ///
    /// These headers are merged on top of the backend client's default headers (which include
    /// `Authorization` and `Accept`) when the request is sent. If a header name matches one of
    /// the defaults (e.g. `Authorization`), the value provided here takes precedence. This is
    /// intentional to support per-request credentials in multi-tenant setups.
    pub fn with_headers(self, headers: HeaderMap) -> Self {
        Self {
            extra_headers: headers,
            ..self
        }
    }

    fn validate(&self) -> Result<(), PreparedRequestError> {
        if !self.request_body.is_object() {
            return Err(PreparedRequestError::NotJsonObject);
        }

        if self.model().is_none() {
            return Err(PreparedRequestError::MissingModelField);
        }

        if self.messages_value().is_none() {
            return Err(PreparedRequestError::MissingMessagesArray);
        }

        Ok(())
    }

    fn messages_value(&self) -> Option<&Vec<Value>> {
        self.request_body.get("messages").and_then(Value::as_array)
    }
}
