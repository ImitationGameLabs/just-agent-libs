use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::error::LlmError;

/// Lightweight summary of a prepared request.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreparedChatRequestPreview {
    /// Number of tools declared in the serialized payload.
    pub tool_count: u32,
    /// Whether the serialized payload enables streaming.
    pub has_stream: bool,
    /// Whether the serialized payload requests a structured response format.
    pub has_response_format: bool,
}

/// Backend-bound prepared chat request.
///
/// The canonical source of truth is the serialized request body. Preview helpers derive from
/// that same JSON payload so preview and execution cannot drift apart. Prepared requests stay as
/// pure data so they can be serialized, persisted, or sent across process boundaries without
/// carrying a live backend handle.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PreparedChatRequest {
    backend_id: String,
    request_body: Value,
}

#[derive(Deserialize)]
struct PreparedChatRequestRepr {
    backend_id: String,
    request_body: Value,
}

impl PreparedChatRequest {
    // Keeping the canonical payload as JSON preserves provider-specific request shape without
    // forcing the client layer to understand every backend extension.
    /// Creates a prepared request from a pre-serialized JSON body.
    pub fn from_request_body(
        backend_id: impl Into<String>,
        request_body: Value,
    ) -> Result<Self, LlmError> {
        let request = Self { backend_id: backend_id.into(), request_body };
        request.validate()?;
        Ok(request)
    }

    /// Returns the backend identifier that prepared this request.
    pub fn backend_id(&self) -> &str {
        &self.backend_id
    }

    /// Returns the serialized `model` value when present and valid.
    pub fn model(&self) -> Option<&str> {
        self.request_body.get("model").and_then(Value::as_str)
    }

    /// Returns the number of serialized messages carried by this prepared request.
    pub fn message_count(&self) -> usize {
        self.messages_value().map_or(0, Vec::len)
    }

    /// Returns a compact summary derived from the canonical payload.
    pub fn preview(&self) -> PreparedChatRequestPreview {
        PreparedChatRequestPreview {
            tool_count: self
                .request_body
                .get("tools")
                .and_then(Value::as_array)
                .map_or(0, |tools| tools.len().try_into().unwrap_or(u32::MAX)),
            has_stream: self.has_stream(),
            has_response_format: self.request_body.get("response_format").is_some(),
        }
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

    /// Verifies that this request is executed by the backend that prepared it.
    pub fn ensure_backend(&self, backend_id: &'static str) -> Result<(), LlmError> {
        if self.backend_id == backend_id {
            return Ok(());
        }

        Err(LlmError::invalid_request(format!(
            "prepared request for backend '{}' cannot be used by '{}'",
            self.backend_id, backend_id
        )))
    }

    fn validate(&self) -> Result<(), LlmError> {
        if !self.request_body.is_object() {
            return Err(LlmError::invalid_request(
                "prepared request body must be a JSON object",
            ));
        }

        if self.model().is_none() {
            return Err(LlmError::invalid_request(
                "prepared request body must include a string model field",
            ));
        }

        if self.messages_value().is_none() {
            return Err(LlmError::invalid_request(
                "prepared request body must include a messages array",
            ));
        }

        Ok(())
    }

    fn messages_value(&self) -> Option<&Vec<Value>> {
        self.request_body.get("messages").and_then(Value::as_array)
    }
}

impl<'de> Deserialize<'de> for PreparedChatRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = PreparedChatRequestRepr::deserialize(deserializer)?;
        Self::from_request_body(repr.backend_id, repr.request_body)
            .map_err(serde::de::Error::custom)
    }
}
