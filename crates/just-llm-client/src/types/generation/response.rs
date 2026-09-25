//! Protocol-neutral generation response.
#![allow(missing_docs)]

use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::{
    message::{AssistantMessage, Reasoning, ToolCall},
    shared::Usage,
};

/// Normalized non-streaming generation response.
///
/// A single response corresponds to one assistant message (no `choices[]` wrapper, which is a
/// chat-completions wire artifact rather than a semantic concept).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GenerationResponse {
    pub id: String,
    pub model: String,
    pub message: AssistantMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<super::shared::FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<ResponseMeta>,
}

/// Transport-level observations attached by the client after a successful call.
///
/// Both fields are best-effort: providers omit or alias these freely, and absence of either
/// does not make a response less usable.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ResponseMeta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency: Option<Duration>,
}

impl GenerationResponse {
    /// Returns the response's text content when present.
    pub fn text(&self) -> Option<&str> {
        self.message.content.as_deref()
    }

    /// Returns the model's tool invocations.
    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.message.tool_calls
    }

    /// Returns the model's reasoning when present.
    pub fn reasoning(&self) -> Option<&Reasoning> {
        self.message.reasoning.as_ref()
    }
}
