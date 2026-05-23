use serde::{Deserialize, Serialize};

use super::{
    ResponseFormat, StopSequence, StreamOptions, ToolChoice, ToolDefinition,
    shared::ChatCompletionToolCall,
};

/// Wire DTO for `POST /chat/completions`.
pub type CreateChatCompletionRequest = ChatCompletionRequest;

/// Wire DTO for `POST /chat/completions`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopSequence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

impl ChatCompletionRequest {
    /// Creates a minimal request with common defaults.
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            thinking: None,
            max_tokens: None,
            response_format: None,
            stop: None,
            stream: None,
            stream_options: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            logprobs: None,
            top_logprobs: None,
            reasoning_effort: None,
            user_id: None,
        }
    }
}

/// One request-side chat message in wire form.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ChatMessage {
    ToolCalls(ToolCallsMessage),
    ToolResult(ToolResultMessage),
    Message(TextMessage),
}

/// Plain role/content message.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TextMessage {
    /// Request-side roles remain free-form strings so callers can use DeepSeek-compatible
    /// extensions such as `developer` without waiting for a curated enum surface.
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// Assistant message that contains tool calls.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ToolCallsMessage {
    /// Outbound roles intentionally stay stringly typed for compatibility with provider-specific
    /// role values layered on top of the shared chat format.
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub tool_calls: Vec<ChatCompletionToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// Tool result message sent back to the model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ToolResultMessage {
    /// Kept as a raw string for parity with the wire protocol and custom role extensions.
    pub role: String,
    pub content: String,
    pub tool_call_id: String,
}

impl ChatMessage {
    /// Creates a message with an explicit role string.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self::Message(TextMessage {
            role: role.into(),
            content: content.into(),
            name: None,
            reasoning_content: None,
        })
    }

    /// Creates a named message for providers that support the `name` field.
    pub fn named(
        role: impl Into<String>,
        content: impl Into<String>,
        name: impl Into<String>,
    ) -> Self {
        Self::Message(TextMessage {
            role: role.into(),
            content: content.into(),
            name: Some(name.into()),
            reasoning_content: None,
        })
    }

    /// Creates a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    /// Creates a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    /// Creates an assistant message without tool calls.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }

    /// Creates an assistant tool-call message without extra text content.
    pub fn assistant_tool_calls(tool_calls: Vec<ChatCompletionToolCall>) -> Self {
        Self::ToolCalls(ToolCallsMessage {
            role: "assistant".to_owned(),
            content: None,
            name: None,
            tool_calls,
            reasoning_content: None,
        })
    }

    /// Creates an assistant tool-call message with accompanying text content.
    pub fn assistant_tool_calls_with_content(
        content: impl Into<String>,
        tool_calls: Vec<ChatCompletionToolCall>,
    ) -> Self {
        Self::ToolCalls(ToolCallsMessage {
            role: "assistant".to_owned(),
            content: Some(content.into()),
            name: None,
            tool_calls,
            reasoning_content: None,
        })
    }

    /// Creates a tool result message.
    pub fn tool_result(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self::ToolResult(ToolResultMessage {
            role: "tool".to_owned(),
            content: content.into(),
            tool_call_id: tool_call_id.into(),
        })
    }

    /// Alias for [`Self::tool_result`].
    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self::tool_result(content, tool_call_id)
    }
}

/// DeepSeek reasoning controls.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub kind: ThinkingMode,
}

/// Whether reasoning is enabled for the request.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingMode {
    Enabled,
    Disabled,
}

/// Requested reasoning effort.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    High,
    Max,
}
