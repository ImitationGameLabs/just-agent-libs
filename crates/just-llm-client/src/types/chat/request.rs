use serde::{Deserialize, Serialize};

use super::{
    ChatToolCall, ResponseFormat, StopSequence, StreamOptions, ToolChoice, ToolDefinition,
};

/// Normalized chat completion request understood by LLM client backends.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub frequency_penalty: Option<f32>,
    pub max_tokens: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub response_format: Option<ResponseFormat>,
    pub stop: Option<StopSequence>,
    pub stream: Option<bool>,
    pub stream_options: Option<StreamOptions>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub tools: Option<Vec<ToolDefinition>>,
    pub tool_choice: Option<ToolChoice>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u8>,
}

impl ChatCompletionRequest {
    /// Creates a minimal request with provider-neutral defaults.
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            frequency_penalty: None,
            max_tokens: None,
            presence_penalty: None,
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
        }
    }

    /// Sets the configured tools for the request.
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Sets the tool-choice behavior for the request.
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Sets the sampling temperature for the request.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the maximum generated-token count for the request.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the response-format hint for the request.
    pub fn with_response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    /// Inserts a system message at the end of the leading system-message block.
    ///
    /// Repeated calls preserve the order they are invoked while keeping system messages grouped
    /// at the front of the request.
    pub fn with_system_prompt(self, content: impl Into<String>) -> Self {
        self.prepend_system_message(content)
    }

    /// Inserts a system message at the end of the leading system-message block.
    pub fn prepend_system_message(mut self, content: impl Into<String>) -> Self {
        let insert_index = self
            .messages
            .iter()
            .take_while(|message| message.role() == "system")
            .count();
        self.messages
            .insert(insert_index, ChatMessage::system(content));
        self
    }

    /// Prepends a message to the request.
    pub fn prepend_message(mut self, message: ChatMessage) -> Self {
        self.messages.insert(0, message);
        self
    }
}

/// One request-side chat message in normalized form.
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
    /// Roles stay as free-form strings so the client layer can pass through provider- or
    /// application-specific roles like `developer` without forcing every extension through a
    /// normalized enum first.
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// Assistant message that includes one or more tool calls.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ToolCallsMessage {
    /// Request-side roles remain strings for the same reason as [`TextMessage::role`]: the
    /// normalized layer keeps outbound role handling open-ended on purpose.
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub tool_calls: Vec<ChatToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// Tool result message sent back to the model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ToolResultMessage {
    /// This stays stringly-typed to preserve wire compatibility with backends that may add
    /// request-side role variants beyond the normalized helper constructors.
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
    pub fn assistant_tool_calls(tool_calls: Vec<ChatToolCall>) -> Self {
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
        tool_calls: Vec<ChatToolCall>,
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

    /// Returns the message role regardless of which normalized variant is stored.
    pub fn role(&self) -> &str {
        match self {
            Self::Message(message) => &message.role,
            Self::ToolCalls(message) => &message.role,
            Self::ToolResult(message) => &message.role,
        }
    }

    /// Returns the message text content when that variant carries it.
    pub fn content(&self) -> Option<&str> {
        match self {
            Self::Message(message) => Some(message.content.as_str()),
            Self::ToolCalls(message) => message.content.as_deref(),
            Self::ToolResult(message) => Some(message.content.as_str()),
        }
    }

    /// Returns the optional provider-level `name` field when present.
    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Message(message) => message.name.as_deref(),
            Self::ToolCalls(message) => message.name.as_deref(),
            Self::ToolResult(_) => None,
        }
    }

    /// Returns the tool calls carried by an assistant tool-call message.
    pub fn tool_calls(&self) -> Option<&[ChatToolCall]> {
        match self {
            Self::ToolCalls(message) => Some(message.tool_calls.as_slice()),
            Self::Message(_) | Self::ToolResult(_) => None,
        }
    }

    /// Returns the tool-call identifier for a tool result message.
    pub fn tool_call_id(&self) -> Option<&str> {
        match self {
            Self::ToolResult(message) => Some(message.tool_call_id.as_str()),
            Self::Message(_) | Self::ToolCalls(_) => None,
        }
    }

    /// Returns the reasoning content carried by an assistant message, if any.
    pub fn reasoning_content(&self) -> Option<&str> {
        match self {
            Self::Message(message) => message.reasoning_content.as_deref(),
            Self::ToolCalls(message) => message.reasoning_content.as_deref(),
            Self::ToolResult(_) => None,
        }
    }

    /// Creates an assistant tool-call message with reasoning content.
    pub fn assistant_tool_calls_with_reasoning(
        tool_calls: Vec<ChatToolCall>,
        reasoning_content: impl Into<String>,
    ) -> Self {
        Self::ToolCalls(ToolCallsMessage {
            role: "assistant".to_owned(),
            content: None,
            name: None,
            tool_calls,
            reasoning_content: Some(reasoning_content.into()),
        })
    }

    /// Creates an assistant message with reasoning content and no tool calls.
    pub fn assistant_with_reasoning(
        content: impl Into<String>,
        reasoning_content: impl Into<String>,
    ) -> Self {
        Self::Message(TextMessage {
            role: "assistant".to_owned(),
            content: content.into(),
            name: None,
            reasoning_content: Some(reasoning_content.into()),
        })
    }
}
