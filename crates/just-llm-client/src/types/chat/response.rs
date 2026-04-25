use serde::{Deserialize, Serialize};

use super::{
    ChatCompletionChunkToolCall, ChatCompletionLogprobs, ChatToolCall, FinishReason, Usage,
};

/// Normalized non-streaming chat completion response.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<ChatChoice>,
    pub created: i64,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    pub object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl ChatCompletionResponse {
    /// Returns the first response choice when present.
    pub fn first_choice(&self) -> Option<&ChatChoice> {
        self.choices.first()
    }

    /// Returns the first assistant message when present.
    pub fn first_message(&self) -> Option<&AssistantMessage> {
        self.first_choice().map(|choice| &choice.message)
    }

    /// Returns the first choice's text content when present.
    pub fn first_choice_content(&self) -> Option<&str> {
        self.first_message()
            .and_then(|message| message.content.as_deref())
    }

    /// Returns the first choice's reasoning content when present.
    pub fn first_choice_reasoning_content(&self) -> Option<&str> {
        self.first_message()
            .and_then(|message| message.reasoning_content.as_deref())
    }

    /// Returns the first choice's tool calls when present.
    pub fn first_choice_tool_calls(&self) -> Option<&[ChatToolCall]> {
        self.first_message()
            .and_then(|message| message.tool_calls.as_deref())
    }
}

/// One choice inside a non-streaming response.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatChoice {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    pub index: u32,
    pub message: AssistantMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatCompletionLogprobs>,
}

/// Assistant message returned in a non-streaming response.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AssistantMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatToolCall>>,
    pub role: AssistantRole,
}

/// Assistant role values currently normalized by the client layer.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AssistantRole {
    Assistant,
}

/// Normalized streaming chunk.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
    pub created: i64,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    pub object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// One choice inside a streaming chunk.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionChunkChoice {
    pub delta: DeltaMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatCompletionLogprobs>,
}

/// Incremental assistant delta payload.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DeltaMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<AssistantRole>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatCompletionChunkToolCall>>,
}
