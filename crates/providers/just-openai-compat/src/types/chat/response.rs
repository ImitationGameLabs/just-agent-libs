use serde::{Deserialize, Serialize};

use super::{
    ChatCompletionChunkToolCall, ChatCompletionLogprobs, ChatCompletionToolCall, FinishReason,
    Usage,
};

/// Wire DTO for a non-streaming chat completion response.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletion {
    pub id: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub created: i64,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    pub object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionChoice {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    pub index: u32,
    pub message: AssistantMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatCompletionLogprobs>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AssistantMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatCompletionToolCall>>,
    pub role: AssistantRole,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AssistantRole {
    Assistant,
}

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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionChunkChoice {
    pub delta: DeltaMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatCompletionLogprobs>,
}

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
