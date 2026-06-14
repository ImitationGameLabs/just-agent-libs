use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Structured response-format request.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub kind: ResponseFormatType,
}

/// Response-format families shared across normalized backends.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormatType {
    Text,
    JsonObject,
}

/// Stop sequence configuration.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

/// Streaming-specific options.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

/// Tool definition passed to the model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub kind: ToolType,
    pub function: FunctionDefinition,
}

/// Callable function schema exposed to the model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Explicit named tool-choice request.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct NamedToolChoice {
    #[serde(rename = "type")]
    pub kind: ToolType,
    pub function: NamedToolChoiceFunction,
}

/// Named function used in a named tool-choice request.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct NamedToolChoiceFunction {
    pub name: String,
}

/// Tool choice configuration.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(ToolChoiceMode),
    Named(NamedToolChoice),
}

/// Common tool-choice modes.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceMode {
    None,
    Auto,
    Required,
}

/// Tool families currently normalized by the client layer.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    Function,
}

/// Tool call emitted in a non-streaming assistant response.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: ToolType,
    pub function: FunctionCall,
}

/// Incremental tool-call payload emitted during streaming.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionChunkToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<ToolType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

/// Function invocation payload.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Incremental function-call payload emitted during streaming.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// Common completion stop reasons normalized by the client layer.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    InsufficientSystemResource,
}

/// Token usage metadata.
///
/// Cache fields are optional because some providers cannot report them faithfully.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Usage {
    pub completion_tokens: u32,
    pub prompt_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_hit_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_miss_tokens: Option<u32>,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

/// Additional completion-token details when the provider exposes them.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CompletionTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
}

/// Logprob payload for completion tokens.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionLogprobs {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<TokenLogprob>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<Vec<TokenLogprob>>,
}

/// Logprob metadata for a single token.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogprob>,
}

/// Top alternative logprob for a token.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}
