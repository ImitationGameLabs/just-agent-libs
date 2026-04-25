use serde::{Deserialize, Serialize};

use crate::types::chat::{ChatMessage, ToolDefinition};

/// Request for backend-specific context metrics.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ContextMetricsRequest {
    /// Model identifier the request would target.
    pub model: String,
    /// Messages that contribute to prompt context.
    pub messages: Vec<ChatMessage>,
    /// Optional tool declarations that also contribute to context usage.
    pub tools: Option<Vec<ToolDefinition>>,
}

/// Best-effort context metadata returned by a backend.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextMetricsResult {
    /// Estimated or reported prompt token count.
    pub prompt_tokens: Option<u32>,
    /// Maximum supported context window when the backend can provide it.
    pub max_context_tokens: Option<u32>,
    /// Maximum supported output tokens when the backend can provide it.
    pub max_output_tokens: Option<u32>,
}
