//! Semantic message types.
//!
//! These are protocol-neutral client-facing types: no wire shape is privileged. The provider
//! backends map them to and from their own wire formats (e.g. chat `{role, content}` messages,
//! Responses items, or Anthropic content blocks).
#![allow(missing_docs)]

use serde::{Deserialize, Serialize};

/// Message text content: a plain string, or an array of content parts (multimodal).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// A single multimodal content part carried by a user/system message.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    Image {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },
}

/// Where the pixels of an image content part come from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ImageSource {
    Url { url: String },
    Base64 { data: String, media_type: String },
    FileId { file_id: String },
}

/// Rendering detail hint for an image input.
///
/// Unrecognized values are preserved verbatim, so a deserialized message
/// round-trips without losing what the original sender specified.
#[derive(Clone, Debug, PartialEq)]
pub enum ImageDetail {
    Auto,
    Low,
    High,
    Original,
    Unknown(String),
}

impl Serialize for ImageDetail {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let value = match self {
            Self::Auto => "auto",
            Self::Low => "low",
            Self::High => "high",
            Self::Original => "original",
            Self::Unknown(value) => value.as_str(),
        };
        serializer.serialize_str(value)
    }
}

impl<'de> Deserialize<'de> for ImageDetail {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "auto" => Self::Auto,
            "low" => Self::Low,
            "high" => Self::High,
            "original" => Self::Original,
            _ => Self::Unknown(value.clone()),
        })
    }
}

/// Assistant message emitted by the model.
///
/// Deserialization treats an absent `tool_calls` list as empty, so a plain text reply
/// round-trips through its own serialized form.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AssistantMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
}

/// Model-initiated tool invocation.
///
/// Flattened across protocols: chat `tool_calls[]` entries, Responses `function_call` items, and
/// Anthropic `tool_use` blocks all reduce to `(id, name, arguments)`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// Reasoning emitted alongside an assistant message.
///
/// Carries both the readable text and protocol-specific fidelity carriers needed for lossless
/// round-tripping: chat `reasoning_content` fills `text`; Responses `reasoning` items fill
/// `id`/`encrypted` (plus `text` from the summary); Anthropic `thinking` blocks fill `text` +
/// `signature`, and `redacted_thinking` blocks fill `redacted`.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Reasoning {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encrypted: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redacted: Option<String>,
}

/// One message in the conversation, discriminated by role.
///
/// System messages must occupy the leading block of the message list; backends that carry the
/// system prompt as a top-level parameter (Responses `instructions`, Anthropic `system`) extract
/// them there.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    System {
        content: MessageContent,
    },
    User {
        content: MessageContent,
    },
    Assistant(AssistantMessage),
    Tool {
        tool_call_id: String,
        content: String,
    },
}

impl Message {
    /// Creates a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: MessageContent::Text(content.into()),
        }
    }

    /// Creates a user message with plain-text content.
    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: MessageContent::Text(content.into()),
        }
    }

    /// Creates a user message with multimodal content parts.
    pub fn user_parts(parts: Vec<ContentPart>) -> Self {
        Self::User {
            content: MessageContent::Parts(parts),
        }
    }

    /// Creates an assistant message with plain-text content.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::Assistant(AssistantMessage {
            content: Some(content.into()),
            tool_calls: Vec::new(),
            reasoning: None,
        })
    }

    /// Creates an assistant message carrying tool calls (and optional content/reasoning).
    pub fn assistant_tool_calls(
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
        reasoning: Option<Reasoning>,
    ) -> Self {
        Self::Assistant(AssistantMessage {
            content,
            tool_calls,
            reasoning,
        })
    }

    /// Creates a tool result message fed back to the model.
    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self::Tool {
            content: content.into(),
            tool_call_id: tool_call_id.into(),
        }
    }

    /// Returns the role regardless of variant.
    pub fn role(&self) -> &str {
        match self {
            Self::System { .. } => "system",
            Self::User { .. } => "user",
            Self::Assistant(_) => "assistant",
            Self::Tool { .. } => "tool",
        }
    }

    /// Returns the text content when the variant carries one.
    ///
    /// Multimodal `Parts` content yields `None` here; use [`Self::content_parts`] for it.
    pub fn content(&self) -> Option<&str> {
        match self {
            Self::System {
                content: MessageContent::Text(text),
            } => Some(text),
            Self::User {
                content: MessageContent::Text(text),
            } => Some(text),
            Self::Assistant(message) => message.content.as_deref(),
            Self::Tool { content, .. } => Some(content),
            Self::System {
                content: MessageContent::Parts(_),
            }
            | Self::User {
                content: MessageContent::Parts(_),
            } => None,
        }
    }

    /// Returns the multimodal content parts when present.
    pub fn content_parts(&self) -> Option<&[ContentPart]> {
        match self {
            Self::System {
                content: MessageContent::Parts(parts),
            }
            | Self::User {
                content: MessageContent::Parts(parts),
            } => Some(parts),
            _ => None,
        }
    }

    /// Returns the tool calls carried by an assistant message.
    pub fn tool_calls(&self) -> &[ToolCall] {
        match self {
            Self::Assistant(message) => &message.tool_calls,
            _ => &[],
        }
    }

    /// Returns the reasoning carried by an assistant message, if any.
    pub fn reasoning(&self) -> Option<&Reasoning> {
        match self {
            Self::Assistant(message) => message.reasoning.as_ref(),
            _ => None,
        }
    }

    /// Returns the tool-call identifier for a tool result message.
    pub fn tool_call_id(&self) -> Option<&str> {
        match self {
            Self::Tool { tool_call_id, .. } => Some(tool_call_id),
            _ => None,
        }
    }
}
