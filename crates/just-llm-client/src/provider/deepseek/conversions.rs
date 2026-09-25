//! Explicit DeepSeek <-> LLM client conversions.
//!
//! These mappings keep the provider wire types independent from the `just-llm-client`
//! normalized layer, so future provider-specific evolution does not need to route through a
//! shared protocol abstraction. Request-side conversions are fallible: the semantic types carry
//! carry fields DeepSeek cannot express (`top_k`, penalties), which surface as an explicit
//! invalid-request error rather than being silently dropped. Multimodal content parts are
//! expressible and passed through for the provider to judge.
//!
use crate::{BackendError, types::generation as client_gen};
use just_deepseek::types::chat as provider_chat;

/// Converts semantic content into chat wire content (string or parts).
///
/// Base64 image sources become data URIs; file-id sources have no chat-completions
/// representation and surface as an explicit error.
fn content_to_wire(
    content: &client_gen::MessageContent,
) -> Result<provider_chat::MessageContent, BackendError> {
    match content {
        client_gen::MessageContent::Text(text) => {
            Ok(provider_chat::MessageContent::Text(text.clone()))
        }
        client_gen::MessageContent::Parts(parts) => {
            let mut wire_parts = Vec::new();
            for part in parts {
                match part {
                    client_gen::ContentPart::Text { text } => {
                        wire_parts.push(provider_chat::ContentPart::Text { text: text.clone() });
                    }
                    client_gen::ContentPart::Image { source, detail } => {
                        let url = match source {
                            client_gen::ImageSource::Url { url } => url.clone(),
                            client_gen::ImageSource::Base64 { data, media_type } => {
                                format!("data:{media_type};base64,{data}")
                            }
                            client_gen::ImageSource::FileId { .. } => {
                                return Err(BackendError::unserializable(
                                    crate::family::DEEPSEEK,
                                    "file-id image sources have no chat-completions representation",
                                ));
                            }
                        };
                        wire_parts.push(provider_chat::ContentPart::ImageUrl {
                            image_url: provider_chat::ImageUrlSource {
                                url,
                                detail: detail_to_wire(detail),
                            },
                        });
                    }
                }
            }
            Ok(provider_chat::MessageContent::Parts(wire_parts))
        }
    }
}

/// Maps a semantic detail hint to the chat wire's free-string detail field;
/// unknown values pass through verbatim.
fn detail_to_wire(detail: &Option<client_gen::ImageDetail>) -> Option<String> {
    detail.as_ref().map(|detail| match detail {
        client_gen::ImageDetail::Auto => "auto".to_owned(),
        client_gen::ImageDetail::Low => "low".to_owned(),
        client_gen::ImageDetail::High => "high".to_owned(),
        client_gen::ImageDetail::Original => "original".to_owned(),
        client_gen::ImageDetail::Unknown(value) => value.clone(),
    })
}

impl TryFrom<client_gen::Message> for provider_chat::ChatMessage {
    type Error = BackendError;

    fn try_from(message: client_gen::Message) -> Result<Self, Self::Error> {
        match message {
            client_gen::Message::System { content } => Ok(provider_chat::ChatMessage::Message(
                provider_chat::TextMessage {
                    role: "system".to_owned(),
                    content: content_to_wire(&content)?,
                    name: None,
                    reasoning_content: None,
                },
            )),
            client_gen::Message::User { content } => Ok(provider_chat::ChatMessage::Message(
                provider_chat::TextMessage {
                    role: "user".to_owned(),
                    content: content_to_wire(&content)?,
                    name: None,
                    reasoning_content: None,
                },
            )),
            client_gen::Message::Assistant(message) => {
                let reasoning_content = message.reasoning.and_then(|r| r.text);
                if message.tool_calls.is_empty() {
                    Ok(provider_chat::ChatMessage::Message(
                        provider_chat::TextMessage {
                            role: "assistant".to_owned(),
                            content: provider_chat::MessageContent::Text(
                                message.content.unwrap_or_default(),
                            ),
                            name: None,
                            reasoning_content,
                        },
                    ))
                } else {
                    Ok(provider_chat::ChatMessage::ToolCalls(
                        provider_chat::ToolCallsMessage {
                            role: "assistant".to_owned(),
                            content: message.content,
                            name: None,
                            tool_calls: message.tool_calls.into_iter().map(Into::into).collect(),
                            reasoning_content,
                        },
                    ))
                }
            }
            client_gen::Message::Tool {
                content,
                tool_call_id,
            } => Ok(provider_chat::ChatMessage::ToolResult(
                provider_chat::ToolResultMessage {
                    role: "tool".to_owned(),
                    content,
                    tool_call_id,
                },
            )),
        }
    }
}

impl From<client_gen::ToolCall> for provider_chat::ChatCompletionToolCall {
    fn from(call: client_gen::ToolCall) -> Self {
        Self {
            id: call.id,
            kind: provider_chat::ToolType::Function,
            function: provider_chat::FunctionCall {
                name: call.name,
                arguments: call.arguments,
            },
        }
    }
}

impl TryFrom<client_gen::GenerationRequest> for provider_chat::ChatCompletionRequest {
    type Error = BackendError;

    fn try_from(request: client_gen::GenerationRequest) -> Result<Self, Self::Error> {
        if request.top_k.is_some() {
            return Err(BackendError::invalid_request(
                "DeepSeek does not support top_k",
            ));
        }
        if request.previous_response_id.is_some() || request.store.is_some() {
            return Err(BackendError::invalid_request(
                "DeepSeek does not support stateful conversation (previous_response_id/store)",
            ));
        }
        if request.frequency_penalty.is_some() || request.presence_penalty.is_some() {
            return Err(BackendError::invalid_request(
                "DeepSeek does not support frequency_penalty or presence_penalty",
            ));
        }
        let reasoning_effort = match request.reasoning_effort {
            None => None,
            Some(client_gen::ReasoningEffort::Low) => Some(provider_chat::ReasoningEffort::Low),
            Some(client_gen::ReasoningEffort::Medium) => {
                Some(provider_chat::ReasoningEffort::Medium)
            }
            Some(client_gen::ReasoningEffort::High) => Some(provider_chat::ReasoningEffort::High),
            Some(client_gen::ReasoningEffort::Xhigh) => Some(provider_chat::ReasoningEffort::Xhigh),
            Some(client_gen::ReasoningEffort::Max) => Some(provider_chat::ReasoningEffort::Max),
        };
        let messages = request
            .messages
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            model: request.model,
            messages,
            thinking: None,
            max_tokens: request.max_tokens,
            response_format: request.response_format.map(Into::into),
            stop: request.stop.map(Into::into),
            stream: request.stream,
            // Request streaming usage so the normalized Usage event can fire: chat-completions
            // only emits usage when include_usage is set.
            stream_options: Some(provider_chat::StreamOptions {
                include_usage: Some(true),
            }),
            temperature: request.temperature,
            top_p: request.top_p,
            tools: request
                .tools
                .map(|tools| tools.into_iter().map(Into::into).collect()),
            tool_choice: request.tool_choice.map(Into::into),
            logprobs: request.logprobs,
            top_logprobs: request.top_logprobs,
            reasoning_effort,
            user_id: None,
        })
    }
}

impl From<client_gen::ResponseFormat> for provider_chat::ResponseFormat {
    fn from(format: client_gen::ResponseFormat) -> Self {
        Self {
            #[allow(unreachable_patterns)]
            kind: match format.kind {
                client_gen::ResponseFormatType::Text => provider_chat::ResponseFormatType::Text,
                client_gen::ResponseFormatType::JsonObject => {
                    provider_chat::ResponseFormatType::JsonObject
                }
                _ => provider_chat::ResponseFormatType::Text,
            },
        }
    }
}

impl From<client_gen::StopSequence> for provider_chat::StopSequence {
    fn from(stop: client_gen::StopSequence) -> Self {
        match stop {
            client_gen::StopSequence::Single(value) => Self::Single(value),
            client_gen::StopSequence::Multiple(values) => Self::Multiple(values),
        }
    }
}

impl From<client_gen::ToolDefinition> for provider_chat::ToolDefinition {
    fn from(tool: client_gen::ToolDefinition) -> Self {
        Self {
            kind: tool.kind.into(),
            function: provider_chat::FunctionDefinition {
                name: tool.function.name,
                description: tool.function.description,
                parameters: tool.function.parameters,
                strict: tool.function.strict,
            },
        }
    }
}

impl From<client_gen::ToolChoice> for provider_chat::ToolChoice {
    fn from(choice: client_gen::ToolChoice) -> Self {
        match choice {
            client_gen::ToolChoice::Mode(mode) => Self::Mode(mode.into()),
            client_gen::ToolChoice::Named(choice) => Self::Named(provider_chat::NamedToolChoice {
                kind: choice.kind.into(),
                function: provider_chat::NamedToolChoiceFunction {
                    name: choice.function.name,
                },
            }),
        }
    }
}

impl From<client_gen::ToolChoiceMode> for provider_chat::ToolChoiceMode {
    fn from(mode: client_gen::ToolChoiceMode) -> Self {
        #[allow(unreachable_patterns)]
        match mode {
            client_gen::ToolChoiceMode::None => Self::None,
            client_gen::ToolChoiceMode::Auto => Self::Auto,
            client_gen::ToolChoiceMode::Required => Self::Required,
            _ => Self::Auto,
        }
    }
}

impl From<client_gen::ToolType> for provider_chat::ToolType {
    fn from(tool_type: client_gen::ToolType) -> Self {
        #[allow(unreachable_patterns)]
        match tool_type {
            client_gen::ToolType::Function => Self::Function,
            _ => Self::Function,
        }
    }
}

// --- response direction ---

impl From<provider_chat::ChatCompletion> for client_gen::GenerationResponse {
    fn from(response: provider_chat::ChatCompletion) -> Self {
        let usage = response.usage.map(Into::into);
        match response.choices.into_iter().next() {
            Some(choice) => Self {
                id: response.id,
                model: response.model,
                message: choice.message.into(),
                finish_reason: choice.finish_reason.map(Into::into),
                usage,
                meta: None,
            },
            None => Self {
                id: response.id,
                model: response.model,
                message: client_gen::AssistantMessage {
                    content: None,
                    tool_calls: Vec::new(),
                    reasoning: None,
                },
                finish_reason: None,
                usage,
                meta: None,
            },
        }
    }
}

impl From<provider_chat::AssistantMessage> for client_gen::AssistantMessage {
    fn from(message: provider_chat::AssistantMessage) -> Self {
        Self {
            content: message.content.filter(|content| !content.is_empty()),
            tool_calls: message
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .map(Into::into)
                .collect(),
            reasoning: message.reasoning_content.map(|text| client_gen::Reasoning {
                text: Some(text),
                id: None,
                encrypted: None,
                signature: None,
                redacted: None,
            }),
        }
    }
}

impl From<provider_chat::ChatCompletionToolCall> for client_gen::ToolCall {
    fn from(call: provider_chat::ChatCompletionToolCall) -> Self {
        Self {
            id: call.id,
            name: call.function.name,
            arguments: call.function.arguments,
        }
    }
}

// NOTE: exhaustive by design — no `_ =>` fallback. The source
// `provider_chat::FinishReason` is not `#[non_exhaustive]`, so the compiler
// proves exhaustiveness and any new provider variant becomes a compile error
// forcing a deliberate mapping here. (The client->provider conversions above use
// `_ =>` arms only because their source — the client enum — IS
// `#[non_exhaustive]`. Different direction, different requirement.)
impl From<provider_chat::FinishReason> for client_gen::FinishReason {
    fn from(reason: provider_chat::FinishReason) -> Self {
        match reason {
            provider_chat::FinishReason::Stop => Self::Stop,
            provider_chat::FinishReason::Length => Self::Length,
            provider_chat::FinishReason::ContentFilter => Self::ContentFilter,
            provider_chat::FinishReason::ToolCalls => Self::ToolCalls,
            provider_chat::FinishReason::InsufficientSystemResource => {
                Self::InsufficientSystemResource
            }
        }
    }
}

impl From<provider_chat::Usage> for client_gen::Usage {
    fn from(usage: provider_chat::Usage) -> Self {
        Self {
            completion_tokens: usage.completion_tokens,
            prompt_tokens: usage.prompt_tokens,
            cache_read_tokens: Some(usage.prompt_cache_hit_tokens),
            cache_write_tokens: Some(usage.prompt_cache_miss_tokens),
            total_tokens: usage.total_tokens,
            completion_tokens_details: usage.completion_tokens_details.map(|details| {
                client_gen::CompletionTokensDetails {
                    reasoning_tokens: details.reasoning_tokens,
                }
            }),
        }
    }
}

/// Flattens a provider streaming chunk into normalized generation events.
///
/// A chat chunk may carry a text delta, a reasoning delta, multiple tool-call deltas, a finish
/// reason, and cumulative usage in any combination, so it maps to zero or more events.
pub fn chunk_to_events(
    chunk: provider_chat::ChatCompletionChunk,
) -> Vec<client_gen::GenerationEvent> {
    let mut events = Vec::new();

    for choice in chunk.choices {
        if let Some(content) = choice.delta.content {
            events.push(client_gen::GenerationEvent::Text { delta: content });
        }
        if let Some(reasoning) = choice.delta.reasoning_content {
            events.push(client_gen::GenerationEvent::Reasoning { delta: reasoning });
        }
        if let Some(tool_calls) = choice.delta.tool_calls {
            for call in tool_calls {
                events.push(client_gen::GenerationEvent::ToolCall {
                    delta: client_gen::ToolCallDelta {
                        index: call.index,
                        id: call.id,
                        name: call.function.as_ref().and_then(|f| f.name.clone()),
                        arguments: call.function.as_ref().and_then(|f| f.arguments.clone()),
                    },
                });
            }
        }
        if let Some(finish_reason) = choice.finish_reason {
            events.push(client_gen::GenerationEvent::End {
                finish_reason: Some(finish_reason.into()),
                response_id: None,
            });
        }
    }

    if let Some(usage) = chunk.usage {
        events.push(client_gen::GenerationEvent::Usage {
            usage: usage.into(),
        });
    }

    events
}
