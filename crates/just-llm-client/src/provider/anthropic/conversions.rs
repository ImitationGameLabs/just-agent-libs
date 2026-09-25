//! Explicit Anthropic Messages <-> LLM client conversions.
//!
//! The Messages API differs structurally from chat-completions: the system prompt is a top-level
//! `system` parameter, tool results are `tool_result` blocks embedded in `user` messages, and
//! extended thinking round-trips through `thinking`/`redacted_thinking` blocks carrying a
//! `signature`. Request-side conversions are fallible: semantic fields Anthropic cannot express
//! (penalties, logprobs, response format) surface as an explicit
//! invalid-request error.
//!
use crate::{BackendError, types::generation as client_gen};
use just_anthropic::types::{
    message::{
        ContentBlock, ContentBlockParam, ImageMediaType, ImageSource,
        MessageContent as WireContent, MessageParam, MessageRole, TextBlockParam,
    },
    output::{OutputConfig, OutputEffort},
    request::{CreateMessageRequest, SystemParam},
    tool::{Tool as WireTool, ToolChoice as WireToolChoice},
};

/// Extracts leading system messages into a top-level `system` parameter (Anthropic carries the
/// system prompt outside the `messages` array), returning the remaining messages.
pub(crate) fn extract_system(
    messages: Vec<client_gen::Message>,
) -> Result<(Option<SystemParam>, Vec<client_gen::Message>), BackendError> {
    let mut split_index = 0;
    while split_index < messages.len()
        && matches!(messages[split_index], client_gen::Message::System { .. })
    {
        split_index += 1;
    }

    if split_index == 0 {
        return Ok((None, messages));
    }

    let mut text_blocks = Vec::new();
    let mut all_text = String::new();
    for message in &messages[..split_index] {
        match message {
            client_gen::Message::System {
                content: client_gen::MessageContent::Text(text),
            } => {
                text_blocks.push(TextBlockParam::new(text.clone()));
                all_text.push_str(text);
                all_text.push('\n');
            }
            client_gen::Message::System {
                content: client_gen::MessageContent::Parts(parts),
            } => {
                for part in parts {
                    match part {
                        client_gen::ContentPart::Text { text } => {
                            text_blocks.push(TextBlockParam::new(text.clone()));
                            all_text.push_str(text);
                            all_text.push('\n');
                        }
                        client_gen::ContentPart::Image { .. } => {
                            return Err(BackendError::unserializable(
                                crate::family::ANTHROPIC,
                                "system prompts must be plain text",
                            ));
                        }
                    }
                }
            }
            _ => unreachable!("leading block is all system messages"),
        }
    }

    // Anthropic accepts a plain string or an array of text blocks; use the plain string for the
    // common single-system-message case.
    let system = if text_blocks.len() == 1 {
        SystemParam::Text(all_text.trim_end().to_owned())
    } else {
        SystemParam::Blocks(text_blocks)
    };

    Ok((Some(system), messages[split_index..].to_vec()))
}

/// Converts semantic content into Anthropic wire content blocks.
fn content_to_wire(content: &client_gen::MessageContent) -> Result<WireContent, BackendError> {
    match content {
        client_gen::MessageContent::Text(text) => Ok(WireContent::Text(text.clone())),
        client_gen::MessageContent::Parts(parts) => {
            let mut blocks = Vec::new();
            for part in parts {
                match part {
                    client_gen::ContentPart::Text { text } => {
                        blocks.push(ContentBlockParam::Text {
                            text: text.clone(),
                            cache_control: None,
                            citations: None,
                        });
                    }
                    client_gen::ContentPart::Image { source, .. } => {
                        let source = match source {
                            client_gen::ImageSource::Url { url } => {
                                ImageSource::Url { url: url.clone() }
                            }
                            client_gen::ImageSource::Base64 { data, media_type } => {
                                ImageSource::Base64 {
                                    data: data.clone(),
                                    media_type: media_type_to_wire(media_type),
                                }
                            }
                            client_gen::ImageSource::FileId { file_id } => ImageSource::File {
                                file_id: file_id.clone(),
                            },
                        };
                        blocks.push(ContentBlockParam::Image {
                            source,
                            cache_control: None,
                        });
                    }
                }
            }
            Ok(WireContent::Blocks(blocks))
        }
    }
}

/// Maps a semantic media-type string to the Anthropic wire enum; unknown
/// values are preserved verbatim for the provider to judge.
fn media_type_to_wire(media_type: &str) -> ImageMediaType {
    match media_type {
        "image/jpeg" => ImageMediaType::Jpeg,
        "image/png" => ImageMediaType::Png,
        "image/gif" => ImageMediaType::Gif,
        "image/webp" => ImageMediaType::Webp,
        other => ImageMediaType::Unknown(other.to_owned()),
    }
}

/// Converts a semantic `ToolCall` arguments string into the value Anthropic expects for a
/// `tool_use` block input. An empty string (a no-argument call) maps to `{}`.
fn arguments_to_value(arguments: &str) -> Result<serde_json::Value, BackendError> {
    if arguments.trim().is_empty() {
        return Ok(serde_json::json!({}));
    }
    serde_json::from_str(arguments).map_err(|_| {
        BackendError::invalid_request(format!(
            "tool call arguments are not valid JSON for Anthropic: {arguments}"
        ))
    })
}

/// Converts one semantic message into an Anthropic message param.
pub(crate) fn message_to_param(message: client_gen::Message) -> Result<MessageParam, BackendError> {
    match message {
        client_gen::Message::System { .. } => Err(BackendError::invalid_request(
            "system messages must be extracted before message conversion",
        )),
        client_gen::Message::User { content } => Ok(MessageParam {
            role: MessageRole::User,
            content: content_to_wire(&content)?,
        }),
        client_gen::Message::Assistant(message) => {
            let mut blocks = Vec::new();

            if let Some(content) = message.content {
                blocks.push(ContentBlockParam::Text {
                    text: content,
                    cache_control: None,
                    citations: None,
                });
            }

            if let Some(reasoning) = message.reasoning {
                if let Some(text) = reasoning.text {
                    blocks.push(ContentBlockParam::Thinking {
                        thinking: text,
                        signature: reasoning.signature.unwrap_or_default(),
                    });
                } else if let Some(redacted) = reasoning.redacted {
                    blocks.push(ContentBlockParam::RedactedThinking { data: redacted });
                }
            }

            for call in message.tool_calls {
                blocks.push(ContentBlockParam::ToolUse {
                    id: call.id,
                    name: call.name,
                    input: arguments_to_value(&call.arguments)?,
                    cache_control: None,
                });
            }

            Ok(MessageParam {
                role: MessageRole::Assistant,
                content: WireContent::Blocks(blocks),
            })
        }
        client_gen::Message::Tool {
            content,
            tool_call_id,
        } => Ok(MessageParam {
            role: MessageRole::User,
            content: WireContent::Blocks(vec![ContentBlockParam::ToolResult {
                tool_use_id: tool_call_id,
                content: Some(WireContent::Text(content)),
                is_error: None,
                cache_control: None,
            }]),
        }),
    }
}

fn wire_tool_choice(choice: client_gen::ToolChoice) -> WireToolChoice {
    match choice {
        client_gen::ToolChoice::Mode(mode) => match mode {
            client_gen::ToolChoiceMode::None => WireToolChoice::None,
            client_gen::ToolChoiceMode::Auto => WireToolChoice::Auto {
                disable_parallel_tool_use: None,
            },
            // Anthropic has no "required" mode; "any" forces tool use, the closest equivalent.
            client_gen::ToolChoiceMode::Required => WireToolChoice::Any {
                disable_parallel_tool_use: None,
            },
            #[allow(unreachable_patterns)]
            _ => WireToolChoice::Auto {
                disable_parallel_tool_use: None,
            },
        },
        client_gen::ToolChoice::Named(choice) => WireToolChoice::Tool {
            name: choice.function.name,
            disable_parallel_tool_use: None,
        },
    }
}

fn wire_effort(effort: client_gen::ReasoningEffort) -> OutputEffort {
    match effort {
        client_gen::ReasoningEffort::Low => OutputEffort::Low,
        client_gen::ReasoningEffort::Medium => OutputEffort::Medium,
        client_gen::ReasoningEffort::High => OutputEffort::High,
        client_gen::ReasoningEffort::Xhigh => OutputEffort::Xhigh,
        client_gen::ReasoningEffort::Max => OutputEffort::Max,
    }
}

pub(crate) fn wire_tool(tool: client_gen::ToolDefinition) -> WireTool {
    WireTool::new(
        tool.function.name,
        tool.function.parameters.unwrap_or_default(),
    )
    .with_description(tool.function.description.unwrap_or_default())
}

impl TryFrom<client_gen::GenerationRequest> for CreateMessageRequest {
    type Error = BackendError;

    fn try_from(request: client_gen::GenerationRequest) -> Result<Self, Self::Error> {
        if request.frequency_penalty.is_some() || request.presence_penalty.is_some() {
            return Err(BackendError::invalid_request(
                "Anthropic does not support frequency_penalty or presence_penalty",
            ));
        }
        if request.previous_response_id.is_some() || request.store.is_some() {
            return Err(BackendError::invalid_request(
                "Anthropic does not support stateful conversation (previous_response_id/store)",
            ));
        }
        if request.logprobs.is_some() || request.top_logprobs.is_some() {
            return Err(BackendError::invalid_request(
                "Anthropic does not support logprobs",
            ));
        }
        if request
            .response_format
            .as_ref()
            .is_some_and(|format| format.kind == client_gen::ResponseFormatType::JsonObject)
        {
            return Err(BackendError::invalid_request(
                "Anthropic structured output requires a JSON schema, which the normalized layer does not carry",
            ));
        }

        let max_tokens = request.max_tokens.ok_or_else(|| {
            BackendError::invalid_request("Anthropic requires max_tokens to be set")
        })?;

        let (system, messages) = extract_system(request.messages)?;
        let messages = messages
            .into_iter()
            .map(message_to_param)
            .collect::<Result<Vec<MessageParam>, _>>()?;

        Ok(CreateMessageRequest {
            model: request.model,
            messages,
            max_tokens,
            system,
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            stop_sequences: match request.stop {
                Some(client_gen::StopSequence::Single(seq)) => Some(vec![seq]),
                Some(client_gen::StopSequence::Multiple(seqs)) => Some(seqs),
                None => None,
            },
            stream: request.stream,
            thinking: None,
            tool_choice: request.tool_choice.map(wire_tool_choice),
            tools: request
                .tools
                .map(|tools| tools.into_iter().map(wire_tool).collect()),
            metadata: None,
            service_tier: None,
            cache_control: None,
            output_config: request.reasoning_effort.map(|effort| OutputConfig {
                effort: Some(wire_effort(effort)),
                format: None,
            }),
        })
    }
}

// --- response direction ---

fn wire_usage_to_client(usage: just_anthropic::types::usage::Usage) -> client_gen::Usage {
    client_gen::Usage {
        completion_tokens: usage.output_tokens as u32,
        prompt_tokens: usage.input_tokens as u32,
        cache_read_tokens: usage.cache_read_input_tokens.map(|tokens| tokens as u32),
        cache_write_tokens: usage
            .cache_creation_input_tokens
            .map(|tokens| tokens as u32),
        total_tokens: (usage.input_tokens + usage.output_tokens) as u32,
        completion_tokens_details: usage.output_tokens_details.map(|details| {
            client_gen::CompletionTokensDetails {
                reasoning_tokens: details.thinking_tokens.map(|tokens| tokens as u32),
            }
        }),
    }
}

pub(crate) fn wire_finish_reason(
    reason: Option<just_anthropic::types::message::StopReason>,
) -> Option<client_gen::FinishReason> {
    use just_anthropic::types::message::StopReason;
    reason.map(|reason| match reason {
        StopReason::EndTurn => client_gen::FinishReason::Stop,
        StopReason::MaxTokens => client_gen::FinishReason::Length,
        StopReason::StopSequence => client_gen::FinishReason::Stop,
        StopReason::ToolUse => client_gen::FinishReason::ToolCalls,
        StopReason::PauseTurn => client_gen::FinishReason::PauseTurn,
        StopReason::Refusal => client_gen::FinishReason::Refusal,
        StopReason::ModelContextWindowExceeded => {
            client_gen::FinishReason::ModelContextWindowExceeded
        }
        StopReason::Unknown => client_gen::FinishReason::Stop,
    })
}
impl From<just_anthropic::types::message::Message> for client_gen::GenerationResponse {
    fn from(message: just_anthropic::types::message::Message) -> Self {
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut reasoning = None;

        for block in message.content {
            match block {
                ContentBlock::Text { text, .. } => content.push_str(&text),
                ContentBlock::Thinking(thinking) => {
                    let entry = reasoning.get_or_insert(client_gen::Reasoning::default());
                    if !thinking.thinking.is_empty() {
                        entry.text = Some(thinking.thinking);
                    }
                    if !thinking.signature.is_empty() {
                        entry.signature = Some(thinking.signature);
                    }
                }
                ContentBlock::RedactedThinking { data } => {
                    let entry = reasoning.get_or_insert(client_gen::Reasoning::default());
                    entry.redacted = Some(data);
                }
                ContentBlock::ToolUse(call) => {
                    tool_calls.push(client_gen::ToolCall {
                        id: call.id,
                        name: call.name,
                        arguments: serde_json::to_string(&call.input)
                            .unwrap_or_else(|_| "{}".to_owned()),
                    });
                }
                ContentBlock::Unknown => {
                    tracing::debug!("dropped unknown anthropic content block");
                }
            }
        }

        Self {
            id: message.id,
            model: message.model,
            message: client_gen::AssistantMessage {
                content: if content.is_empty() {
                    None
                } else {
                    Some(content)
                },
                tool_calls,
                reasoning,
            },
            finish_reason: wire_finish_reason(message.stop_reason),
            usage: Some(wire_usage_to_client(message.usage)),
            meta: None,
        }
    }
}

/// Maps a streaming Anthropic event to normalized generation events (zero or more per event).
pub(crate) fn event_to_generation_events(
    event: just_anthropic::types::event::StreamEvent,
) -> Vec<client_gen::GenerationEvent> {
    use just_anthropic::types::{
        event::{ContentBlockDelta, StreamEvent},
        message::ContentBlock as WireBlock,
    };

    match event {
        StreamEvent::ContentBlockStart {
            content_block: WireBlock::ToolUse(call),
            index,
            ..
        } => vec![client_gen::GenerationEvent::ToolCall {
            delta: client_gen::ToolCallDelta {
                index: Some(index as u32),
                id: Some(call.id),
                name: Some(call.name),
                arguments: None,
            },
        }],
        StreamEvent::ContentBlockDelta {
            delta: ContentBlockDelta::TextDelta { text },
            ..
        } => vec![client_gen::GenerationEvent::Text { delta: text }],
        StreamEvent::ContentBlockDelta {
            delta: ContentBlockDelta::ThinkingDelta { thinking },
            ..
        } => vec![client_gen::GenerationEvent::Reasoning { delta: thinking }],
        StreamEvent::ContentBlockDelta {
            delta: ContentBlockDelta::InputJsonDelta { partial_json },
            index,
            ..
        } => vec![client_gen::GenerationEvent::ToolCall {
            delta: client_gen::ToolCallDelta {
                index: Some(index as u32),
                id: None,
                name: None,
                arguments: Some(partial_json),
            },
        }],
        // Message-level events (message_start/message_delta) are handled by the backend's
        // stateful scan so it can merge the final usage; signature deltas and non-tool block
        // boundaries carry no normalized event.
        _ => Vec::new(),
    }
}
