//! Explicit OpenAI Responses <-> LLM client conversions.
//!
//! The Responses API differs structurally from the chat-completions shape the other two backends
//! target: system instructions live in a top-level `instructions` field, tool results are
//! `function_call_output` input items, and reasoning round-trips through `reasoning` items carrying
//! `id`/`encrypted_content`. Request-side conversions are fallible: semantic fields Responses
//! cannot express (`top_k`, `stop`, penalties) surface as an explicit invalid-request error.
//!
use crate::{BackendError, types::generation as client_gen};
use just_openai_responses::types::{
    item::{
        FunctionCall as WireFunctionCall, FunctionCallOutput, InputItem, OutputItem, ReasoningItem,
    },
    message::{
        ImageDetail as WireImageDetail, InputContentPart, InputMessage,
        MessageContent as WireMessageContent, MessageRole, SummaryText,
    },
    request::{CreateResponseRequest, ResponseInput},
    response::{IncompleteReason, Response, ResponseStatus, ResponseUsage},
    shared::{
        ReasoningConfig, ResponseIncludable, TextConfig, TextFormat, ToolChoice as WireToolChoice,
        ToolChoiceMode as WireToolChoiceMode,
    },
    tool::ResponseTool,
};

/// Extracts leading system messages into an `instructions` string (Responses carries the system
/// prompt as a top-level string), returning the remaining messages for `input`.
pub(crate) fn extract_instructions(
    messages: Vec<client_gen::Message>,
) -> Result<(Option<String>, Vec<client_gen::Message>), BackendError> {
    let mut split_index = 0;
    while split_index < messages.len()
        && matches!(messages[split_index], client_gen::Message::System { .. })
    {
        split_index += 1;
    }

    if split_index == 0 {
        return Ok((None, messages));
    }

    let mut parts = Vec::new();
    for message in &messages[..split_index] {
        match message {
            client_gen::Message::System {
                content: client_gen::MessageContent::Text(text),
            } => parts.push(text.clone()),
            client_gen::Message::System {
                content: client_gen::MessageContent::Parts(_),
            } => {
                return Err(BackendError::unserializable(
                    crate::family::OPENAI_RESPONSES,
                    "instructions require plain-text system content",
                ));
            }
            _ => unreachable!("leading block is all system messages"),
        }
    }

    Ok((Some(parts.join("\n\n")), messages[split_index..].to_vec()))
}

/// Converts semantic content into Responses wire message content.
fn message_content_to_wire(
    content: &client_gen::MessageContent,
) -> Result<WireMessageContent, BackendError> {
    match content {
        client_gen::MessageContent::Text(text) => Ok(WireMessageContent::Text(text.clone())),
        client_gen::MessageContent::Parts(parts) => {
            let mut wire_parts = Vec::new();
            for part in parts {
                match part {
                    client_gen::ContentPart::Text { text } => {
                        wire_parts.push(InputContentPart::InputText {
                            text: text.clone(),
                            prompt_cache_breakpoint: None,
                        });
                    }
                    client_gen::ContentPart::Image { source, detail } => {
                        let (image_url, file_id) = match source {
                            client_gen::ImageSource::Url { url } => (Some(url.clone()), None),
                            client_gen::ImageSource::Base64 { data, media_type } => {
                                (Some(format!("data:{media_type};base64,{data}")), None)
                            }
                            client_gen::ImageSource::FileId { file_id } => {
                                (None, Some(file_id.clone()))
                            }
                        };
                        wire_parts.push(InputContentPart::InputImage {
                            detail: detail.as_ref().map(image_detail_to_wire),
                            image_url,
                            file_id,
                            prompt_cache_breakpoint: None,
                        });
                    }
                }
            }
            Ok(WireMessageContent::Parts(wire_parts))
        }
    }
}

/// Maps a semantic image detail to the Responses wire enum; unknown values
/// pass through verbatim (the wire variant carries the original string).
fn image_detail_to_wire(detail: &client_gen::ImageDetail) -> WireImageDetail {
    match detail {
        client_gen::ImageDetail::Auto => WireImageDetail::Auto,
        client_gen::ImageDetail::Low => WireImageDetail::Low,
        client_gen::ImageDetail::High => WireImageDetail::High,
        client_gen::ImageDetail::Original => WireImageDetail::Original,
        client_gen::ImageDetail::Unknown(value) => WireImageDetail::Unknown(value.clone()),
    }
}

/// Converts one semantic message into the corresponding Responses input items.
///
/// An assistant message maps to multiple items: an output-text message, one `function_call` per
/// tool call, and a `reasoning` item when fidelity carriers are present.
pub(crate) fn message_to_input_items(
    message: client_gen::Message,
) -> Result<Vec<InputItem>, BackendError> {
    match message {
        client_gen::Message::System { .. } => Err(BackendError::invalid_request(
            "system messages must be extracted before input conversion",
        )),
        client_gen::Message::User { content } => Ok(vec![InputItem::Message(InputMessage {
            role: MessageRole::User,
            content: message_content_to_wire(&content)?,
            phase: None,
        })]),
        client_gen::Message::Assistant(message) => {
            let mut items = Vec::new();

            if let Some(content) = message.content {
                items.push(InputItem::Message(InputMessage {
                    role: MessageRole::Assistant,
                    content: WireMessageContent::Text(content),
                    phase: None,
                }));
            }

            for call in message.tool_calls {
                items.push(InputItem::FunctionCall(WireFunctionCall::new(
                    call.name,
                    call.id,
                    call.arguments,
                )));
            }

            if let Some(reasoning) = message.reasoning {
                if let Some(id) = reasoning.id {
                    items.push(InputItem::Reasoning(ReasoningItem {
                        id,
                        summary: reasoning
                            .text
                            .map(|text| vec![SummaryText::Summary { text }])
                            .unwrap_or_default(),
                        content: None,
                        encrypted_content: reasoning.encrypted,
                        status: None,
                    }));
                }
            }

            Ok(items)
        }
        client_gen::Message::Tool {
            content,
            tool_call_id,
        } => Ok(vec![InputItem::FunctionCallOutput(
            FunctionCallOutput::new(tool_call_id, content),
        )]),
    }
}

fn wire_tool_choice(choice: client_gen::ToolChoice) -> WireToolChoice {
    match choice {
        client_gen::ToolChoice::Mode(mode) => WireToolChoice::Mode({
            #[allow(unreachable_patterns)]
            match mode {
                client_gen::ToolChoiceMode::None => WireToolChoiceMode::None,
                client_gen::ToolChoiceMode::Auto => WireToolChoiceMode::Auto,
                client_gen::ToolChoiceMode::Required => WireToolChoiceMode::Required,
                _ => WireToolChoiceMode::Auto,
            }
        }),
        client_gen::ToolChoice::Named(choice) => WireToolChoice::Object(
            just_openai_responses::types::shared::ToolChoiceObject::Function {
                name: choice.function.name,
            },
        ),
    }
}

pub(crate) fn wire_tool(tool: client_gen::ToolDefinition) -> ResponseTool {
    ResponseTool::Function {
        name: tool.function.name,
        description: tool.function.description,
        parameters: tool.function.parameters,
        strict: tool.function.strict,
        output_schema: None,
        defer_loading: None,
        allowed_callers: None,
    }
}

fn wire_text_config(format: &client_gen::ResponseFormat) -> TextConfig {
    let kind = match format.kind {
        client_gen::ResponseFormatType::JsonObject => TextFormat::JsonObject,
        _ => TextFormat::Text,
    };
    TextConfig {
        format: Some(kind),
        verbosity: None,
    }
}

fn wire_reasoning(effort: &client_gen::ReasoningEffort) -> ReasoningConfig {
    use just_openai_responses::types::shared::ReasoningEffort as WireEffort;
    let effort = match effort {
        client_gen::ReasoningEffort::Low => WireEffort::Low,
        client_gen::ReasoningEffort::Medium => WireEffort::Medium,
        client_gen::ReasoningEffort::High => WireEffort::High,
        client_gen::ReasoningEffort::Xhigh => WireEffort::Xhigh,
        client_gen::ReasoningEffort::Max => WireEffort::Max,
    };
    ReasoningConfig {
        effort: Some(effort),
        summary: None,
        context: None,
        mode: None,
    }
}

impl TryFrom<client_gen::GenerationRequest> for CreateResponseRequest {
    type Error = BackendError;

    fn try_from(request: client_gen::GenerationRequest) -> Result<Self, Self::Error> {
        if request.top_k.is_some() {
            return Err(BackendError::invalid_request(
                "the Responses API does not support top_k",
            ));
        }
        if request.stop.is_some() {
            return Err(BackendError::invalid_request(
                "the Responses API does not support stop sequences",
            ));
        }
        if request.frequency_penalty.is_some() || request.presence_penalty.is_some() {
            return Err(BackendError::invalid_request(
                "the Responses API does not support frequency_penalty or presence_penalty",
            ));
        }

        let previous_response_id = request.previous_response_id;
        let (extracted_instructions, messages) = extract_instructions(request.messages)?;
        let input = messages
            .into_iter()
            .map(message_to_input_items)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        // On continuation, the system prompt is carried by the stored conversation (and xAI
        // forbids passing instructions alongside previous_response_id), so drop it. A stored
        // response is required for the chain, so default to storing unless the caller opted out.
        let instructions = if previous_response_id.is_some() {
            None
        } else {
            extracted_instructions
        };
        let store = if previous_response_id.is_some() {
            request.store.or(Some(true))
        } else {
            request.store
        };

        Ok(CreateResponseRequest {
            model: request.model,
            input: Some(ResponseInput::Items(input)),
            instructions,
            tools: request
                .tools
                .map(|tools| tools.into_iter().map(wire_tool).collect()),
            tool_choice: request.tool_choice.map(wire_tool_choice),
            parallel_tool_calls: None,
            previous_response_id,
            conversation: None,
            store,
            stream: request.stream,
            stream_options: None,
            temperature: request.temperature,
            top_p: request.top_p,
            top_logprobs: request.top_logprobs,
            max_output_tokens: request.max_tokens,
            max_tool_calls: None,
            text: request.response_format.as_ref().map(wire_text_config),
            reasoning: request.reasoning_effort.as_ref().map(wire_reasoning),
            include: request
                .reasoning_effort
                .as_ref()
                .map(|_| vec![ResponseIncludable::ReasoningEncryptedContent]),
            truncation: None,
            metadata: None,
            user: None,
            background: None,
            service_tier: None,
        })
    }
}

// --- response direction ---

/// Derives the normalized finish reason from a Responses `Response`.
///
/// The Responses API has no `finish_reason` field; it is reconstructed from the item set and the
/// status/incomplete details.
fn finish_reason(response: &Response) -> Option<client_gen::FinishReason> {
    let has_tool_calls = response
        .output
        .iter()
        .any(|item| matches!(item, OutputItem::FunctionCall(_)));
    if has_tool_calls {
        return Some(client_gen::FinishReason::ToolCalls);
    }

    match response.status {
        ResponseStatus::Completed => Some(client_gen::FinishReason::Stop),
        ResponseStatus::Incomplete => match response
            .incomplete_details
            .as_ref()
            .and_then(|details| details.reason)
        {
            Some(IncompleteReason::MaxOutputTokens) => Some(client_gen::FinishReason::Length),
            Some(IncompleteReason::ContentFilter) => Some(client_gen::FinishReason::ContentFilter),
            _ => None,
        },
        _ => None,
    }
}

fn wire_usage_to_client(usage: ResponseUsage) -> client_gen::Usage {
    client_gen::Usage {
        completion_tokens: usage.output_tokens as u32,
        prompt_tokens: usage.input_tokens as u32,
        cache_read_tokens: usage
            .input_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens)
            .map(|tokens| tokens as u32),
        cache_write_tokens: usage
            .input_tokens_details
            .as_ref()
            .and_then(|details| details.cache_write_tokens)
            .map(|tokens| tokens as u32),
        total_tokens: usage.total_tokens as u32,
        completion_tokens_details: usage
            .output_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens)
            .map(|tokens| client_gen::CompletionTokensDetails {
                reasoning_tokens: Some(tokens as u32),
            }),
    }
}

impl From<Response> for client_gen::GenerationResponse {
    fn from(response: Response) -> Self {
        let finish_reason = finish_reason(&response);
        let usage = response.usage.map(wire_usage_to_client);

        let mut content = String::new();
        let mut tool_calls = Vec::new();
        let mut reasoning = None;

        for item in response.output {
            match item {
                OutputItem::Message(message) => {
                    for part in message.content {
                        if let just_openai_responses::types::message::OutputContentPart::OutputText {
                            text,
                            ..
                        } = part
                        {
                            content.push_str(&text);
                        }
                    }
                }
                OutputItem::FunctionCall(call) => {
                    tool_calls.push(client_gen::ToolCall {
                        id: call.call_id,
                        name: call.name,
                        arguments: call.arguments,
                    });
                }
                OutputItem::Reasoning(item) => {
                    reasoning = Some(client_gen::Reasoning {
                        text: {
                            let mut summary = String::new();
                            for entry in item.summary {
                                if let SummaryText::Summary { text } = entry {
                                    summary.push_str(&text);
                                }
                            }
                            if summary.is_empty() {
                                None
                            } else {
                                Some(summary)
                            }
                        },
                        id: Some(item.id),
                        encrypted: item.encrypted_content,
                        signature: None,
                        redacted: None,
                    });
                }
                _ => {}
            }
        }

        Self {
            id: response.id,
            model: response.model,
            message: client_gen::AssistantMessage {
                content: if content.is_empty() {
                    None
                } else {
                    Some(content)
                },
                tool_calls,
                reasoning,
            },
            finish_reason,
            usage,
            meta: None,
        }
    }
}

/// Maps a streaming Responses event to normalized generation events (zero or more per event).
pub(crate) fn event_to_generation_events(
    event: just_openai_responses::types::event::StreamEvent,
) -> Vec<client_gen::GenerationEvent> {
    use just_openai_responses::types::event::StreamEvent;

    match event {
        StreamEvent::ResponseOutputTextDelta { delta, .. } => {
            vec![client_gen::GenerationEvent::Text { delta }]
        }
        StreamEvent::ResponseReasoningTextDelta { delta, .. }
        | StreamEvent::ResponseReasoningSummaryTextDelta { delta, .. } => {
            vec![client_gen::GenerationEvent::Reasoning { delta }]
        }
        StreamEvent::ResponseOutputItemAdded {
            item: OutputItem::FunctionCall(call),
            output_index,
            ..
        } => vec![client_gen::GenerationEvent::ToolCall {
            delta: client_gen::ToolCallDelta {
                index: Some(output_index),
                id: Some(call.call_id),
                name: Some(call.name),
                arguments: None,
            },
        }],
        StreamEvent::ResponseFunctionCallArgumentsDelta {
            delta,
            output_index,
            ..
        } => vec![client_gen::GenerationEvent::ToolCall {
            delta: client_gen::ToolCallDelta {
                index: Some(output_index),
                id: None,
                name: None,
                arguments: Some(delta),
            },
        }],
        StreamEvent::ResponseCompleted { response, .. } => {
            let mut events = vec![client_gen::GenerationEvent::End {
                finish_reason: finish_reason(&response),
                response_id: Some(response.id),
            }];
            if let Some(usage) = response.usage {
                events.push(client_gen::GenerationEvent::Usage {
                    usage: wire_usage_to_client(usage),
                });
            }
            events
        }
        StreamEvent::ResponseIncomplete { response, .. } => {
            vec![client_gen::GenerationEvent::End {
                finish_reason: finish_reason(&response),
                response_id: Some(response.id),
            }]
        }
        StreamEvent::ResponseFailed { response, .. } => {
            vec![client_gen::GenerationEvent::End {
                finish_reason: finish_reason(&response),
                response_id: Some(response.id),
            }]
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_usage_details_map_to_none() {
        let usage = ResponseUsage {
            input_tokens: 10,
            input_tokens_details: None,
            output_tokens: 5,
            output_tokens_details: None,
            total_tokens: 15,
        };

        let client_usage = wire_usage_to_client(usage);

        assert_eq!(client_usage.prompt_tokens, 10);
        assert_eq!(client_usage.completion_tokens, 5);
        assert_eq!(client_usage.total_tokens, 15);
        assert_eq!(client_usage.cache_read_tokens, None);
        assert_eq!(client_usage.cache_write_tokens, None);
        assert_eq!(client_usage.completion_tokens_details, None);
    }
}
