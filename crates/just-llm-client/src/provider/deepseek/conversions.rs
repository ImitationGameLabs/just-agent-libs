//! Explicit DeepSeek <-> LLM client conversions.
//!
//! Many of these mappings look mechanically similar to the OpenAI-compatible adapter on purpose:
//! the duplication keeps the provider wire types independent from the `just-llm-client` normalized layer, so
//! future provider-specific evolution does not need to route through a shared protocol abstraction.
//!
use crate::types::chat as client_chat;
use just_deepseek::types::chat as provider_chat;

impl From<client_chat::ChatCompletionRequest> for provider_chat::ChatCompletionRequest {
    fn from(request: client_chat::ChatCompletionRequest) -> Self {
        Self {
            model: request.model,
            messages: request.messages.into_iter().map(Into::into).collect(),
            thinking: None,
            max_tokens: request.max_tokens,
            response_format: request.response_format.map(Into::into),
            stop: request.stop.map(Into::into),
            stream: request.stream,
            stream_options: request.stream_options.map(Into::into),
            temperature: request.temperature,
            top_p: request.top_p,
            tools: request
                .tools
                .map(|tools| tools.into_iter().map(Into::into).collect()),
            tool_choice: request.tool_choice.map(Into::into),
            logprobs: request.logprobs,
            top_logprobs: request.top_logprobs,
            reasoning_effort: None,
            user_id: None,
        }
    }
}

impl From<client_chat::ChatMessage> for provider_chat::ChatMessage {
    fn from(message: client_chat::ChatMessage) -> Self {
        match message {
            client_chat::ChatMessage::Message(message) => {
                provider_chat::ChatMessage::Message(provider_chat::TextMessage {
                    role: message.role,
                    content: message.content,
                    name: message.name,
                    reasoning_content: message.reasoning_content,
                })
            }
            client_chat::ChatMessage::ToolCalls(message) => {
                provider_chat::ChatMessage::ToolCalls(provider_chat::ToolCallsMessage {
                    role: message.role,
                    content: message.content,
                    name: message.name,
                    reasoning_content: message.reasoning_content,
                    tool_calls: message.tool_calls.into_iter().map(Into::into).collect(),
                })
            }
            client_chat::ChatMessage::ToolResult(message) => {
                provider_chat::ChatMessage::ToolResult(provider_chat::ToolResultMessage {
                    role: message.role,
                    content: message.content,
                    tool_call_id: message.tool_call_id,
                })
            }
        }
    }
}

impl From<client_chat::ResponseFormat> for provider_chat::ResponseFormat {
    fn from(format: client_chat::ResponseFormat) -> Self {
        Self {
            #[allow(unreachable_patterns)]
            kind: match format.kind {
                client_chat::ResponseFormatType::Text => provider_chat::ResponseFormatType::Text,
                client_chat::ResponseFormatType::JsonObject => {
                    provider_chat::ResponseFormatType::JsonObject
                }
                _ => provider_chat::ResponseFormatType::Text,
            },
        }
    }
}

impl From<client_chat::StopSequence> for provider_chat::StopSequence {
    fn from(stop: client_chat::StopSequence) -> Self {
        match stop {
            client_chat::StopSequence::Single(value) => Self::Single(value),
            client_chat::StopSequence::Multiple(values) => Self::Multiple(values),
        }
    }
}

impl From<client_chat::StreamOptions> for provider_chat::StreamOptions {
    fn from(options: client_chat::StreamOptions) -> Self {
        Self {
            include_usage: options.include_usage,
        }
    }
}

impl From<client_chat::ToolDefinition> for provider_chat::ToolDefinition {
    fn from(tool: client_chat::ToolDefinition) -> Self {
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

impl From<client_chat::ToolChoice> for provider_chat::ToolChoice {
    fn from(choice: client_chat::ToolChoice) -> Self {
        match choice {
            client_chat::ToolChoice::Mode(mode) => Self::Mode(mode.into()),
            client_chat::ToolChoice::Named(choice) => Self::Named(provider_chat::NamedToolChoice {
                kind: choice.kind.into(),
                function: provider_chat::NamedToolChoiceFunction {
                    name: choice.function.name,
                },
            }),
        }
    }
}

impl From<client_chat::ToolChoiceMode> for provider_chat::ToolChoiceMode {
    fn from(mode: client_chat::ToolChoiceMode) -> Self {
        #[allow(unreachable_patterns)]
        match mode {
            client_chat::ToolChoiceMode::None => Self::None,
            client_chat::ToolChoiceMode::Auto => Self::Auto,
            client_chat::ToolChoiceMode::Required => Self::Required,
            _ => Self::Auto,
        }
    }
}

impl From<client_chat::ToolType> for provider_chat::ToolType {
    fn from(tool_type: client_chat::ToolType) -> Self {
        #[allow(unreachable_patterns)]
        match tool_type {
            client_chat::ToolType::Function => Self::Function,
            _ => Self::Function,
        }
    }
}

impl From<client_chat::ChatToolCall> for provider_chat::ChatCompletionToolCall {
    fn from(call: client_chat::ChatToolCall) -> Self {
        Self {
            id: call.id,
            kind: call.kind.into(),
            function: provider_chat::FunctionCall {
                name: call.function.name,
                arguments: call.function.arguments,
            },
        }
    }
}

impl From<provider_chat::ChatCompletion> for client_chat::ChatCompletionResponse {
    fn from(response: provider_chat::ChatCompletion) -> Self {
        Self {
            id: response.id,
            choices: response.choices.into_iter().map(Into::into).collect(),
            created: response.created,
            model: response.model,
            system_fingerprint: response.system_fingerprint,
            object: response.object,
            usage: response.usage.map(Into::into),
        }
    }
}

impl From<provider_chat::ChatCompletionChoice> for client_chat::ChatChoice {
    fn from(choice: provider_chat::ChatCompletionChoice) -> Self {
        Self {
            finish_reason: choice.finish_reason.map(Into::into),
            index: choice.index,
            message: client_chat::AssistantMessage {
                content: choice.message.content,
                reasoning_content: choice.message.reasoning_content,
                tool_calls: choice
                    .message
                    .tool_calls
                    .map(|calls| calls.into_iter().map(Into::into).collect()),
                role: client_chat::AssistantRole::Assistant,
            },
            logprobs: choice.logprobs.map(Into::into),
        }
    }
}

impl From<provider_chat::ChatCompletionChunk> for client_chat::ChatCompletionChunk {
    fn from(chunk: provider_chat::ChatCompletionChunk) -> Self {
        Self {
            id: chunk.id,
            choices: chunk.choices.into_iter().map(Into::into).collect(),
            created: chunk.created,
            model: chunk.model,
            system_fingerprint: chunk.system_fingerprint,
            object: chunk.object,
            usage: chunk.usage.map(Into::into),
        }
    }
}

impl From<provider_chat::ChatCompletionChunkChoice> for client_chat::ChatCompletionChunkChoice {
    fn from(choice: provider_chat::ChatCompletionChunkChoice) -> Self {
        Self {
            delta: client_chat::DeltaMessage {
                content: choice.delta.content,
                reasoning_content: choice.delta.reasoning_content,
                role: choice
                    .delta
                    .role
                    .map(|_| client_chat::AssistantRole::Assistant),
                tool_calls: choice
                    .delta
                    .tool_calls
                    .map(|calls| calls.into_iter().map(Into::into).collect()),
            },
            finish_reason: choice.finish_reason.map(Into::into),
            index: choice.index,
            logprobs: choice.logprobs.map(Into::into),
        }
    }
}

impl From<provider_chat::ChatCompletionChunkToolCall> for client_chat::ChatCompletionChunkToolCall {
    fn from(call: provider_chat::ChatCompletionChunkToolCall) -> Self {
        Self {
            index: call.index,
            id: call.id,
            kind: call.kind.map(Into::into),
            function: call.function.map(Into::into),
        }
    }
}

impl From<provider_chat::FunctionCallDelta> for client_chat::FunctionCallDelta {
    fn from(function: provider_chat::FunctionCallDelta) -> Self {
        Self {
            name: function.name,
            arguments: function.arguments,
        }
    }
}

impl From<provider_chat::FinishReason> for client_chat::FinishReason {
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

impl From<provider_chat::Usage> for client_chat::Usage {
    fn from(usage: provider_chat::Usage) -> Self {
        Self {
            completion_tokens: usage.completion_tokens,
            prompt_tokens: usage.prompt_tokens,
            prompt_cache_hit_tokens: Some(usage.prompt_cache_hit_tokens),
            prompt_cache_miss_tokens: Some(usage.prompt_cache_miss_tokens),
            total_tokens: usage.total_tokens,
            completion_tokens_details: usage.completion_tokens_details.map(|details| {
                client_chat::CompletionTokensDetails {
                    reasoning_tokens: details.reasoning_tokens,
                }
            }),
        }
    }
}

impl From<provider_chat::ChatCompletionLogprobs> for client_chat::ChatCompletionLogprobs {
    fn from(logprobs: provider_chat::ChatCompletionLogprobs) -> Self {
        Self {
            content: logprobs
                .content
                .map(|items| items.into_iter().map(Into::into).collect()),
            reasoning_content: logprobs
                .reasoning_content
                .map(|items| items.into_iter().map(Into::into).collect()),
        }
    }
}

impl From<provider_chat::TokenLogprob> for client_chat::TokenLogprob {
    fn from(value: provider_chat::TokenLogprob) -> Self {
        Self {
            token: value.token,
            logprob: value.logprob,
            bytes: value.bytes,
            top_logprobs: value.top_logprobs.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<provider_chat::TopLogprob> for client_chat::TopLogprob {
    fn from(entry: provider_chat::TopLogprob) -> Self {
        Self {
            token: entry.token,
            logprob: entry.logprob,
            bytes: entry.bytes,
        }
    }
}

impl From<provider_chat::ChatCompletionToolCall> for client_chat::ChatToolCall {
    fn from(call: provider_chat::ChatCompletionToolCall) -> Self {
        Self {
            id: call.id,
            kind: client_chat::ToolType::Function,
            function: client_chat::FunctionCall {
                name: call.function.name,
                arguments: call.function.arguments,
            },
        }
    }
}

impl From<provider_chat::ToolType> for client_chat::ToolType {
    fn from(_: provider_chat::ToolType) -> Self {
        Self::Function
    }
}
