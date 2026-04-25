//! Normalized client-facing chat-completion types.
//!
//! Field names intentionally mirror the common OpenAI-like shape, but these are normalized
//! client-layer types rather than provider-specific wire DTOs.
#![allow(missing_docs)]

mod request;
mod response;
mod shared;

pub use request::{
    ChatCompletionRequest, ChatMessage, TextMessage, ToolCallsMessage, ToolResultMessage,
};
pub use response::{
    AssistantMessage, AssistantRole, ChatChoice, ChatCompletionChunk, ChatCompletionChunkChoice,
    ChatCompletionResponse, DeltaMessage,
};
pub use shared::{
    ChatCompletionChunkToolCall, ChatCompletionLogprobs, ChatToolCall, CompletionTokensDetails,
    FinishReason, FunctionCall, FunctionCallDelta, FunctionDefinition, NamedToolChoice,
    NamedToolChoiceFunction, ResponseFormat, ResponseFormatType, StopSequence, StreamOptions,
    TokenLogprob, ToolChoice, ToolChoiceMode, ToolDefinition, ToolType, TopLogprob, Usage,
};

#[cfg(test)]
mod tests {
    use super::{
        AssistantMessage, AssistantRole, ChatChoice, ChatCompletionRequest, ChatCompletionResponse,
        ChatMessage, ChatToolCall, FunctionCall, ResponseFormat, ResponseFormatType, ToolChoice,
        ToolChoiceMode, ToolType,
    };

    #[test]
    fn serializes_custom_roles_and_explicit_tool_messages() {
        let messages = vec![
            ChatMessage::new("developer", "Keep answers short."),
            ChatMessage::assistant_tool_calls(vec![ChatToolCall {
                id: "call_1".to_owned(),
                kind: ToolType::Function,
                function: FunctionCall {
                    name: "lookup_weather".to_owned(),
                    arguments: "{\"city\":\"Shanghai\"}".to_owned(),
                },
            }]),
            ChatMessage::tool_result("{\"temperature\":26}", "call_1"),
        ];

        let json = serde_json::to_value(messages).unwrap();

        assert_eq!(json[0]["role"], "developer");
        assert_eq!(json[1]["role"], "assistant");
        assert_eq!(
            json[1]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"Shanghai\"}"
        );
        assert_eq!(json[2]["role"], "tool");
        assert_eq!(json[2]["tool_call_id"], "call_1");
    }

    #[test]
    fn chat_message_accessors_cover_all_variants() {
        let named = ChatMessage::named("developer", "Keep answers short.", "planner");
        let tool_call = ChatToolCall {
            id: "call_1".to_owned(),
            kind: ToolType::Function,
            function: FunctionCall {
                name: "lookup_weather".to_owned(),
                arguments: "{\"city\":\"Shanghai\"}".to_owned(),
            },
        };
        let assistant = ChatMessage::assistant_tool_calls_with_content(
            "Calling the weather tool.",
            vec![tool_call.clone()],
        );
        let tool_result = ChatMessage::tool_result("{\"temperature\":26}", "call_1");

        assert_eq!(named.role(), "developer");
        assert_eq!(named.content(), Some("Keep answers short."));
        assert_eq!(named.name(), Some("planner"));
        assert!(named.tool_calls().is_none());
        assert!(named.tool_call_id().is_none());

        assert_eq!(assistant.role(), "assistant");
        assert_eq!(assistant.content(), Some("Calling the weather tool."));
        assert_eq!(assistant.tool_calls(), Some([tool_call].as_slice()));
        assert!(assistant.tool_call_id().is_none());

        assert_eq!(tool_result.role(), "tool");
        assert_eq!(tool_result.content(), Some("{\"temperature\":26}"));
        assert_eq!(tool_result.tool_call_id(), Some("call_1"));
        assert!(tool_result.tool_calls().is_none());
    }

    #[test]
    fn request_helpers_preserve_explicit_request_shape() {
        let request = ChatCompletionRequest::new(
            "gpt-4.1-mini",
            vec![ChatMessage::user("Say hello in one sentence.")],
        )
        .with_temperature(0.2)
        .with_max_tokens(64)
        .with_tool_choice(ToolChoice::Mode(ToolChoiceMode::Auto))
        .with_response_format(ResponseFormat { kind: ResponseFormatType::Text })
        .with_system_prompt("You are a concise assistant.");

        assert_eq!(request.messages[0].role(), "system");
        assert_eq!(
            request.messages[0].content(),
            Some("You are a concise assistant.")
        );
        assert_eq!(request.messages[1].role(), "user");
        assert_eq!(request.temperature, Some(0.2));
        assert_eq!(request.max_tokens, Some(64));
        assert_eq!(
            request.tool_choice,
            Some(ToolChoice::Mode(ToolChoiceMode::Auto))
        );
        assert_eq!(
            request.response_format,
            Some(ResponseFormat { kind: ResponseFormatType::Text })
        );
    }

    #[test]
    fn repeated_system_prompt_helpers_preserve_insertion_order() {
        let request = ChatCompletionRequest::new(
            "gpt-4.1-mini",
            vec![
                ChatMessage::system("Existing system prompt."),
                ChatMessage::user("Say hello in one sentence."),
            ],
        )
        .with_system_prompt("First injected prompt.")
        .with_system_prompt("Second injected prompt.");

        assert_eq!(
            request
                .messages
                .iter()
                .map(|message| message.content().unwrap_or_default())
                .collect::<Vec<_>>(),
            vec![
                "Existing system prompt.",
                "First injected prompt.",
                "Second injected prompt.",
                "Say hello in one sentence.",
            ]
        );
    }

    #[test]
    fn response_accessors_expose_first_choice_content_and_tool_calls() {
        let response = ChatCompletionResponse {
            id: "chatcmpl-1".to_owned(),
            choices: vec![ChatChoice {
                finish_reason: None,
                index: 0,
                message: AssistantMessage {
                    content: Some("hello".to_owned()),
                    reasoning_content: Some("Let me think this through.".to_owned()),
                    tool_calls: Some(vec![ChatToolCall {
                        id: "call_1".to_owned(),
                        kind: ToolType::Function,
                        function: FunctionCall {
                            name: "lookup_weather".to_owned(),
                            arguments: "{\"city\":\"Shanghai\"}".to_owned(),
                        },
                    }]),
                    role: AssistantRole::Assistant,
                },
                logprobs: None,
            }],
            created: 1,
            model: "gpt-4.1-mini".to_owned(),
            system_fingerprint: None,
            object: "chat.completion".to_owned(),
            usage: None,
        };

        assert_eq!(response.first_choice().map(|choice| choice.index), Some(0));
        assert_eq!(response.first_choice_content(), Some("hello"));
        assert_eq!(
            response.first_choice_reasoning_content(),
            Some("Let me think this through.")
        );
        assert_eq!(
            response
                .first_choice_tool_calls()
                .and_then(|tool_calls| tool_calls.first())
                .map(|tool_call| tool_call.function.name.as_str()),
            Some("lookup_weather")
        );
    }
}
