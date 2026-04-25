//! OpenAI-compatible chat-completion DTOs.
//!
//! These types intentionally mirror the wire format closely. They are suitable when callers need
//! direct control over request payloads or access to provider-specific response fields.
#![allow(missing_docs)]

mod request;
mod response;
mod shared;

pub use request::{
    ChatCompletionRequest, ChatMessage, CreateChatCompletionRequest, TextMessage, ToolCallsMessage,
    ToolResultMessage,
};
pub use response::{
    AssistantMessage, AssistantRole, ChatCompletion, ChatCompletionChoice, ChatCompletionChunk,
    ChatCompletionChunkChoice, DeltaMessage,
};
pub use shared::{
    ChatCompletionChunkToolCall, ChatCompletionLogprobs, ChatCompletionToolCall,
    CompletionTokensDetails, FinishReason, FunctionCall, FunctionCallDelta, FunctionDefinition,
    NamedToolChoice, NamedToolChoiceFunction, ResponseFormat, ResponseFormatType, StopSequence,
    StreamOptions, TokenLogprob, ToolChoice, ToolChoiceMode, ToolDefinition, ToolType, TopLogprob,
    Usage,
};

#[cfg(test)]
mod tests {
    use super::{
        ChatCompletionToolCall, ChatMessage, CreateChatCompletionRequest, FunctionCall,
        StopSequence, ToolChoice, ToolChoiceMode, ToolType,
    };

    #[test]
    fn serializes_minimal_request() {
        let request = CreateChatCompletionRequest::new(
            "gpt-4.1-mini",
            vec![ChatMessage::system("You are helpful."), ChatMessage::user("Hi")],
        );

        let json = serde_json::to_value(request).unwrap();

        assert_eq!(json["model"], "gpt-4.1-mini");
        assert_eq!(json["messages"][0]["role"], "system");
        assert_eq!(json["messages"][1]["role"], "user");
        assert!(json.get("stream").is_none());
    }

    #[test]
    fn serializes_untagged_variants() {
        let mut request =
            CreateChatCompletionRequest::new("gpt-4.1-mini", vec![ChatMessage::user("Hi")]);
        request.stop = Some(StopSequence::Multiple(vec!["END".to_owned()]));
        request.tool_choice = Some(ToolChoice::Mode(ToolChoiceMode::Auto));

        let json = serde_json::to_value(request).unwrap();

        assert_eq!(json["stop"][0], "END");
        assert_eq!(json["tool_choice"], "auto");
    }

    #[test]
    fn serializes_custom_roles_and_explicit_tool_messages() {
        let messages = vec![
            ChatMessage::new("developer", "Keep answers short."),
            ChatMessage::assistant_tool_calls(vec![ChatCompletionToolCall {
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
}
