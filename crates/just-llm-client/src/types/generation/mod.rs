//! Normalized client-facing generation types.
//!
//! These are semantic, protocol-neutral types: no provider wire shape is privileged. Backends map
//! between these and their provider-specific DTOs. Field names borrow the common LLM vocabulary
//! but the shapes are deliberately not a mirror of any single protocol.
#![allow(missing_docs)]

mod message;
mod request;
mod response;
mod shared;
mod stream;

pub use message::{
    AssistantMessage, ContentPart, ImageDetail, ImageSource, Message, MessageContent, Reasoning,
    ToolCall,
};
pub use request::GenerationRequest;
pub use response::{GenerationResponse, ResponseMeta};
pub use shared::{
    CompletionTokensDetails, FinishReason, FunctionDefinition, NamedToolChoice,
    NamedToolChoiceFunction, ReasoningEffort, ResponseFormat, ResponseFormatType, StopSequence,
    ToolChoice, ToolChoiceMode, ToolDefinition, ToolType, Usage,
};
pub use stream::{GenerationEvent, ToolCallDelta};

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        AssistantMessage, ContentPart, FinishReason, GenerationEvent, GenerationRequest,
        ImageDetail, ImageSource, Message, Reasoning, ReasoningEffort, ResponseFormat,
        ResponseFormatType, ToolCall, ToolCallDelta, ToolChoice, ToolChoiceMode,
    };

    #[test]
    fn serializes_messages_discriminated_by_role() {
        let messages = vec![
            Message::system("You are concise."),
            Message::user("Hello"),
            Message::assistant("Hi!"),
            Message::tool("{\"temp\":26}", "call_1"),
        ];

        let json = serde_json::to_value(messages).unwrap();

        assert_eq!(json[0]["role"], "system");
        assert_eq!(json[0]["content"], "You are concise.");
        assert_eq!(json[1]["role"], "user");
        assert_eq!(json[2]["role"], "assistant");
        assert_eq!(json[2]["content"], "Hi!");
        assert_eq!(json[3]["role"], "tool");
        assert_eq!(json[3]["tool_call_id"], "call_1");
    }

    #[test]
    fn reasoning_effort_serde_round_trips_xhigh() {
        assert_eq!(
            serde_json::to_value(ReasoningEffort::Xhigh).unwrap(),
            json!("xhigh")
        );
        assert_eq!(
            serde_json::from_value::<ReasoningEffort>(json!("xhigh")).unwrap(),
            ReasoningEffort::Xhigh
        );
    }

    #[test]
    fn message_accessors_cover_all_variants() {
        let system = Message::system("Be brief.");
        let user = Message::user("hello");
        let assistant = Message::assistant_tool_calls(
            Some("Calling the weather tool.".to_owned()),
            vec![ToolCall {
                id: "call_1".to_owned(),
                name: "get_weather".to_owned(),
                arguments: "{\"city\":\"Shanghai\"}".to_owned(),
            }],
            Some(Reasoning {
                text: Some("reasoned".to_owned()),
                id: None,
                encrypted: None,
                signature: None,
                redacted: None,
            }),
        );
        let tool = Message::tool("{\"temperature\":26}", "call_1");

        assert_eq!(system.role(), "system");
        assert_eq!(system.content(), Some("Be brief."));
        assert_eq!(user.role(), "user");
        assert_eq!(user.content(), Some("hello"));

        assert_eq!(assistant.role(), "assistant");
        assert_eq!(assistant.content(), Some("Calling the weather tool."));
        assert_eq!(assistant.tool_calls().len(), 1);
        assert_eq!(assistant.tool_calls()[0].name, "get_weather");
        assert_eq!(
            assistant.reasoning().and_then(|r| r.text.as_deref()),
            Some("reasoned")
        );

        assert_eq!(tool.role(), "tool");
        assert_eq!(tool.tool_call_id(), Some("call_1"));
        assert_eq!(tool.content(), Some("{\"temperature\":26}"));
        assert!(tool.tool_calls().is_empty());
    }

    #[test]
    fn serializes_multimodal_content_parts() {
        let message = Message::user_parts(vec![
            ContentPart::Text {
                text: "What is this?".to_owned(),
            },
            ContentPart::Image {
                source: ImageSource::Url {
                    url: "https://example.com/cat.png".to_owned(),
                },
                detail: Some(ImageDetail::High),
            },
        ]);

        let json = serde_json::to_value(&message).unwrap();

        assert_eq!(json["role"], "user");
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][1]["type"], "image");
        assert_eq!(
            json["content"][1]["source"]["url"],
            "https://example.com/cat.png"
        );
        assert_eq!(json["content"][1]["detail"], "high");
        assert!(matches!(
            message.content_parts(),
            Some(parts) if parts.len() == 2
        ));
        assert_eq!(message.content(), None);
    }

    #[test]
    fn serializes_base64_and_file_id_image_sources() {
        let message = Message::user_parts(vec![
            ContentPart::Image {
                source: ImageSource::Base64 {
                    data: "aGVsbG8=".to_owned(),
                    media_type: "image/png".to_owned(),
                },
                detail: None,
            },
            ContentPart::Image {
                source: ImageSource::FileId {
                    file_id: "file-123".to_owned(),
                },
                detail: None,
            },
        ]);

        let json = serde_json::to_value(&message).unwrap();

        assert_eq!(json["content"][0]["source"]["data"], "aGVsbG8=");
        assert_eq!(json["content"][0]["source"]["media_type"], "image/png");
        assert!(json["content"][0].get("detail").is_none());
        assert_eq!(json["content"][1]["source"]["file_id"], "file-123");
    }

    #[test]
    fn image_parts_round_trip_all_sources() {
        let parts = vec![
            ContentPart::Image {
                source: ImageSource::Url {
                    url: "https://example.com/cat.png".to_owned(),
                },
                detail: Some(ImageDetail::Low),
            },
            ContentPart::Image {
                source: ImageSource::Base64 {
                    data: "aGVsbG8=".to_owned(),
                    media_type: "image/jpeg".to_owned(),
                },
                detail: None,
            },
            ContentPart::Image {
                source: ImageSource::FileId {
                    file_id: "file-123".to_owned(),
                },
                detail: Some(ImageDetail::Original),
            },
        ];

        let json = serde_json::to_value(&parts).unwrap();
        let back: Vec<ContentPart> = serde_json::from_value(json).unwrap();
        assert_eq!(back, parts);
    }

    #[test]
    fn image_detail_preserves_unknown_values() {
        assert_eq!(
            serde_json::to_value(ImageDetail::Unknown("hd".to_owned())).unwrap(),
            json!("hd")
        );
        let back: ImageDetail = serde_json::from_value(json!("hd")).unwrap();
        assert_eq!(back, ImageDetail::Unknown("hd".to_owned()));

        for (wire, known) in [
            ("auto", ImageDetail::Auto),
            ("low", ImageDetail::Low),
            ("high", ImageDetail::High),
            ("original", ImageDetail::Original),
        ] {
            assert_eq!(serde_json::to_value(&known).unwrap(), json!(wire));
            let back: ImageDetail = serde_json::from_value(json!(wire)).unwrap();
            assert_eq!(back, known);
        }
    }

    #[test]
    fn request_helpers_preserve_explicit_request_shape() {
        let request = GenerationRequest::new(
            "gpt-4.1-mini",
            vec![Message::user("Say hello in one sentence.")],
        )
        .with_temperature(0.2)
        .with_max_tokens(64)
        .with_tool_choice(ToolChoice::Mode(ToolChoiceMode::Auto))
        .with_response_format(ResponseFormat {
            kind: ResponseFormatType::Text,
        })
        .with_system_prompt("You are a concise assistant.")
        .with_reasoning_effort(ReasoningEffort::High)
        .with_previous_response_id("resp_1")
        .with_store(true);

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
        assert_eq!(request.reasoning_effort, Some(ReasoningEffort::High));
        assert_eq!(request.previous_response_id, Some("resp_1".to_owned()));
        assert_eq!(request.store, Some(true));
    }

    #[test]
    fn repeated_system_prompt_helpers_preserve_insertion_order() {
        let request = GenerationRequest::new(
            "gpt-4.1-mini",
            vec![
                Message::system("Existing system prompt."),
                Message::user("Say hello in one sentence."),
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
    fn response_accessors_expose_text_tool_calls_and_reasoning() {
        let response = super::GenerationResponse {
            id: "gen-1".to_owned(),
            model: "gpt-4.1-mini".to_owned(),
            message: AssistantMessage {
                content: Some("hello".to_owned()),
                tool_calls: vec![ToolCall {
                    id: "call_1".to_owned(),
                    name: "get_weather".to_owned(),
                    arguments: "{\"city\":\"Paris\"}".to_owned(),
                }],
                reasoning: Some(Reasoning {
                    text: Some("Let me think this through.".to_owned()),
                    id: None,
                    encrypted: None,
                    signature: None,
                    redacted: None,
                }),
            },
            finish_reason: Some(FinishReason::Stop),
            usage: None,
            meta: None,
        };

        assert_eq!(response.text(), Some("hello"));
        assert_eq!(response.tool_calls()[0].name, "get_weather");
        assert_eq!(
            response.reasoning().and_then(|r| r.text.as_deref()),
            Some("Let me think this through.")
        );
    }

    #[test]
    fn reasoning_round_trips_fidelity_carriers() {
        let reasoning = Reasoning {
            text: Some("summarized".to_owned()),
            id: Some("rs_1".to_owned()),
            encrypted: Some("encrypted==".to_owned()),
            signature: Some("sig_1".to_owned()),
            redacted: Some("redacted".to_owned()),
        };

        let json = serde_json::to_value(&reasoning).unwrap();
        let back: Reasoning = serde_json::from_value(json).unwrap();
        assert_eq!(back, reasoning);
    }

    #[test]
    fn generation_events_round_trip() {
        let events = [
            GenerationEvent::Text {
                delta: "hi".to_owned(),
            },
            GenerationEvent::Reasoning {
                delta: "think".to_owned(),
            },
            GenerationEvent::ToolCall {
                delta: ToolCallDelta {
                    index: Some(0),
                    id: Some("call_1".to_owned()),
                    name: Some("get_weather".to_owned()),
                    arguments: Some("{\"city\":\"Paris\"}".to_owned()),
                },
            },
            GenerationEvent::End {
                finish_reason: Some(FinishReason::Refusal),
                response_id: Some("resp_1".to_owned()),
            },
        ];

        for event in events {
            let json = serde_json::to_value(&event).unwrap();
            let back: GenerationEvent = serde_json::from_value(json).unwrap();
            assert_eq!(back, event);
        }
    }

    #[test]
    fn finish_reason_new_variants_deserialize() {
        for (wire, expected) in [
            ("refusal", FinishReason::Refusal),
            (
                "model_context_window_exceeded",
                FinishReason::ModelContextWindowExceeded,
            ),
            ("pause_turn", FinishReason::PauseTurn),
        ] {
            let reason: FinishReason = serde_json::from_value(json!(wire)).unwrap();
            assert_eq!(reason, expected, "wire {wire:?}");
        }
    }

    #[test]
    fn text_only_assistant_message_round_trips() {
        let message = Message::assistant("Plain text reply with no tool calls.");

        let json = serde_json::to_value(&message).unwrap();
        assert!(json.get("tool_calls").is_none());

        let back: Message = serde_json::from_value(json).unwrap();
        assert_eq!(back, message);
    }
}
