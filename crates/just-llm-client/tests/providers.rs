#[cfg(feature = "openai-compat")]
use just_llm_client::error::Capability;

#[cfg(feature = "openai-compat")]
use futures_util::StreamExt;
#[cfg(any(feature = "deepseek", feature = "openai-compat"))]
use just_llm_client::CapabilityNegotiation;
#[cfg(feature = "deepseek")]
use just_llm_client::provider::DeepSeekBackend;
#[cfg(feature = "openai-compat")]
use just_llm_client::provider::OpenAiCompatBackend;
#[cfg(feature = "deepseek")]
use just_llm_client::types::chat::{ToolChoice, ToolChoiceMode};
#[cfg(any(feature = "deepseek", feature = "openai-compat"))]
use just_llm_client::{
    LlmBackend,
    error::LlmError,
    types::chat::{ChatCompletionRequest, ChatMessage},
};
#[cfg(any(feature = "deepseek", feature = "openai-compat"))]
use serde_json::json;
#[cfg(any(feature = "deepseek", feature = "openai-compat"))]
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

#[cfg(feature = "deepseek")]
fn deepseek_backend(server: &MockServer) -> DeepSeekBackend {
    let client = just_deepseek::DeepSeekClient::builder()
        .api_key("test-key")
        .base_url(server.uri())
        .build()
        .unwrap();
    DeepSeekBackend::new(client)
}

#[cfg(feature = "openai-compat")]
fn openai_backend(server: &MockServer) -> OpenAiCompatBackend {
    let client = just_openai_compat::OpenAiCompatClient::builder()
        .api_key("test-key")
        .base_url(server.uri())
        .build()
        .unwrap();
    OpenAiCompatBackend::new(client)
}

// --- DeepSeek tests ---

#[cfg(feature = "deepseek")]
#[tokio::test]
async fn deepseek_adapter_maps_chat_and_balance() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "hello"
                    }
                }
            ],
            "usage": {
                "completion_tokens": 1,
                "prompt_tokens": 2,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 2,
                "total_tokens": 3
            }
        })))
        .mount(&server)
        .await;

    Mock::given(method("GET"))
        .and(path("/user/balance"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "is_available": true,
            "balance_infos": [
                {
                    "currency": "USD",
                    "total_balance": "10.00",
                    "granted_balance": "1.00",
                    "topped_up_balance": "9.00"
                }
            ]
        })))
        .mount(&server)
        .await;

    let backend = deepseek_backend(&server);
    let response = backend
        .chat_completion(ChatCompletionRequest::new(
            "deepseek-v4-pro",
            vec![ChatMessage::user("hello")],
        ))
        .await
        .unwrap();
    let balance = backend.balance().unwrap().get_balance().await.unwrap();

    assert_eq!(response.first_choice_content(), Some("hello"));
    assert!(balance.is_available);
}

#[cfg(feature = "deepseek")]
#[tokio::test]
async fn preparation_rejects_invalid_request_combinations() {
    let server = MockServer::start().await;
    let backend = deepseek_backend(&server);
    let mut request = ChatCompletionRequest::new("deepseek-v4-pro", vec![ChatMessage::user("x")]);
    request.tool_choice = Some(ToolChoice::Mode(ToolChoiceMode::Auto));

    let error = backend.prepare(request).unwrap_err();

    assert!(matches!(error, LlmError::InvalidRequest(_)));
}

#[cfg(feature = "deepseek")]
#[tokio::test]
async fn deepseek_adapter_preserves_cache_usage_when_reported() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-5",
            "object": "chat.completion",
            "created": 1,
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "hello"
                    }
                }
            ],
            "usage": {
                "completion_tokens": 1,
                "prompt_tokens": 2,
                "prompt_cache_hit_tokens": 5,
                "prompt_cache_miss_tokens": 7,
                "total_tokens": 3
            }
        })))
        .mount(&server)
        .await;

    let backend = deepseek_backend(&server);
    let response = backend
        .chat_completion(ChatCompletionRequest::new(
            "deepseek-v4-pro",
            vec![ChatMessage::user("hello")],
        ))
        .await
        .unwrap();

    let usage = response.usage.expect("usage should be present");
    assert_eq!(usage.prompt_cache_hit_tokens, Some(5));
    assert_eq!(usage.prompt_cache_miss_tokens, Some(7));
}

// --- OpenAI-compatible tests ---

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn openai_compat_adapter_maps_models_and_marks_balance_unsupported() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [
                {
                    "id": "gpt-4.1-mini",
                    "object": "model",
                    "owned_by": "example"
                }
            ]
        })))
        .mount(&server)
        .await;

    let backend = openai_backend(&server);
    let models = backend
        .model_catalog()
        .unwrap()
        .list_models()
        .await
        .unwrap();
    let error = match backend.balance() {
        Ok(_) => panic!("balance negotiation should fail for openai-compatible"),
        Err(error) => error,
    };

    assert_eq!(models.data[0].id, "gpt-4.1-mini");
    assert!(matches!(
        error,
        LlmError::UnsupportedCapability {
            backend: "openai-compatible",
            capability: Capability::Balance,
        }
    ));
}

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn prepared_requests_can_be_previewed_and_executed() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-2",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4.1-mini",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "prepared"
                    }
                }
            ]
        })))
        .mount(&server)
        .await;

    let backend = openai_backend(&server);
    let prepared = backend
        .prepare(ChatCompletionRequest::new(
            "gpt-4.1-mini",
            vec![ChatMessage::user("hello from prepared")],
        ))
        .unwrap();

    assert_eq!(prepared.backend_id(), "openai-compatible");
    assert_eq!(prepared.model(), Some("gpt-4.1-mini"));
    assert_eq!(prepared.message_count(), 1);
    assert!(prepared.request_body_text().contains("\"messages\""));

    let response = backend.send(&prepared).await.unwrap();

    assert_eq!(response.first_choice_content(), Some("prepared"));
}

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn prepared_requests_round_trip_and_execute_through_backend() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-roundtrip",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4.1-mini",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "round trip"
                    }
                }
            ]
        })))
        .mount(&server)
        .await;

    let backend = openai_backend(&server);
    let prepared = backend
        .prepare(ChatCompletionRequest::new(
            "gpt-4.1-mini",
            vec![ChatMessage::user("round trip")],
        ))
        .unwrap();

    // Verify field accessors are consistent.
    assert_eq!(prepared.backend_id(), "openai-compatible");
    assert_eq!(prepared.model(), Some("gpt-4.1-mini"));
    assert_eq!(prepared.message_count(), 1);
    assert!(!prepared.has_stream());
    assert_eq!(prepared.headers().len(), 0);

    let response = backend.send(&prepared).await.unwrap();

    assert_eq!(response.first_choice_content(), Some("round trip"));
}

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn chat_completion_rejects_streaming_requests() {
    let server = MockServer::start().await;
    let backend = openai_backend(&server);
    let mut request = ChatCompletionRequest::new("gpt-4.1-mini", vec![ChatMessage::user("x")]);
    request.stream = Some(true);

    let error = backend.chat_completion(request).await.unwrap_err();

    assert!(matches!(error, LlmError::InvalidRequest(_)));
}

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn stream_chat_completion_promotes_stream_flag() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(
                    "data: {\"id\":\"chatcmpl-3\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"}}]}\n\ndata: [DONE]\n",
                    "text/event-stream",
                ),
        )
        .mount(&server)
        .await;

    let backend = openai_backend(&server);
    let _stream = backend
        .stream_chat_completion(ChatCompletionRequest::new(
            "gpt-4.1-mini",
            vec![ChatMessage::user("stream please")],
        ))
        .await
        .unwrap();
}

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn prepared_streaming_requests_can_be_previewed_and_executed() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(
                    "data: {\"id\":\"chatcmpl-stream-prepared\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n",
                    "text/event-stream",
                ),
        )
        .mount(&server)
        .await;

    let backend = openai_backend(&server);
    let prepared = backend
        .prepare_streaming(ChatCompletionRequest::new(
            "gpt-4.1-mini",
            vec![ChatMessage::user("stream please")],
        ))
        .unwrap();

    assert!(prepared.has_stream());
    assert_eq!(prepared.headers().len(), 0);

    let mut stream = backend.send_streaming(&prepared).await.unwrap();
    let first = stream.next().await.unwrap().unwrap();

    assert_eq!(first.choices[0].delta.content.as_deref(), Some("hi"));
}

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn non_stream_sender_rejects_prepared_streaming_requests() {
    let server = MockServer::start().await;
    let backend = openai_backend(&server);
    let prepared = backend
        .prepare_streaming(ChatCompletionRequest::new(
            "gpt-4.1-mini",
            vec![ChatMessage::user("stream please")],
        ))
        .unwrap();

    let error = backend.send(&prepared).await.unwrap_err();

    assert!(matches!(error, LlmError::InvalidRequest(_)));
}

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn stream_sender_rejects_non_stream_prepared_requests() {
    let server = MockServer::start().await;
    let backend = openai_backend(&server);
    let prepared = backend
        .prepare(ChatCompletionRequest::new(
            "gpt-4.1-mini",
            vec![ChatMessage::user("hello")],
        ))
        .unwrap();

    let error = match backend.send_streaming(&prepared).await {
        Ok(_) => panic!("stream sender should reject non-stream prepared requests"),
        Err(error) => error,
    };

    assert!(matches!(error, LlmError::InvalidRequest(_)));
}

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn openai_compat_adapter_leaves_unknown_cache_usage_empty() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-4",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4.1-mini",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "hello"
                    }
                }
            ],
            "usage": {
                "completion_tokens": 1,
                "prompt_tokens": 2,
                "total_tokens": 3
            }
        })))
        .mount(&server)
        .await;

    let backend = openai_backend(&server);
    let response = backend
        .chat_completion(ChatCompletionRequest::new(
            "gpt-4.1-mini",
            vec![ChatMessage::user("hello")],
        ))
        .await
        .unwrap();

    let usage = response.usage.expect("usage should be present");
    assert_eq!(usage.prompt_cache_hit_tokens, None);
    assert_eq!(usage.prompt_cache_miss_tokens, None);
}

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn openai_compat_adapter_maps_streaming_tool_call_deltas() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(
                    concat!(
                        "data: {\"id\":\"chatcmpl-tool-2\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"lookup_weather\",\"arguments\":\"\"}}]}}]}\n\n",
                        "data: {\"id\":\"chatcmpl-tool-2\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\\\"Shanghai\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
                        "data: [DONE]\n\n"
                    ),
                    "text/event-stream",
                ),
        )
        .mount(&server)
        .await;

    let backend = openai_backend(&server);
    let mut stream = backend
        .stream_chat_completion(ChatCompletionRequest::new(
            "gpt-4.1-mini",
            vec![ChatMessage::user("use tools")],
        ))
        .await
        .unwrap();

    let first = stream.next().await.unwrap().unwrap();
    let second = stream.next().await.unwrap().unwrap();

    let first_function = &first.choices[0].delta.tool_calls.as_ref().unwrap()[0]
        .function
        .as_ref()
        .unwrap();
    assert_eq!(first_function.name.as_deref(), Some("lookup_weather"));
    assert_eq!(first_function.arguments.as_deref(), Some(""));

    let second_function = &second.choices[0].delta.tool_calls.as_ref().unwrap()[0]
        .function
        .as_ref()
        .unwrap();
    assert_eq!(second_function.name, None);
    assert_eq!(
        second_function.arguments.as_deref(),
        Some("{\"city\":\"Shanghai\"}")
    );
}

// --- Cross-backend tests ---

#[cfg(all(feature = "deepseek", feature = "openai-compat"))]
#[tokio::test]
async fn prepared_requests_reject_cross_backend_execution() {
    let server = MockServer::start().await;
    let deepseek = deepseek_backend(&server);
    let openai = openai_backend(&server);
    let prepared = deepseek
        .prepare(ChatCompletionRequest::new(
            "deepseek-v4-pro",
            vec![ChatMessage::user("cross backend")],
        ))
        .unwrap();

    let error = openai.send(&prepared).await.unwrap_err();

    assert!(matches!(error, LlmError::InvalidRequest(_)));
}

// --- Provider-agnostic tests ---

#[test]
fn prepared_requests_reject_invalid_payloads() {
    use just_llm_client::types::prepared::PreparedChatRequest;
    use serde_json::json;

    // Missing messages array.
    let error = PreparedChatRequest::from_request_body(
        "openai-compatible",
        json!({"model": "gpt-4.1-mini"}),
    )
    .unwrap_err();

    assert!(error.to_string().contains("messages array"));

    // Missing model field.
    let error = PreparedChatRequest::from_request_body(
        "openai-compatible",
        json!({"messages": [{"role": "user", "content": "hi"}]}),
    )
    .unwrap_err();

    assert!(error.to_string().contains("model field"));

    // Body is not a JSON object.
    let error = PreparedChatRequest::from_request_body("openai-compatible", json!(42)).unwrap_err();

    assert!(error.to_string().contains("JSON object"));
}

// --- Extra headers tests ---

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn prepared_requests_send_extra_headers_to_backend() {
    use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(wiremock::matchers::header("x-trace-id", "test-trace-123"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-hdr",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4.1-mini",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "with headers"
                    }
                }
            ]
        })))
        .mount(&server)
        .await;

    let backend = openai_backend(&server);

    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static("x-trace-id"),
        HeaderValue::from_static("test-trace-123"),
    );

    let prepared = backend
        .prepare(ChatCompletionRequest::new(
            "gpt-4.1-mini",
            vec![ChatMessage::user("hello")],
        ))
        .unwrap()
        .with_headers(headers);

    assert_eq!(prepared.headers().len(), 1);

    let response = backend.send(&prepared).await.unwrap();
    assert_eq!(response.first_choice_content(), Some("with headers"));
}

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn prepared_requests_work_without_extra_headers() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-nohdr",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4.1-mini",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "no headers"
                    }
                }
            ]
        })))
        .mount(&server)
        .await;

    let backend = openai_backend(&server);
    let prepared = backend
        .prepare(ChatCompletionRequest::new(
            "gpt-4.1-mini",
            vec![ChatMessage::user("hello")],
        ))
        .unwrap();

    assert_eq!(prepared.headers().len(), 0);

    let response = backend.send(&prepared).await.unwrap();
    assert_eq!(response.first_choice_content(), Some("no headers"));
}
