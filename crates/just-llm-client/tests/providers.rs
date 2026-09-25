#[cfg(feature = "openai-compat")]
use just_llm_client::error::{Capability, CapabilityError};

#[cfg(any(
    feature = "openai-compat",
    feature = "responses",
    feature = "anthropic"
))]
use futures_util::StreamExt;
#[cfg(feature = "anthropic")]
use just_llm_client::provider::AnthropicBackend;
#[cfg(feature = "deepseek")]
use just_llm_client::provider::DeepSeekBackend;
#[cfg(feature = "openai-compat")]
use just_llm_client::provider::OpenAiCompatBackend;
#[cfg(feature = "responses")]
use just_llm_client::provider::OpenAiResponsesBackend;
#[cfg(feature = "anthropic")]
use just_llm_client::types::generation::MessageContent;
#[cfg(feature = "responses")]
use just_llm_client::types::generation::StopSequence;
#[cfg(any(
    feature = "deepseek",
    feature = "openai-compat",
    feature = "responses",
    feature = "anthropic"
))]
use just_llm_client::types::generation::ToolCall;
#[cfg(any(feature = "deepseek", feature = "openai-compat"))]
use just_llm_client::types::generation::{FunctionDefinition, ToolDefinition, ToolType};
#[cfg(feature = "deepseek")]
use just_llm_client::types::generation::{ToolChoice, ToolChoiceMode};
#[cfg(any(
    feature = "deepseek",
    feature = "openai-compat",
    feature = "responses",
    feature = "anthropic"
))]
use just_llm_client::{
    LlmBackend, captured_body,
    error::BackendError,
    provider_rejection,
    types::generation::{GenerationEvent, GenerationRequest, Message},
};
#[cfg(any(
    feature = "deepseek",
    feature = "openai-compat",
    feature = "responses",
    feature = "anthropic"
))]
use serde_json::json;
#[cfg(any(
    feature = "deepseek",
    feature = "openai-compat",
    feature = "responses",
    feature = "anthropic"
))]
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

#[cfg(any(
    feature = "deepseek",
    feature = "openai-compat",
    feature = "responses",
    feature = "anthropic"
))]
use just_llm_client::types::generation::{ContentPart, ImageDetail, ImageSource};
#[cfg(any(
    feature = "deepseek",
    feature = "openai-compat",
    feature = "responses",
    feature = "anthropic"
))]
use std::sync::Arc;

#[cfg(feature = "deepseek")]
fn deepseek_backend(server: &MockServer) -> Arc<dyn LlmBackend> {
    let uri = server.uri();
    DeepSeekBackend::new(
        reqwest::Client::builder().use_rustls_tls(),
        "test-key",
        Some(&uri),
    )
    .expect("failed to build deepseek backend")
}

#[cfg(feature = "deepseek")]
fn deepseek_backend_no_server() -> Arc<dyn LlmBackend> {
    DeepSeekBackend::new(
        reqwest::Client::builder().use_rustls_tls(),
        "test-key",
        Some("http://127.0.0.1:0"),
    )
    .expect("failed to build deepseek backend")
}

#[cfg(feature = "openai-compat")]
fn openai_backend(server: &MockServer) -> Arc<dyn LlmBackend> {
    let uri = server.uri();
    OpenAiCompatBackend::new(
        reqwest::Client::builder().use_rustls_tls(),
        "test-key",
        Some(&uri),
    )
    .expect("failed to build openai backend")
}

#[cfg(feature = "openai-compat")]
fn openai_backend_no_server() -> Arc<dyn LlmBackend> {
    OpenAiCompatBackend::new(
        reqwest::Client::builder().use_rustls_tls(),
        "test-key",
        Some("http://127.0.0.1:0"),
    )
    .expect("failed to build openai backend")
}

// --- DeepSeek tests ---

#[cfg(feature = "deepseek")]
#[tokio::test]
async fn deepseek_adapter_maps_generation_and_balance() {
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
        .generate(GenerationRequest::new(
            "deepseek-v4-pro",
            vec![Message::user("hello")],
        ))
        .await
        .unwrap();
    let balance = backend.balance().unwrap().get_balance().await.unwrap();

    assert_eq!(response.text(), Some("hello"));
    assert!(balance.is_available);
}

#[cfg(feature = "deepseek")]
#[tokio::test]
async fn preparation_rejects_invalid_request_combinations() {
    let server = MockServer::start().await;
    let backend = deepseek_backend(&server);
    let mut request = GenerationRequest::new("deepseek-v4-pro", vec![Message::user("x")]);
    request.tool_choice = Some(ToolChoice::Mode(ToolChoiceMode::Auto));

    let error = backend.prepare(request).unwrap_err();

    assert!(matches!(error, BackendError::InvalidRequest(_)));
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
        .generate(GenerationRequest::new(
            "deepseek-v4-pro",
            vec![Message::user("hello")],
        ))
        .await
        .unwrap();

    let usage = response.usage.expect("usage should be present");
    assert_eq!(usage.cache_read_tokens, Some(5));
    assert_eq!(usage.cache_write_tokens, Some(7));
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
        CapabilityError::Unsupported {
            family: "openai-compatible",
            capability: Capability::Balance,
        }
    ));
}

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn prepare_send_returns_raw_response_with_accessible_headers() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "30")
                .set_body_json(json!({
                    "error": {"message": "rate limited", "type": "rate_limit_error"}
                })),
        )
        .mount(&server)
        .await;

    let backend = openai_backend(&server);
    let builder = backend
        .prepare(GenerationRequest::new(
            "gpt-4.1-mini",
            vec![Message::user("hello")],
        ))
        .unwrap();

    // send() returns the raw reqwest::Response — headers are accessible.
    let response = backend.send(builder).await.unwrap();

    assert_eq!(response.status(), reqwest::StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response.headers().get("retry-after").unwrap(),
        reqwest::header::HeaderValue::from_static("30")
    );
}

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn prepare_send_parse_roundtrips_normalized_response() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-rt",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4.1-mini",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hi"}
            }]
        })))
        .mount(&server)
        .await;

    let backend = openai_backend(&server);
    let prepared = backend
        .prepare(GenerationRequest::new(
            "gpt-4.1-mini",
            vec![Message::user("hello")],
        ))
        .unwrap();
    let response = backend.send(prepared).await.unwrap();
    assert!(response.status().is_success());

    // parse deserializes via dyn dispatch on the right backend.
    let generation = backend.parse(response).await.unwrap();
    assert_eq!(generation.text(), Some("hi"));
}

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn parse_surfaces_http_status_for_error_response() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "30")
                .set_body_json(json!({
                    "error": {"message": "rate limited", "type": "rate_limit_error"}
                })),
        )
        .mount(&server)
        .await;

    let backend = openai_backend(&server);
    let prepared = backend
        .prepare(GenerationRequest::new(
            "gpt-4.1-mini",
            vec![Message::user("hello")],
        ))
        .unwrap();
    let response = backend.send(prepared).await.unwrap();
    // Headers are still readable before parse consumes the body.
    assert_eq!(
        response.headers().get("retry-after").unwrap(),
        reqwest::header::HeaderValue::from_static("30")
    );

    // parse runs ensure_success; the backend wraps the provider's ProviderError, whose Transport
    // variant carries the HttpStatus.
    let error = backend.parse(response).await.unwrap_err();
    let BackendError::Provider { family: id, source } = error else {
        panic!("expected BackendError::Provider");
    };
    assert_eq!(id, "openai-compatible");
    let provider_err = source
        .downcast_ref::<just_common::error::ProviderError>()
        .expect("source should be a ProviderError");
    assert!(matches!(
        provider_err,
        just_common::error::ProviderError::Transport(
            just_common::error::TransportError::HttpStatus { status, .. }
        ) if *status == reqwest::StatusCode::TOO_MANY_REQUESTS
    ));
}

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn parse_surfaces_deserialize_error_for_malformed_body() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not valid json"))
        .mount(&server)
        .await;

    let backend = openai_backend(&server);
    let prepared = backend
        .prepare(GenerationRequest::new(
            "gpt-4.1-mini",
            vec![Message::user("hello")],
        ))
        .unwrap();
    let response = backend.send(prepared).await.unwrap();

    // A 2xx body that fails to deserialize surfaces as BackendError::Provider { .. }.
    let error = backend.parse(response).await.unwrap_err();
    let BackendError::Provider { source, .. } = error else {
        panic!("expected BackendError::Provider");
    };
    let provider_err = source
        .downcast_ref::<just_common::error::ProviderError>()
        .expect("source should be a ProviderError");
    assert!(matches!(
        provider_err,
        just_common::error::ProviderError::Deserialize { .. }
    ));
}

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn parse_streaming_yields_normalized_events() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(
                    "data: {\"id\":\"chatcmpl-s\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4.1-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n",
                    "text/event-stream",
                ),
        )
        .mount(&server)
        .await;

    let backend = openai_backend(&server);
    let prepared = backend
        .prepare_streaming(GenerationRequest::new(
            "gpt-4.1-mini",
            vec![Message::user("hi")],
        ))
        .unwrap();
    let response = backend.send(prepared).await.unwrap();

    let mut stream = backend.parse_streaming(response).await.unwrap();
    let event = stream.next().await.unwrap().unwrap();
    assert!(matches!(
        event,
        GenerationEvent::Text { delta } if delta == "hi"
    ));
}

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn generate_rejects_streaming_requests() {
    let server = MockServer::start().await;
    let backend = openai_backend(&server);
    let mut request = GenerationRequest::new("gpt-4.1-mini", vec![Message::user("x")]);
    request.stream = Some(true);

    let error = backend.generate(request).await.unwrap_err();

    assert!(matches!(error, BackendError::InvalidRequest(_)));
}

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn stream_generate_promotes_stream_flag() {
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
    let mut stream = backend
        .stream_generate(GenerationRequest::new(
            "gpt-4.1-mini",
            vec![Message::user("stream please")],
        ))
        .await
        .unwrap();

    let event = stream.next().await.unwrap().unwrap();
    assert!(matches!(
        event,
        GenerationEvent::Text { delta } if delta == "hi"
    ));
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
        .generate(GenerationRequest::new(
            "gpt-4.1-mini",
            vec![Message::user("hello")],
        ))
        .await
        .unwrap();

    let usage = response.usage.expect("usage should be present");
    assert_eq!(usage.cache_read_tokens, None);
    assert_eq!(usage.cache_write_tokens, None);
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
        .stream_generate(GenerationRequest::new(
            "gpt-4.1-mini",
            vec![Message::user("use tools")],
        ))
        .await
        .unwrap();

    let first = stream.next().await.unwrap().unwrap();
    let second = stream.next().await.unwrap().unwrap();
    let end = stream.next().await.unwrap().unwrap();

    let GenerationEvent::ToolCall { delta: first_delta } = first else {
        panic!("expected ToolCall event, got {first:?}");
    };
    assert_eq!(first_delta.id.as_deref(), Some("call_1"));
    assert_eq!(first_delta.name.as_deref(), Some("lookup_weather"));
    assert_eq!(first_delta.arguments.as_deref(), Some(""));

    let GenerationEvent::ToolCall {
        delta: second_delta,
    } = second
    else {
        panic!("expected ToolCall event, got {second:?}");
    };
    assert_eq!(second_delta.name, None);
    assert_eq!(
        second_delta.arguments.as_deref(),
        Some("{\"city\":\"Shanghai\"}")
    );

    assert!(matches!(
        end,
        GenerationEvent::End {
            finish_reason: Some(just_llm_client::types::generation::FinishReason::ToolCalls),
            ..
        }
    ));
}

// --- render_messages / render_tools tests ---

#[cfg(feature = "deepseek")]
#[test]
fn deepseek_render_messages_produces_provider_json() {
    let backend = deepseek_backend_no_server();
    let messages = vec![Message::system("You are helpful."), Message::user("Hello")];

    let json = backend.render_messages(&messages).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.as_array().unwrap().len(), 2);
    assert_eq!(parsed[0]["role"], "system");
    assert_eq!(parsed[0]["content"], "You are helpful.");
    assert_eq!(parsed[1]["role"], "user");
    assert_eq!(parsed[1]["content"], "Hello");
}

#[cfg(feature = "deepseek")]
#[test]
fn deepseek_render_tools_produces_provider_json() {
    let backend = deepseek_backend_no_server();
    let tools = vec![ToolDefinition {
        kind: ToolType::Function,
        function: FunctionDefinition {
            name: "get_weather".to_owned(),
            description: Some("Get weather".to_owned()),
            parameters: None,
            strict: None,
        },
    }];

    let json = backend.render_tools(&tools).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.as_array().unwrap().len(), 1);
    assert_eq!(parsed[0]["type"], "function");
    assert_eq!(parsed[0]["function"]["name"], "get_weather");
}

#[cfg(feature = "deepseek")]
#[test]
fn deepseek_render_messages_empty_slice_returns_empty_array() {
    let backend = deepseek_backend_no_server();
    let json = backend.render_messages(&[]).unwrap();
    assert_eq!(json, "[]");
}

#[cfg(feature = "openai-compat")]
#[test]
fn openai_compat_render_messages_produces_provider_json() {
    let backend = openai_backend_no_server();
    let messages = vec![Message::system("You are helpful."), Message::user("Hello")];

    let json = backend.render_messages(&messages).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.as_array().unwrap().len(), 2);
    assert_eq!(parsed[0]["role"], "system");
    assert_eq!(parsed[0]["content"], "You are helpful.");
    assert_eq!(parsed[1]["role"], "user");
    assert_eq!(parsed[1]["content"], "Hello");
}

#[cfg(feature = "openai-compat")]
#[test]
fn openai_compat_render_tools_produces_provider_json() {
    let backend = openai_backend_no_server();
    let tools = vec![ToolDefinition {
        kind: ToolType::Function,
        function: FunctionDefinition {
            name: "get_weather".to_owned(),
            description: Some("Get weather".to_owned()),
            parameters: None,
            strict: None,
        },
    }];

    let json = backend.render_tools(&tools).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.as_array().unwrap().len(), 1);
    assert_eq!(parsed[0]["type"], "function");
    assert_eq!(parsed[0]["function"]["name"], "get_weather");
}

#[cfg(feature = "openai-compat")]
#[test]
fn openai_compat_render_messages_empty_slice_returns_empty_array() {
    let backend = openai_backend_no_server();
    let json = backend.render_messages(&[]).unwrap();
    assert_eq!(json, "[]");
}

// --- render_messages with tool-call variants ---

#[cfg(feature = "deepseek")]
#[test]
fn deepseek_render_messages_with_tool_calls() {
    let backend = deepseek_backend_no_server();
    let messages = vec![Message::assistant_tool_calls(
        None,
        vec![ToolCall {
            id: "call_1".to_owned(),
            name: "get_weather".to_owned(),
            arguments: "{\"city\":\"Shanghai\"}".to_owned(),
        }],
        None,
    )];

    let json = backend.render_messages(&messages).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed[0]["role"], "assistant");
    assert_eq!(parsed[0]["tool_calls"][0]["id"], "call_1");
    assert_eq!(parsed[0]["tool_calls"][0]["type"], "function");
    assert_eq!(
        parsed[0]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
    assert_eq!(
        parsed[0]["tool_calls"][0]["function"]["arguments"],
        "{\"city\":\"Shanghai\"}"
    );
}

#[cfg(feature = "deepseek")]
#[test]
fn deepseek_render_messages_with_tool_result() {
    let backend = deepseek_backend_no_server();
    let messages = vec![Message::tool("{\"temperature\":26}", "call_1")];

    let json = backend.render_messages(&messages).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed[0]["role"], "tool");
    assert_eq!(parsed[0]["content"], "{\"temperature\":26}");
    assert_eq!(parsed[0]["tool_call_id"], "call_1");
}

#[cfg(feature = "deepseek")]
#[test]
fn deepseek_render_tools_with_parameters() {
    let backend = deepseek_backend_no_server();
    let tools = vec![ToolDefinition {
        kind: ToolType::Function,
        function: FunctionDefinition {
            name: "get_weather".to_owned(),
            description: Some("Get current weather".to_owned()),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            })),
            strict: None,
        },
    }];

    let json = backend.render_tools(&tools).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed[0]["function"]["parameters"]["type"], "object");
    assert_eq!(parsed[0]["function"]["parameters"]["required"][0], "city");
    assert!(parsed[0]["function"]["parameters"]["properties"]["city"].is_object());
}

#[cfg(feature = "deepseek")]
#[test]
fn deepseek_render_tools_empty_slice_returns_empty_array() {
    let backend = deepseek_backend_no_server();
    let json = backend.render_tools(&[]).unwrap();
    assert_eq!(json, "[]");
}

#[cfg(feature = "openai-compat")]
#[test]
fn openai_compat_render_messages_with_tool_calls() {
    let backend = openai_backend_no_server();
    let messages = vec![Message::assistant_tool_calls(
        None,
        vec![ToolCall {
            id: "call_1".to_owned(),
            name: "get_weather".to_owned(),
            arguments: "{\"city\":\"Shanghai\"}".to_owned(),
        }],
        None,
    )];

    let json = backend.render_messages(&messages).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed[0]["role"], "assistant");
    assert_eq!(parsed[0]["tool_calls"][0]["id"], "call_1");
    assert_eq!(parsed[0]["tool_calls"][0]["type"], "function");
    assert_eq!(
        parsed[0]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
    assert_eq!(
        parsed[0]["tool_calls"][0]["function"]["arguments"],
        "{\"city\":\"Shanghai\"}"
    );
}

#[cfg(feature = "openai-compat")]
#[test]
fn openai_compat_render_messages_with_tool_result() {
    let backend = openai_backend_no_server();
    let messages = vec![Message::tool("{\"temperature\":26}", "call_1")];

    let json = backend.render_messages(&messages).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed[0]["role"], "tool");
    assert_eq!(parsed[0]["content"], "{\"temperature\":26}");
    assert_eq!(parsed[0]["tool_call_id"], "call_1");
}

#[cfg(feature = "openai-compat")]
#[test]
fn openai_compat_render_tools_with_parameters() {
    let backend = openai_backend_no_server();
    let tools = vec![ToolDefinition {
        kind: ToolType::Function,
        function: FunctionDefinition {
            name: "get_weather".to_owned(),
            description: Some("Get current weather".to_owned()),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            })),
            strict: None,
        },
    }];

    let json = backend.render_tools(&tools).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed[0]["function"]["parameters"]["type"], "object");
    assert_eq!(parsed[0]["function"]["parameters"]["required"][0], "city");
    assert!(parsed[0]["function"]["parameters"]["properties"]["city"].is_object());
}

#[cfg(feature = "openai-compat")]
#[test]
fn openai_compat_render_tools_empty_slice_returns_empty_array() {
    let backend = openai_backend_no_server();
    let json = backend.render_tools(&[]).unwrap();
    assert_eq!(json, "[]");
}

// --- OpenAI Responses tests ---

#[cfg(feature = "responses")]
fn responses_backend(server: &MockServer) -> Arc<dyn LlmBackend> {
    let uri = server.uri();
    OpenAiResponsesBackend::new(
        reqwest::Client::builder().use_rustls_tls(),
        "test-key",
        Some(&uri),
    )
    .expect("failed to build responses backend")
}

#[cfg(feature = "responses")]
#[tokio::test]
async fn responses_adapter_maps_generation_with_tools_and_reasoning() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-5.6",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        { "type": "output_text", "text": "Checking the weather.", "annotations": [] }
                    ]
                },
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{ "type": "summary_text", "text": "checked forecast" }],
                    "encrypted_content": "encrypted==",
                    "status": "completed"
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{\"city\":\"Paris\"}"
                }
            ],
            "usage": {
                "input_tokens": 10,
                "input_tokens_details": { "cache_write_tokens": 128, "cached_tokens": 512 },
                "output_tokens": 5,
                "output_tokens_details": { "reasoning_tokens": 2 },
                "total_tokens": 15
            }
        })))
        .mount(&server)
        .await;

    let backend = responses_backend(&server);
    let response = backend
        .generate(GenerationRequest::new(
            "gpt-5.6",
            vec![Message::user("weather?")],
        ))
        .await
        .unwrap();

    assert_eq!(response.text(), Some("Checking the weather."));
    assert_eq!(
        response.finish_reason,
        Some(just_llm_client::types::generation::FinishReason::ToolCalls)
    );
    assert_eq!(response.tool_calls().len(), 1);
    assert_eq!(response.tool_calls()[0].name, "get_weather");
    assert_eq!(response.tool_calls()[0].arguments, "{\"city\":\"Paris\"}");
    let reasoning = response.reasoning().expect("reasoning should be present");
    assert_eq!(reasoning.id.as_deref(), Some("rs_1"));
    assert_eq!(reasoning.encrypted.as_deref(), Some("encrypted=="));
    assert_eq!(reasoning.text.as_deref(), Some("checked forecast"));
    assert_eq!(
        response
            .usage
            .as_ref()
            .unwrap()
            .completion_tokens_details
            .as_ref()
            .unwrap()
            .reasoning_tokens,
        Some(2)
    );
    let cache_usage = response.usage.as_ref().unwrap();
    assert_eq!(cache_usage.cache_read_tokens, Some(512));
    assert_eq!(cache_usage.cache_write_tokens, Some(128));
}

#[cfg(feature = "responses")]
#[test]
fn responses_prepare_extracts_instructions_and_input_items() {
    let backend = OpenAiResponsesBackend::new(
        reqwest::Client::builder().use_rustls_tls(),
        "test-key",
        Some("http://127.0.0.1:0"),
    )
    .expect("failed to build responses backend");

    let request = GenerationRequest::new(
        "gpt-5.6",
        vec![
            Message::system("You are concise."),
            Message::user("What is the weather?"),
            Message::assistant_tool_calls(
                None,
                vec![ToolCall {
                    id: "call_1".to_owned(),
                    name: "get_weather".to_owned(),
                    arguments: "{\"city\":\"Paris\"}".to_owned(),
                }],
                None,
            ),
            Message::tool("25C", "call_1"),
        ],
    );

    let prepared = backend.prepare(request).unwrap();
    let body = prepared.body().and_then(|b| b.as_bytes()).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(body).unwrap();

    assert_eq!(parsed["instructions"], "You are concise.");
    assert_eq!(parsed["input"][0]["type"], "message");
    assert_eq!(parsed["input"][0]["role"], "user");
    assert_eq!(parsed["input"][1]["type"], "function_call");
    assert_eq!(parsed["input"][1]["call_id"], "call_1");
    assert_eq!(parsed["input"][1]["name"], "get_weather");
    assert_eq!(parsed["input"][2]["type"], "function_call_output");
    assert_eq!(parsed["input"][2]["call_id"], "call_1");
    assert_eq!(parsed["input"][2]["output"], "25C");
}

#[cfg(feature = "responses")]
#[test]
fn responses_prepare_rejects_unsupported_fields() {
    let backend = OpenAiResponsesBackend::new(
        reqwest::Client::builder().use_rustls_tls(),
        "test-key",
        Some("http://127.0.0.1:0"),
    )
    .expect("failed to build responses backend");

    let request = GenerationRequest::new("gpt-5.6", vec![Message::user("x")]).with_top_k(10);
    let error = backend.prepare(request).unwrap_err();
    assert!(matches!(error, BackendError::InvalidRequest(_)));

    let request = GenerationRequest::new("gpt-5.6", vec![Message::user("x")])
        .with_stop_sequences(StopSequence::Single("END".to_owned()));
    let error = backend.prepare(request).unwrap_err();
    assert!(matches!(error, BackendError::InvalidRequest(_)));
}

#[cfg(feature = "responses")]
#[test]
fn responses_prepare_maps_stateful_continuation() {
    let backend = OpenAiResponsesBackend::new(
        reqwest::Client::builder().use_rustls_tls(),
        "test-key",
        Some("http://127.0.0.1:0"),
    )
    .expect("failed to build responses backend");

    // A continuation carries the previous id, forces storage, drops the system prompt (the stored
    // conversation owns it), and requests encrypted reasoning when reasoning effort is set.
    let request = GenerationRequest::new(
        "gpt-5.6",
        vec![
            Message::system("You are concise."),
            Message::user("And now?"),
        ],
    )
    .with_previous_response_id("resp_1")
    .with_reasoning_effort(just_llm_client::types::generation::ReasoningEffort::High);

    let prepared = backend.prepare(request).unwrap();
    let body = prepared.body().and_then(|b| b.as_bytes()).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(body).unwrap();

    assert_eq!(parsed["previous_response_id"], "resp_1");
    assert_eq!(parsed["store"], true);
    assert!(parsed.get("instructions").is_none());
    assert_eq!(parsed["include"], json!(["reasoning.encrypted_content"]));
}

#[cfg(feature = "responses")]
#[test]
fn responses_prepare_maps_xhigh_effort() {
    let backend = OpenAiResponsesBackend::new(
        reqwest::Client::builder().use_rustls_tls(),
        "test-key",
        Some("http://127.0.0.1:0"),
    )
    .expect("failed to build responses backend");

    let request = GenerationRequest::new("gpt-5.6", vec![Message::user("x")])
        .with_reasoning_effort(just_llm_client::types::generation::ReasoningEffort::Xhigh);
    let prepared = backend.prepare(request).unwrap();
    let body = prepared.body().and_then(|b| b.as_bytes()).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(body).unwrap();

    assert_eq!(parsed["reasoning"]["effort"], "xhigh");
}

#[cfg(feature = "openai-compat")]
#[test]
fn openai_compat_prepare_rejects_stateful_fields() {
    let backend = OpenAiCompatBackend::new(
        reqwest::Client::builder().use_rustls_tls(),
        "test-key",
        Some("http://127.0.0.1:0"),
    )
    .expect("failed to build openai-compat backend");

    let request = GenerationRequest::new("gpt-4.1-mini", vec![Message::user("x")])
        .with_previous_response_id("resp_1");
    assert!(matches!(
        backend.prepare(request),
        Err(BackendError::InvalidRequest(_))
    ));

    let request = GenerationRequest::new("gpt-4.1-mini", vec![Message::user("x")]).with_store(true);
    assert!(matches!(
        backend.prepare(request),
        Err(BackendError::InvalidRequest(_))
    ));
}

#[cfg(feature = "openai-compat")]
#[test]
fn openai_compat_prepare_maps_xhigh_effort() {
    let backend = OpenAiCompatBackend::new(
        reqwest::Client::builder().use_rustls_tls(),
        "test-key",
        Some("http://127.0.0.1:0"),
    )
    .expect("failed to build openai-compat backend");

    let request = GenerationRequest::new("gpt-4.1-mini", vec![Message::user("x")])
        .with_reasoning_effort(just_llm_client::types::generation::ReasoningEffort::Xhigh);
    let prepared = backend.prepare(request).unwrap();
    let body = prepared.body().and_then(|b| b.as_bytes()).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(body).unwrap();

    assert_eq!(parsed["reasoning_effort"], "xhigh");
}

#[cfg(feature = "deepseek")]
#[test]
fn deepseek_prepare_rejects_stateful_fields() {
    let backend = DeepSeekBackend::new(
        reqwest::Client::builder().use_rustls_tls(),
        "test-key",
        Some("http://127.0.0.1:0"),
    )
    .expect("failed to build deepseek backend");

    let request = GenerationRequest::new("deepseek-reasoner", vec![Message::user("x")])
        .with_previous_response_id("resp_1");
    assert!(matches!(
        backend.prepare(request),
        Err(BackendError::InvalidRequest(_))
    ));
}

#[cfg(feature = "deepseek")]
#[test]
fn deepseek_prepare_maps_all_effort_levels() {
    let backend = DeepSeekBackend::new(
        reqwest::Client::builder().use_rustls_tls(),
        "test-key",
        Some("http://127.0.0.1:0"),
    )
    .expect("failed to build deepseek backend");

    let levels = [
        (
            just_llm_client::types::generation::ReasoningEffort::Low,
            "low",
        ),
        (
            just_llm_client::types::generation::ReasoningEffort::Medium,
            "medium",
        ),
        (
            just_llm_client::types::generation::ReasoningEffort::High,
            "high",
        ),
        (
            just_llm_client::types::generation::ReasoningEffort::Xhigh,
            "xhigh",
        ),
        (
            just_llm_client::types::generation::ReasoningEffort::Max,
            "max",
        ),
    ];
    for (effort, wire) in levels {
        let request = GenerationRequest::new("deepseek-v4-pro", vec![Message::user("x")])
            .with_reasoning_effort(effort);
        let prepared = backend.prepare(request).unwrap();
        let body = prepared.body().and_then(|b| b.as_bytes()).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(body).unwrap();

        assert_eq!(parsed["reasoning_effort"], wire);
    }

    // No effort preference means the field is omitted entirely.
    let request = GenerationRequest::new("deepseek-v4-pro", vec![Message::user("x")]);
    let prepared = backend.prepare(request).unwrap();
    let body = prepared.body().and_then(|b| b.as_bytes()).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(body).unwrap();
    assert!(parsed.get("reasoning_effort").is_none());
}

#[cfg(feature = "anthropic")]
#[test]
fn anthropic_prepare_rejects_stateful_fields() {
    let backend = AnthropicBackend::new(
        reqwest::Client::builder().use_rustls_tls(),
        "test-key",
        Some("http://127.0.0.1:0"),
    )
    .expect("failed to build anthropic backend");

    let request = GenerationRequest::new("claude-opus-5", vec![Message::user("x")])
        .with_max_tokens(256)
        .with_store(true);
    assert!(matches!(
        backend.prepare(request),
        Err(BackendError::InvalidRequest(_))
    ));
}

#[cfg(feature = "anthropic")]
#[test]
fn anthropic_prepare_maps_all_effort_levels() {
    let backend = AnthropicBackend::new(
        reqwest::Client::builder().use_rustls_tls(),
        "test-key",
        Some("http://127.0.0.1:0"),
    )
    .expect("failed to build anthropic backend");

    let levels = [
        (
            just_llm_client::types::generation::ReasoningEffort::Low,
            "low",
        ),
        (
            just_llm_client::types::generation::ReasoningEffort::Medium,
            "medium",
        ),
        (
            just_llm_client::types::generation::ReasoningEffort::High,
            "high",
        ),
        (
            just_llm_client::types::generation::ReasoningEffort::Xhigh,
            "xhigh",
        ),
        (
            just_llm_client::types::generation::ReasoningEffort::Max,
            "max",
        ),
    ];
    for (effort, wire) in levels {
        let request = GenerationRequest::new("claude-opus-5", vec![Message::user("x")])
            .with_max_tokens(256)
            .with_reasoning_effort(effort);
        let prepared = backend.prepare(request).unwrap();
        let body = prepared.body().and_then(|b| b.as_bytes()).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(body).unwrap();

        assert_eq!(parsed["output_config"]["effort"], wire);
    }

    // No effort preference means no output_config is emitted.
    let request =
        GenerationRequest::new("claude-opus-5", vec![Message::user("x")]).with_max_tokens(256);
    let prepared = backend.prepare(request).unwrap();
    let body = prepared.body().and_then(|b| b.as_bytes()).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(body).unwrap();
    assert!(parsed.get("output_config").is_none());
}

#[cfg(feature = "responses")]
#[tokio::test]
async fn responses_adapter_streams_events() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hel\"}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"lo\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\"status\":\"completed\",\"model\":\"gpt-5.6\",\"output\":[]}}\n\n",
        "data: [DONE]\n\n"
    );

    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(body, "text/event-stream"),
        )
        .mount(&server)
        .await;

    let backend = responses_backend(&server);
    let mut stream = backend
        .stream_generate(GenerationRequest::new("gpt-5.6", vec![Message::user("hi")]))
        .await
        .unwrap();

    let first = stream.next().await.unwrap().unwrap();
    assert!(matches!(first, GenerationEvent::Text { delta } if delta == "Hel"));

    let second = stream.next().await.unwrap().unwrap();
    assert!(matches!(second, GenerationEvent::Text { delta } if delta == "lo"));

    let end = stream.next().await.unwrap().unwrap();
    assert!(matches!(
        end,
        GenerationEvent::End {
            finish_reason: Some(just_llm_client::types::generation::FinishReason::Stop),
            ..
        }
    ));

    assert!(stream.next().await.is_none());
}

// --- Anthropic tests ---

#[cfg(feature = "anthropic")]
fn anthropic_backend(server: &MockServer) -> Arc<dyn LlmBackend> {
    let uri = server.uri();
    AnthropicBackend::new(
        reqwest::Client::builder().use_rustls_tls(),
        "test-key",
        Some(&uri),
    )
    .expect("failed to build anthropic backend")
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_adapter_maps_generation_with_tools_and_thinking() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [
                { "type": "thinking", "thinking": "need the weather", "signature": "sig_1" },
                { "type": "text", "text": "Checking the weather." },
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": { "city": "Paris" }
                }
            ],
            "model": "claude-opus-5",
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 8,
                "cache_creation_input_tokens": 512,
                "cache_read_input_tokens": 2048,
                "output_tokens_details": { "thinking_tokens": 3 }
            }
        })))
        .mount(&server)
        .await;

    let backend = anthropic_backend(&server);
    let response = backend
        .generate(
            GenerationRequest::new("claude-opus-5", vec![Message::user("weather?")])
                .with_max_tokens(1024),
        )
        .await
        .unwrap();

    assert_eq!(response.text(), Some("Checking the weather."));
    assert_eq!(
        response.finish_reason,
        Some(just_llm_client::types::generation::FinishReason::ToolCalls)
    );
    assert_eq!(response.tool_calls().len(), 1);
    assert_eq!(response.tool_calls()[0].id, "toolu_1");
    assert_eq!(response.tool_calls()[0].name, "get_weather");
    assert_eq!(response.tool_calls()[0].arguments, r#"{"city":"Paris"}"#);
    let reasoning = response.reasoning().expect("thinking should be present");
    assert_eq!(reasoning.text.as_deref(), Some("need the weather"));
    assert_eq!(reasoning.signature.as_deref(), Some("sig_1"));
    assert_eq!(
        response
            .usage
            .as_ref()
            .unwrap()
            .completion_tokens_details
            .as_ref()
            .unwrap()
            .reasoning_tokens,
        Some(3)
    );
    let cache_usage = response.usage.as_ref().unwrap();
    assert_eq!(cache_usage.cache_read_tokens, Some(2048));
    assert_eq!(cache_usage.cache_write_tokens, Some(512));
}

#[cfg(feature = "anthropic")]
#[test]
fn anthropic_prepare_extracts_system_and_requires_max_tokens() {
    let backend = AnthropicBackend::new(
        reqwest::Client::builder().use_rustls_tls(),
        "test-key",
        Some("http://127.0.0.1:0"),
    )
    .expect("failed to build anthropic backend");

    let request = GenerationRequest::new(
        "claude-opus-5",
        vec![Message::system("You are concise."), Message::user("Hi")],
    )
    .with_max_tokens(256);
    let prepared = backend.prepare(request).unwrap();
    let body = prepared.body().and_then(|b| b.as_bytes()).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(body).unwrap();

    assert_eq!(parsed["system"], "You are concise.");
    assert_eq!(parsed["max_tokens"], 256);
    assert_eq!(parsed["messages"][0]["role"], "user");
    assert_eq!(parsed["messages"][0]["content"], "Hi");

    // max_tokens is required by Anthropic; its absence is an explicit error.
    let request = GenerationRequest::new("claude-opus-5", vec![Message::user("Hi")]);
    let error = backend.prepare(request).unwrap_err();
    assert!(matches!(error, BackendError::InvalidRequest(_)));
}

#[cfg(feature = "anthropic")]
#[test]
fn anthropic_prepare_maps_tool_result_to_user_message() {
    let backend = AnthropicBackend::new(
        reqwest::Client::builder().use_rustls_tls(),
        "test-key",
        Some("http://127.0.0.1:0"),
    )
    .expect("failed to build anthropic backend");

    let request = GenerationRequest::new(
        "claude-opus-5",
        vec![
            Message::assistant_tool_calls(
                None,
                vec![ToolCall {
                    id: "toolu_1".to_owned(),
                    name: "get_weather".to_owned(),
                    arguments: r#"{"city":"Paris"}"#.to_owned(),
                }],
                None,
            ),
            Message::tool("25C", "toolu_1"),
        ],
    )
    .with_max_tokens(256);

    let prepared = backend.prepare(request).unwrap();
    let body = prepared.body().and_then(|b| b.as_bytes()).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(body).unwrap();

    // The assistant message carries a tool_use block.
    assert_eq!(parsed["messages"][0]["role"], "assistant");
    assert_eq!(parsed["messages"][0]["content"][0]["type"], "tool_use");
    assert_eq!(parsed["messages"][0]["content"][0]["name"], "get_weather");
    assert_eq!(
        parsed["messages"][0]["content"][0]["input"]["city"],
        "Paris"
    );

    // The tool result becomes a user message with a tool_result block.
    assert_eq!(parsed["messages"][1]["role"], "user");
    assert_eq!(parsed["messages"][1]["content"][0]["type"], "tool_result");
    assert_eq!(
        parsed["messages"][1]["content"][0]["tool_use_id"],
        "toolu_1"
    );
    assert_eq!(parsed["messages"][1]["content"][0]["content"], "25C");
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_adapter_streams_events() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-opus-5\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{}}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}\n\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":5}}\n\n",
        "data: {\"type\":\"message_stop\"}\n\n"
    );

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(body, "text/event-stream"),
        )
        .mount(&server)
        .await;

    let backend = anthropic_backend(&server);
    let mut stream = backend
        .stream_generate(
            GenerationRequest::new("claude-opus-5", vec![Message::user("hi")])
                .with_max_tokens(1024),
        )
        .await
        .unwrap();

    // The tool_use block start carries id/name and the block index.
    let tool_start = stream.next().await.unwrap().unwrap();
    assert!(matches!(
        tool_start,
        GenerationEvent::ToolCall { delta } if delta.index == Some(0) && delta.id.as_deref() == Some("toolu_1") && delta.name.as_deref() == Some("get_weather")
    ));

    // input_json_delta appends arguments.
    let args = stream.next().await.unwrap().unwrap();
    assert!(matches!(
        args,
        GenerationEvent::ToolCall { delta } if delta.arguments.as_deref() == Some("{\"city\":\"Paris\"}")
    ));

    // message_delta terminates with the finish reason and the final merged usage.
    let end = stream.next().await.unwrap().unwrap();
    assert!(matches!(
        end,
        GenerationEvent::End {
            finish_reason: Some(just_llm_client::types::generation::FinishReason::ToolCalls),
            ..
        }
    ));

    let usage = stream.next().await.unwrap().unwrap();
    assert!(matches!(
        usage,
        GenerationEvent::Usage { usage } if usage.prompt_tokens == 10 && usage.completion_tokens == 5
    ));

    assert!(stream.next().await.is_none());
}

#[cfg(feature = "deepseek")]
#[tokio::test]
async fn deepseek_adapter_streams_events() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(
                    concat!(
                        "data: {\"id\":\"chatcmpl-s\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-v4-pro\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"think\"}}]}\n\n",
                        "data: {\"id\":\"chatcmpl-s\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-v4-pro\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n",
                        "data: {\"id\":\"chatcmpl-s\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-v4-pro\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                        "data: [DONE]\n"
                    ),
                    "text/event-stream",
                ),
        )
        .mount(&server)
        .await;

    let backend = deepseek_backend(&server);
    let mut stream = backend
        .stream_generate(GenerationRequest::new(
            "deepseek-v4-pro",
            vec![Message::user("hi")],
        ))
        .await
        .unwrap();

    let reasoning = stream.next().await.unwrap().unwrap();
    assert!(matches!(reasoning, GenerationEvent::Reasoning { delta } if delta == "think"));

    let text = stream.next().await.unwrap().unwrap();
    assert!(matches!(text, GenerationEvent::Text { delta } if delta == "hi"));

    let end = stream.next().await.unwrap().unwrap();
    assert!(matches!(
        end,
        GenerationEvent::End {
            finish_reason: Some(just_llm_client::types::generation::FinishReason::Stop),
            ..
        }
    ));

    assert!(stream.next().await.is_none());
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_adapter_maps_usage_without_cache_fields() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [ { "type": "text", "text": "Hello." } ],
            "model": "claude-opus-5",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 8
            }
        })))
        .mount(&server)
        .await;

    let backend = anthropic_backend(&server);
    let response = backend
        .generate(
            GenerationRequest::new("claude-opus-5", vec![Message::user("hi")])
                .with_max_tokens(1024),
        )
        .await
        .unwrap();

    let usage = response.usage.expect("usage should be present");
    assert_eq!(usage.cache_read_tokens, None);
    assert_eq!(usage.cache_write_tokens, None);
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_stream_surfaces_cache_fields_from_message_delta_usage() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-opus-5\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":5,\"cache_read_input_tokens\":2048,\"cache_creation_input_tokens\":512}}\n\n",
        "data: {\"type\":\"message_stop\"}\n\n"
    );

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(body, "text/event-stream"),
        )
        .mount(&server)
        .await;

    let backend = anthropic_backend(&server);
    let mut stream = backend
        .stream_generate(
            GenerationRequest::new("claude-opus-5", vec![Message::user("hi")])
                .with_max_tokens(1024),
        )
        .await
        .unwrap();

    let end = stream.next().await.unwrap().unwrap();
    assert!(matches!(
        end,
        GenerationEvent::End {
            finish_reason: Some(just_llm_client::types::generation::FinishReason::Stop),
            ..
        }
    ));

    // message_delta usage carries the cache fields; the streaming path
    // surfaces them instead of discarding them.
    let usage = stream.next().await.unwrap().unwrap();
    assert!(matches!(
        usage,
        GenerationEvent::Usage { usage } if usage.prompt_tokens == 10 && usage.completion_tokens == 5
            && usage.cache_read_tokens == Some(2048)
            && usage.cache_write_tokens == Some(512)
    ));

    assert!(stream.next().await.is_none());
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_stream_omits_usage_event_without_stream_usage() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-opus-5\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null}}\n\n",
        "data: {\"type\":\"message_stop\"}\n\n"
    );

    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(body, "text/event-stream"),
        )
        .mount(&server)
        .await;

    let backend = anthropic_backend(&server);
    let mut stream = backend
        .stream_generate(
            GenerationRequest::new("claude-opus-5", vec![Message::user("hi")])
                .with_max_tokens(1024),
        )
        .await
        .unwrap();

    let end = stream.next().await.unwrap().unwrap();
    assert!(matches!(
        end,
        GenerationEvent::End {
            finish_reason: Some(just_llm_client::types::generation::FinishReason::Stop),
            ..
        }
    ));

    // No Usage event follows when message_delta carries no usage.
    assert!(stream.next().await.is_none());
}

// --- Multimodal image input (unified ContentPart::Image) ---

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_prepare_maps_image_sources_and_drops_detail() {
    let backend = anthropic_backend(&MockServer::start().await);
    let request = GenerationRequest::new(
        "claude-opus-5",
        vec![Message::user_parts(vec![
            ContentPart::Text {
                text: "what is this?".to_string(),
            },
            ContentPart::Image {
                source: ImageSource::Url {
                    url: "https://example.com/cat.png".to_string(),
                },
                detail: Some(ImageDetail::High),
            },
            ContentPart::Image {
                source: ImageSource::Base64 {
                    data: "aGVsbG8=".to_string(),
                    media_type: "image/png".to_string(),
                },
                detail: None,
            },
            ContentPart::Image {
                source: ImageSource::FileId {
                    file_id: "file_123".to_string(),
                },
                detail: Some(ImageDetail::Low),
            },
        ])],
    )
    .with_max_tokens(256);

    let prepared = backend.prepare(request).unwrap();
    let body = prepared.body().and_then(|b| b.as_bytes()).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(body).unwrap();

    let content = &parsed["messages"][0]["content"];
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[1]["type"], "image");
    assert_eq!(content[1]["source"]["type"], "url");
    assert_eq!(content[1]["source"]["url"], "https://example.com/cat.png");
    assert!(content[1].get("detail").is_none());
    assert_eq!(content[2]["source"]["type"], "base64");
    assert_eq!(content[2]["source"]["data"], "aGVsbG8=");
    assert_eq!(content[2]["source"]["media_type"], "image/png");
    assert!(content[2].get("detail").is_none());
    assert_eq!(content[3]["source"]["type"], "file");
    assert_eq!(content[3]["source"]["file_id"], "file_123");
    assert!(content[3].get("detail").is_none());
}

#[cfg(feature = "responses")]
#[tokio::test]
async fn responses_prepare_maps_image_fields_and_passes_detail_through() {
    let backend = responses_backend(&MockServer::start().await);
    let request = GenerationRequest::new(
        "gpt-5.6",
        vec![Message::user_parts(vec![
            ContentPart::Text {
                text: "describe".to_string(),
            },
            ContentPart::Image {
                source: ImageSource::Url {
                    url: "https://example.com/cat.png".to_string(),
                },
                detail: Some(ImageDetail::High),
            },
            ContentPart::Image {
                source: ImageSource::Base64 {
                    data: "aGVsbG8=".to_string(),
                    media_type: "image/jpeg".to_string(),
                },
                detail: None,
            },
            ContentPart::Image {
                source: ImageSource::FileId {
                    file_id: "file-abc".to_string(),
                },
                detail: Some(ImageDetail::Unknown("hd".to_string())),
            },
        ])],
    );

    let prepared = backend.prepare(request).unwrap();
    let body = prepared.body().and_then(|b| b.as_bytes()).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(body).unwrap();

    let content = &parsed["input"][0]["content"];
    assert_eq!(content[0]["type"], "input_text");
    assert_eq!(content[1]["type"], "input_image");
    assert_eq!(content[1]["image_url"], "https://example.com/cat.png");
    assert_eq!(content[1]["detail"], "high");
    assert_eq!(content[2]["image_url"], "data:image/jpeg;base64,aGVsbG8=");
    assert!(content[2].get("detail").is_none());
    assert_eq!(content[3]["file_id"], "file-abc");
    assert_eq!(content[3]["detail"], "hd");
}

#[cfg(feature = "deepseek")]
#[tokio::test]
async fn deepseek_prepare_maps_multimodal_content() {
    let backend = deepseek_backend(&MockServer::start().await);
    let request = GenerationRequest::new(
        "deepseek-v4-pro",
        vec![Message::user_parts(vec![
            ContentPart::Text {
                text: "describe".to_string(),
            },
            ContentPart::Image {
                source: ImageSource::Url {
                    url: "https://example.com/cat.png".to_string(),
                },
                detail: Some(ImageDetail::Low),
            },
            ContentPart::Image {
                source: ImageSource::Base64 {
                    data: "aGVsbG8=".to_string(),
                    media_type: "image/png".to_string(),
                },
                detail: None,
            },
        ])],
    );

    let prepared = backend.prepare(request).unwrap();
    let body = prepared.body().and_then(|b| b.as_bytes()).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(body).unwrap();

    let content = &parsed["messages"][0]["content"];
    assert!(content.is_array());
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[1]["type"], "image_url");
    assert_eq!(
        content[1]["image_url"]["url"],
        "https://example.com/cat.png"
    );
    assert_eq!(content[1]["image_url"]["detail"], "low");
    assert_eq!(
        content[2]["image_url"]["url"],
        "data:image/png;base64,aGVsbG8="
    );
    assert!(content[2]["image_url"].get("detail").is_none());
}

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn openai_compat_prepare_rejects_file_id_image_source() {
    let backend = openai_backend(&MockServer::start().await);
    let request = GenerationRequest::new(
        "gpt-4.1-mini",
        vec![Message::user_parts(vec![ContentPart::Image {
            source: ImageSource::FileId {
                file_id: "file-abc".to_string(),
            },
            detail: None,
        }])],
    );

    let error = backend.prepare(request).unwrap_err();
    assert!(matches!(error, BackendError::Unserializable { .. }));
}

#[cfg(feature = "deepseek")]
#[tokio::test]
async fn deepseek_prepare_rejects_file_id_image_source() {
    let backend = deepseek_backend(&MockServer::start().await);
    let request = GenerationRequest::new(
        "deepseek-v4-pro",
        vec![Message::user_parts(vec![ContentPart::Image {
            source: ImageSource::FileId {
                file_id: "file-abc".to_string(),
            },
            detail: None,
        }])],
    );

    let error = backend.prepare(request).unwrap_err();
    assert!(matches!(error, BackendError::Unserializable { .. }));
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_send_preserves_raw_response_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [{ "type": "text", "text": "raw body stays reachable" }],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        })))
        .mount(&server)
        .await;

    let backend = anthropic_backend(&server);
    let prepared = backend
        .prepare(
            GenerationRequest::new("claude-opus-5", vec![Message::user("hi")]).with_max_tokens(256),
        )
        .unwrap();

    // send() hands back the raw reqwest::Response — callers may read the
    // original body bytes without going through parse().
    let response = backend.send(prepared).await.unwrap();
    assert!(response.status().is_success());
    let body = response.bytes().await.unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed["id"], "msg_1");
    assert_eq!(parsed["content"][0]["text"], "raw body stays reachable");
}

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn openai_compat_roundtrips_multimodal_message() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-mm",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4.1-mini",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "a cat"}
            }]
        })))
        .mount(&server)
        .await;

    let backend = openai_backend(&server);
    let prepared = backend
        .prepare(GenerationRequest::new(
            "gpt-4.1-mini",
            vec![Message::user_parts(vec![
                ContentPart::Text {
                    text: "what is it?".to_string(),
                },
                ContentPart::Image {
                    source: ImageSource::Url {
                        url: "https://example.com/cat.png".to_string(),
                    },
                    detail: Some(ImageDetail::Low),
                },
            ])],
        ))
        .unwrap();
    let response = backend.send(prepared).await.unwrap();
    let generation = backend.parse(response).await.unwrap();

    assert_eq!(generation.text(), Some("a cat"));
}

#[cfg(feature = "deepseek")]
#[tokio::test]
async fn deepseek_roundtrips_multimodal_message() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-mm",
            "object": "chat.completion",
            "created": 1,
            "model": "deepseek-v4-pro",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "a cat"}
            }]
        })))
        .mount(&server)
        .await;

    let backend = deepseek_backend(&server);
    let prepared = backend
        .prepare(GenerationRequest::new(
            "deepseek-v4-pro",
            vec![Message::user_parts(vec![
                ContentPart::Text {
                    text: "what is it?".to_string(),
                },
                ContentPart::Image {
                    source: ImageSource::Url {
                        url: "https://example.com/cat.png".to_string(),
                    },
                    detail: Some(ImageDetail::Low),
                },
            ])],
        ))
        .unwrap();
    let response = backend.send(prepared).await.unwrap();
    let generation = backend.parse(response).await.unwrap();

    assert_eq!(generation.text(), Some("a cat"));
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_roundtrips_multimodal_message() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [{ "type": "text", "text": "a cat" }],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        })))
        .mount(&server)
        .await;

    let backend = anthropic_backend(&server);
    let prepared = backend
        .prepare(
            GenerationRequest::new(
                "claude-opus-5",
                vec![Message::user_parts(vec![
                    ContentPart::Text {
                        text: "what is it?".to_string(),
                    },
                    ContentPart::Image {
                        source: ImageSource::Url {
                            url: "https://example.com/cat.png".to_string(),
                        },
                        detail: None,
                    },
                ])],
            )
            .with_max_tokens(256),
        )
        .unwrap();
    let response = backend.send(prepared).await.unwrap();
    let generation = backend.parse(response).await.unwrap();

    assert_eq!(generation.text(), Some("a cat"));
}

#[cfg(feature = "responses")]
#[tokio::test]
async fn responses_roundtrips_multimodal_message() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-5.6",
            "output": [{
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    { "type": "output_text", "text": "a cat", "annotations": [] }
                ]
            }],
            "usage": { "input_tokens": 1, "output_tokens": 1, "total_tokens": 2 }
        })))
        .mount(&server)
        .await;

    let backend = responses_backend(&server);
    let prepared = backend
        .prepare(GenerationRequest::new(
            "gpt-5.6",
            vec![Message::user_parts(vec![
                ContentPart::Text {
                    text: "what is it?".to_string(),
                },
                ContentPart::Image {
                    source: ImageSource::Url {
                        url: "https://example.com/cat.png".to_string(),
                    },
                    detail: None,
                },
            ])],
        ))
        .unwrap();
    let response = backend.send(prepared).await.unwrap();
    let generation = backend.parse(response).await.unwrap();

    assert_eq!(generation.text(), Some("a cat"));
}

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn openai_compat_prepare_maps_multimodal_content() {
    let backend = openai_backend(&MockServer::start().await);
    let request = GenerationRequest::new(
        "gpt-4.1-mini",
        vec![Message::user_parts(vec![
            ContentPart::Text {
                text: "describe".to_string(),
            },
            ContentPart::Image {
                source: ImageSource::Url {
                    url: "https://example.com/cat.png".to_string(),
                },
                detail: Some(ImageDetail::Low),
            },
            ContentPart::Image {
                source: ImageSource::Base64 {
                    data: "aGVsbG8=".to_string(),
                    media_type: "image/png".to_string(),
                },
                detail: None,
            },
        ])],
    );

    let prepared = backend.prepare(request).unwrap();
    let body = prepared.body().and_then(|b| b.as_bytes()).unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(body).unwrap();

    let message = &parsed["messages"][0];
    assert_eq!(message["role"], "user");
    let content = &message["content"];
    assert!(content.is_array());
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[1]["type"], "image_url");
    assert_eq!(
        content[1]["image_url"]["url"],
        "https://example.com/cat.png"
    );
    assert_eq!(content[1]["image_url"]["detail"], "low");
    assert_eq!(
        content[2]["image_url"]["url"],
        "data:image/png;base64,aGVsbG8="
    );
    assert!(content[2]["image_url"].get("detail").is_none());
}

#[cfg(feature = "openai-compat")]
#[tokio::test]
async fn openai_compat_surfaces_provider_rejection_from_error_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": {
                "code": "image_invalid",
                "type": "invalid_request_error"
            }
        })))
        .mount(&server)
        .await;

    let backend = openai_backend(&server);
    let error = backend
        .generate(GenerationRequest::new(
            "gpt-4.1-mini",
            vec![Message::user_parts(vec![ContentPart::Image {
                source: ImageSource::Url {
                    url: "https://example.com/cat.png".to_string(),
                },
                detail: None,
            }])],
        ))
        .await
        .unwrap_err();

    let rejection = provider_rejection(&error).expect("http rejection must be reachable");
    assert_eq!(rejection.status, reqwest::StatusCode::BAD_REQUEST);
    assert_eq!(rejection.code.as_deref(), Some("image_invalid"));
    assert_eq!(
        rejection.error_type.as_deref(),
        Some("invalid_request_error")
    );

    // The original body is still recoverable verbatim via captured_body.
    let body = captured_body(&error).expect("error body must be captured");
    let value: serde_json::Value = serde_json::from_str(body).unwrap();
    assert_eq!(value["error"]["code"], "image_invalid");
    assert_eq!(value["error"]["type"], "invalid_request_error");
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_prepare_rejects_system_image_parts_as_unserializable() {
    let backend = anthropic_backend(&MockServer::start().await);
    let request = GenerationRequest::new(
        "claude-opus-5",
        vec![
            Message::System {
                content: MessageContent::Parts(vec![ContentPart::Image {
                    source: ImageSource::Url {
                        url: "https://example.com/cat.png".to_string(),
                    },
                    detail: None,
                }]),
            },
            Message::user("hi"),
        ],
    )
    .with_max_tokens(256);

    let error = backend.prepare(request).unwrap_err();
    assert!(matches!(error, BackendError::Unserializable { .. }));
}
