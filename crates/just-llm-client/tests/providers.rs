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
#[cfg(any(feature = "deepseek", feature = "openai-compat"))]
use just_llm_client::types::chat::{ChatToolCall, FunctionCall, FunctionDefinition, ToolType};
#[cfg(feature = "deepseek")]
use just_llm_client::types::chat::{ToolChoice, ToolChoiceMode};
#[cfg(any(feature = "deepseek", feature = "openai-compat"))]
use just_llm_client::{
    LlmBackend,
    error::LlmError,
    types::chat::{ChatCompletionRequest, ChatMessage, ToolDefinition},
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
    let builder = reqwest::Client::builder().use_rustls_tls();
    let http = just_common::transport::http::build_client(builder, "test-key").unwrap();
    DeepSeekBackend::new(http, server.uri())
}

#[cfg(feature = "deepseek")]
fn deepseek_backend_no_server() -> DeepSeekBackend {
    let builder = reqwest::Client::builder().use_rustls_tls();
    let http = just_common::transport::http::build_client(builder, "test-key").unwrap();
    DeepSeekBackend::new(http, "http://127.0.0.1:0".to_owned())
}

#[cfg(feature = "openai-compat")]
fn openai_backend(server: &MockServer) -> OpenAiCompatBackend {
    let builder = reqwest::Client::builder().use_rustls_tls();
    let http = just_common::transport::http::build_client(builder, "test-key").unwrap();
    OpenAiCompatBackend::new(http, server.uri())
}

#[cfg(feature = "openai-compat")]
fn openai_backend_no_server() -> OpenAiCompatBackend {
    let builder = reqwest::Client::builder().use_rustls_tls();
    let http = just_common::transport::http::build_client(builder, "test-key").unwrap();
    OpenAiCompatBackend::new(http, "http://127.0.0.1:0".to_owned())
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
        .prepare(ChatCompletionRequest::new(
            "gpt-4.1-mini",
            vec![ChatMessage::user("hello")],
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

// --- render_messages / render_tools tests ---

#[cfg(feature = "deepseek")]
#[test]
fn deepseek_render_messages_produces_provider_json() {
    let backend = deepseek_backend_no_server();
    let messages = vec![
        ChatMessage::system("You are helpful."),
        ChatMessage::user("Hello"),
    ];

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
    let messages = vec![
        ChatMessage::system("You are helpful."),
        ChatMessage::user("Hello"),
    ];

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
    let messages = vec![ChatMessage::assistant_tool_calls(vec![ChatToolCall {
        id: "call_1".to_owned(),
        kind: ToolType::Function,
        function: FunctionCall {
            name: "get_weather".to_owned(),
            arguments: "{\"city\":\"Shanghai\"}".to_owned(),
        },
    }])];

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
    let messages = vec![ChatMessage::tool_result("{\"temperature\":26}", "call_1")];

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
    let messages = vec![ChatMessage::assistant_tool_calls(vec![ChatToolCall {
        id: "call_1".to_owned(),
        kind: ToolType::Function,
        function: FunctionCall {
            name: "get_weather".to_owned(),
            arguments: "{\"city\":\"Shanghai\"}".to_owned(),
        },
    }])];

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
    let messages = vec![ChatMessage::tool_result("{\"temperature\":26}", "call_1")];

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
