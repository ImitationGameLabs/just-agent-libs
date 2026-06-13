use futures_util::StreamExt;
use just_common::error::TransportError;
use just_deepseek::{
    ChatCompletionStream, DeepSeekClient, Error,
    types::chat::{AssistantRole, ChatCompletionRequest, ChatMessage},
};
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{header, method, path},
};

fn client(server: &MockServer) -> DeepSeekClient {
    DeepSeekClient::builder()
        .api_key("test-key")
        .base_url(server.uri())
        .build()
        .unwrap()
}

fn client_with_http(server: &MockServer) -> DeepSeekClient {
    DeepSeekClient::builder()
        .api_key("test-key")
        .base_url(server.uri())
        .http_client(reqwest::Client::builder())
        .build()
        .unwrap()
}

fn basic_request() -> ChatCompletionRequest {
    ChatCompletionRequest::new(
        "deepseek-v4-pro",
        vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ],
    )
}

#[tokio::test]
async fn lists_models() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [
                {
                    "id": "deepseek-v4-pro",
                    "object": "model",
                    "owned_by": "deepseek"
                }
            ]
        })))
        .mount(&server)
        .await;

    let response = client(&server).list_models().await.unwrap();

    assert_eq!(response.data[0].id, "deepseek-v4-pro");
}

#[tokio::test]
async fn gets_user_balance() {
    let server = MockServer::start().await;
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

    let response = client(&server).get_user_balance().await.unwrap();

    assert!(response.is_available);
    assert_eq!(response.balance_infos.len(), 1);
}

#[tokio::test]
async fn creates_non_streaming_chat_completion() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
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
                        "content": "Hello!",
                        "reasoning_content": "thinking"
                    },
                    "logprobs": null
                }
            ],
            "usage": {
                "completion_tokens": 1,
                "prompt_tokens": 1,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 1,
                "total_tokens": 2
            }
        })))
        .mount(&server)
        .await;

    let response = client(&server)
        .chat_completion(basic_request())
        .await
        .unwrap();

    assert_eq!(response.choices[0].message.role, AssistantRole::Assistant);
    assert_eq!(
        response.choices[0].message.content.as_deref(),
        Some("Hello!")
    );
}

#[tokio::test]
async fn rejects_stream_flag_on_non_stream_method() {
    let request = ChatCompletionRequest {
        stream: Some(true),
        ..basic_request()
    };

    let error = DeepSeekClient::builder()
        .api_key("test-key")
        .build()
        .unwrap()
        .chat_completion(request)
        .await
        .unwrap_err();

    assert!(matches!(error, Error::InvalidRequest(_)));
}

#[tokio::test]
async fn streams_chat_completion_chunks() {
    let server = MockServer::start().await;
    let body = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-v4-pro\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"},\"finish_reason\":null}],\"usage\":null}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-v4-pro\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":1,\"prompt_tokens\":1,\"prompt_cache_hit_tokens\":0,\"prompt_cache_miss_tokens\":1,\"total_tokens\":2}}\n\n",
        "data: [DONE]\n\n"
    );

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(body, "text/event-stream"),
        )
        .mount(&server)
        .await;

    let mut stream: ChatCompletionStream = client(&server)
        .stream_chat_completion(basic_request())
        .await
        .unwrap();

    let first = stream.next().await.unwrap().unwrap();
    let second = stream.next().await.unwrap().unwrap();

    assert_eq!(first.choices[0].delta.role, Some(AssistantRole::Assistant));
    assert_eq!(first.choices[0].delta.content.as_deref(), Some("Hel"));
    assert_eq!(second.choices[0].delta.content.as_deref(), Some("lo"));
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn streams_tool_call_deltas_and_ignores_heartbeat_events() {
    let server = MockServer::start().await;
    let body = concat!(
        "\n\n",
        ": ping\n\n",
        "data: {\"id\":\"chatcmpl-tool-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-v4-pro\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"lookup_weather\",\"arguments\":\"\"}}]},\"finish_reason\":null}],\"usage\":null}\n\n",
        "data: {\"id\":\"chatcmpl-tool-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-v4-pro\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\\\"Shanghai\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":null}\n\n",
        "data: [DONE]\n\n"
    );

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(body, "text/event-stream"),
        )
        .mount(&server)
        .await;

    let mut stream: ChatCompletionStream = client(&server)
        .stream_chat_completion(basic_request())
        .await
        .unwrap();

    let first = stream.next().await.unwrap().unwrap();
    let second = stream.next().await.unwrap().unwrap();

    let first_call = &first.choices[0].delta.tool_calls.as_ref().unwrap()[0];
    assert_eq!(first_call.id.as_deref(), Some("call_1"));
    let first_function = first_call.function.as_ref().unwrap();
    assert_eq!(first_function.name.as_deref(), Some("lookup_weather"));
    assert_eq!(first_function.arguments.as_deref(), Some(""));

    let second_call = &second.choices[0].delta.tool_calls.as_ref().unwrap()[0];
    let second_function = second_call.function.as_ref().unwrap();
    assert_eq!(second_function.name, None);
    assert_eq!(
        second_function.arguments.as_deref(),
        Some("{\"city\":\"Shanghai\"}")
    );
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn preserves_http_error_body() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(ResponseTemplate::new(401).set_body_string("invalid auth"))
        .mount(&server)
        .await;

    let error = client(&server).list_models().await.unwrap_err();

    match error {
        Error::Transport(TransportError::HttpStatus { status, body }) => {
            assert_eq!(status.as_u16(), 401);
            assert_eq!(body, "invalid auth");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn stream_chat_completion_preserves_http_error_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("invalid auth"))
        .mount(&server)
        .await;

    let error = client(&server)
        .stream_chat_completion(basic_request())
        .await
        .unwrap_err();

    match error {
        Error::Transport(TransportError::HttpStatus { status, body }) => {
            assert_eq!(status.as_u16(), 401);
            assert_eq!(body, "invalid auth");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn rejects_streaming_response_without_sse_content_type() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "not-a-stream"
        })))
        .mount(&server)
        .await;

    let error = client(&server)
        .stream_chat_completion(basic_request())
        .await
        .unwrap_err();

    assert!(matches!(
        error,
        Error::Transport(TransportError::InvalidResponse(_))
    ));
}

#[tokio::test]
async fn error_display_does_not_dump_raw_response_body() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(ResponseTemplate::new(401).set_body_string("sensitive response body"))
        .mount(&server)
        .await;

    let error = client(&server).list_models().await.unwrap_err();

    assert!(!error.to_string().contains("sensitive response body"));
}

#[tokio::test]
async fn lists_models_via_injected_http_client() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [
                {
                    "id": "deepseek-v4-pro",
                    "object": "model",
                    "owned_by": "deepseek"
                }
            ]
        })))
        .mount(&server)
        .await;

    let response = client_with_http(&server).list_models().await.unwrap();
    assert_eq!(response.data[0].id, "deepseek-v4-pro");
}
