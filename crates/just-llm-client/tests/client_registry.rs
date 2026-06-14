use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use async_trait::async_trait;
use futures_util::stream;
use just_llm_client::{
    CapabilityNegotiation, ChatClientOptions, ChatCompletionStream, ProviderEntry,
    ProviderRegistry,
    error::LlmError,
    provider::LlmBackend,
    types::chat::{
        AssistantMessage, AssistantRole, ChatChoice, ChatCompletionRequest, ChatCompletionResponse,
        ChatMessage, ToolDefinition,
    },
};

struct TestProvider {
    id: String,
    family: &'static str,
    connect_count: Arc<AtomicUsize>,
}

impl TestProvider {
    fn new(id: impl Into<String>, connect_count: Arc<AtomicUsize>) -> Self {
        Self {
            id: id.into(),
            family: "test-provider",
            connect_count,
        }
    }
}

impl ProviderEntry for TestProvider {
    fn instance_id(&self) -> &str {
        &self.id
    }

    fn connect(&self) -> Result<Arc<dyn LlmBackend>, LlmError> {
        self.connect_count.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(TestBackend {
            family: self.family,
            http: reqwest::Client::new(),
        }))
    }
}

struct TestBackend {
    family: &'static str,
    http: reqwest::Client,
}

impl just_llm_client::Identifiable for TestBackend {
    fn family(&self) -> &'static str {
        self.family
    }
}

impl CapabilityNegotiation for TestBackend {}

#[async_trait]
impl LlmBackend for TestBackend {
    fn prepare(&self, request: ChatCompletionRequest) -> Result<reqwest::Request, LlmError> {
        // Build a request pointing at an unreachable URL — the test
        // overrides chat_completion() below so this is never actually sent.
        let url = "http://127.0.0.1:0/chat/completions";
        self.http
            .post(url)
            .json(&serde_json::json!({
                "model": request.model,
                "messages": request.messages,
            }))
            .build()
            .map_err(|e| {
                LlmError::backend(
                    self.family,
                    just_common::error::TransportError::Transport(e),
                )
            })
    }

    fn prepare_streaming(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<reqwest::Request, LlmError> {
        self.prepare(request)
    }

    async fn send(&self, _prepared: reqwest::Request) -> Result<reqwest::Response, LlmError> {
        // Never called — chat_completion() is overridden below.
        todo!("send is not used by these tests")
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        Ok(response_for_model(request.model))
    }

    async fn stream_chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> Result<ChatCompletionStream, LlmError> {
        Ok(ChatCompletionStream::new(Box::pin(stream::empty())))
    }

    async fn parse(
        &self,
        _response: reqwest::Response,
    ) -> Result<ChatCompletionResponse, LlmError> {
        // chat_completion is overridden above, so the default prepare->send->parse path
        // is never taken; this exists only to satisfy the trait surface.
        unreachable!("TestBackend.parse is unreachable: chat_completion is overridden")
    }

    async fn parse_streaming(
        &self,
        _response: reqwest::Response,
    ) -> Result<ChatCompletionStream, LlmError> {
        unreachable!(
            "TestBackend.parse_streaming is unreachable: stream_chat_completion is overridden"
        )
    }

    fn render_messages(&self, _messages: &[ChatMessage]) -> Result<String, LlmError> {
        Ok("[]".to_owned())
    }

    fn render_tools(&self, _tools: &[ToolDefinition]) -> Result<String, LlmError> {
        Ok("[]".to_owned())
    }
}

fn response_for_model(model: String) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: "test-response".to_owned(),
        choices: vec![ChatChoice {
            finish_reason: None,
            index: 0,
            logprobs: None,
            message: AssistantMessage {
                content: Some("ok".to_owned()),
                reasoning_content: None,
                tool_calls: None,
                role: AssistantRole::Assistant,
            },
        }],
        created: 0,
        model,
        system_fingerprint: None,
        object: "chat.completion".to_owned(),
        usage: None,
    }
}

#[test]
fn chat_client_request_injects_model_and_system_prompt() {
    let connect_count = Arc::new(AtomicUsize::new(0));
    let mut registry = ProviderRegistry::new();
    registry.register(TestProvider::new("alpha", connect_count));

    let client = registry
        .chat(
            "alpha",
            ChatClientOptions::new("test-model").with_system_prompt("Be concise."),
        )
        .unwrap();
    let request = client.create_request(vec![ChatMessage::user("hello")]);

    assert_eq!(client.instance_id(), "alpha");
    assert_eq!(client.model(), "test-model");
    assert_eq!(client.system_prompt(), Some("Be concise."));
    assert_eq!(request.model, "test-model");
    assert_eq!(request.messages.len(), 2);
    assert_eq!(request.messages[0].role(), "system");
    assert_eq!(request.messages[0].content(), Some("Be concise."));
    assert_eq!(request.messages[1].role(), "user");
    assert_eq!(request.messages[1].content(), Some("hello"));
}

#[test]
fn provider_registry_reuses_connected_backend() {
    let connect_count = Arc::new(AtomicUsize::new(0));
    let mut registry = ProviderRegistry::new();
    registry.register(TestProvider::new("alpha", connect_count.clone()));

    let first = registry
        .chat("alpha", ChatClientOptions::new("model-a"))
        .unwrap();
    let second = registry
        .chat("alpha", ChatClientOptions::new("model-b"))
        .unwrap();

    assert_eq!(connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(first.instance_id(), "alpha");
    assert_eq!(second.instance_id(), "alpha");
    assert_eq!(first.model(), "model-a");
    assert_eq!(second.model(), "model-b");
}

#[test]
fn provider_registry_replaces_existing_provider_with_same_id() {
    let first_count = Arc::new(AtomicUsize::new(0));
    let second_count = Arc::new(AtomicUsize::new(0));
    let mut registry = ProviderRegistry::new();
    registry.register(TestProvider::new("alpha", first_count.clone()));
    registry.register(TestProvider::new("alpha", second_count.clone()));

    let client = registry
        .chat("alpha", ChatClientOptions::new("model"))
        .unwrap();

    assert_eq!(client.instance_id(), "alpha");
    assert_eq!(first_count.load(Ordering::SeqCst), 0);
    assert_eq!(second_count.load(Ordering::SeqCst), 1);
}

#[test]
fn provider_registry_rejects_unknown_instance_id() {
    let registry = ProviderRegistry::new();
    let error = registry
        .chat("missing", ChatClientOptions::new("model"))
        .unwrap_err();

    assert!(matches!(error, LlmError::InvalidRequest(_)));
    assert_eq!(
        error.to_string(),
        "invalid request: unknown instance id: missing"
    );
}
