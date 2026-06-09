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
    types::{
        chat::{
            AssistantMessage, AssistantRole, ChatChoice, ChatCompletionRequest,
            ChatCompletionResponse, ChatMessage,
        },
        prepared::PreparedChatRequest,
    },
};
use serde_json::json;

struct TestProvider {
    id: String,
    provider: &'static str,
    connect_count: Arc<AtomicUsize>,
}

impl TestProvider {
    fn new(id: impl Into<String>, connect_count: Arc<AtomicUsize>) -> Self {
        Self { id: id.into(), provider: "test-provider", connect_count }
    }
}

impl ProviderEntry for TestProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn provider(&self) -> &str {
        self.provider
    }

    fn connect(&self) -> Result<Arc<dyn LlmBackend>, LlmError> {
        self.connect_count.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(TestBackend { backend_id: self.provider }))
    }
}

struct TestBackend {
    backend_id: &'static str,
}

impl just_llm_client::Identifiable for TestBackend {
    fn backend_id(&self) -> &'static str {
        self.backend_id
    }
}

impl CapabilityNegotiation for TestBackend {}

#[async_trait]
impl LlmBackend for TestBackend {
    fn prepare(&self, request: ChatCompletionRequest) -> Result<PreparedChatRequest, LlmError> {
        PreparedChatRequest::from_request_body(
            self.backend_id,
            json!({
                "model": request.model,
                "messages": request.messages,
            }),
        )
        .map_err(LlmError::from)
    }

    async fn send(
        &self,
        prepared: &PreparedChatRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        Ok(response_for_model(
            prepared.model().unwrap_or("missing-model").to_owned(),
        ))
    }

    fn prepare_streaming(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<PreparedChatRequest, LlmError> {
        self.prepare(request)
    }

    async fn send_streaming(
        &self,
        _prepared: &PreparedChatRequest,
    ) -> Result<ChatCompletionStream, LlmError> {
        Ok(Box::pin(stream::empty()))
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
    let request = client.request(vec![ChatMessage::user("hello")]);

    assert_eq!(client.provider_id(), "alpha");
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
    assert_eq!(first.provider_id(), "alpha");
    assert_eq!(second.provider_id(), "alpha");
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

    assert_eq!(client.provider_id(), "alpha");
    assert_eq!(first_count.load(Ordering::SeqCst), 0);
    assert_eq!(second_count.load(Ordering::SeqCst), 1);
}

#[test]
fn provider_registry_rejects_unknown_provider_id() {
    let registry = ProviderRegistry::new();
    let error = registry
        .chat("missing", ChatClientOptions::new("model"))
        .unwrap_err();

    assert!(matches!(error, LlmError::InvalidRequest(_)));
    assert_eq!(
        error.to_string(),
        "invalid request: unknown provider id: missing"
    );
}
