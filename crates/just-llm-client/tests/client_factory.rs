//! `BackendFactory` behavior: family dispatch, replace-on-reregister, unknown-family rejection.

use std::{marker::PhantomData, sync::Arc};

use async_trait::async_trait;
use futures_util::stream;
use just_llm_client::{
    BackendConstructError, BackendError, BackendFactory, CapabilityNegotiation, ChatClient,
    ChatClientOptions, ChatCompletionStream, Identifiable, LlmBackend,
    types::chat::{
        AssistantMessage, AssistantRole, ChatChoice, ChatCompletionRequest, ChatCompletionResponse,
        ChatMessage, ToolDefinition,
    },
};

/// Per-type test family: the family string the backend reports, and the `render_messages` output
/// that distinguishes one type from another (used to prove which constructor won).
trait TestBackendFamily: Send + Sync + 'static {
    const FAMILY: &'static str;
    const RENDER: &'static str;
}

struct TestBackend<F: TestBackendFamily> {
    http: reqwest::Client,
    _f: PhantomData<F>,
}

impl<F: TestBackendFamily> Identifiable for TestBackend<F> {
    fn family(&self) -> &'static str {
        F::FAMILY
    }
}

impl<F: TestBackendFamily> CapabilityNegotiation for TestBackend<F> {}

#[async_trait]
impl<F: TestBackendFamily> LlmBackend for TestBackend<F> {
    fn prepare(&self, request: ChatCompletionRequest) -> Result<reqwest::Request, BackendError> {
        // Points at an unreachable URL; `chat_completion` is overridden below so this is never sent.
        self.http
            .post("http://127.0.0.1:0/chat/completions")
            .json(&serde_json::json!({
                "model": request.model,
                "messages": request.messages,
            }))
            .build()
            .map_err(|e| BackendError::provider(F::FAMILY, e))
    }

    fn prepare_streaming(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<reqwest::Request, BackendError> {
        self.prepare(request)
    }

    async fn send(&self, _prepared: reqwest::Request) -> Result<reqwest::Response, BackendError> {
        unreachable!("send is not used: chat_completion is overridden")
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, BackendError> {
        Ok(response_for_model(request.model))
    }

    async fn stream_chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> Result<ChatCompletionStream, BackendError> {
        Ok(ChatCompletionStream::new(Box::pin(stream::empty())))
    }

    async fn parse(
        &self,
        _response: reqwest::Response,
    ) -> Result<ChatCompletionResponse, BackendError> {
        unreachable!("parse is not used: chat_completion is overridden")
    }

    async fn parse_streaming(
        &self,
        _response: reqwest::Response,
    ) -> Result<ChatCompletionStream, BackendError> {
        unreachable!("parse_streaming is not used: stream_chat_completion is overridden")
    }

    fn render_messages(&self, _messages: &[ChatMessage]) -> Result<String, BackendError> {
        Ok(F::RENDER.to_owned())
    }

    fn render_tools(&self, _tools: &[ToolDefinition]) -> Result<String, BackendError> {
        Ok("[]".to_owned())
    }

    fn family() -> &'static str
    where
        Self: Sized,
    {
        F::FAMILY
    }

    #[allow(clippy::new_ret_no_self)]
    fn new(
        _http: reqwest::ClientBuilder,
        _api_key: &str,
        _base_url: Option<&str>,
    ) -> Result<Arc<dyn LlmBackend>, BackendConstructError>
    where
        Self: Sized,
    {
        Ok(Arc::new(Self {
            http: reqwest::Client::new(),
            _f: PhantomData,
        }))
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

// --- marker families ---

struct Solo;
impl TestBackendFamily for Solo {
    const FAMILY: &'static str = "solo";
    const RENDER: &'static str = "solo-render";
}

struct Alpha;
impl TestBackendFamily for Alpha {
    const FAMILY: &'static str = "dup";
    const RENDER: &'static str = "alpha";
}

struct Beta;
impl TestBackendFamily for Beta {
    const FAMILY: &'static str = "dup"; // same as Alpha — both key under "dup"
    const RENDER: &'static str = "beta";
}

// --- tests ---

#[test]
fn factory_create_builds_named_backend() {
    let mut factory = BackendFactory::empty();
    factory.register::<TestBackend<Solo>>();

    let backend = factory
        .create("solo", reqwest::Client::builder(), "key", None)
        .unwrap_or_else(|e| panic!("create failed: {e}"));
    let client = ChatClient::new(
        backend,
        ChatClientOptions::new("test-model").with_system_prompt("Be concise."),
    );
    let request = client.create_request(vec![ChatMessage::user("hello")]);

    assert_eq!(client.family(), "solo");
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
fn factory_register_replaces_same_family() {
    let mut factory = BackendFactory::empty();
    factory.register::<TestBackend<Alpha>>();
    factory.register::<TestBackend<Beta>>();

    let backend = factory
        .create("dup", reqwest::Client::builder(), "key", None)
        .unwrap_or_else(|e| panic!("create failed: {e}"));
    // Beta was registered last under "dup" — its constructor must be the one that won.
    assert_eq!(backend.render_messages(&[]).unwrap(), "beta");
    // Replaced, not appended.
    assert_eq!(factory.families().count(), 1);
}

#[test]
fn factory_create_rejects_unknown_family() {
    let factory = BackendFactory::empty();
    let error = factory
        .create("missing", reqwest::Client::builder(), "key", None)
        .err()
        .expect("expected unknown-family error");

    assert!(matches!(error, BackendConstructError::UnknownFamily { .. }));
    assert_eq!(
        error.to_string(),
        "no backend registered for family 'missing'"
    );
}
