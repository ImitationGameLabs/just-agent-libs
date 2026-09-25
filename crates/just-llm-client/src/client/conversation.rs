//! Stateful conversation continuation for Responses-family backends.
//!
//! [`Conversation`] is a thin "continuation injector", not a history manager: the caller owns the
//! conversation and passes the **full logical message list** on every turn. The [`Conversation`]
//! only remembers enough to decide what the wire payload should be:
//!
//! - On a stateless backend (chat completions, Anthropic) it passes the caller's list through
//!   unchanged — the classic full-resend replay.
//! - On a Responses-family backend (OpenAI, xAI) it detects whether the caller's list is a pure
//!   append to the previous turn (`last_input ++ [last_assistant] ++ delta`) and, when it is, sends
//!   only `delta` plus `previous_response_id` + `store: true`, avoiding re-transmitting the stored
//!   prefix.
//!
//! Any divergence — trimming, rewriting, pinning, a changed generation setting, or an unknown
//! anchor after a streamed turn — makes the append detection fail conservatively and the request
//! goes out as a full stateless send (which re-anchors the chain on stateful backends).
//!
//! # Contract
//!
//! - Callers always pass the full logical context (system + history + new messages), mirroring the
//!   previous assistant turn verbatim via [`Conversation::last_message`] so the anchor can be
//!   recognized.
//! - Generation settings (model, tools, sampling, reasoning, response format) must stay constant
//!   across a chain; changing any of them is treated as a rewrite and forces a stateless send.
//! - The two request fields [`previous_response_id`](crate::types::generation::GenerationRequest::previous_response_id)
//!   and [`store`](crate::types::generation::GenerationRequest::store) are owned by the
//!   [`Conversation`]; [`Conversation::prepare`] and
//!   [`Conversation::prepare_streaming`](Conversation::prepare_streaming) write them into the
//!   wire payload, and callers must not set them directly.
//! - `store` (default `true`) is required for chaining: `with_store(false)` makes every turn a
//!   plain stateless send.
//!
//! # Streaming
//!
//! [`Conversation::stream_generate`] returns a [`ConversationStream`] that forwards every event
//! unchanged while assembling the assistant message from the deltas. Recording the turn is
//! explicit: consume the stream through its terminal `End` event, take the assembled response
//! with `into_response`, and pass it to [`Conversation::adopt`]. A stream that never reaches
//! `End`, whether dropped or cut short by an error, leaves nothing to adopt, so the previous
//! anchor stays valid and the next turn continues from it.
//!
//! # Failures
//!
//! A failed turn never touches the anchor: the previous successful turn's stored response and id
//! remain valid server-side, so the next turn keeps delta continuation. If the provider's stored
//! context expires (for example past a retention window), delta continuation keeps failing until
//! the caller rebuilds the conversation or disables storage with [`Conversation::with_store`];
//! the library never silently falls back and hides the loss.

use std::{collections::HashSet, fmt, sync::Arc, time::Instant};

pub use super::assembly::ConversationStream;
use crate::{
    BackendError,
    provider::{LlmBackend, attach_latency},
    types::generation::{
        AssistantMessage, GenerationRequest, GenerationResponse, Message, ReasoningEffort,
        ResponseFormat, StopSequence, ToolChoice, ToolDefinition,
    },
};

/// Generation settings that must stay constant across a stateful chain.
///
/// Excludes `messages` (the payload), `stream` (per-call), and the Conversation-managed
/// `previous_response_id`/`store`.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Settings {
    pub(super) model: String,
    tools: Option<Vec<ToolDefinition>>,
    tool_choice: Option<ToolChoice>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    max_tokens: Option<u32>,
    stop: Option<StopSequence>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    response_format: Option<ResponseFormat>,
    logprobs: Option<bool>,
    top_logprobs: Option<u8>,
    reasoning_effort: Option<ReasoningEffort>,
}

impl Settings {
    fn from_request(request: &GenerationRequest) -> Self {
        Self {
            model: request.model.clone(),
            tools: request.tools.clone(),
            tool_choice: request.tool_choice.clone(),
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            max_tokens: request.max_tokens,
            stop: request.stop.clone(),
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            response_format: request.response_format.clone(),
            logprobs: request.logprobs,
            top_logprobs: request.top_logprobs,
            reasoning_effort: request.reasoning_effort.clone(),
        }
    }
}

/// Anchor recorded after a completed turn; `S = last_input ++ [last_assistant]` is the server-side
/// conversation the caller's next full list is compared against.
struct Anchor {
    settings: Settings,
    last_input: Vec<Message>,
    last_assistant: AssistantMessage,
    previous_response_id: String,
}

/// Stateful conversation continuation on Responses-family backends.
///
/// Construct via [`GenerationClient::conversation`](crate::GenerationClient::conversation). The
/// caller owns the message list and passes the full logical context each turn; see the module
/// documentation for the contract.
pub struct Conversation {
    backend: Arc<dyn LlmBackend>,
    store: bool,
    anchor: Option<Anchor>,
    memo: Option<GenerationRequest>,
}

impl Conversation {
    /// Creates a conversation bound to `backend`, with server-side storage enabled.
    pub(crate) fn new(backend: Arc<dyn LlmBackend>) -> Self {
        Self {
            backend,
            store: true,
            anchor: None,
            memo: None,
        }
    }

    /// Sets whether server-side storage is enabled for this conversation.
    ///
    /// `false` disables stateful continuation entirely: every turn becomes a plain stateless send.
    /// Chaining via [`GenerationRequest::previous_response_id`] requires the referenced response
    /// to be stored, so storage cannot be disabled mid-chain.
    pub fn with_store(mut self, store: bool) -> Self {
        self.store = store;
        self
    }

    /// Whether this conversation will attempt stateful continuation.
    ///
    /// `true` only on Responses-family backends with server-side storage enabled; callers can use
    /// this for cost/behavior observability.
    pub fn is_stateful(&self) -> bool {
        self.store && self.backend.supports_stateful_conversation()
    }

    /// The assistant message of the last settled turn.
    ///
    /// Non-streaming turns settle when [`generate`](Self::generate) returns; streamed turns
    /// settle at [`adopt`](Self::adopt). Until a streamed turn is adopted, this keeps reporting
    /// the previous settled turn: the in-flight stream's assembled message is available from
    /// `into_response` on the stream itself. Mirror the settled turn verbatim into the next
    /// turn's message list so the anchor can be recognized.
    pub fn last_message(&self) -> Option<Message> {
        self.anchor
            .as_ref()
            .map(|anchor| Message::Assistant(anchor.last_assistant.clone()))
    }

    /// Judges the turn and renders the wire request without sending it.
    ///
    /// On success the request is recorded as the memo that the next [`adopt`](Self::adopt) pairs
    /// against; every successful call overwrites the memo, so adopt always pairs with the most
    /// recent prepare. The returned [`reqwest::Request`] is a plain HTTP request: the caller may
    /// add or rewrite headers (tracing, gateway routing) but must send the body verbatim, since
    /// it carries the judgment result, including the conversation-owned `previous_response_id`
    /// and `store`. To resend for a retry, clone it with [`reqwest::Request::try_clone`]. A
    /// failed prepare writes nothing: memo and anchor are untouched.
    pub fn prepare(
        &mut self,
        request: GenerationRequest,
    ) -> Result<reqwest::Request, BackendError> {
        let snapshot = request.clone();
        validate_tool_results(&request.messages)?;
        let wire = self.wire_request(request)?;
        let prepared = self.backend.prepare(wire)?;
        self.memo = Some(snapshot);
        Ok(prepared)
    }

    /// Judges a streaming turn and renders the SSE wire request without opening a stream.
    ///
    /// Same contract as [`prepare`](Self::prepare), except the rendered request asks the backend
    /// for a streamed response. The memo is recorded here exactly as for `prepare`.
    pub fn prepare_streaming(
        &mut self,
        request: GenerationRequest,
    ) -> Result<reqwest::Request, BackendError> {
        let snapshot = request.clone();
        validate_tool_results(&request.messages)?;
        let wire = self.wire_request(request)?;
        let prepared = self.backend.prepare_streaming(wire)?;
        self.memo = Some(snapshot);
        Ok(prepared)
    }

    /// Records a completed turn as the new anchor. This is the only method that writes
    /// the anchor after construction.
    ///
    /// Pairing contract: pass the response of the most recent successful
    /// [`prepare`](Self::prepare), or the response assembled from the stream opened by the most
    /// recent [`prepare_streaming`](Self::prepare_streaming). Adopting a response produced by an
    /// earlier prepare cannot be detected here and may chain the conversation to server-side
    /// context the caller never built. With storage disabled or a stateless backend this is a
    /// no-op. Adopting without a memo, or a response with an empty id, is an invalid request. Do
    /// not adopt a failed turn: a turn that is never adopted leaves the previous anchor valid.
    pub fn adopt(&mut self, response: GenerationResponse) -> Result<(), BackendError> {
        if !self.is_stateful() {
            return Ok(());
        }
        let memo = self.memo.as_ref().ok_or_else(|| {
            BackendError::invalid_request(
                "no prepared request to adopt against; call prepare or prepare_streaming first",
            )
        })?;
        if response.id.is_empty() {
            return Err(BackendError::invalid_request(
                "response id is empty; refusing to anchor the conversation on it",
            ));
        }
        self.anchor = Some(Anchor {
            settings: Settings::from_request(memo),
            last_input: memo.messages.clone(),
            last_assistant: response.message,
            previous_response_id: response.id,
        });
        Ok(())
    }

    /// Executes a non-streaming generation turn: prepare, send, parse, adopt.
    ///
    /// On a stateful backend, a pure append to the previous turn is sent as `previous_response_id`
    /// plus only the new messages; anything else is sent as a full request. A failed turn keeps
    /// the anchor: the next turn retries the delta against the same stored context.
    ///
    /// This is the convenience path: it returns a parsed [`GenerationResponse`] and does not
    /// expose response headers or the raw body. For header-aware retries or custom status
    /// handling, drive [`prepare`](Self::prepare) / [`send`](LlmBackend::send) /
    /// [`parse`](LlmBackend::parse) / [`adopt`](Self::adopt) yourself.
    ///
    /// The returned response's `meta.latency` measures this call end-to-end.
    pub async fn generate(
        &mut self,
        request: GenerationRequest,
    ) -> Result<GenerationResponse, BackendError> {
        let started = Instant::now();
        let prepared = self.prepare(request)?;
        let raw = self.backend.send(prepared).await?;
        let mut response = self.backend.parse(raw).await?;
        attach_latency(&mut response, started.elapsed());
        self.adopt(response.clone())?;
        Ok(response)
    }

    /// Executes a streaming generation turn: prepare_streaming, send, parse_streaming.
    ///
    /// The returned [`ConversationStream`] forwards events unchanged while assembling the
    /// assistant message. Recording the turn is explicit: consume the stream through its terminal
    /// `End` event, take the assembled response with
    /// `into_response`, and pass it to
    /// [`adopt`](Self::adopt). Until then [`last_message`](Self::last_message) keeps reporting
    /// the previous settled turn.
    pub async fn stream_generate(
        &mut self,
        request: GenerationRequest,
    ) -> Result<ConversationStream, BackendError> {
        let settings = Settings::from_request(&request);
        let prepared = self.prepare_streaming(request)?;
        let raw = self.backend.send(prepared).await?;
        let inner = self.backend.parse_streaming(raw).await?;
        Ok(ConversationStream::new(inner, settings))
    }

    /// Builds the wire request: full send, delta continuation, or a stateless re-anchor.
    fn wire_request(
        &self,
        mut request: GenerationRequest,
    ) -> Result<GenerationRequest, BackendError> {
        // The Conversation owns the two stateful fields on every wire payload.
        if !self.is_stateful() {
            request.previous_response_id = None;
            request.store = None;
            return Ok(request);
        }
        request.store = Some(true);

        let Some(anchor) = &self.anchor else {
            // First turn of a stateful conversation: full send, stored so the chain can continue.
            return Ok(request);
        };

        // A changed generation setting means the caller rewrote the request intent: go stateless.
        if anchor.settings != Settings::from_request(&request) {
            request.previous_response_id = None;
            return Ok(request);
        }

        // S = last_input ++ [last_assistant]. Append only when the caller's list extends S exactly.
        let split = anchor.last_input.len();
        if request.messages.len() <= split
            || request.messages[..split] != anchor.last_input[..]
            || request.messages[split] != Message::Assistant(anchor.last_assistant.clone())
        {
            request.previous_response_id = None;
            return Ok(request);
        }

        let delta = request.messages[split + 1..].to_vec();
        if delta.is_empty() {
            // Regenerate of the identical list: re-send statelessly rather than replay a stored turn.
            request.previous_response_id = None;
            return Ok(request);
        }

        validate_append_tools(&delta, &anchor.last_assistant)?;

        request.messages = delta;
        // Tools and tool_choice stay as the caller sent them: the settings fingerprint above
        // already guarantees they match the anchor, and the stored prefix does not inherit
        // request parameters such as the tool set.
        request.previous_response_id = Some(anchor.previous_response_id.clone());
        Ok(request)
    }
}

impl fmt::Debug for Conversation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Conversation")
            .field("family", &self.backend.family())
            .field("store", &self.store)
            .field("stateful", &self.is_stateful())
            .field("has_anchor", &self.anchor.is_some())
            .finish_non_exhaustive()
    }
}

/// Ensures every tool result in a message list answers a tool call issued earlier in the list.
///
/// A `tool` message whose `tool_call_id` matches no preceding assistant `tool_calls` is malformed
/// on every wire protocol (chat completions rejects a dangling `tool_call_id`; Responses rejects a
/// `function_call_output` without the corresponding `function_call`), so it fails fast here.
fn validate_tool_results(messages: &[Message]) -> Result<(), BackendError> {
    let mut issued = HashSet::new();
    for message in messages {
        match message {
            Message::Assistant(assistant) => {
                for call in &assistant.tool_calls {
                    issued.insert(call.id.as_str());
                }
            }
            Message::Tool { tool_call_id, .. } if !issued.contains(tool_call_id.as_str()) => {
                return Err(BackendError::invalid_request(format!(
                    "tool result {tool_call_id} has no preceding assistant tool call in the message list"
                )));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Ensures the delta of an append answers exactly the previous assistant turn's tool calls.
fn validate_append_tools(
    delta: &[Message],
    last_assistant: &AssistantMessage,
) -> Result<(), BackendError> {
    let issued: HashSet<&str> = last_assistant
        .tool_calls
        .iter()
        .map(|call| call.id.as_str())
        .collect();
    for message in delta {
        if let Message::Tool { tool_call_id, .. } = message {
            if !issued.contains(tool_call_id.as_str()) {
                return Err(BackendError::invalid_request(format!(
                    "tool result {tool_call_id} does not answer any tool call of the previous assistant turn"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{Arc, Mutex},
    };

    use async_trait::async_trait;
    use futures_util::StreamExt;
    use just_common::error::TransportError;

    use super::*;
    use crate::{
        capability::{GenerationStream, Identifiable},
        error::BackendConstructError,
        types::generation::{
            FinishReason, FunctionDefinition, GenerationEvent, ToolCall, ToolCallDelta, ToolChoice,
            ToolChoiceMode, ToolDefinition, ToolType, Usage,
        },
    };

    struct MockBackend {
        stateful: bool,
        requests: Mutex<Vec<GenerationRequest>>,
        responses: Mutex<VecDeque<Result<GenerationResponse, BackendError>>>,
        streams: Mutex<VecDeque<Vec<GenerationEvent>>>,
        send_script: Mutex<VecDeque<Result<(), BackendError>>>,
    }

    impl MockBackend {
        fn new(stateful: bool) -> Self {
            Self {
                stateful,
                requests: Mutex::new(Vec::new()),
                responses: Mutex::new(VecDeque::new()),
                streams: Mutex::new(VecDeque::new()),
                send_script: Mutex::new(VecDeque::new()),
            }
        }

        fn response(id: &str, message: AssistantMessage) -> GenerationResponse {
            GenerationResponse {
                id: id.to_owned(),
                model: "mock".to_owned(),
                message,
                finish_reason: Some(FinishReason::Stop),
                usage: None,
                meta: None,
            }
        }

        fn assistant(text: &str) -> AssistantMessage {
            AssistantMessage {
                content: Some(text.to_owned()),
                tool_calls: Vec::new(),
                reasoning: None,
            }
        }

        fn take_requests(&self) -> Vec<GenerationRequest> {
            std::mem::take(&mut *self.requests.lock().unwrap())
        }

        fn queue_response(&self, response: GenerationResponse) {
            self.responses.lock().unwrap().push_back(Ok(response));
        }

        fn queue_error(&self, error: BackendError) {
            self.responses.lock().unwrap().push_back(Err(error));
        }

        fn queue_stream(&self, events: Vec<GenerationEvent>) {
            self.streams.lock().unwrap().push_back(events);
        }

        fn queue_send_error(&self, error: BackendError) {
            self.send_script.lock().unwrap().push_back(Err(error));
        }

        fn dummy_request(&self) -> reqwest::Request {
            reqwest::Request::new(
                reqwest::Method::POST,
                "http://mock.local/".parse().expect("valid mock url"),
            )
        }

        fn dummy_response() -> reqwest::Response {
            reqwest::Response::from(http::Response::new(""))
        }
    }

    impl Identifiable for MockBackend {
        fn family(&self) -> &'static str {
            "mock"
        }
    }

    impl crate::capability::CapabilityNegotiation for MockBackend {
        fn supports_stateful_conversation(&self) -> bool {
            self.stateful
        }
    }

    #[async_trait]
    impl LlmBackend for MockBackend {
        fn prepare(&self, request: GenerationRequest) -> Result<reqwest::Request, BackendError> {
            self.requests.lock().unwrap().push(request);
            Ok(self.dummy_request())
        }

        fn prepare_streaming(
            &self,
            request: GenerationRequest,
        ) -> Result<reqwest::Request, BackendError> {
            self.requests.lock().unwrap().push(request);
            Ok(self.dummy_request())
        }

        async fn send(
            &self,
            _prepared: reqwest::Request,
        ) -> Result<reqwest::Response, BackendError> {
            match self.send_script.lock().unwrap().pop_front() {
                Some(Err(error)) => Err(error),
                _ => Ok(Self::dummy_response()),
            }
        }

        async fn parse(
            &self,
            _response: reqwest::Response,
        ) -> Result<GenerationResponse, BackendError> {
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .expect("no response queued for mock backend")
        }

        async fn parse_streaming(
            &self,
            _response: reqwest::Response,
        ) -> Result<GenerationStream, BackendError> {
            let events = self
                .streams
                .lock()
                .unwrap()
                .pop_front()
                .expect("no stream queued for mock backend");
            let stream =
                futures_util::stream::iter(events.into_iter().map(Ok::<_, TransportError>));
            Ok(GenerationStream::new(Box::pin(stream)))
        }

        fn render_messages(&self, _messages: &[Message]) -> Result<String, BackendError> {
            unimplemented!("not exercised by tests")
        }

        fn render_tools(&self, _tools: &[ToolDefinition]) -> Result<String, BackendError> {
            unimplemented!("not exercised by tests")
        }

        fn family() -> &'static str {
            "mock"
        }

        fn new(
            _http: reqwest::ClientBuilder,
            _api_key: &str,
            _base_url: Option<&str>,
        ) -> Result<Arc<dyn LlmBackend>, BackendConstructError> {
            unimplemented!("not exercised by tests")
        }
    }

    #[tokio::test]
    async fn stateless_backend_replays_full_requests() {
        let backend = Arc::new(MockBackend::new(false));
        backend.queue_response(MockBackend::response("r1", MockBackend::assistant("a1")));
        backend.queue_response(MockBackend::response("r2", MockBackend::assistant("a2")));
        let mut conv = Conversation::new(backend.clone());
        assert!(!conv.is_stateful());

        let m1 = vec![Message::user("q1")];
        conv.generate(GenerationRequest::new("mock", m1.clone()))
            .await
            .unwrap();
        let m2 = vec![
            Message::user("q1"),
            Message::assistant("a1"),
            Message::user("q2"),
        ];
        conv.generate(GenerationRequest::new("mock", m2.clone()))
            .await
            .unwrap();

        let requests = backend.take_requests();
        assert_eq!(requests.len(), 2);
        assert!(requests[0].previous_response_id.is_none());
        assert!(requests[0].store.is_none());
        assert_eq!(requests[0].messages, m1);
        assert!(requests[1].previous_response_id.is_none());
        assert!(requests[1].store.is_none());
        assert_eq!(requests[1].messages, m2);
    }

    #[tokio::test]
    async fn generate_attaches_latency_to_the_returned_response() {
        let backend = Arc::new(MockBackend::new(false));
        backend.queue_response(MockBackend::response("r1", MockBackend::assistant("a1")));
        let mut conv = Conversation::new(backend);

        let response = conv
            .generate(GenerationRequest::new("mock", vec![Message::user("q1")]))
            .await
            .unwrap();

        let meta = response
            .meta
            .expect("Conversation::generate measures latency");
        assert!(meta.latency.is_some());
        assert_eq!(meta.request_id, None);
    }

    #[tokio::test]
    async fn backend_generate_attaches_latency_to_the_returned_response() {
        let backend = Arc::new(MockBackend::new(false));
        backend.queue_response(MockBackend::response("r1", MockBackend::assistant("a1")));

        let response = backend
            .generate(GenerationRequest::new("mock", vec![Message::user("q1")]))
            .await
            .unwrap();

        assert!(response.meta.as_ref().unwrap().latency.is_some());
    }

    #[tokio::test]
    async fn stateful_first_turn_stores_then_append_sends_delta() {
        let backend = Arc::new(MockBackend::new(true));
        backend.queue_response(MockBackend::response("r1", MockBackend::assistant("a1")));
        backend.queue_response(MockBackend::response("r2", MockBackend::assistant("a2")));
        let mut conv = Conversation::new(backend.clone());
        assert!(conv.is_stateful());

        let m1 = vec![Message::user("q1")];
        conv.generate(GenerationRequest::new("mock", m1.clone()))
            .await
            .unwrap();

        let mirror = conv.last_message().unwrap();
        let m2 = vec![Message::user("q1"), mirror, Message::user("q2")];
        conv.generate(GenerationRequest::new("mock", m2))
            .await
            .unwrap();

        let requests = backend.take_requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].previous_response_id, None);
        assert_eq!(requests[0].store, Some(true));
        assert_eq!(requests[0].messages, m1);
        assert_eq!(requests[1].previous_response_id.as_deref(), Some("r1"));
        assert_eq!(requests[1].store, Some(true));
        assert_eq!(requests[1].messages, vec![Message::user("q2")]);
    }

    #[tokio::test]
    async fn append_delta_carries_tools_and_tool_choice() {
        let backend = Arc::new(MockBackend::new(true));
        backend.queue_response(MockBackend::response("r1", MockBackend::assistant("a1")));
        backend.queue_response(MockBackend::response("r2", MockBackend::assistant("a2")));
        let mut conv = Conversation::new(backend.clone());

        let tools = vec![ToolDefinition {
            kind: ToolType::Function,
            function: FunctionDefinition {
                name: "get_weather".to_owned(),
                description: None,
                parameters: None,
                strict: None,
            },
        }];
        let request = GenerationRequest::new("mock", vec![Message::user("q1")])
            .with_tools(tools.clone())
            .with_tool_choice(ToolChoice::Mode(ToolChoiceMode::Auto));
        conv.generate(request).await.unwrap();

        let mirror = conv.last_message().unwrap();
        let m2 = vec![Message::user("q1"), mirror, Message::user("q2")];
        let request = GenerationRequest::new("mock", m2)
            .with_tools(tools.clone())
            .with_tool_choice(ToolChoice::Mode(ToolChoiceMode::Auto));
        conv.generate(request).await.unwrap();

        let requests = backend.take_requests();
        assert_eq!(requests[1].previous_response_id.as_deref(), Some("r1"));
        // The delta keeps the caller's tools: the model cannot use a tool the wire request omits.
        assert_eq!(requests[1].tools.as_deref(), Some(tools.as_slice()));
        assert_eq!(
            requests[1].tool_choice,
            Some(ToolChoice::Mode(ToolChoiceMode::Auto))
        );
    }

    #[tokio::test]
    async fn append_without_mirror_falls_back_to_stateless() {
        let backend = Arc::new(MockBackend::new(true));
        backend.queue_response(MockBackend::response("r1", MockBackend::assistant("a1")));
        backend.queue_response(MockBackend::response("r2", MockBackend::assistant("a2")));
        let mut conv = Conversation::new(backend.clone());

        let m1 = vec![Message::user("q1")];
        conv.generate(GenerationRequest::new("mock", m1.clone()))
            .await
            .unwrap();

        let m2 = vec![Message::user("q1"), Message::user("q2")];
        conv.generate(GenerationRequest::new("mock", m2.clone()))
            .await
            .unwrap();

        let requests = backend.take_requests();
        assert!(requests[1].previous_response_id.is_none());
        assert_eq!(requests[1].messages, m2);
    }

    #[tokio::test]
    async fn trimmed_context_falls_back_to_stateless() {
        let backend = Arc::new(MockBackend::new(true));
        backend.queue_response(MockBackend::response("r1", MockBackend::assistant("a1")));
        backend.queue_response(MockBackend::response("r2", MockBackend::assistant("a2")));
        let mut conv = Conversation::new(backend.clone());

        let m1 = vec![Message::user("q1")];
        conv.generate(GenerationRequest::new("mock", m1))
            .await
            .unwrap();

        // Caller dropped the assistant turn and prepended a fresh system prompt: not an append.
        let m2 = vec![Message::system("new rules"), Message::user("q2")];
        conv.generate(GenerationRequest::new("mock", m2.clone()))
            .await
            .unwrap();

        let requests = backend.take_requests();
        assert!(requests[1].previous_response_id.is_none());
        assert_eq!(requests[1].messages, m2);
    }

    #[tokio::test]
    async fn settings_change_forces_stateless() {
        let backend = Arc::new(MockBackend::new(true));
        backend.queue_response(MockBackend::response("r1", MockBackend::assistant("a1")));
        backend.queue_response(MockBackend::response("r2", MockBackend::assistant("a2")));
        let mut conv = Conversation::new(backend.clone());

        conv.generate(
            GenerationRequest::new("mock", vec![Message::user("q1")]).with_temperature(0.2),
        )
        .await
        .unwrap();

        let mirror = conv.last_message().unwrap();
        let m2 = vec![Message::user("q1"), mirror, Message::user("q2")];
        conv.generate(GenerationRequest::new("mock", m2.clone()).with_temperature(0.8))
            .await
            .unwrap();

        let requests = backend.take_requests();
        assert!(requests[1].previous_response_id.is_none());
        assert_eq!(requests[1].messages, m2);
    }

    #[tokio::test]
    async fn identical_list_regenerates_stateless() {
        let backend = Arc::new(MockBackend::new(true));
        backend.queue_response(MockBackend::response("r1", MockBackend::assistant("a1")));
        backend.queue_response(MockBackend::response("r2", MockBackend::assistant("a2")));
        let mut conv = Conversation::new(backend.clone());

        let m1 = vec![Message::user("q1")];
        conv.generate(GenerationRequest::new("mock", m1.clone()))
            .await
            .unwrap();

        // Re-send the identical list: a fresh generation, not a continuation of the stored turn.
        conv.generate(GenerationRequest::new("mock", m1.clone()))
            .await
            .unwrap();

        let requests = backend.take_requests();
        assert!(requests[1].previous_response_id.is_none());
        assert_eq!(requests[1].messages, m1);
    }

    #[tokio::test]
    async fn store_disabled_never_chains() {
        let backend = Arc::new(MockBackend::new(true));
        backend.queue_response(MockBackend::response("r1", MockBackend::assistant("a1")));
        backend.queue_response(MockBackend::response("r2", MockBackend::assistant("a2")));
        let mut conv = Conversation::new(backend.clone()).with_store(false);
        assert!(!conv.is_stateful());

        conv.generate(GenerationRequest::new("mock", vec![Message::user("q1")]))
            .await
            .unwrap();
        conv.generate(GenerationRequest::new(
            "mock",
            vec![
                Message::user("q1"),
                Message::assistant("a1"),
                Message::user("q2"),
            ],
        ))
        .await
        .unwrap();

        let requests = backend.take_requests();
        assert!(requests.iter().all(|r| r.previous_response_id.is_none()));
        assert!(requests.iter().all(|r| r.store.is_none()));
    }

    #[tokio::test]
    async fn dangling_tool_result_is_rejected() {
        let backend = Arc::new(MockBackend::new(true));
        let mut conv = Conversation::new(backend.clone());

        let error = conv
            .generate(GenerationRequest::new(
                "mock",
                vec![Message::tool("25C", "call_missing")],
            ))
            .await
            .unwrap_err();

        assert!(matches!(error, BackendError::InvalidRequest(_)));
        assert!(backend.take_requests().is_empty());
    }

    #[tokio::test]
    async fn tool_loop_appends_results_against_previous_assistant() {
        let backend = Arc::new(MockBackend::new(true));
        backend.queue_response(MockBackend::response(
            "r1",
            AssistantMessage {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_1".to_owned(),
                    name: "get_weather".to_owned(),
                    arguments: r#"{"city":"Paris"}"#.to_owned(),
                }],
                reasoning: None,
            },
        ));
        backend.queue_response(MockBackend::response("r2", MockBackend::assistant("25C")));
        let mut conv = Conversation::new(backend.clone());

        conv.generate(GenerationRequest::new(
            "mock",
            vec![Message::user("weather?")],
        ))
        .await
        .unwrap();

        let mirror = conv.last_message().unwrap();
        let m2 = vec![
            Message::user("weather?"),
            mirror,
            Message::tool("25C", "call_1"),
        ];
        conv.generate(GenerationRequest::new("mock", m2))
            .await
            .unwrap();

        let requests = backend.take_requests();
        assert_eq!(requests[1].previous_response_id.as_deref(), Some("r1"));
        // The delta carries only the tool result, not the mirror or the original user message.
        assert_eq!(requests[1].messages, vec![Message::tool("25C", "call_1")]);
    }

    #[tokio::test]
    async fn tool_loop_rejects_unrelated_tool_result_in_delta() {
        let backend = Arc::new(MockBackend::new(true));
        backend.queue_response(MockBackend::response(
            "r1",
            AssistantMessage {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "call_1".to_owned(),
                    name: "get_weather".to_owned(),
                    arguments: "{}".to_owned(),
                }],
                reasoning: None,
            },
        ));
        let mut conv = Conversation::new(backend.clone());

        conv.generate(GenerationRequest::new(
            "mock",
            vec![Message::user("weather?")],
        ))
        .await
        .unwrap();

        let mirror = conv.last_message().unwrap();
        let m2 = vec![
            Message::user("weather?"),
            mirror,
            Message::tool("25C", "call_other"),
        ];
        let error = conv
            .generate(GenerationRequest::new("mock", m2))
            .await
            .unwrap_err();
        assert!(matches!(error, BackendError::InvalidRequest(_)));
    }

    #[tokio::test]
    async fn failed_turn_keeps_anchor_and_next_turn_continues_delta() {
        let backend = Arc::new(MockBackend::new(true));
        backend.queue_response(MockBackend::response("r1", MockBackend::assistant("a1")));
        backend.queue_error(BackendError::provider(
            "mock",
            just_common::error::ProviderError::InvalidRequest("boom".to_owned()),
        ));
        backend.queue_response(MockBackend::response("r3", MockBackend::assistant("a3")));
        let mut conv = Conversation::new(backend.clone());

        conv.generate(GenerationRequest::new("mock", vec![Message::user("q1")]))
            .await
            .unwrap();
        assert!(
            conv.generate(GenerationRequest::new(
                "mock",
                vec![
                    Message::user("q1"),
                    conv.last_message().unwrap(),
                    Message::user("q2")
                ],
            ))
            .await
            .is_err()
        );

        // The anchor survives the failed turn: the next turn still chains from r1 as a delta.
        let m3 = vec![
            Message::user("q1"),
            Message::assistant("a1"),
            Message::user("q3"),
        ];
        conv.generate(GenerationRequest::new("mock", m3))
            .await
            .unwrap();

        let requests = backend.take_requests();
        assert_eq!(requests.len(), 3);
        assert_eq!(requests[1].previous_response_id.as_deref(), Some("r1"));
        assert_eq!(requests[2].previous_response_id.as_deref(), Some("r1"));
        assert_eq!(requests[2].messages, vec![Message::user("q3")]);
    }

    #[tokio::test]
    async fn streamed_turn_assembles_and_is_adopted_next() {
        let backend = Arc::new(MockBackend::new(true));
        backend.queue_stream(vec![
            GenerationEvent::Text {
                delta: "The answer is ".to_owned(),
            },
            GenerationEvent::Text {
                delta: "42.".to_owned(),
            },
            GenerationEvent::ToolCall {
                delta: ToolCallDelta {
                    index: Some(0),
                    id: Some("call_1".to_owned()),
                    name: Some("get_weather".to_owned()),
                    arguments: None,
                },
            },
            GenerationEvent::ToolCall {
                delta: ToolCallDelta {
                    index: Some(0),
                    id: None,
                    name: None,
                    arguments: Some(r#"{"city":"Paris"}"#.to_owned()),
                },
            },
            GenerationEvent::End {
                finish_reason: Some(FinishReason::Stop),
                response_id: Some("rs_1".to_owned()),
            },
        ]);
        backend.queue_response(MockBackend::response("r2", MockBackend::assistant("a2")));
        let mut conv = Conversation::new(backend.clone());

        let mut stream = conv
            .stream_generate(GenerationRequest::new("mock", vec![Message::user("q1")]))
            .await
            .unwrap();
        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            events.push(event.unwrap());
        }
        assert_eq!(events.len(), 5);

        // The assembled response carries the turn's identity and message until adopt records it.
        let response = stream.into_response().expect("End carried a response id");
        assert_eq!(response.id, "rs_1");
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        let mirror = Message::Assistant(response.message.clone());
        assert_eq!(mirror.content(), Some("The answer is 42."));
        assert_eq!(mirror.tool_calls().len(), 1);
        assert_eq!(mirror.tool_calls()[0].name, "get_weather");
        assert_eq!(mirror.tool_calls()[0].arguments, r#"{"city":"Paris"}"#);
        conv.adopt(response).unwrap();

        let m2 = vec![Message::user("q1"), mirror, Message::user("q2")];
        conv.generate(GenerationRequest::new("mock", m2))
            .await
            .unwrap();

        let requests = backend.take_requests();
        assert_eq!(requests[0].previous_response_id, None);
        assert_eq!(requests[0].store, Some(true));
        assert_eq!(requests[1].previous_response_id.as_deref(), Some("rs_1"));
        assert_eq!(requests[1].messages, vec![Message::user("q2")]);
    }

    #[tokio::test]
    async fn abandoned_stream_does_not_corrupt_anchor() {
        let backend = Arc::new(MockBackend::new(true));
        backend.queue_response(MockBackend::response("r1", MockBackend::assistant("a1")));
        backend.queue_stream(vec![GenerationEvent::End {
            finish_reason: Some(FinishReason::Stop),
            response_id: Some("rs_1".to_owned()),
        }]);
        backend.queue_response(MockBackend::response("r3", MockBackend::assistant("a3")));
        let mut conv = Conversation::new(backend.clone());

        conv.generate(GenerationRequest::new("mock", vec![Message::user("q1")]))
            .await
            .unwrap();

        // Start a streamed turn that cannot be recognized as an append, and abandon it before
        // its terminal event: `into_response` yields nothing, so nothing is adopted and the
        // anchor from r1 stays valid.
        let stream = conv
            .stream_generate(GenerationRequest::new("mock", vec![Message::user("q2")]))
            .await
            .unwrap();
        assert!(stream.into_response().is_none());
        // The abandoned turn's wire request went out stateless, and the next turn still chains
        // from the last completed anchor (r1) since nothing adopted the stream.
        let m3 = vec![
            Message::user("q1"),
            Message::assistant("a1"),
            Message::user("q3"),
        ];
        conv.generate(GenerationRequest::new("mock", m3))
            .await
            .unwrap();

        let requests = backend.take_requests();
        assert_eq!(requests.len(), 3);
        assert_eq!(requests[1].previous_response_id, None);
        assert_eq!(requests[1].store, Some(true));
        assert_eq!(requests[1].messages, vec![Message::user("q2")]);
        assert_eq!(requests[2].previous_response_id.as_deref(), Some("r1"));
        assert_eq!(requests[2].messages, vec![Message::user("q3")]);
    }

    #[tokio::test]
    async fn last_message_reflects_last_completed_turn() {
        let backend = Arc::new(MockBackend::new(true));
        backend.queue_response(MockBackend::response("r1", MockBackend::assistant("a1")));
        let mut conv = Conversation::new(backend.clone());
        assert!(conv.last_message().is_none());

        conv.generate(GenerationRequest::new("mock", vec![Message::user("q1")]))
            .await
            .unwrap();
        assert_eq!(conv.last_message().unwrap().content(), Some("a1"));
    }

    #[tokio::test]
    async fn stateless_stream_yields_no_adoptable_response() {
        let backend = Arc::new(MockBackend::new(false));
        backend.queue_stream(vec![
            GenerationEvent::Text {
                delta: "hi".to_owned(),
            },
            GenerationEvent::End {
                finish_reason: Some(FinishReason::Stop),
                response_id: None,
            },
        ]);
        let mut conv = Conversation::new(backend.clone());

        let mut stream = conv
            .stream_generate(GenerationRequest::new("mock", vec![Message::user("q1")]))
            .await
            .unwrap();
        while stream.next().await.is_some() {}

        // A stateless backend's End carries no id, so there is nothing to adopt, and the
        // never-adopted turn does not show up in last_message.
        assert!(stream.into_response().is_none());
        assert!(conv.last_message().is_none());
    }

    #[tokio::test]
    async fn adopt_without_prepare_is_rejected() {
        let backend = Arc::new(MockBackend::new(true));
        let mut conv = Conversation::new(backend);

        let error = conv
            .adopt(MockBackend::response("r1", MockBackend::assistant("a1")))
            .unwrap_err();
        assert!(matches!(error, BackendError::InvalidRequest(_)));
    }

    #[tokio::test]
    async fn adopt_rejects_empty_response_id() {
        let backend = Arc::new(MockBackend::new(true));
        let mut conv = Conversation::new(backend);
        conv.prepare(GenerationRequest::new("mock", vec![Message::user("q1")]))
            .unwrap();

        let error = conv
            .adopt(MockBackend::response("", MockBackend::assistant("a1")))
            .unwrap_err();
        assert!(matches!(error, BackendError::InvalidRequest(_)));
    }

    #[tokio::test]
    async fn adopt_on_stateless_backend_is_noop() {
        let backend = Arc::new(MockBackend::new(false));
        backend.queue_response(MockBackend::response("r1", MockBackend::assistant("a1")));
        let mut conv = Conversation::new(backend.clone());

        conv.adopt(MockBackend::response("r0", MockBackend::assistant("a0")))
            .unwrap();
        conv.generate(GenerationRequest::new(
            "mock",
            vec![
                Message::user("q1"),
                Message::assistant("a1"),
                Message::user("q2"),
            ],
        ))
        .await
        .unwrap();

        let requests = backend.take_requests();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].previous_response_id.is_none());
        assert!(requests[0].store.is_none());
    }

    #[tokio::test]
    async fn failed_prepare_leaves_state_untouched() {
        let backend = Arc::new(MockBackend::new(true));
        backend.queue_response(MockBackend::response("r1", MockBackend::assistant("a1")));
        backend.queue_response(MockBackend::response("r2", MockBackend::assistant("a2")));
        let mut conv = Conversation::new(backend.clone());

        let error = conv
            .prepare(GenerationRequest::new(
                "mock",
                vec![Message::tool("25C", "call_missing")],
            ))
            .unwrap_err();
        assert!(matches!(error, BackendError::InvalidRequest(_)));
        assert!(backend.take_requests().is_empty());

        let prepared = conv
            .prepare(GenerationRequest::new("mock", vec![Message::user("q1")]))
            .unwrap();
        let raw = backend.send(prepared).await.unwrap();
        conv.adopt(backend.parse(raw).await.unwrap()).unwrap();

        let mirror = conv.last_message().unwrap();
        let m2 = vec![Message::user("q1"), mirror, Message::user("q2")];
        conv.generate(GenerationRequest::new("mock", m2))
            .await
            .unwrap();

        let requests = backend.take_requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].previous_response_id.as_deref(), Some("r1"));
        assert_eq!(requests[1].messages, vec![Message::user("q2")]);
    }

    #[tokio::test]
    async fn reprepare_overwrites_memo() {
        let backend = Arc::new(MockBackend::new(true));
        backend.queue_response(MockBackend::response("r2", MockBackend::assistant("a2")));
        backend.queue_response(MockBackend::response("r3", MockBackend::assistant("a3")));
        let mut conv = Conversation::new(backend.clone());

        conv.prepare(GenerationRequest::new("mock", vec![Message::user("q1")]))
            .unwrap();
        // The second prepare overwrites the memo, so adopt pairs with this request.
        let prepared = conv
            .prepare(GenerationRequest::new("mock", vec![Message::user("q9")]))
            .unwrap();
        let raw = backend.send(prepared).await.unwrap();
        conv.adopt(backend.parse(raw).await.unwrap()).unwrap();

        let m2 = vec![
            Message::user("q9"),
            Message::assistant("a2"),
            Message::user("q10"),
        ];
        conv.generate(GenerationRequest::new("mock", m2))
            .await
            .unwrap();

        let requests = backend.take_requests();
        assert_eq!(requests.len(), 3);
        assert_eq!(requests[2].previous_response_id.as_deref(), Some("r2"));
        assert_eq!(requests[2].messages, vec![Message::user("q10")]);
    }

    #[tokio::test]
    async fn retry_with_try_clone_adopts_after_send_failure() {
        let backend = Arc::new(MockBackend::new(true));
        backend.queue_response(MockBackend::response("r1", MockBackend::assistant("a1")));
        backend.queue_response(MockBackend::response("r2", MockBackend::assistant("a2")));
        backend.queue_send_error(BackendError::provider(
            "mock",
            just_common::error::ProviderError::InvalidRequest("boom".to_owned()),
        ));
        let mut conv = Conversation::new(backend.clone());

        let prepared = conv
            .prepare(GenerationRequest::new("mock", vec![Message::user("q1")]))
            .unwrap();

        // First send fails; the caller resends a clone of the same prepared request and the
        // turn settles normally.
        assert!(backend.send(prepared.try_clone().unwrap()).await.is_err());
        let raw = backend.send(prepared.try_clone().unwrap()).await.unwrap();
        conv.adopt(backend.parse(raw).await.unwrap()).unwrap();

        let mirror = conv.last_message().unwrap();
        let m2 = vec![Message::user("q1"), mirror, Message::user("q2")];
        conv.generate(GenerationRequest::new("mock", m2))
            .await
            .unwrap();

        let requests = backend.take_requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].previous_response_id.as_deref(), Some("r1"));
        assert_eq!(requests[1].messages, vec![Message::user("q2")]);
    }

    #[tokio::test]
    async fn mismatched_adopt_degrades_to_stateless() {
        let backend = Arc::new(MockBackend::new(true));
        backend.queue_response(MockBackend::response("r2", MockBackend::assistant("a2")));
        let mut conv = Conversation::new(backend.clone());

        conv.prepare(GenerationRequest::new("mock", vec![Message::user("q1")]))
            .unwrap();
        // Adopt the response of a different logical turn (the detectable direction of the
        // pairing contract): the anchor now pairs A's list with B's response.
        conv.adopt(MockBackend::response("rB", MockBackend::assistant("aB")))
            .unwrap();

        // The caller's list mirrors B, not the anchor's list: append detection fails and the
        // turn degrades to a full stateless send instead of erroring or chaining wrongly.
        let m2 = vec![
            Message::user("q9"),
            Message::assistant("aB"),
            Message::user("q10"),
        ];
        conv.generate(GenerationRequest::new("mock", m2.clone()))
            .await
            .unwrap();

        let requests = backend.take_requests();
        assert_eq!(requests.len(), 2);
        assert!(requests[1].previous_response_id.is_none());
        assert_eq!(requests[1].messages, m2);
    }

    #[tokio::test]
    async fn header_mutation_leaves_wire_body_unchanged() {
        let backend = Arc::new(MockBackend::new(true));
        backend.queue_response(MockBackend::response("r1", MockBackend::assistant("a1")));
        backend.queue_response(MockBackend::response("r2", MockBackend::assistant("a2")));
        let mut conv = Conversation::new(backend.clone());

        let request = GenerationRequest::new("mock", vec![Message::user("q1")]);
        let mut mutated = conv.prepare(request).unwrap();
        mutated.headers_mut().insert(
            http::HeaderName::from_static("x-trace-id"),
            http::HeaderValue::from_static("trace-1"),
        );
        assert_eq![
            mutated.headers_mut().get("x-trace-id"),
            Some(&http::HeaderValue::from_static("trace-1"))
        ];

        // The mutated copy still flows through send/parse/adopt, and the chain continues.
        let raw = backend.send(mutated).await.unwrap();
        conv.adopt(backend.parse(raw).await.unwrap()).unwrap();
        let mirror = conv.last_message().unwrap();
        let m2 = vec![Message::user("q1"), mirror, Message::user("q2")];
        conv.generate(GenerationRequest::new("mock", m2))
            .await
            .unwrap();
        let requests = backend.take_requests();
        assert_eq!(requests[1].previous_response_id.as_deref(), Some("r1"));
        assert_eq!(requests[1].messages, vec![Message::user("q2")]);
    }

    #[tokio::test]
    async fn into_response_is_none_before_the_end_event() {
        let backend = Arc::new(MockBackend::new(true));
        backend.queue_stream(vec![
            GenerationEvent::Text {
                delta: "hi".to_owned(),
            },
            GenerationEvent::End {
                finish_reason: Some(FinishReason::Stop),
                response_id: Some("rs_1".to_owned()),
            },
        ]);
        let mut conv = Conversation::new(backend);

        let mut stream = conv
            .stream_generate(GenerationRequest::new("mock", vec![Message::user("q1")]))
            .await
            .unwrap();
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            GenerationEvent::Text { .. }
        ));

        // The terminal event has not been consumed yet: nothing to adopt.
        assert!(stream.into_response().is_none());
    }

    #[tokio::test]
    async fn last_usage_report_wins() {
        let backend = Arc::new(MockBackend::new(true));
        backend.queue_stream(vec![
            GenerationEvent::Text {
                delta: "hi".to_owned(),
            },
            GenerationEvent::Usage {
                usage: Usage {
                    completion_tokens: 7,
                    prompt_tokens: 11,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                    total_tokens: 18,
                    completion_tokens_details: None,
                },
            },
            GenerationEvent::Usage {
                usage: Usage {
                    completion_tokens: 5,
                    prompt_tokens: 100,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                    total_tokens: 105,
                    completion_tokens_details: None,
                },
            },
            GenerationEvent::End {
                finish_reason: Some(FinishReason::Stop),
                response_id: Some("rs_1".to_owned()),
            },
        ]);
        let mut conv = Conversation::new(backend);

        let mut stream = conv
            .stream_generate(GenerationRequest::new("mock", vec![Message::user("q1")]))
            .await
            .unwrap();
        while stream.next().await.is_some() {}

        // A late report supersedes the earlier one instead of merging with it.
        let usage = stream.into_response().unwrap().usage.unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 105);
    }

    #[tokio::test]
    async fn assembled_response_carries_all_fields() {
        let backend = Arc::new(MockBackend::new(true));
        backend.queue_stream(vec![
            GenerationEvent::Text {
                delta: "The answer is ".to_owned(),
            },
            GenerationEvent::Reasoning {
                delta: "thinking".to_owned(),
            },
            GenerationEvent::ToolCall {
                delta: ToolCallDelta {
                    index: Some(0),
                    id: Some("call_1".to_owned()),
                    name: Some("get_weather".to_owned()),
                    arguments: None,
                },
            },
            GenerationEvent::ToolCall {
                delta: ToolCallDelta {
                    index: Some(0),
                    id: None,
                    name: None,
                    arguments: Some(r#"{"city":"Paris"}"#.to_owned()),
                },
            },
            GenerationEvent::ToolCall {
                delta: ToolCallDelta {
                    index: Some(1),
                    id: Some("call_2".to_owned()),
                    name: None,
                    arguments: None,
                },
            },
            GenerationEvent::Usage {
                usage: Usage {
                    completion_tokens: 7,
                    prompt_tokens: 11,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                    total_tokens: 18,
                    completion_tokens_details: None,
                },
            },
            GenerationEvent::End {
                finish_reason: Some(FinishReason::Stop),
                response_id: Some("rs_9".to_owned()),
            },
        ]);
        let mut conv = Conversation::new(backend);

        let mut stream = conv
            .stream_generate(GenerationRequest::new("mock", vec![Message::user("q1")]))
            .await
            .unwrap();
        while stream.next().await.is_some() {}

        let response = stream.into_response().unwrap();
        assert_eq!(response.id, "rs_9");
        assert_eq!(response.model, "mock");
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(response.message.content.as_deref(), Some("The answer is "));
        let reasoning = response.message.reasoning.as_ref().unwrap();
        assert_eq!(reasoning.text.as_deref(), Some("thinking"));
        // The nameless call_2 fragment is dropped; call_1 assembles completely.
        assert_eq!(response.message.tool_calls.len(), 1);
        assert_eq!(response.message.tool_calls[0].id, "call_1");
        assert_eq!(
            response.message.tool_calls[0].arguments,
            r#"{"city":"Paris"}"#
        );
        let usage = response.usage.unwrap();
        assert_eq!(usage.completion_tokens, 7);
        assert_eq!(usage.prompt_tokens, 11);
        assert_eq!(usage.total_tokens, 18);
        assert_eq!(usage.cache_read_tokens, None);
    }
}
