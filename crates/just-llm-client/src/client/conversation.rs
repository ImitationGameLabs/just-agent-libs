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
//!   [`Conversation`]; callers do not set them directly.
//! - `store` (default `true`) is required for chaining: `with_store(false)` makes every turn a
//!   plain stateless send.
//!
//! # Streaming
//!
//! [`Conversation::stream_generate`] returns a [`ConversationStream`] that forwards every event
//! unchanged while assembling the assistant message from the deltas. On the terminal `End` event
//! the stream records the assembled message plus the response id; the next turn adopts them as the
//! new anchor. A stream that never reaches `End` (dropped, or cut short by an error) leaves the
//! anchor unadopted, so the following turn goes stateless.

use std::{
    collections::HashSet,
    fmt,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
};

use futures_core::Stream;
use futures_util::StreamExt;

use crate::{
    BackendError,
    capability::GenerationStream,
    provider::LlmBackend,
    types::generation::{
        AssistantMessage, GenerationEvent, GenerationRequest, GenerationResponse, Message,
        Reasoning, ReasoningEffort, ResponseFormat, StopSequence, ToolCall, ToolChoice,
        ToolDefinition,
    },
};

/// Generation settings that must stay constant across a stateful chain.
///
/// Excludes `messages` (the payload), `stream` (per-call), and the Conversation-managed
/// `previous_response_id`/`store`.
#[derive(Clone, Debug, PartialEq)]
struct Settings {
    model: String,
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

/// Capture written by a completed [`ConversationStream`] and adopted on the next turn.
///
/// `response_id` is `None` on stateless backends, whose `End` events carry no id; such captures
/// surface the assembled message via [`Conversation::last_message`] but are never adopted as an
/// anchor.
struct StreamCapture {
    settings: Settings,
    messages: Vec<Message>,
    assistant: AssistantMessage,
    response_id: Option<String>,
}

/// Stream-side accumulation of the assistant message from normalized events.
#[derive(Default)]
struct StreamState {
    settings: Option<Settings>,
    messages: Vec<Message>,
    content: String,
    reasoning: String,
    tool_calls: Vec<AccumulatingCall>,
}

/// A single in-flight tool call being assembled from `ToolCall` deltas (keyed by `index`).
#[derive(Default)]
struct AccumulatingCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
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
    stream_slot: Arc<Mutex<Option<StreamCapture>>>,
}

impl Conversation {
    /// Creates a conversation bound to `backend`, with server-side storage enabled.
    pub(crate) fn new(backend: Arc<dyn LlmBackend>) -> Self {
        Self {
            backend,
            store: true,
            anchor: None,
            stream_slot: Arc::new(Mutex::new(None)),
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

    /// The assistant message of the last completed turn.
    ///
    /// Non-streaming turns return the parsed `response.message`; streamed turns return the message
    /// assembled from the events (available once the stream has been consumed through its `End`
    /// event). Mirror this verbatim into the next turn's message list so the anchor can be
    /// recognized.
    pub fn last_message(&self) -> Option<Message> {
        if let Ok(slot) = self.stream_slot.lock() {
            if let Some(capture) = slot.as_ref() {
                return Some(Message::Assistant(capture.assistant.clone()));
            }
        }
        self.anchor
            .as_ref()
            .map(|anchor| Message::Assistant(anchor.last_assistant.clone()))
    }

    /// Executes a non-streaming generation turn.
    ///
    /// On a stateful backend, a pure append to the previous turn is sent as `previous_response_id`
    /// plus only the new messages; anything else is sent as a full request.
    ///
    /// This is the convenience path for simple turns: it validates the request, manages the
    /// continuation anchor, and returns a parsed [`GenerationResponse`]. Response headers and
    /// the raw body are not exposed. For retries keyed on `Retry-After`, header inspection,
    /// or custom status handling, hold a backend handle and use the three-step trait methods
    /// [`prepare`](LlmBackend::prepare) / [`send`](LlmBackend::send) / [`parse`](LlmBackend::parse)
    /// directly; they are defined on [`LlmBackend`], not on `Conversation`.
    pub async fn generate(
        &mut self,
        request: GenerationRequest,
    ) -> Result<GenerationResponse, BackendError> {
        self.adopt_pending_stream();
        validate_tool_results(&request.messages)?;

        let settings = Settings::from_request(&request);
        let last_input = request.messages.clone();
        let wire_request = self.wire_request(request)?;

        match self.backend.generate(wire_request).await {
            Ok(response) => {
                self.record_anchor(settings, last_input, &response);
                Ok(response)
            }
            Err(error) => {
                self.clear_anchor();
                Err(error)
            }
        }
    }

    /// Executes a streaming generation turn.
    ///
    /// The returned [`ConversationStream`] forwards events unchanged and, once consumed through its
    /// terminal `End` event, records the assembled assistant message and response id for the next
    /// turn to adopt.
    ///
    /// As with [`generate`](Self::generate), this is the convenience path: response headers and
    /// the raw body are not exposed. For header-aware retries or custom status handling,
    /// use the three-step trait methods [`prepare_streaming`](LlmBackend::prepare_streaming) /
    /// [`send`](LlmBackend::send) / [`parse_streaming`](LlmBackend::parse_streaming) directly.
    pub async fn stream_generate(
        &mut self,
        request: GenerationRequest,
    ) -> Result<ConversationStream, BackendError> {
        self.adopt_pending_stream();
        validate_tool_results(&request.messages)?;

        let settings = Settings::from_request(&request);
        let last_input = request.messages.clone();
        let wire_request = self.wire_request(request)?;

        let inner = match self.backend.stream_generate(wire_request).await {
            Ok(stream) => stream,
            Err(error) => {
                self.clear_anchor();
                return Err(error);
            }
        };

        Ok(ConversationStream {
            inner,
            state: Arc::new(Mutex::new(StreamState {
                settings: Some(settings),
                messages: last_input,
                content: String::new(),
                reasoning: String::new(),
                tool_calls: Vec::new(),
            })),
            slot: self.stream_slot.clone(),
        })
    }

    /// Adopts a completed streamed turn as the anchor (lazily, on the next call).
    ///
    /// A capture without a response id (stateless backend) is discarded: it carries no continuation
    /// handle, so the previous anchor, if any, stays valid.
    fn adopt_pending_stream(&mut self) {
        let Ok(mut slot) = self.stream_slot.lock() else {
            return;
        };
        let Some(capture) = slot.take() else {
            return;
        };
        let Some(response_id) = capture.response_id else {
            return;
        };
        self.anchor = Some(Anchor {
            settings: capture.settings,
            last_input: capture.messages,
            last_assistant: capture.assistant,
            previous_response_id: response_id,
        });
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
        request.tools = None;
        request.tool_choice = None;
        request.previous_response_id = Some(anchor.previous_response_id.clone());
        Ok(request)
    }

    fn record_anchor(
        &mut self,
        settings: Settings,
        last_input: Vec<Message>,
        response: &GenerationResponse,
    ) {
        self.anchor = Some(Anchor {
            settings,
            last_input,
            last_assistant: response.message.clone(),
            previous_response_id: response.id.clone(),
        });
    }

    fn clear_anchor(&mut self) {
        self.anchor = None;
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

/// Streaming wrapper returned by [`Conversation::stream_generate`].
///
/// Forwards every event unchanged while assembling the assistant message from the deltas; on the
/// terminal `End` event the assembled message and response id are recorded for the next turn.
pub struct ConversationStream {
    inner: GenerationStream,
    state: Arc<Mutex<StreamState>>,
    slot: Arc<Mutex<Option<StreamCapture>>>,
}

impl Stream for ConversationStream {
    type Item = Result<GenerationEvent, just_common::error::TransportError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let item = this.inner.poll_next_unpin(cx);

        match &item {
            Poll::Ready(Some(Ok(GenerationEvent::Text { delta }))) => {
                this.state.lock().unwrap().content.push_str(delta);
            }
            Poll::Ready(Some(Ok(GenerationEvent::Reasoning { delta }))) => {
                this.state.lock().unwrap().reasoning.push_str(delta);
            }
            Poll::Ready(Some(Ok(GenerationEvent::ToolCall { delta }))) => {
                let mut state = this.state.lock().unwrap();
                let index = delta.index.unwrap_or(0) as usize;
                while state.tool_calls.len() <= index {
                    state.tool_calls.push(AccumulatingCall::default());
                }
                if let Some(id) = &delta.id {
                    state.tool_calls[index].id = Some(id.clone());
                }
                if let Some(name) = &delta.name {
                    state.tool_calls[index].name = Some(name.clone());
                }
                if let Some(arguments) = &delta.arguments {
                    state.tool_calls[index].arguments.push_str(arguments);
                }
            }
            Poll::Ready(Some(Ok(GenerationEvent::End { response_id, .. }))) => {
                let state = this.state.lock().unwrap();
                if let (Some(settings), Some(assistant)) =
                    (state.settings.clone(), build_assistant(&state))
                {
                    let capture = StreamCapture {
                        settings,
                        messages: state.messages.clone(),
                        assistant,
                        response_id: response_id.clone(),
                    };
                    if let Ok(mut slot) = this.slot.lock() {
                        *slot = Some(capture);
                    }
                }
            }
            _ => {}
        }

        item
    }
}

impl fmt::Debug for ConversationStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConversationStream").finish_non_exhaustive()
    }
}

/// Assembles the assistant message from the accumulated stream state.
fn build_assistant(state: &StreamState) -> Option<AssistantMessage> {
    let tool_calls = state
        .tool_calls
        .iter()
        .filter_map(|call| {
            let id = call.id.as_ref()?;
            let name = call.name.as_ref()?;
            Some(ToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: call.arguments.clone(),
            })
        })
        .collect();
    let reasoning = if state.reasoning.is_empty() {
        None
    } else {
        Some(Reasoning {
            text: Some(state.reasoning.clone()),
            id: None,
            encrypted: None,
            signature: None,
            redacted: None,
        })
    };

    Some(AssistantMessage {
        content: if state.content.is_empty() {
            None
        } else {
            Some(state.content.clone())
        },
        tool_calls,
        reasoning,
    })
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
            Message::Tool { tool_call_id, .. } => {
                if !issued.contains(tool_call_id.as_str()) {
                    return Err(BackendError::invalid_request(format!(
                        "tool result {tool_call_id} has no preceding assistant tool call in the message list"
                    )));
                }
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
        capability::Identifiable,
        error::BackendConstructError,
        types::generation::{FinishReason, ToolCallDelta, ToolDefinition},
    };

    struct MockBackend {
        stateful: bool,
        requests: Mutex<Vec<GenerationRequest>>,
        responses: Mutex<VecDeque<Result<GenerationResponse, BackendError>>>,
        streams: Mutex<VecDeque<Vec<GenerationEvent>>>,
    }

    impl MockBackend {
        fn new(stateful: bool) -> Self {
            Self {
                stateful,
                requests: Mutex::new(Vec::new()),
                responses: Mutex::new(VecDeque::new()),
                streams: Mutex::new(VecDeque::new()),
            }
        }

        fn response(id: &str, message: AssistantMessage) -> GenerationResponse {
            GenerationResponse {
                id: id.to_owned(),
                model: "mock".to_owned(),
                message,
                finish_reason: Some(FinishReason::Stop),
                usage: None,
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
        fn prepare(&self, _request: GenerationRequest) -> Result<reqwest::Request, BackendError> {
            unimplemented!("not exercised by tests")
        }

        fn prepare_streaming(
            &self,
            _request: GenerationRequest,
        ) -> Result<reqwest::Request, BackendError> {
            unimplemented!("not exercised by tests")
        }

        async fn send(
            &self,
            _prepared: reqwest::Request,
        ) -> Result<reqwest::Response, BackendError> {
            unimplemented!("not exercised by tests")
        }

        async fn parse(
            &self,
            _response: reqwest::Response,
        ) -> Result<GenerationResponse, BackendError> {
            unimplemented!("not exercised by tests")
        }

        async fn parse_streaming(
            &self,
            _response: reqwest::Response,
        ) -> Result<GenerationStream, BackendError> {
            unimplemented!("not exercised by tests")
        }

        async fn generate(
            &self,
            request: GenerationRequest,
        ) -> Result<GenerationResponse, BackendError> {
            self.requests.lock().unwrap().push(request);
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .expect("no response queued for mock backend")
        }

        async fn stream_generate(
            &self,
            request: GenerationRequest,
        ) -> Result<GenerationStream, BackendError> {
            self.requests.lock().unwrap().push(request);
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
    async fn error_clears_anchor_and_next_turn_reanchors() {
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

        // After the error the anchor is gone: the next turn is a full re-anchoring send.
        let m3 = vec![
            Message::user("q1"),
            Message::assistant("a1"),
            Message::user("q3"),
        ];
        conv.generate(GenerationRequest::new("mock", m3.clone()))
            .await
            .unwrap();

        let requests = backend.take_requests();
        assert_eq!(requests.len(), 3);
        assert!(requests[1].previous_response_id.is_some());
        assert!(requests[2].previous_response_id.is_none());
        assert_eq!(requests[2].messages, m3);
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

        let stream = conv
            .stream_generate(GenerationRequest::new("mock", vec![Message::user("q1")]))
            .await
            .unwrap();
        let mut events = Vec::new();
        let mut stream = Box::pin(stream);
        while let Some(event) = stream.next().await {
            events.push(event.unwrap());
        }
        assert_eq!(events.len(), 5);

        // The assembled message is available before the next turn adopts it.
        let mirror = conv.last_message().unwrap();
        assert_eq!(mirror.content(), Some("The answer is 42."));
        assert_eq!(mirror.tool_calls().len(), 1);
        assert_eq!(mirror.tool_calls()[0].name, "get_weather");
        assert_eq!(mirror.tool_calls()[0].arguments, r#"{"city":"Paris"}"#);

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

        // Start a streamed turn that cannot be recognized as an append, but never poll it to its
        // terminal event, so it never adopts an anchor.
        let _dropped = conv
            .stream_generate(GenerationRequest::new("mock", vec![Message::user("q2")]))
            .await
            .unwrap();

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
    async fn streamed_last_message_works_on_stateless_backend() {
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

        let mut stream = Box::pin(
            conv.stream_generate(GenerationRequest::new("mock", vec![Message::user("q1")]))
                .await
                .unwrap(),
        );
        while stream.next().await.is_some() {}

        // The assembled message is available even though a stateless backend can never chain.
        assert_eq!(conv.last_message().unwrap().content(), Some("hi"));
    }
}
