#[cfg(feature = "deepseek")]
mod deepseek;
#[cfg(feature = "openai-compat")]
mod openai_compat;
/// Request validation helpers for building custom backends.
pub mod validation;

use async_trait::async_trait;

use self::validation::{into_validated_streaming_request, validate_non_streaming_request};
use crate::{
    CapabilityNegotiation, Identifiable,
    capability::ChatCompletionStream,
    error::LlmError,
    types::chat::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ToolDefinition},
};

#[cfg(feature = "deepseek")]
pub use deepseek::DeepSeekBackend;
#[cfg(feature = "openai-compat")]
pub use openai_compat::OpenAiCompatBackend;

/// Unified trait for the runtime-selected LLM provider surface.
///
/// Combines chat completion operations (prepare/send/chat_completion and their streaming
/// counterparts) with identity and capability negotiation. All types are concrete (no associated
/// types) for object safety.
///
/// # Why `chat_completion` and `stream_chat_completion` are default impls
///
/// They compose [`prepare`](LlmBackend::prepare) + [`send`](LlmBackend::send) +
/// [`parse`](LlmBackend::parse). The provider-specific deserialization lives in the required
/// [`parse`](LlmBackend::parse) / [`parse_streaming`](LlmBackend::parse_streaming) methods, which
/// each backend implements against its own provider-native type and lifts to normalized types via
/// `From`. Override the defaults only for non-HTTP backends that cannot express a completion as
/// prepare/send/parse.
///
/// Callers typically access this through [`ChatClient`](crate::ChatClient) which implements
/// [`Deref`](std::ops::Deref) to `dyn LlmBackend`, so all methods are available without importing
/// the trait explicitly.
///
/// # Prepare-send-parse pattern
///
/// Use [`prepare`](LlmBackend::prepare) (or [`prepare_streaming`](LlmBackend::prepare_streaming))
/// to obtain a `reqwest::Request`, then [`send`](LlmBackend::send) to execute it. The returned
/// `reqwest::Request` is `Send + Sync + Clone`: store it, clone it for retry, or send it across
/// threads. [`send`](LlmBackend::send) returns the raw `reqwest::Response` without checking status,
/// so headers like `retry-after` and `x-ratelimit-*` are readable before the body is consumed.
/// Finally [`parse`](LlmBackend::parse) (or [`parse_streaming`](LlmBackend::parse_streaming))
/// deserializes the response into a normalized type, dispatched to the right backend through the
/// trait object.
///
/// ```ignore
/// let prepared = backend.prepare(req)?;
/// // The prepared request is Clone; keep a copy for retry.
/// let response = backend.send(prepared.clone()).await?;
/// // Inspect status / headers before consuming the body.
/// let retry_after = response.headers().get("retry-after");
/// if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
///     // back off, then re-send the cloned `prepared` ...
/// }
/// // Deserialize with the right backend (dyn dispatch on `self`).
/// let completion = backend.parse(response).await?;
/// ```
#[async_trait]
pub trait LlmBackend: Identifiable + CapabilityNegotiation + Send + Sync {
    /// Prepare a non-streaming request for later execution.
    ///
    /// Returns a `reqwest::Request` with the URL, Content-Type, auth headers, and serialized
    /// body already set.
    ///
    /// This is a synchronous operation — it validates and serializes the request but performs no
    /// IO.
    fn prepare(&self, request: ChatCompletionRequest) -> Result<reqwest::Request, LlmError>;

    /// Prepare a streaming request for later execution.
    ///
    /// Same as [`prepare`](LlmBackend::prepare) but forces `stream = true` on the request.
    fn prepare_streaming(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<reqwest::Request, LlmError>;

    /// Send a prepared request and return the raw HTTP response.
    ///
    /// Returns the response without checking HTTP status — callers must handle non-success
    /// statuses (4xx, 5xx) themselves. This allows inspecting response headers (e.g.
    /// `retry-after`, `x-ratelimit-*`) before consuming the body.
    ///
    /// For automatic status checking and deserialization, use [`chat_completion`](LlmBackend::chat_completion)
    /// or [`stream_chat_completion`](LlmBackend::stream_chat_completion) instead.
    async fn send(&self, prepared: reqwest::Request) -> Result<reqwest::Response, LlmError>;

    /// Parse a raw response into a normalized non-streaming completion.
    ///
    /// Implementations must check HTTP status before deserializing (use
    /// [`ensure_success`](just_common::transport::http::ensure_success)); [`send`](LlmBackend::send)
    /// intentionally does not, so headers stay readable before the body is consumed.
    ///
    /// Pair with [`prepare`](LlmBackend::prepare) + [`send`](LlmBackend::send) when you need
    /// the raw response in hand, e.g. to inspect `retry-after` / `x-ratelimit-*` headers, or
    /// to clone the prepared request for retry, before deserializing.
    async fn parse(&self, response: reqwest::Response) -> Result<ChatCompletionResponse, LlmError>;

    /// Parse a raw response into a normalized streaming chunk stream.
    ///
    /// Implementations must check HTTP status first (use
    /// [`ensure_success`](just_common::transport::http::ensure_success)); the SSE parser assumes a
    /// 2xx event stream.
    async fn parse_streaming(
        &self,
        response: reqwest::Response,
    ) -> Result<ChatCompletionStream, LlmError>;

    /// Execute a non-streaming chat completion: validate -> prepare -> send -> parse.
    ///
    /// Default impl; override only for non-HTTP backends. Validation is repeated here and again
    /// inside [`prepare`](LlmBackend::prepare) deliberately: this entry point attributes
    /// invalid-request errors to `chat_completion`, while `prepare` attributes them to itself,
    /// so both messages stay correct.
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        validate_non_streaming_request(&request, "chat_completion", "stream_chat_completion")?;
        let prepared = self.prepare(request)?;
        let response = self.send(prepared).await?;
        self.parse(response).await
    }

    /// Execute a streaming chat completion: validate -> prepare_streaming -> send -> parse_streaming.
    ///
    /// Default impl; override only for non-HTTP backends. Validation is repeated here and again
    /// inside [`prepare_streaming`](LlmBackend::prepare_streaming) deliberately so invalid-request
    /// errors attribute to `stream_chat_completion` rather than `prepare_streaming`.
    async fn stream_chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionStream, LlmError> {
        let request = into_validated_streaming_request(request, "stream_chat_completion")?;
        let prepared = self.prepare_streaming(request)?;
        let response = self.send(prepared).await?;
        self.parse_streaming(response).await
    }

    /// Render messages to their provider-specific JSON string representation.
    ///
    /// The returned string matches exactly what the provider would receive in the `messages`
    /// field of a chat completion request body. Useful for token estimation.
    fn render_messages(&self, messages: &[ChatMessage]) -> Result<String, LlmError>;

    /// Render tool definitions to their provider-specific JSON string representation.
    ///
    /// The returned string matches exactly what the provider would receive in the `tools`
    /// field of a chat completion request body. Useful for token estimation.
    fn render_tools(&self, tools: &[ToolDefinition]) -> Result<String, LlmError>;
}
