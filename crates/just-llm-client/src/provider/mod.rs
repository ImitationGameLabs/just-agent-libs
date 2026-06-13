#[cfg(feature = "deepseek")]
mod deepseek;
#[cfg(feature = "openai-compat")]
mod openai_compat;
/// Request validation helpers for building custom backends.
pub mod validation;

use async_trait::async_trait;

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
/// # Why `chat_completion` and `stream_chat_completion` are required methods
///
/// These could theoretically be default implementations that compose `prepare` + `send` +
/// deserialization. However, the type conversion between normalized and provider-specific types is
/// inherently provider-specific, and providing defaults would require either associated types
/// (which break object safety) or generic conversion traits (more complex than the current ~5-line
/// per-backend implementation). Revisit when 3+ providers exist.
///
/// Callers typically access this through [`ChatClient`](crate::ChatClient) which implements
/// [`Deref`](std::ops::Deref) to `dyn LlmBackend`, so all methods are available without importing
/// the trait explicitly.
///
/// # Prepare-send pattern
///
/// Use [`prepare`](LlmBackend::prepare) or [`prepare_streaming`](LlmBackend::prepare_streaming)
/// to obtain a `reqwest::Request`, then call [`send`](LlmBackend::send) to execute it. The
/// returned `reqwest::Request` is `Send + Sync + Clone` — it can be stored, cloned for retry,
/// or sent across threads.
///
/// ```ignore
/// let request = backend.prepare(req)?;
/// // Inspect or clone the request as needed...
/// let response = backend.send(request).await?;
/// // response is a reqwest::Response — full header access available
/// let retry_after = response.headers().get("retry-after");
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

    /// Execute a non-streaming chat completion: prepare + send + deserialize.
    ///
    /// Each adapter implements this to handle provider-specific deserialization.
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError>;

    /// Execute a streaming chat completion: prepare + send + stream.
    ///
    /// Each adapter implements this to handle provider-specific stream parsing.
    async fn stream_chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionStream, LlmError>;

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
