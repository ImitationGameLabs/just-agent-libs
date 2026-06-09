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
    types::chat::{ChatCompletionRequest, ChatCompletionResponse},
    types::prepared::PreparedChatRequest,
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
/// Callers typically access this through [`ChatClient`](crate::ChatClient) which implements
/// [`Deref`](std::ops::Deref) to `dyn LlmBackend`, so all methods are available without importing
/// the trait explicitly.
#[async_trait]
pub trait LlmBackend: Identifiable + CapabilityNegotiation + Send + Sync {
    /// Prepare a non-streaming request for later execution.
    ///
    /// This is a synchronous operation — it validates and serializes the request but performs no
    /// IO.
    fn prepare(&self, request: ChatCompletionRequest) -> Result<PreparedChatRequest, LlmError>;

    /// Send a previously prepared non-streaming request.
    async fn send(
        &self,
        prepared: &PreparedChatRequest,
    ) -> Result<ChatCompletionResponse, LlmError>;

    /// Convenience: prepare + send in one step.
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let prepared = self.prepare(request)?;
        self.send(&prepared).await
    }

    /// Prepare a streaming request for later execution.
    ///
    /// This is a synchronous operation — it validates and serializes the request but performs no
    /// IO.
    fn prepare_streaming(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<PreparedChatRequest, LlmError>;

    /// Send a previously prepared streaming request.
    async fn send_streaming(
        &self,
        prepared: &PreparedChatRequest,
    ) -> Result<ChatCompletionStream, LlmError>;

    /// Convenience: prepare_streaming + send_streaming in one step.
    async fn stream_chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionStream, LlmError> {
        let prepared = self.prepare_streaming(request)?;
        self.send_streaming(&prepared).await
    }
}
