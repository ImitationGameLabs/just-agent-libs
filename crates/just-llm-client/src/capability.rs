use std::pin::Pin;

use async_trait::async_trait;
use futures_core::Stream;

use crate::{
    error::{Capability, LlmError},
    types::{
        balance::BalanceSnapshot,
        chat::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse},
        model::ModelCatalogResponse,
        prepared::PreparedChatRequest,
    },
};

/// Boxed stream of normalized chat-completion chunks.
pub type ChatCompletionStream = Pin<
    Box<dyn Stream<Item = Result<ChatCompletionChunk, just_common::error::TransportError>> + Send>,
>;

/// Root identity trait shared by all client capabilities.
pub trait Identifiable: Send + Sync {
    /// Returns the stable backend identifier used in error attribution and prepared-request binding.
    fn backend_id(&self) -> &'static str;
}

/// Non-streaming chat completion, with both direct execution and explicit prepare/send paths.
#[async_trait]
pub trait ChatCompletion: Identifiable {
    /// Executes a non-streaming chat completion request.
    async fn create_chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError>;

    /// Converts a normalized request into a backend-bound [`PreparedChatRequest`].
    async fn prepared_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<PreparedChatRequest, LlmError>;

    /// Executes a prepared request that was produced for the same backend.
    async fn send_prepared(
        &self,
        request: &PreparedChatRequest,
    ) -> Result<ChatCompletionResponse, LlmError>;
}

/// Streaming chat completion, with both direct execution and explicit prepare/send paths.
#[async_trait]
pub trait StreamingChatCompletion: Identifiable {
    /// Starts a streaming chat completion request.
    async fn stream_chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionStream, LlmError>;

    /// Converts a streaming request into a backend-bound [`PreparedChatRequest`].
    async fn prepared_streaming_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<PreparedChatRequest, LlmError>;

    /// Executes a prepared streaming request that was produced for the same backend.
    async fn send_prepared_stream(
        &self,
        request: &PreparedChatRequest,
    ) -> Result<ChatCompletionStream, LlmError>;
}

/// List available models from the provider.
#[async_trait]
pub trait ModelCatalog: Identifiable {
    /// Returns the provider's current model catalog.
    async fn list_models(&self) -> Result<ModelCatalogResponse, LlmError>;
}

/// Query account balance or quota state.
#[async_trait]
pub trait Balance: Identifiable {
    /// Returns the provider's current balance snapshot.
    async fn get_balance(&self) -> Result<BalanceSnapshot, LlmError>;
}

/// Explicit capability negotiation for runtime-selected or otherwise abstract backends.
///
/// Each successful negotiation returns a handle that only exposes the requested behavior. If a
/// backend does not support a capability, the unsupported error is surfaced here instead of on the
/// capability trait itself.
pub trait CapabilityNegotiation: Identifiable {
    /// Returns a handle for model catalog inspection when the backend supports it.
    fn model_catalog(&self) -> Result<&dyn ModelCatalog, LlmError> {
        Err(LlmError::unsupported(
            self.backend_id(),
            Capability::ModelCatalog,
        ))
    }

    /// Returns a handle for balance inspection when the backend supports it.
    fn balance(&self) -> Result<&dyn Balance, LlmError> {
        Err(LlmError::unsupported(
            self.backend_id(),
            Capability::Balance,
        ))
    }
}
