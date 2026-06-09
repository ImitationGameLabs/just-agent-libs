use std::pin::Pin;

use async_trait::async_trait;
use futures_core::Stream;

use crate::{
    error::{Capability, LlmError},
    types::{balance::BalanceSnapshot, chat::ChatCompletionChunk, model::ModelCatalogResponse},
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
