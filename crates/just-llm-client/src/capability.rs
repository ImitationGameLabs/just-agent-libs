use std::{
    fmt,
    pin::Pin,
    task::{Context, Poll},
};

use async_trait::async_trait;
use futures_core::Stream;

use crate::{
    error::{Capability, LlmError},
    types::{balance::BalanceSnapshot, chat::ChatCompletionChunk, model::ModelCatalogResponse},
};

/// Stream of normalized chat-completion chunks.
///
/// Wrapper around a `Pin<Box<dyn Stream<...>>>` that implements [`Stream`] so all `StreamExt`
/// methods (`.next()`, `.map()`, `.collect()`, etc.) work as expected.
#[must_use = "streams are lazy; call .next() to drive them"]
pub struct ChatCompletionStream {
    inner: Pin<
        Box<
            dyn Stream<Item = Result<ChatCompletionChunk, just_common::error::TransportError>>
                + Send,
        >,
    >,
}

impl ChatCompletionStream {
    /// Wraps a boxed, pinned stream of chunks into a typed [`ChatCompletionStream`].
    pub fn new(
        inner: Pin<
            Box<
                dyn Stream<Item = Result<ChatCompletionChunk, just_common::error::TransportError>>
                    + Send,
            >,
        >,
    ) -> Self {
        Self { inner }
    }
}

impl fmt::Debug for ChatCompletionStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatCompletionStream")
            .finish_non_exhaustive()
    }
}

impl Stream for ChatCompletionStream {
    type Item = Result<ChatCompletionChunk, just_common::error::TransportError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

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
