//! Unified chat client and backend factory.
//!
//! Two building blocks on top of the concrete backend adapters:
//!
//! - [`BackendFactory`] dispatches a backend family string to that backend's
//!   [`LlmBackend::new`] constructor — a composable `family ->
//!   constructor` primitive with no caching or held configuration.
//! - [`ChatClient`] pairs per-call request defaults (model, system prompt) with a shared
//!   [`LlmBackend`](crate::LlmBackend) and derefs to `dyn LlmBackend` so chat and capability
//!   methods are reachable directly.
//!
//! Backends can also be constructed directly via
//! [`LlmBackend::new`] (the trait constructor, with the trait in scope),
//! or from a pre-built provider client via each backend's `from_provider_client`.

mod factory;

use std::{ops::Deref, sync::Arc};

use crate::{
    provider::LlmBackend,
    types::chat::{ChatCompletionRequest, ChatMessage},
};

pub use factory::BackendFactory;

/// Per-call defaults for constructing a [`ChatClient`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatClientOptions {
    model: String,
    system_prompt: Option<String>,
}

impl ChatClientOptions {
    /// Create options with a required default model.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            system_prompt: None,
        }
    }

    /// Add a default system prompt that will be injected into requests built by the client.
    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    /// Returns the configured default model.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns the configured default system prompt.
    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }
}

/// Thin facade that pairs per-call default request values with a shared backend.
///
/// Constructed via [`ChatClient::new`], typically with a backend produced by a
/// [`BackendFactory`] or by [`LlmBackend::new`] directly. Implements
/// [`Deref`] to [`dyn LlmBackend`](crate::LlmBackend) so the direct and prepared chat execution
/// paths, plus capability negotiation methods, are accessible directly.
#[derive(Clone)]
pub struct ChatClient {
    model: String,
    system_prompt: Option<String>,
    backend: Arc<dyn LlmBackend>,
}

impl ChatClient {
    /// Create a client pairing per-call defaults with a shared backend.
    ///
    /// The backend identity (family) is available via [`family`](crate::Identifiable::family)
    /// through the [`Deref`] to [`LlmBackend`].
    pub fn new(backend: Arc<dyn LlmBackend>, options: ChatClientOptions) -> Self {
        Self {
            model: options.model,
            system_prompt: options.system_prompt,
            backend,
        }
    }

    /// Returns the resolved model string (explicit or provider default).
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns the configured default system prompt, if any.
    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    /// Returns the client with a new default model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Returns the client with a new default system prompt.
    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    /// Returns the client without a default system prompt.
    pub fn clear_system_prompt(mut self) -> Self {
        self.system_prompt = None;
        self
    }

    /// Create a request pre-filled with this client's default model and system prompt.
    #[must_use = "the returned request must be passed to a backend method to have any effect"]
    pub fn create_request(&self, messages: Vec<ChatMessage>) -> ChatCompletionRequest {
        let mut request = ChatCompletionRequest::new(self.model.clone(), messages);
        if let Some(system_prompt) = &self.system_prompt {
            request = request.with_system_prompt(system_prompt.clone());
        }
        request
    }
}

impl Deref for ChatClient {
    type Target = dyn crate::LlmBackend;

    fn deref(&self) -> &Self::Target {
        &*self.backend
    }
}

impl std::fmt::Debug for ChatClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatClient")
            .field("family", &self.backend.family())
            .field("model", &self.model)
            .field("has_system_prompt", &self.system_prompt.is_some())
            .finish()
    }
}
