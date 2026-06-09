//! Unified chat client and provider registry helpers.
//!
//! This module provides a convenience layer on top of the concrete backend adapters.
//! For provider-specific behavior not covered by [`LlmBackend`](crate::LlmBackend),
//! construct the concrete backend directly via
//! [`DeepSeekBackend`](crate::provider::DeepSeekBackend) or
//! [`OpenAiCompatBackend`](crate::provider::OpenAiCompatBackend).

#[cfg(feature = "deepseek")]
mod deepseek;
#[cfg(feature = "openai-compat")]
mod openai_compat;
mod provider;
mod registry;

use std::{ops::Deref, sync::Arc};

use crate::{
    provider::LlmBackend,
    types::chat::{ChatCompletionRequest, ChatMessage},
};

#[cfg(feature = "deepseek")]
pub use deepseek::DeepSeekProvider;
#[cfg(feature = "openai-compat")]
pub use openai_compat::OpenAiCompatProvider;
pub use provider::ProviderEntry;
pub use registry::ProviderRegistry;

/// Per-call defaults for constructing a [`ChatClient`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatClientOptions {
    model: String,
    system_prompt: Option<String>,
}

impl ChatClientOptions {
    /// Create options with a required default model.
    pub fn new(model: impl Into<String>) -> Self {
        Self { model: model.into(), system_prompt: None }
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

/// Thin facade that pairs a provider entry id, default request values, and a shared backend.
///
/// Constructed via [`ProviderRegistry::chat`].
/// Implements [`Deref`] to [`dyn LlmBackend`](crate::LlmBackend) so the direct and prepared chat
/// execution paths, plus capability negotiation methods, are accessible directly. The underlying
/// backend is the one produced by [`ProviderEntry::connect`] — e.g. a
/// [`DeepSeekBackend`](crate::provider::DeepSeekBackend).
#[derive(Clone)]
pub struct ChatClient {
    provider_id: String,
    model: String,
    system_prompt: Option<String>,
    backend: Arc<dyn LlmBackend>,
}

impl ChatClient {
    pub(crate) fn new(
        provider_id: String,
        options: ChatClientOptions,
        backend: Arc<dyn LlmBackend>,
    ) -> Self {
        Self { provider_id, model: options.model, system_prompt: options.system_prompt, backend }
    }

    /// Returns the provider entry id used to create this client.
    pub fn provider_id(&self) -> &str {
        &self.provider_id
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
            .field("provider_id", &self.provider_id)
            .field("model", &self.model)
            .field("has_system_prompt", &self.system_prompt.is_some())
            .field("backend_id", &self.backend.backend_id())
            .finish()
    }
}
