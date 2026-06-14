use std::sync::{Arc, Mutex};

#[cfg(feature = "deepseek")]
use crate::client::deepseek::DeepSeekProvider;
#[cfg(feature = "openai-compat")]
use crate::client::openai_compat::OpenAiCompatProvider;

use crate::client::provider::ProviderEntry;
use crate::client::{ChatClient, ChatClientOptions};
use crate::error::LlmError;
use crate::provider::LlmBackend;

struct StoredProvider {
    provider: Box<dyn ProviderEntry>,
    backend: Mutex<Option<Arc<dyn LlmBackend>>>,
}

impl StoredProvider {
    fn new(provider: impl ProviderEntry) -> Self {
        Self {
            provider: Box::new(provider),
            backend: Mutex::new(None),
        }
    }

    fn instance_id(&self) -> &str {
        self.provider.instance_id()
    }
}

/// Registry of programmatically configured provider entries.
///
/// Each entry is registered with an application-defined identifier such as `"deepseek"` or
/// `"deepseek-local"`. The first call to [`chat`](ProviderRegistry::chat) lazily connects a shared
/// backend for that entry; later calls reuse the same backend while creating fresh
/// [`ChatClient`] wrappers with per-call defaults.
#[derive(Default)]
pub struct ProviderRegistry {
    providers: Vec<StoredProvider>,
}

impl ProviderRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    /// Create a registry with a single built-in DeepSeek provider entry.
    #[cfg(feature = "deepseek")]
    pub fn with_deepseek(provider: DeepSeekProvider) -> Self {
        let mut registry = Self::new();
        registry.register(provider);
        registry
    }

    /// Create a registry with a single built-in OpenAI-compatible provider entry.
    #[cfg(feature = "openai-compat")]
    pub fn with_openai_compat(provider: OpenAiCompatProvider) -> Self {
        let mut registry = Self::new();
        registry.register(provider);
        registry
    }

    /// Register (or replace) a configured provider entry.
    ///
    /// If another provider with the same [`ProviderEntry::instance_id`] already exists, it is replaced and
    /// any previously cached backend for that identifier is discarded.
    pub fn register(&mut self, provider: impl ProviderEntry) -> &mut Self {
        let stored = StoredProvider::new(provider);
        if let Some(existing_index) = self
            .providers
            .iter()
            .position(|entry| entry.instance_id() == stored.instance_id())
        {
            self.providers[existing_index] = stored;
        } else {
            self.providers.push(stored);
        }
        self
    }

    /// Create a [`ChatClient`] for a registered provider entry.
    pub fn chat(&self, id: &str, options: ChatClientOptions) -> Result<ChatClient, LlmError> {
        let stored = self
            .providers
            .iter()
            .find(|provider| provider.instance_id() == id)
            .ok_or_else(|| LlmError::invalid_request(format!("unknown instance id: {id}")))?;
        let backend = {
            let mut cached = stored.backend.lock().map_err(|_| {
                LlmError::invalid_request(format!("provider backend cache poisoned: {id}"))
            })?;
            if let Some(backend) = cached.as_ref() {
                backend.clone()
            } else {
                let backend = stored.provider.connect()?;
                *cached = Some(backend.clone());
                backend
            }
        };
        Ok(ChatClient::new(id.to_owned(), options, backend))
    }

    /// Returns the instance ids of all registered provider entries.
    pub fn instance_ids(&self) -> impl Iterator<Item = &str> {
        self.providers.iter().map(StoredProvider::instance_id)
    }
}
