use std::sync::Arc;

use crate::client::provider::ProviderEntry;
use crate::error::LlmError;
use crate::provider::{LlmBackend, OpenAiCompatBackend};

/// Programmatically configured OpenAI-compatible provider entry for use with [`crate::ProviderRegistry`].
#[derive(Clone, Debug)]
pub struct OpenAiCompatProvider {
    id: String,
    api_key: String,
    base_url: Option<String>,
}

impl OpenAiCompatProvider {
    /// Create a configured provider entry from an explicit id and API key.
    pub fn new(id: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self { id: id.into(), api_key: api_key.into(), base_url: None }
    }

    /// Alias for [`Self::new`] that keeps the old constructor naming.
    pub fn from_api_key(id: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self::new(id, api_key)
    }

    /// Override the base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }
}

impl ProviderEntry for OpenAiCompatProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn provider(&self) -> &str {
        "openai-compatible"
    }

    fn connect(&self) -> Result<Arc<dyn LlmBackend>, LlmError> {
        match &self.base_url {
            Some(url) => Ok(Arc::new(OpenAiCompatBackend::with_base_url(
                &self.api_key,
                url,
            )?)),
            None => Ok(Arc::new(OpenAiCompatBackend::with_config(
                just_openai_compat::OpenAiCompatConfig::new(&self.api_key),
            )?)),
        }
    }
}
