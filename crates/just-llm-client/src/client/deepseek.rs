use std::sync::Arc;

use crate::client::provider::ProviderEntry;
use crate::error::LlmError;
use crate::provider::{DeepSeekBackend, LlmBackend};

/// Programmatically configured DeepSeek provider entry for use with [`crate::ProviderRegistry`].
#[derive(Clone, Debug)]
pub struct DeepSeekProvider {
    id: String,
    api_key: String,
    base_url: Option<String>,
}

impl DeepSeekProvider {
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

impl ProviderEntry for DeepSeekProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn provider(&self) -> &str {
        "deepseek"
    }

    fn connect(&self) -> Result<Arc<dyn LlmBackend>, LlmError> {
        let mut builder = just_deepseek::DeepSeekClient::builder().api_key(&self.api_key);
        if let Some(url) = &self.base_url {
            builder = builder.base_url(url);
        }
        let client = builder
            .build()
            .map_err(|source| LlmError::backend("deepseek", source))?;
        Ok(Arc::new(DeepSeekBackend::new(client)))
    }
}
