use std::sync::Arc;

use crate::client::provider::ProviderEntry;
use crate::error::LlmError;
use crate::provider::{LlmBackend, OpenAiCompatBackend};

/// Programmatically configured OpenAI-compatible provider entry for use with [`crate::ProviderRegistry`].
#[derive(Clone, Debug)]
pub struct OpenAiCompatProvider {
    id: String,
    api_key: String,
    base_url: String,
}

impl OpenAiCompatProvider {
    /// Create a configured provider entry with an explicit base URL.
    pub fn new(
        id: impl Into<String>,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self { id: id.into(), api_key: api_key.into(), base_url: base_url.into() }
    }

    /// Alias for [`Self::new`] that keeps the old constructor naming.
    pub fn from_api_key(
        id: impl Into<String>,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self::new(id, api_key, base_url)
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
        Ok(Arc::new(OpenAiCompatBackend::with_base_url(
            &self.api_key,
            &self.base_url,
        )?))
    }
}
