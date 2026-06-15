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
        Self {
            id: id.into(),
            api_key: api_key.into(),
            base_url: base_url.into(),
        }
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
    fn instance_id(&self) -> &str {
        &self.id
    }

    fn family(&self) -> &'static str {
        OpenAiCompatBackend::FAMILY
    }

    fn connect(&self) -> Result<Arc<dyn LlmBackend>, LlmError> {
        let client = just_openai_compat::OpenAiCompatClient::builder()
            .api_key(&self.api_key)
            .base_url(&self.base_url)
            .build()
            .map_err(|e| LlmError::backend(self.family(), e))?;
        Ok(Arc::new(OpenAiCompatBackend::from_provider_client(client)))
    }
}
