//! DeepSeek LLM backend adapter.
//!
//! [`DeepSeekBackend`] wraps a [`just_deepseek::DeepSeekClient`] and implements [`LlmBackend`]
//! so it can be used through [`dyn LlmBackend`](crate::LlmBackend) or directly.
//!
//! Construct from raw inputs (API key + optional base URL) via the [`LlmBackend::new`] trait
//! method ([`LlmBackend`] must be in scope), via [`DeepSeekBackend::from_provider_client`] with
//! a pre-built provider client, or through a [`BackendFactory`](crate::BackendFactory) that
//! registers this backend.

mod conversions;

use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;

use crate::{
    capability::{
        Balance, CapabilityNegotiation, ChatCompletionStream, Identifiable, ModelCatalog,
    },
    error::{BackendConstructError, BackendError, CapabilityError},
    provider::validation::{into_validated_streaming_request, validate_non_streaming_request},
    types::{
        balance::{BalanceEntry, BalanceSnapshot, Currency},
        chat::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ToolDefinition},
        model::{ModelCatalogResponse, ModelInfo},
    },
};

use super::LlmBackend;

/// `just-llm-client` adapter for the DeepSeek provider.
///
/// Delegates HTTP dispatch to a [`just_deepseek::DeepSeekClient`] and handles type conversion
/// between normalized client types and provider-specific wire types.
#[derive(Clone, Debug)]
pub struct DeepSeekBackend {
    client: just_deepseek::DeepSeekClient,
}

impl DeepSeekBackend {
    /// Creates a new backend from a pre-built provider client.
    pub fn from_provider_client(client: just_deepseek::DeepSeekClient) -> Self {
        Self { client }
    }
}

impl Identifiable for DeepSeekBackend {
    fn family(&self) -> &'static str {
        crate::family::DEEPSEEK
    }
}

impl CapabilityNegotiation for DeepSeekBackend {
    fn model_catalog(&self) -> Result<&dyn ModelCatalog, CapabilityError> {
        Ok(self)
    }

    fn balance(&self) -> Result<&dyn Balance, CapabilityError> {
        Ok(self)
    }
}

#[async_trait]
impl LlmBackend for DeepSeekBackend {
    // --- prepare / send (raw HTTP surface) ---

    fn prepare(&self, request: ChatCompletionRequest) -> Result<reqwest::Request, BackendError> {
        validate_non_streaming_request(&request, "prepare", "prepare_streaming")?;
        let provider_req: just_deepseek::types::chat::ChatCompletionRequest = request.into();
        self.client
            .prepare(provider_req)
            .map_err(|e| BackendError::provider(self.family(), e))
    }

    fn prepare_streaming(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<reqwest::Request, BackendError> {
        let request = into_validated_streaming_request(request, "prepare_streaming")?;
        let provider_req: just_deepseek::types::chat::ChatCompletionRequest = request.into();
        self.client
            .prepare_streaming(provider_req)
            .map_err(|e| BackendError::provider(self.family(), e))
    }

    async fn send(&self, prepared: reqwest::Request) -> Result<reqwest::Response, BackendError> {
        self.client
            .send(prepared)
            .await
            .map_err(|e| BackendError::provider(self.family(), e))
    }

    // --- parse + rendering ---

    async fn parse(
        &self,
        response: reqwest::Response,
    ) -> Result<ChatCompletionResponse, BackendError> {
        // Deserialize into the provider-native type, then lift to the normalized client type.
        let native: just_deepseek::types::chat::ChatCompletion = self
            .client
            .parse(response)
            .await
            .map_err(|e| BackendError::provider(self.family(), e))?;
        Ok(native.into())
    }

    async fn parse_streaming(
        &self,
        response: reqwest::Response,
    ) -> Result<ChatCompletionStream, BackendError> {
        // The provider stream yields provider-native chunks; map to normalized types.
        let stream = self
            .client
            .parse_streaming(response)
            .await
            .map_err(|e| BackendError::provider(self.family(), e))?;
        let mapped = stream.map(|chunk| chunk.map(Into::into));
        Ok(ChatCompletionStream::new(Box::pin(mapped)))
    }

    fn render_messages(&self, messages: &[ChatMessage]) -> Result<String, BackendError> {
        let provider_messages: Vec<just_deepseek::types::chat::ChatMessage> =
            messages.iter().cloned().map(Into::into).collect();
        serde_json::to_string(&provider_messages).map_err(BackendError::serialization)
    }

    fn render_tools(&self, tools: &[ToolDefinition]) -> Result<String, BackendError> {
        let provider_tools: Vec<just_deepseek::types::chat::ToolDefinition> =
            tools.iter().cloned().map(Into::into).collect();
        serde_json::to_string(&provider_tools).map_err(BackendError::serialization)
    }

    fn family() -> &'static str
    where
        Self: Sized,
    {
        crate::family::DEEPSEEK
    }

    /// Build a shared DeepSeek backend from raw inputs.
    ///
    /// `base_url = None` uses the provider default (`https://api.deepseek.com`).
    #[allow(clippy::new_ret_no_self)]
    fn new(
        http: reqwest::ClientBuilder,
        api_key: &str,
        base_url: Option<&str>,
    ) -> Result<Arc<dyn LlmBackend>, BackendConstructError>
    where
        Self: Sized,
    {
        let mut builder = just_deepseek::DeepSeekClient::builder()
            .api_key(api_key)
            .http_client(http);
        if let Some(url) = base_url {
            builder = builder.base_url(url);
        }
        let client = builder
            .build()
            .map_err(|e| BackendConstructError::provider(crate::family::DEEPSEEK, e))?;
        Ok(Arc::new(Self::from_provider_client(client)))
    }
}

#[async_trait]
impl ModelCatalog for DeepSeekBackend {
    async fn list_models(&self) -> Result<ModelCatalogResponse, BackendError> {
        let models = self
            .client
            .list_models()
            .await
            .map_err(|e| BackendError::provider(self.family(), e))?;

        Ok(ModelCatalogResponse {
            data: models
                .data
                .into_iter()
                .map(|model| ModelInfo {
                    id: model.id,
                    object: Some(model.object),
                    owned_by: Some(model.owned_by),
                })
                .collect(),
        })
    }
}

#[async_trait]
impl Balance for DeepSeekBackend {
    async fn get_balance(&self) -> Result<BalanceSnapshot, BackendError> {
        let balance = self
            .client
            .get_user_balance()
            .await
            .map_err(|e| BackendError::provider(self.family(), e))?;

        Ok(BalanceSnapshot {
            is_available: balance.is_available,
            entries: balance
                .balance_infos
                .into_iter()
                .map(|entry| BalanceEntry {
                    currency: match entry.currency {
                        just_deepseek::types::balance::Currency::Cny => Currency::Cny,
                        just_deepseek::types::balance::Currency::Usd => Currency::Usd,
                    },
                    total_balance: entry.total_balance,
                    granted_balance: entry.granted_balance,
                    topped_up_balance: entry.topped_up_balance,
                })
                .collect(),
        })
    }
}
