//! DeepSeek LLM backend adapter.
//!
//! [`DeepSeekBackend`] wraps a `just_deepseek::DeepSeekClient` and implements [`LlmBackend`]
//! so it can be used through [`dyn LlmBackend`](crate::LlmBackend) or directly.
//!
//! Construct via [`DeepSeekBackend::new`] with a pre-built SDK client, or let
//! [`DeepSeekProvider`](crate::DeepSeekProvider) build one through the registry.

mod conversions;

use async_trait::async_trait;
use futures_util::StreamExt;

use crate::{
    capability::{
        Balance, CapabilityNegotiation, ChatCompletionStream, Identifiable, ModelCatalog,
    },
    error::LlmError,
    provider::validation::{
        into_validated_streaming_request, validate_non_streaming_request,
        validate_prepared_non_streaming_request, validate_prepared_streaming_request,
    },
    types::{
        balance::{BalanceEntry, BalanceSnapshot, Currency},
        chat::{ChatCompletionRequest, ChatCompletionResponse},
        model::{ModelCatalogResponse, ModelInfo},
        prepared::PreparedChatRequest,
    },
};

use super::LlmBackend;

/// `just-llm-client` adapter for the DeepSeek provider crate.
#[derive(Clone, Debug)]
pub struct DeepSeekBackend {
    client: just_deepseek::DeepSeekClient,
}

impl DeepSeekBackend {
    /// Wraps an existing DeepSeek client.
    pub fn new(client: just_deepseek::DeepSeekClient) -> Self {
        Self { client }
    }
}

impl Identifiable for DeepSeekBackend {
    fn backend_id(&self) -> &'static str {
        "deepseek"
    }
}

impl CapabilityNegotiation for DeepSeekBackend {
    fn model_catalog(&self) -> Result<&dyn ModelCatalog, LlmError> {
        Ok(self)
    }

    fn balance(&self) -> Result<&dyn Balance, LlmError> {
        Ok(self)
    }
}

#[async_trait]
impl LlmBackend for DeepSeekBackend {
    fn prepare(&self, request: ChatCompletionRequest) -> Result<PreparedChatRequest, LlmError> {
        validate_non_streaming_request(&request, "prepare", "prepare_streaming")?;
        let provider_request: just_deepseek::types::chat::CreateChatCompletionRequest =
            request.into();
        let inner = self
            .client
            .prepare(provider_request)
            .map_err(|source| LlmError::backend(self.backend_id(), source))?;
        Ok(PreparedChatRequest::from_common(self.backend_id(), inner))
    }

    async fn send(
        &self,
        prepared: &PreparedChatRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        validate_prepared_non_streaming_request(prepared, "send", "send_streaming")?;
        prepared.ensure_backend(self.backend_id())?;
        let response: just_deepseek::types::chat::ChatCompletion = self
            .client
            .send(prepared.inner())
            .await
            .map_err(|source| LlmError::backend(self.backend_id(), source))?;
        Ok(response.into())
    }

    fn prepare_streaming(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<PreparedChatRequest, LlmError> {
        let request = into_validated_streaming_request(request, "prepare_streaming")?;
        let provider_request: just_deepseek::types::chat::CreateChatCompletionRequest =
            request.into();
        let inner = self
            .client
            .prepare_streaming(provider_request)
            .map_err(|source| LlmError::backend(self.backend_id(), source))?;
        Ok(PreparedChatRequest::from_common(self.backend_id(), inner))
    }

    async fn send_streaming(
        &self,
        prepared: &PreparedChatRequest,
    ) -> Result<ChatCompletionStream, LlmError> {
        validate_prepared_streaming_request(prepared, "send_streaming", "send")?;
        prepared.ensure_backend(self.backend_id())?;
        let stream = self
            .client
            .send_streaming(prepared.inner())
            .await
            .map_err(|source| LlmError::backend(self.backend_id(), source))?;
        Ok(ChatCompletionStream::new(Box::pin(
            stream.map(|chunk| chunk.map(Into::into)),
        )))
    }
}

#[async_trait]
impl ModelCatalog for DeepSeekBackend {
    async fn list_models(&self) -> Result<ModelCatalogResponse, LlmError> {
        let response = self
            .client
            .list_models()
            .await
            .map_err(|source| LlmError::backend(self.backend_id(), source))?;

        Ok(ModelCatalogResponse {
            data: response
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
    async fn get_balance(&self) -> Result<BalanceSnapshot, LlmError> {
        let response = self
            .client
            .get_user_balance()
            .await
            .map_err(|source| LlmError::backend(self.backend_id(), source))?;

        Ok(BalanceSnapshot {
            is_available: response.is_available,
            entries: response
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
