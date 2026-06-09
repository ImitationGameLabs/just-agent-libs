//! DeepSeek LLM backend adapter.
//!
//! [`DeepSeekBackend`] wraps a `just_deepseek::DeepSeekClient` and implements the shared
//! capability traits ([`ChatCompletion`](crate::ChatCompletion), [`Balance`](crate::Balance),
//! etc.) so it can be used through [`LlmBackend`](crate::LlmBackend) or directly.
//!
//! Construct via [`DeepSeekBackend::new`] with a pre-built SDK client, or let
//! [`DeepSeekProvider`](crate::DeepSeekProvider) build one through the registry.

mod conversions;
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Method;

use crate::{
    capability::{
        Balance, CapabilityNegotiation, ChatCompletion, ChatCompletionStream, Identifiable,
        ModelCatalog, StreamingChatCompletion,
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
impl ChatCompletion for DeepSeekBackend {
    async fn create_chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        validate_non_streaming_request(
            &request,
            "create_chat_completion",
            "stream_chat_completion",
        )?;

        let response = self
            .client
            .create_chat_completion(request.into())
            .await
            .map_err(|source| LlmError::backend(self.backend_id(), source))?;

        Ok(response.into())
    }

    async fn prepared_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<PreparedChatRequest, LlmError> {
        validate_non_streaming_request(&request, "prepared_request", "prepared_streaming_request")?;

        let provider_request: just_deepseek::types::chat::CreateChatCompletionRequest =
            request.into();
        let request_body = serde_json::to_value(&provider_request)
            .map_err(|source| LlmError::backend(self.backend_id(), source))?;

        PreparedChatRequest::from_request_body(self.backend_id(), request_body)
    }

    async fn send_prepared(
        &self,
        request: &PreparedChatRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        validate_prepared_non_streaming_request(request, "send_prepared", "send_prepared_stream")?;
        request.ensure_backend(self.backend_id())?;

        let response: just_deepseek::types::chat::ChatCompletion = self
            .client
            .send_raw_json(
                Method::POST,
                "/chat/completions",
                request.request_body(),
                request.headers(),
            )
            .await
            .map_err(|source| LlmError::backend(self.backend_id(), source))?;

        Ok(response.into())
    }
}

#[async_trait]
impl StreamingChatCompletion for DeepSeekBackend {
    async fn stream_chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionStream, LlmError> {
        let request = into_validated_streaming_request(request, "stream_chat_completion")?;
        let stream = self
            .client
            .stream_chat_completion(request.into())
            .await
            .map_err(|source| LlmError::backend(self.backend_id(), source))?;
        Ok(Box::pin(stream.map(|chunk| chunk.map(Into::into))))
    }

    async fn prepared_streaming_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<PreparedChatRequest, LlmError> {
        let request = into_validated_streaming_request(request, "prepared_streaming_request")?;
        let provider_request: just_deepseek::types::chat::CreateChatCompletionRequest =
            request.into();
        let request_body = serde_json::to_value(&provider_request)
            .map_err(|source| LlmError::backend(self.backend_id(), source))?;

        PreparedChatRequest::from_request_body(self.backend_id(), request_body)
    }

    async fn send_prepared_stream(
        &self,
        request: &PreparedChatRequest,
    ) -> Result<ChatCompletionStream, LlmError> {
        validate_prepared_streaming_request(request, "send_prepared_stream", "send_prepared")?;
        request.ensure_backend(self.backend_id())?;

        let response = self
            .client
            .stream_raw_json(
                Method::POST,
                "/chat/completions",
                request.request_body(),
                request.headers(),
            )
            .await
            .map_err(|source| LlmError::backend(self.backend_id(), source))?;

        let stream = just_deepseek::ChatCompletionStream::from_response(response)
            .map_err(|source| LlmError::backend(self.backend_id(), source))?;
        Ok(Box::pin(stream.map(|chunk| chunk.map(Into::into))))
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
