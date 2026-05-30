//! OpenAI-compatible LLM backend adapter.
//!
//! [`OpenAiCompatBackend`] wraps a `just_openai_compat::OpenAiCompatClient` and
//! implements the shared capability traits. Notable: balance inspection is negotiated
//! explicitly and returns [`UnsupportedCapability`](crate::LlmError::UnsupportedCapability)
//! during negotiation because the generic OpenAI-compatible API does not expose a balance
//! endpoint.
//!
//! Construct via [`OpenAiCompatBackend::with_config`] or [`OpenAiCompatBackend::with_base_url`],
//! or let [`OpenAiCompatProvider`](crate::OpenAiCompatProvider) build one through the registry.

mod conversions;

use async_trait::async_trait;
use futures_util::StreamExt;

use crate::{
    capability::{
        CapabilityNegotiation, ChatCompletion, ChatCompletionStream, Identifiable, ModelCatalog,
        StreamingChatCompletion,
    },
    error::LlmError,
    provider::validation::{
        into_validated_streaming_request, validate_non_streaming_request,
        validate_prepared_non_streaming_request, validate_prepared_streaming_request,
    },
    types::{
        chat::{ChatCompletionRequest, ChatCompletionResponse},
        model::{ModelCatalogResponse, ModelInfo},
        prepared::PreparedChatRequest,
    },
};

/// `just-llm-client` adapter for OpenAI-compatible provider crates.
#[derive(Clone, Debug)]
pub struct OpenAiCompatBackend {
    client: just_openai_compat::OpenAiCompatClient,
}

impl OpenAiCompatBackend {
    /// Wraps an existing OpenAI-compatible client.
    pub fn new(client: just_openai_compat::OpenAiCompatClient) -> Self {
        Self { client }
    }

    /// Builds a backend adapter from an OpenAI-compatible configuration value.
    pub fn with_config(config: just_openai_compat::OpenAiCompatConfig) -> Result<Self, LlmError> {
        let client = just_openai_compat::OpenAiCompatClient::with_config(config)
            .map_err(|source| LlmError::backend("openai-compatible", source))?;

        Ok(Self::new(client))
    }

    /// Builds a backend adapter from an API key and custom base URL.
    pub fn with_base_url(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Result<Self, LlmError> {
        let client = just_openai_compat::OpenAiCompatClient::new(api_key, base_url)
            .map_err(|source| LlmError::backend("openai-compatible", source))?;

        Ok(Self::new(client))
    }
}

impl Identifiable for OpenAiCompatBackend {
    fn backend_id(&self) -> &'static str {
        "openai-compatible"
    }
}

impl CapabilityNegotiation for OpenAiCompatBackend {
    fn model_catalog(&self) -> Result<&dyn ModelCatalog, LlmError> {
        Ok(self)
    }
}

#[async_trait]
impl ChatCompletion for OpenAiCompatBackend {
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

        let provider_request: just_openai_compat::types::chat::CreateChatCompletionRequest =
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

        let provider_request: just_openai_compat::types::chat::CreateChatCompletionRequest =
            serde_json::from_value(request.request_body().clone())
                .map_err(|source| LlmError::backend(self.backend_id(), source))?;

        let response = self
            .client
            .create_chat_completion(provider_request)
            .await
            .map_err(|source| LlmError::backend(self.backend_id(), source))?;

        Ok(response.into())
    }
}

#[async_trait]
impl StreamingChatCompletion for OpenAiCompatBackend {
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
        let provider_request: just_openai_compat::types::chat::CreateChatCompletionRequest =
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

        let provider_request: just_openai_compat::types::chat::CreateChatCompletionRequest =
            serde_json::from_value(request.request_body().clone())
                .map_err(|source| LlmError::backend(self.backend_id(), source))?;
        let stream = self
            .client
            .stream_chat_completion(provider_request)
            .await
            .map_err(|source| LlmError::backend(self.backend_id(), source))?;
        Ok(Box::pin(stream.map(|chunk| chunk.map(Into::into))))
    }
}

#[async_trait]
impl ModelCatalog for OpenAiCompatBackend {
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
