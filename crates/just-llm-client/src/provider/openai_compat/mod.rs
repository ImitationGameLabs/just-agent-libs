//! OpenAI-compatible LLM backend adapter.
//!
//! [`OpenAiCompatBackend`] wraps a `just_openai_compat::OpenAiCompatClient` and implements
//! [`LlmBackend`]. Notable: balance inspection is negotiated explicitly and returns
//! [`UnsupportedCapability`](crate::LlmError::UnsupportedCapability) during negotiation because
//! the generic OpenAI-compatible API does not expose a balance endpoint.
//!
//! Construct via [`OpenAiCompatBackend::new`] with a pre-built SDK client, or let
//! [`OpenAiCompatProvider`](crate::OpenAiCompatProvider) build one through the registry.

mod conversions;

use async_trait::async_trait;
use futures_util::StreamExt;

use crate::{
    capability::{CapabilityNegotiation, ChatCompletionStream, Identifiable, ModelCatalog},
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

use super::LlmBackend;

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
impl LlmBackend for OpenAiCompatBackend {
    fn prepare(&self, request: ChatCompletionRequest) -> Result<PreparedChatRequest, LlmError> {
        validate_non_streaming_request(&request, "prepare", "prepare_streaming")?;
        let provider_request: just_openai_compat::types::chat::CreateChatCompletionRequest =
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
        let response: just_openai_compat::types::chat::ChatCompletion = self
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
        let provider_request: just_openai_compat::types::chat::CreateChatCompletionRequest =
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
