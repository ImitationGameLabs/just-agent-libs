//! OpenAI-compatible LLM backend adapter.
//!
//! [`OpenAiCompatBackend`] wraps a [`just_openai_compat::OpenAiCompatClient`] and implements
//! [`LlmBackend`] for any service that exposes an OpenAI-like chat completion surface.
//! Balance inspection is negotiated explicitly and returns
//! [`UnsupportedCapability`](crate::LlmError::UnsupportedCapability) because the generic
//! OpenAI-compatible API does not expose a balance endpoint.
//!
//! Construct via [`OpenAiCompatBackend::from_client`] with a pre-built provider client, or let
//! [`OpenAiCompatProvider`](crate::OpenAiCompatProvider) build one through the registry.

mod conversions;

use async_trait::async_trait;
use futures_util::StreamExt;

use crate::{
    capability::{CapabilityNegotiation, ChatCompletionStream, Identifiable, ModelCatalog},
    error::LlmError,
    provider::validation::{into_validated_streaming_request, validate_non_streaming_request},
    types::{
        chat::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ToolDefinition},
        model::{ModelCatalogResponse, ModelInfo},
    },
};

use super::LlmBackend;

/// `just-llm-client` adapter for OpenAI-compatible providers.
///
/// Delegates HTTP dispatch to a [`just_openai_compat::OpenAiCompatClient`] and handles type
/// conversion between normalized client types and provider-specific wire types.
#[derive(Clone, Debug)]
pub struct OpenAiCompatBackend {
    client: just_openai_compat::OpenAiCompatClient,
}

impl OpenAiCompatBackend {
    /// Creates a new backend from a pre-built provider client.
    pub fn from_client(client: just_openai_compat::OpenAiCompatClient) -> Self {
        Self { client }
    }

    /// Creates a new backend from raw HTTP components.
    ///
    /// The HTTP client should already have auth headers set (e.g. via
    /// [`build_client`](just_common::transport::http::build_client)).
    pub fn new(http: reqwest::Client, base_url: String) -> Self {
        Self {
            client: just_openai_compat::OpenAiCompatClient::new(http, base_url),
        }
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
    // --- prepare / send (raw HTTP surface) ---

    fn prepare(&self, request: ChatCompletionRequest) -> Result<reqwest::Request, LlmError> {
        validate_non_streaming_request(&request, "prepare", "prepare_streaming")?;
        let provider_req: just_openai_compat::types::chat::ChatCompletionRequest = request.into();
        self.client
            .prepare(provider_req)
            .map_err(|e| LlmError::backend(self.backend_id(), e))
    }

    fn prepare_streaming(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<reqwest::Request, LlmError> {
        let request = into_validated_streaming_request(request, "prepare_streaming")?;
        let provider_req: just_openai_compat::types::chat::ChatCompletionRequest = request.into();
        self.client
            .prepare_streaming(provider_req)
            .map_err(|e| LlmError::backend(self.backend_id(), e))
    }

    async fn send(&self, prepared: reqwest::Request) -> Result<reqwest::Response, LlmError> {
        self.client
            .send(prepared)
            .await
            .map_err(|e| LlmError::backend(self.backend_id(), e))
    }

    // --- typed operations + rendering ---

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        validate_non_streaming_request(&request, "chat_completion", "stream_chat_completion")?;
        let provider_req: just_openai_compat::types::chat::ChatCompletionRequest = request.into();
        let response = self
            .client
            .chat_completion(provider_req)
            .await
            .map_err(|e| LlmError::backend(self.backend_id(), e))?;
        Ok(response.into())
    }

    async fn stream_chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionStream, LlmError> {
        let request = into_validated_streaming_request(request, "stream_chat_completion")?;
        let provider_req: just_openai_compat::types::chat::ChatCompletionRequest = request.into();
        let stream = self
            .client
            .stream_chat_completion(provider_req)
            .await
            .map_err(|e| LlmError::backend(self.backend_id(), e))?;
        let mapped = stream.map(|chunk| chunk.map(Into::into));
        Ok(ChatCompletionStream::new(Box::pin(mapped)))
    }

    fn render_messages(&self, messages: &[ChatMessage]) -> Result<String, LlmError> {
        let provider_messages: Vec<just_openai_compat::types::chat::ChatMessage> =
            messages.iter().cloned().map(Into::into).collect();
        serde_json::to_string(&provider_messages).map_err(LlmError::serialization)
    }

    fn render_tools(&self, tools: &[ToolDefinition]) -> Result<String, LlmError> {
        let provider_tools: Vec<just_openai_compat::types::chat::ToolDefinition> =
            tools.iter().cloned().map(Into::into).collect();
        serde_json::to_string(&provider_tools).map_err(LlmError::serialization)
    }
}

#[async_trait]
impl ModelCatalog for OpenAiCompatBackend {
    async fn list_models(&self) -> Result<ModelCatalogResponse, LlmError> {
        let models = self
            .client
            .list_models()
            .await
            .map_err(|e| LlmError::backend(self.backend_id(), e))?;

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
