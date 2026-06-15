//! OpenAI-compatible LLM backend adapter.
//!
//! [`OpenAiCompatBackend`] wraps a [`just_openai_compat::OpenAiCompatClient`] and implements
//! [`LlmBackend`] for any service that exposes an OpenAI-like chat completion surface.
//! Balance inspection is negotiated explicitly and returns
//! [`UnsupportedCapability`](crate::LlmError::UnsupportedCapability) because the generic
//! OpenAI-compatible API does not expose a balance endpoint.
//!
//! Construct via [`OpenAiCompatBackend::new`] from raw inputs (API key + base URL), via
//! [`OpenAiCompatBackend::from_provider_client`] with a pre-built provider client, or let
//! [`OpenAiCompatProvider`](crate::OpenAiCompatProvider) build one through the registry.

mod conversions;

use std::sync::Arc;

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
    /// The backend family string — the single source shared with
    /// [`OpenAiCompatProvider`](crate::OpenAiCompatProvider) for connect-time error attribution.
    pub(crate) const FAMILY: &'static str = "openai-compatible";

    /// Primary constructor: build a shared OpenAI-compatible backend from raw inputs.
    ///
    /// The Bearer auth header is configured from the API key and injected when the underlying
    /// client is built (via [`build_client`](just_common::transport::http::build_client), inside
    /// the SDK builder). Unlike DeepSeek, this provider has no default base URL — `base_url =
    /// None` surfaces as [`LlmError::Backend`](crate::LlmError::Backend) carrying a
    /// `TransportError::InvalidConfig("base url is required")` source. Returns the shared trait
    /// object that the registry and [`ChatClient`](crate::ChatClient) consume.
    // Intentional: yields the shared `Arc<dyn LlmBackend>` that the registry and `ChatClient`
    // consume, rather than a concrete `Self`.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        http: reqwest::ClientBuilder,
        api_key: &str,
        base_url: Option<&str>,
    ) -> Result<Arc<dyn LlmBackend>, LlmError> {
        let mut builder = just_openai_compat::OpenAiCompatClient::builder()
            .api_key(api_key)
            .http_client(http);
        if let Some(url) = base_url {
            builder = builder.base_url(url);
        }
        let client = builder
            .build()
            .map_err(|e| LlmError::backend(Self::FAMILY, e))?;
        Ok(Arc::new(OpenAiCompatBackend::from_provider_client(client)))
    }

    /// Creates a new backend from a pre-built provider client.
    pub fn from_provider_client(client: just_openai_compat::OpenAiCompatClient) -> Self {
        Self { client }
    }
}

impl Identifiable for OpenAiCompatBackend {
    fn family(&self) -> &'static str {
        Self::FAMILY
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
            .map_err(|e| LlmError::backend(self.family(), e))
    }

    fn prepare_streaming(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<reqwest::Request, LlmError> {
        let request = into_validated_streaming_request(request, "prepare_streaming")?;
        let provider_req: just_openai_compat::types::chat::ChatCompletionRequest = request.into();
        self.client
            .prepare_streaming(provider_req)
            .map_err(|e| LlmError::backend(self.family(), e))
    }

    async fn send(&self, prepared: reqwest::Request) -> Result<reqwest::Response, LlmError> {
        self.client
            .send(prepared)
            .await
            .map_err(|e| LlmError::backend(self.family(), e))
    }

    // --- parse + rendering ---

    async fn parse(&self, response: reqwest::Response) -> Result<ChatCompletionResponse, LlmError> {
        // Deserialize into the provider-native type, then lift to the normalized client type.
        let native: just_openai_compat::types::chat::ChatCompletion = self
            .client
            .parse(response)
            .await
            .map_err(|e| LlmError::backend(self.family(), e))?;
        Ok(native.into())
    }

    async fn parse_streaming(
        &self,
        response: reqwest::Response,
    ) -> Result<ChatCompletionStream, LlmError> {
        // The provider stream yields provider-native chunks; map to normalized types.
        let stream = self
            .client
            .parse_streaming(response)
            .await
            .map_err(|e| LlmError::backend(self.family(), e))?;
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
            .map_err(|e| LlmError::backend(self.family(), e))?;

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
