//! OpenAI-compatible LLM backend adapter.
//!
//! [`OpenAiCompatBackend`] wraps a [`just_openai_compat::OpenAiCompatClient`] and implements
//! [`LlmBackend`] for any service that exposes an OpenAI-like chat completion surface.
//! Balance inspection is negotiated explicitly and returns
//! [`CapabilityError::Unsupported`](crate::CapabilityError::Unsupported) because the generic
//! OpenAI-compatible API does not expose a balance endpoint.
//!
//! Construct from raw inputs (API key + base URL) via the [`LlmBackend::new`] trait method
//! ([`LlmBackend`] must be in scope), via [`OpenAiCompatBackend::from_provider_client`] with a
//! pre-built provider client, or through a [`BackendFactory`](crate::BackendFactory) that
//! registers this backend.

mod conversions;

use std::sync::Arc;

use async_trait::async_trait;

use crate::{
    capability::{CapabilityNegotiation, GenerationStream, Identifiable, ModelCatalog},
    error::{BackendConstructError, BackendError, CapabilityError},
    provider::validation::{into_validated_streaming_request, validate_non_streaming_request},
    types::{
        generation::{GenerationRequest, GenerationResponse, Message, ToolDefinition},
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
    pub fn from_provider_client(client: just_openai_compat::OpenAiCompatClient) -> Self {
        Self { client }
    }
}

impl Identifiable for OpenAiCompatBackend {
    fn family(&self) -> &'static str {
        crate::family::OPENAI_COMPATIBLE
    }
}

impl CapabilityNegotiation for OpenAiCompatBackend {
    fn model_catalog(&self) -> Result<&dyn ModelCatalog, CapabilityError> {
        Ok(self)
    }
}

#[async_trait]
impl LlmBackend for OpenAiCompatBackend {
    // --- prepare / send (raw HTTP surface) ---

    fn prepare(&self, request: GenerationRequest) -> Result<reqwest::Request, BackendError> {
        validate_non_streaming_request(&request, "prepare", "prepare_streaming")?;
        let provider_req: just_openai_compat::types::chat::ChatCompletionRequest =
            request.try_into()?;
        self.client
            .prepare(provider_req)
            .map_err(|e| BackendError::provider(self.family(), e))
    }

    fn prepare_streaming(
        &self,
        request: GenerationRequest,
    ) -> Result<reqwest::Request, BackendError> {
        let request = into_validated_streaming_request(request, "prepare_streaming")?;
        let provider_req: just_openai_compat::types::chat::ChatCompletionRequest =
            request.try_into()?;
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

    async fn parse(&self, response: reqwest::Response) -> Result<GenerationResponse, BackendError> {
        // Borrow transport headers before the SDK consumes the response body.
        let request_id = response
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        // Deserialize into the provider-native type, then lift to the normalized client type.
        let native: just_openai_compat::types::chat::ChatCompletion = self
            .client
            .parse(response)
            .await
            .map_err(|e| BackendError::provider(self.family(), e))?;
        let mut normalized: GenerationResponse = native.into();
        if let Some(id) = request_id {
            normalized.meta.get_or_insert_default().request_id = Some(id);
        }
        Ok(normalized)
    }

    async fn parse_streaming(
        &self,
        response: reqwest::Response,
    ) -> Result<GenerationStream, BackendError> {
        // The provider stream yields provider-native chunks; flatten each into normalized events.
        let stream = self
            .client
            .parse_streaming(response)
            .await
            .map_err(|e| BackendError::provider(self.family(), e))?;
        let mapped = super::flatten_events(stream, conversions::chunk_to_events);
        Ok(GenerationStream::new(mapped))
    }

    fn render_messages(&self, messages: &[Message]) -> Result<String, BackendError> {
        let provider_messages = messages
            .iter()
            .cloned()
            .map(just_openai_compat::types::chat::ChatMessage::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        serde_json::to_string(&provider_messages).map_err(BackendError::serialization)
    }

    fn render_tools(&self, tools: &[ToolDefinition]) -> Result<String, BackendError> {
        let provider_tools: Vec<just_openai_compat::types::chat::ToolDefinition> =
            tools.iter().cloned().map(Into::into).collect();
        serde_json::to_string(&provider_tools).map_err(BackendError::serialization)
    }

    fn family() -> &'static str
    where
        Self: Sized,
    {
        crate::family::OPENAI_COMPATIBLE
    }

    /// Build a shared OpenAI-compatible backend from raw inputs.
    ///
    /// This provider has no default base URL — `base_url = None` surfaces as
    /// [`BackendConstructError::Provider`](crate::BackendConstructError::Provider) carrying a
    /// `TransportError::InvalidConfig("base url is required")` source.
    #[allow(clippy::new_ret_no_self)]
    fn new(
        http: reqwest::ClientBuilder,
        api_key: &str,
        base_url: Option<&str>,
    ) -> Result<Arc<dyn LlmBackend>, BackendConstructError>
    where
        Self: Sized,
    {
        let mut builder = just_openai_compat::OpenAiCompatClient::builder()
            .api_key(api_key)
            .http_client(http);
        if let Some(url) = base_url {
            builder = builder.base_url(url);
        }
        let client = builder
            .build()
            .map_err(|e| BackendConstructError::provider(crate::family::OPENAI_COMPATIBLE, e))?;
        Ok(Arc::new(Self::from_provider_client(client)))
    }
}

#[async_trait]
impl ModelCatalog for OpenAiCompatBackend {
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

#[cfg(test)]
mod parse_meta_tests {
    use super::*;

    fn backend() -> OpenAiCompatBackend {
        OpenAiCompatBackend::from_provider_client(just_openai_compat::OpenAiCompatClient::new(
            reqwest::Client::new(),
            "https://api.example.com/v1".to_owned(),
        ))
    }

    fn wire_response(header: Option<(&str, &str)>) -> reqwest::Response {
        let body = r#"{"id":"chatcmpl-1","choices":[{"index":0,"message":{"role":"assistant","content":"hi"}}],"created":0,"model":"gpt-4.1-mini","object":"chat.completion"}"#;
        let mut builder = http::Response::builder().status(200);
        if let Some((name, value)) = header {
            builder = builder.header(name, value);
        }
        reqwest::Response::from(builder.body(body.to_owned()).unwrap())
    }

    #[tokio::test]
    async fn parse_attaches_transport_request_id_header() {
        let parsed = backend()
            .parse(wire_response(Some(("x-request-id", "req-123"))))
            .await
            .unwrap();
        assert_eq!(
            parsed.meta.as_ref().unwrap().request_id.as_deref(),
            Some("req-123")
        );
    }

    #[tokio::test]
    async fn parse_without_header_leaves_request_id_absent() {
        let parsed = backend().parse(wire_response(None)).await.unwrap();
        assert_eq!(parsed.meta, None);
    }
}
