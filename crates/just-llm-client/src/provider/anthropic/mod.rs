//! Anthropic Messages LLM backend adapter.
//!
//! [`AnthropicBackend`] wraps a [`just_anthropic::AnthropicClient`] and implements
//! [`LlmBackend`], mapping the semantic [`GenerationRequest`] to the Messages API. System prompts
//! are extracted to the top-level `system` parameter, tool results round-trip through `tool_result`
//! blocks in `user` messages, and extended thinking preserves its `signature`/`redacted`
//! fidelity carriers. Balance inspection is negotiated explicitly and returns
//! [`CapabilityError::Unsupported`](crate::CapabilityError::Unsupported) because the Messages API
//! does not expose a balance endpoint.
//!
//! Construct from raw inputs (API key + optional base URL) via the [`LlmBackend::new`] trait
//! method ([`LlmBackend`] must be in scope), via [`AnthropicBackend::from_provider_client`] with a
//! pre-built provider client, or through a [`BackendFactory`](crate::BackendFactory) that
//! registers this backend.

mod conversions;

use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;

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

/// `just-llm-client` adapter for the Anthropic Messages API.
///
/// Delegates HTTP dispatch to a [`just_anthropic::AnthropicClient`] and handles type conversion
/// between normalized client types and the Messages wire format.
#[derive(Clone, Debug)]
pub struct AnthropicBackend {
    client: just_anthropic::AnthropicClient,
}

impl AnthropicBackend {
    /// Creates a new backend from a pre-built provider client.
    pub fn from_provider_client(client: just_anthropic::AnthropicClient) -> Self {
        Self { client }
    }
}

impl Identifiable for AnthropicBackend {
    fn family(&self) -> &'static str {
        crate::family::ANTHROPIC
    }
}

impl CapabilityNegotiation for AnthropicBackend {
    fn model_catalog(&self) -> Result<&dyn ModelCatalog, CapabilityError> {
        Ok(self)
    }
}

#[async_trait]
impl LlmBackend for AnthropicBackend {
    // --- prepare / send (raw HTTP surface) ---

    fn prepare(&self, request: GenerationRequest) -> Result<reqwest::Request, BackendError> {
        validate_non_streaming_request(&request, "prepare", "prepare_streaming")?;
        let provider_req: just_anthropic::types::request::CreateMessageRequest =
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
        let provider_req: just_anthropic::types::request::CreateMessageRequest =
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
            .get("request-id")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let native: just_anthropic::types::message::Message = self
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
        let stream = self
            .client
            .parse_streaming(response)
            .await
            .map_err(|e| BackendError::provider(self.family(), e))?;

        // Anthropic splits usage across events: message_start carries the input-side counts and
        // message_delta carries the final output_tokens. Track the input count in the closure and
        // emit a merged Usage alongside the End event when message_delta arrives.
        let input_tokens = std::cell::Cell::new(None);
        let mapped = stream.flat_map(move |event| {
            let events: Vec<Result<crate::types::generation::GenerationEvent, _>> = match event {
                Ok(just_anthropic::types::event::StreamEvent::MessageStart { message }) => {
                    input_tokens.set(Some(message.usage.input_tokens));
                    Vec::new()
                }
                Ok(just_anthropic::types::event::StreamEvent::MessageDelta { delta, usage }) => {
                    let input = input_tokens.take().unwrap_or(0);
                    let mut events = vec![crate::types::generation::GenerationEvent::End {
                        finish_reason: conversions::wire_finish_reason(delta.stop_reason),
                        response_id: None,
                    }];
                    if let Some(stream_usage) = usage {
                        let output = stream_usage.output_tokens;
                        // Inline, not wire_usage_to_client: delta usage is a narrower wire type (no input/total fields).
                        events.push(crate::types::generation::GenerationEvent::Usage {
                            usage: crate::types::generation::Usage {
                                completion_tokens: output as u32,
                                prompt_tokens: input as u32,
                                cache_read_tokens: stream_usage
                                    .cache_read_input_tokens
                                    .map(|tokens| tokens as u32),
                                cache_write_tokens: stream_usage
                                    .cache_creation_input_tokens
                                    .map(|tokens| tokens as u32),
                                total_tokens: (input + output) as u32,
                                completion_tokens_details: None,
                            },
                        });
                    }
                    events.into_iter().map(Ok).collect()
                }
                Ok(event) => conversions::event_to_generation_events(event)
                    .into_iter()
                    .map(Ok)
                    .collect(),
                Err(e) => vec![Err(e)],
            };
            futures_util::stream::iter(events)
        });

        Ok(GenerationStream::new(Box::pin(mapped)))
    }

    fn render_messages(&self, messages: &[Message]) -> Result<String, BackendError> {
        // Anthropic carries the system prompt as a top-level `system` parameter, so only the
        // non-system messages belong in the `messages` array this render targets.
        let (_, remaining) = conversions::extract_system(messages.to_vec())?;
        let provider_messages = remaining
            .into_iter()
            .map(conversions::message_to_param)
            .collect::<Result<Vec<just_anthropic::types::message::MessageParam>, _>>()?;
        serde_json::to_string(&provider_messages).map_err(BackendError::serialization)
    }

    fn render_tools(&self, tools: &[ToolDefinition]) -> Result<String, BackendError> {
        let provider_tools: Vec<just_anthropic::types::tool::Tool> =
            tools.iter().cloned().map(conversions::wire_tool).collect();
        serde_json::to_string(&provider_tools).map_err(BackendError::serialization)
    }

    fn family() -> &'static str
    where
        Self: Sized,
    {
        crate::family::ANTHROPIC
    }

    /// Build a shared Anthropic backend from raw inputs.
    ///
    /// `base_url = None` uses the provider default (`https://api.anthropic.com/v1`).
    #[allow(clippy::new_ret_no_self)]
    fn new(
        http: reqwest::ClientBuilder,
        api_key: &str,
        base_url: Option<&str>,
    ) -> Result<Arc<dyn LlmBackend>, BackendConstructError>
    where
        Self: Sized,
    {
        let mut builder = just_anthropic::AnthropicClient::builder()
            .api_key(api_key)
            .http_client(http);
        if let Some(url) = base_url {
            builder = builder.base_url(url);
        }
        let client = builder
            .build()
            .map_err(|e| BackendConstructError::provider(crate::family::ANTHROPIC, e))?;
        Ok(Arc::new(Self::from_provider_client(client)))
    }
}

#[async_trait]
impl ModelCatalog for AnthropicBackend {
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
                    object: Some(model.r#type.clone()),
                    owned_by: None,
                })
                .collect(),
        })
    }
}

#[cfg(test)]
mod parse_meta_tests {
    use super::*;

    fn backend() -> AnthropicBackend {
        AnthropicBackend::from_provider_client(just_anthropic::AnthropicClient::new(
            reqwest::Client::new(),
            "https://api.anthropic.com".to_owned(),
        ))
    }

    fn wire_response(header: Option<(&str, &str)>) -> reqwest::Response {
        let body = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"hi"}],"model":"claude-sonnet-4","usage":{"input_tokens":1,"output_tokens":1}}"#;
        let mut builder = http::Response::builder().status(200);
        if let Some((name, value)) = header {
            builder = builder.header(name, value);
        }
        reqwest::Response::from(builder.body(body.to_owned()).unwrap())
    }

    #[tokio::test]
    async fn parse_attaches_transport_request_id_header() {
        let parsed = backend()
            .parse(wire_response(Some(("request-id", "req-123"))))
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
