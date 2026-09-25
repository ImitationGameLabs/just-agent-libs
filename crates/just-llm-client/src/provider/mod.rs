#[cfg(feature = "anthropic")]
mod anthropic;
#[cfg(feature = "deepseek")]
mod deepseek;
#[cfg(feature = "openai-compat")]
mod openai_compat;
#[cfg(feature = "responses")]
mod openai_responses;
/// Request validation helpers for building custom backends.
pub mod validation;

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;

use self::validation::{into_validated_streaming_request, validate_non_streaming_request};
use crate::{
    CapabilityNegotiation, Identifiable,
    capability::GenerationStream,
    error::{BackendConstructError, BackendError},
    types::generation::{GenerationRequest, GenerationResponse, Message, ToolDefinition},
};

#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicBackend;
#[cfg(feature = "deepseek")]
pub use deepseek::DeepSeekBackend;
#[cfg(feature = "openai-compat")]
pub use openai_compat::OpenAiCompatBackend;
#[cfg(feature = "responses")]
pub use openai_responses::OpenAiResponsesBackend;

use futures_core::Stream;

use crate::types::generation::GenerationEvent;

/// Flattens a stream of provider-native items into a stream of normalized generation events.
///
/// Each provider item may map to zero or more events (e.g. a chat chunk carrying both a text delta
/// and a finish reason). Errors pass through unchanged. Shared by the backends whose streaming
/// conversion is stateless per item.
pub(crate) fn flatten_events<T, F>(
    stream: impl Stream<Item = Result<T, just_common::error::TransportError>> + Send + 'static,
    convert: F,
) -> std::pin::Pin<
    Box<dyn Stream<Item = Result<GenerationEvent, just_common::error::TransportError>> + Send>,
>
where
    T: 'static,
    F: Fn(T) -> Vec<GenerationEvent> + Send + 'static,
{
    Box::pin(futures_util::StreamExt::flat_map(stream, move |item| {
        let events: Vec<Result<GenerationEvent, just_common::error::TransportError>> = match item {
            Ok(item) => convert(item).into_iter().map(Ok).collect(),
            Err(e) => vec![Err(e)],
        };
        futures_util::stream::iter(events)
    }))
}

/// Fill `meta.latency` on a parsed response without clobbering transport metadata the
/// adapter already extracted (e.g. `request_id`), so the library's timing points compose.
pub(crate) fn attach_latency(response: &mut GenerationResponse, latency: Duration) {
    response.meta.get_or_insert_default().latency = Some(latency);
}

/// Unified trait for the runtime-selected LLM provider surface.
///
/// Combines generation operations (prepare/send/generate and their streaming counterparts) with
/// identity and capability negotiation. All types are concrete (no associated types) for object
/// safety.
///
/// # Why `generate` and `stream_generate` are default impls
///
/// They compose [`prepare`](LlmBackend::prepare) + [`send`](LlmBackend::send) +
/// [`parse`](LlmBackend::parse). The provider-specific deserialization lives in the required
/// [`parse`](LlmBackend::parse) / [`parse_streaming`](LlmBackend::parse_streaming) methods, which
/// each backend implements against its own provider-native type and lifts to normalized types via
/// `From`. Override the defaults only for non-HTTP backends that cannot express a generation as
/// prepare/send/parse.
///
/// Callers typically access this through [`GenerationClient`](crate::GenerationClient) which
/// implements [`Deref`](std::ops::Deref) to `dyn LlmBackend`, so all methods are available without
/// importing the trait explicitly.
///
/// # Prepare-send-parse pattern
///
/// Use [`prepare`](LlmBackend::prepare) (or [`prepare_streaming`](LlmBackend::prepare_streaming))
/// to obtain a `reqwest::Request`, then [`send`](LlmBackend::send) to execute it. The returned
/// `reqwest::Request` is `Send + Sync`; it has no `Clone` impl, but
/// [`Request::try_clone`](reqwest::Request::try_clone) returns `Some` for the buffered-JSON bodies
/// the built-in backends produce, so callers can store one prepared request and re-send a clone on
/// retry. [`send`](LlmBackend::send) returns the raw `reqwest::Response` without checking status,
/// so headers like `retry-after` and `x-ratelimit-*` are readable before the body is consumed.
/// Finally [`parse`](LlmBackend::parse) (or [`parse_streaming`](LlmBackend::parse_streaming))
/// deserializes the response into a normalized type, dispatched to the right backend through the
/// trait object.
///
/// ```ignore
/// let prepared = backend.prepare(req)?;
/// // Re-send a clone on each attempt (bodies are buffered bytes, so try_clone succeeds).
/// let response = backend.send(prepared.try_clone().expect("buffered body")).await?;
/// // Inspect status / headers before consuming the body.
/// let retry_after = response.headers().get("retry-after");
/// if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
///     // back off, then re-send a fresh clone of `prepared` ...
/// }
/// // Deserialize with the right backend (dyn dispatch on `self`).
/// let generation = backend.parse(response).await?;
/// ```
#[async_trait]
pub trait LlmBackend: Identifiable + CapabilityNegotiation + Send + Sync {
    /// Prepare a non-streaming request for later execution.
    ///
    /// Returns a `reqwest::Request` with the URL, Content-Type, auth headers, and serialized
    /// body already set.
    ///
    /// This is a synchronous operation — it validates and serializes the request but performs no
    /// IO.
    fn prepare(&self, request: GenerationRequest) -> Result<reqwest::Request, BackendError>;

    /// Prepare a streaming request for later execution.
    ///
    /// Same as [`prepare`](LlmBackend::prepare) but forces `stream = true` on the request.
    fn prepare_streaming(
        &self,
        request: GenerationRequest,
    ) -> Result<reqwest::Request, BackendError>;

    /// Send a prepared request and return the raw HTTP response.
    ///
    /// Returns the response without checking HTTP status — callers must handle non-success
    /// statuses (4xx, 5xx) themselves. This allows inspecting response headers (e.g.
    /// `retry-after`, `x-ratelimit-*`) before consuming the body.
    ///
    /// For automatic status checking and deserialization, use [`generate`](LlmBackend::generate)
    /// or [`stream_generate`](LlmBackend::stream_generate) instead.
    async fn send(&self, prepared: reqwest::Request) -> Result<reqwest::Response, BackendError>;

    /// Parse a raw response into a normalized non-streaming generation.
    ///
    /// Implementations must check HTTP status before deserializing (use
    /// [`ensure_success`](just_common::transport::http::ensure_success)); [`send`](LlmBackend::send)
    /// intentionally does not, so headers stay readable before the body is consumed.
    ///
    /// Pair with [`prepare`](LlmBackend::prepare) + [`send`](LlmBackend::send) when you need
    /// the raw response in hand, e.g. to inspect `retry-after` / `x-ratelimit-*` headers, or
    /// to re-send a clone of the prepared request (via try_clone) for retry, before deserializing.
    ///
    /// Transport metadata (e.g. `x-request-id`) lands in [`GenerationResponse::meta`]
    /// when the provider sends it. No latency is measured on this path: call sites driving
    /// [`prepare`](LlmBackend::prepare) + [`send`](LlmBackend::send) + `parse` by hand can time
    /// the call themselves; [`generate`](LlmBackend::generate) attaches it.
    async fn parse(&self, response: reqwest::Response) -> Result<GenerationResponse, BackendError>;

    /// Parse a raw response into a normalized streaming event stream.
    ///
    /// Implementations must check HTTP status first (use
    /// [`ensure_success`](just_common::transport::http::ensure_success)); the SSE parser assumes a
    /// 2xx event stream.
    async fn parse_streaming(
        &self,
        response: reqwest::Response,
    ) -> Result<GenerationStream, BackendError>;

    /// Execute a non-streaming generation: validate -> prepare -> send -> parse.
    ///
    /// Default impl; override only for non-HTTP backends. Validation is repeated here and again
    /// inside [`prepare`](LlmBackend::prepare) deliberately: this entry point attributes
    /// invalid-request errors to `generate`, while `prepare` attributes them to itself,
    /// so both messages stay correct.
    ///
    /// The returned response's `meta.latency` measures this call end-to-end; drivers that
    /// run [`prepare`](LlmBackend::prepare) + [`send`](LlmBackend::send) +
    /// [`parse`](LlmBackend::parse) themselves can time it instead.
    async fn generate(
        &self,
        request: GenerationRequest,
    ) -> Result<GenerationResponse, BackendError> {
        let started = Instant::now();
        validate_non_streaming_request(&request, "generate", "stream_generate")?;
        let prepared = self.prepare(request)?;
        let response = self.send(prepared).await?;
        let mut parsed = self.parse(response).await?;
        attach_latency(&mut parsed, started.elapsed());
        Ok(parsed)
    }

    /// Execute a streaming generation: validate -> prepare_streaming -> send -> parse_streaming.
    ///
    /// Default impl; override only for non-HTTP backends. Validation is repeated here and again
    /// inside [`prepare_streaming`](LlmBackend::prepare_streaming) deliberately so invalid-request
    /// errors attribute to `stream_generate` rather than `prepare_streaming`.
    async fn stream_generate(
        &self,
        request: GenerationRequest,
    ) -> Result<GenerationStream, BackendError> {
        let request = into_validated_streaming_request(request, "stream_generate")?;
        let prepared = self.prepare_streaming(request)?;
        let response = self.send(prepared).await?;
        self.parse_streaming(response).await
    }

    /// Render messages to their provider-specific JSON string representation.
    ///
    /// The returned string matches the `messages`/`input` field of a generation request body —
    /// excluding top-level fields such as Anthropic's `system` or Responses' `instructions` —
    /// exactly as the provider would receive it. Useful for token estimation.
    fn render_messages(&self, messages: &[Message]) -> Result<String, BackendError>;

    /// Render tool definitions to their provider-specific JSON string representation.
    ///
    /// The returned string matches exactly what the provider would receive in the `tools`
    /// field of a generation request body. Useful for token estimation.
    fn render_tools(&self, tools: &[ToolDefinition]) -> Result<String, BackendError>;

    /// The backend family this type produces (e.g. [`crate::family::DEEPSEEK`]).
    ///
    /// Static — no instance needed — so a [`BackendFactory`](crate::BackendFactory) can key on it
    /// before constructing anything. It mirrors the instance [`Identifiable::family`], which
    /// reports the family of an already-built backend; both return the same centralized
    /// [`family`](crate::family) constant.
    fn family() -> &'static str
    where
        Self: Sized;

    /// Build a shared backend from raw inputs.
    ///
    /// Static, so not object-safe: the `Self: Sized` bound excludes it from `dyn LlmBackend`
    /// (which keeps working for the behavior methods above). Used by
    /// [`BackendFactory`](crate::BackendFactory); callable on concrete backend types (the trait
    /// must be in scope for the `Type::new(...)` call). Returns the shared trait object rather than
    /// a concrete `Self`.
    ///
    /// Configure `http` with `connect_timeout` + `read_timeout` rather than a single
    /// `timeout` so streaming responses are not aborted mid-flight. Unlike direct provider
    /// construction, no library transport defaults apply on this path: `http` is forwarded
    /// as an explicit client, so timeouts and the TLS backend are the caller's to set; see
    /// `docs/usage/timeouts.md`.
    #[allow(clippy::new_ret_no_self)]
    fn new(
        http: reqwest::ClientBuilder,
        api_key: &str,
        base_url: Option<&str>,
    ) -> Result<Arc<dyn LlmBackend>, BackendConstructError>
    where
        Self: Sized;
}

#[cfg(test)]
mod attach_latency_tests {
    use super::*;
    use crate::types::generation::{AssistantMessage, ResponseMeta};

    fn response(meta: Option<ResponseMeta>) -> GenerationResponse {
        GenerationResponse {
            id: "r".to_owned(),
            model: "m".to_owned(),
            message: AssistantMessage {
                content: None,
                tool_calls: Vec::new(),
                reasoning: None,
            },
            finish_reason: None,
            usage: None,
            meta,
        }
    }

    #[test]
    fn attach_latency_preserves_request_id_and_fills_missing_meta() {
        let mut with_id = response(Some(ResponseMeta {
            request_id: Some("req-1".to_owned()),
            latency: None,
        }));
        attach_latency(&mut with_id, Duration::from_millis(7));
        assert_eq!(
            with_id.meta.as_ref().unwrap().request_id.as_deref(),
            Some("req-1")
        );
        assert_eq!(
            with_id.meta.unwrap().latency,
            Some(Duration::from_millis(7))
        );

        let mut bare = response(None);
        attach_latency(&mut bare, Duration::from_millis(7));
        assert_eq!(bare.meta.as_ref().unwrap().request_id, None);
        assert_eq!(bare.meta.unwrap().latency, Some(Duration::from_millis(7)));
    }
}
