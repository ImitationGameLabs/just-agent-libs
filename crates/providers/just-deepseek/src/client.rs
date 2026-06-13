//! DeepSeek API client.

use just_common::transport::http::{endpoint_url, ensure_success, parse_json};
use reqwest::header::CONTENT_TYPE;

use crate::{
    Error,
    types::{
        balance::GetUserBalanceResponse,
        chat::{ChatCompletion, ChatCompletionRequest},
        models::ListModelsResponse,
    },
};

/// Async DeepSeek API client.
///
/// Holds a pre-configured `reqwest::Client` and base URL. Construct via
/// [`DeepSeekClient::builder()`] or [`DeepSeekClient::new()`].
#[derive(Clone, Debug)]
pub struct DeepSeekClient {
    http: reqwest::Client,
    base_url: String,
}

impl DeepSeekClient {
    // --- construction, accessors, and the prepare/send (raw HTTP) surface ---

    /// Creates a new client from pre-built components.
    ///
    /// The HTTP client should already have auth headers set (e.g. via
    /// [`just_common::transport::http::build_client`]).
    pub fn new(http: reqwest::Client, base_url: String) -> Self {
        Self { http, base_url }
    }

    /// Returns a builder for constructing a new client.
    pub fn builder() -> crate::client_builder::DeepSeekClientBuilder {
        crate::client_builder::DeepSeekClientBuilder::new()
    }

    /// Returns the underlying HTTP client.
    pub fn http_client(&self) -> &reqwest::Client {
        &self.http
    }

    /// Returns the configured base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Prepares a non-streaming chat completion request for later execution.
    ///
    /// Serializes the request body and builds a complete `reqwest::Request`.
    /// This is a synchronous operation (no IO).
    pub fn prepare(&self, request: ChatCompletionRequest) -> Result<reqwest::Request, Error> {
        if request.stream.unwrap_or(false) {
            return Err(Error::InvalidRequest(
                "stream=true is not supported by prepare; use prepare_streaming instead".into(),
            ));
        }
        self.build_request(request, "/chat/completions")
    }

    /// Prepares a streaming chat completion request for later execution.
    ///
    /// Forces `stream = true` on the request, then serializes and builds.
    /// This is a synchronous operation (no IO).
    pub fn prepare_streaming(
        &self,
        mut request: ChatCompletionRequest,
    ) -> Result<reqwest::Request, Error> {
        request.stream = Some(true);
        self.build_request(request, "/chat/completions")
    }

    /// Sends a prepared request and returns the raw HTTP response without checking status.
    ///
    /// Callers must handle non-success statuses themselves. For automatic status checking and
    /// deserialization, use [`chat_completion`](Self::chat_completion) or
    /// [`stream_chat_completion`](Self::stream_chat_completion).
    pub async fn send(&self, request: reqwest::Request) -> Result<reqwest::Response, Error> {
        self.http
            .execute(request)
            .await
            .map_err(just_common::error::TransportError::Transport)
            .map_err(Error::from)
    }

    /// Builds a `reqwest::Request` from a serializable body and endpoint path.
    fn build_request(
        &self,
        body: impl serde::Serialize,
        path: &str,
    ) -> Result<reqwest::Request, Error> {
        let url = endpoint_url(&self.base_url, path)?;
        let payload = serde_json::to_vec(&body)?;
        self.http
            .post(url)
            .header(CONTENT_TYPE, "application/json")
            .body(payload)
            .build()
            .map_err(just_common::error::TransportError::Transport)
            .map_err(Error::from)
    }
}

impl DeepSeekClient {
    // --- typed operations (hide HTTP entirely) ---

    /// Parses a raw HTTP response into a provider-native non-streaming completion.
    ///
    /// Performs HTTP status checking and JSON deserialization. Combine with
    /// [`prepare`](Self::prepare) and [`send`](Self::send) for full control over the request
    /// lifecycle, e.g. to inspect response headers before consuming the body.
    pub async fn parse(&self, response: reqwest::Response) -> Result<ChatCompletion, Error> {
        parse_json(response).await
    }

    /// Parses a raw HTTP response into a provider-native streaming chunk stream.
    ///
    /// Checks the HTTP status first (the SSE stream parser assumes a 2xx event stream).
    pub async fn parse_streaming(
        &self,
        response: reqwest::Response,
    ) -> Result<crate::ChatCompletionStream, Error> {
        let response = ensure_success(response).await?;
        crate::ChatCompletionStream::from_response(response).map_err(Error::Transport)
    }

    /// Executes a non-streaming chat completion request.
    pub async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletion, Error> {
        let response = self.send(self.prepare(request)?).await?;
        self.parse(response).await
    }

    /// Starts a streaming chat completion request.
    pub async fn stream_chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<crate::ChatCompletionStream, Error> {
        let response = self.send(self.prepare_streaming(request)?).await?;
        self.parse_streaming(response).await
    }

    /// Lists models currently exposed by the configured endpoint.
    pub async fn list_models(&self) -> Result<ListModelsResponse, Error> {
        let url = endpoint_url(&self.base_url, "/models")?;
        let response = self
            .http
            .get(url)
            .send()
            .await
            .map_err(just_common::error::TransportError::Transport)?;
        parse_json(response).await
    }

    /// Returns the current user balance state.
    pub async fn get_user_balance(&self) -> Result<GetUserBalanceResponse, Error> {
        let url = endpoint_url(&self.base_url, "/user/balance")?;
        let response = self
            .http
            .get(url)
            .send()
            .await
            .map_err(just_common::error::TransportError::Transport)?;
        parse_json(response).await
    }
}
