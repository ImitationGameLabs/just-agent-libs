use just_common::prepared::PreparedChatRequest;
use just_common::transport::http;
use reqwest::{Method, header::HeaderMap};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::{
    Error,
    types::{
        chat::{ChatCompletion, CreateChatCompletionRequest},
        models::ListModelsResponse,
    },
};

/// Async client for OpenAI-compatible APIs.
#[derive(Clone, Debug)]
pub struct OpenAiCompatClient {
    http: reqwest::Client,
    base_url: String,
}

impl OpenAiCompatClient {
    /// Creates a new client from pre-built components.
    pub(crate) fn new(http: reqwest::Client, base_url: String) -> Self {
        Self { http, base_url }
    }

    /// Returns a builder for constructing a new client.
    pub fn builder() -> crate::client_builder::OpenAiCompatClientBuilder {
        crate::client_builder::OpenAiCompatClientBuilder::new()
    }

    /// Prepares a non-streaming chat completion request for later execution.
    ///
    /// Serializes the request to JSON and validates that required fields are present.
    /// This is a synchronous operation (no IO).
    pub fn prepare(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<PreparedChatRequest, Error> {
        if request.stream.unwrap_or(false) {
            return Err(Error::InvalidRequest(
                "stream=true is not supported by prepare; use prepare_streaming instead".into(),
            ));
        }
        let value = serde_json::to_value(&request)?;
        PreparedChatRequest::from_request_body(value).map_err(Error::from)
    }

    /// Sends a previously prepared non-streaming request.
    pub async fn send(&self, prepared: &PreparedChatRequest) -> Result<ChatCompletion, Error> {
        self.send_raw_json(
            Method::POST,
            "/chat/completions",
            prepared.request_body(),
            prepared.headers(),
        )
        .await
    }

    /// Prepares a streaming chat completion request for later execution.
    ///
    /// Forces `stream = true` on the request, then serializes and validates.
    /// This is a synchronous operation (no IO).
    pub fn prepare_streaming(
        &self,
        mut request: CreateChatCompletionRequest,
    ) -> Result<PreparedChatRequest, Error> {
        request.stream = Some(true);
        let value = serde_json::to_value(&request)?;
        PreparedChatRequest::from_request_body(value).map_err(Error::from)
    }

    /// Sends a previously prepared streaming request.
    pub async fn send_streaming(
        &self,
        prepared: &PreparedChatRequest,
    ) -> Result<crate::ChatCompletionStream, Error> {
        let response = self
            .stream_raw_json(
                Method::POST,
                "/chat/completions",
                prepared.request_body(),
                prepared.headers(),
            )
            .await?;
        crate::ChatCompletionStream::from_response(response).map_err(Error::Transport)
    }

    /// Executes a non-streaming chat completion request.
    pub async fn create_chat_completion(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<ChatCompletion, Error> {
        let prepared = self.prepare(request)?;
        self.send(&prepared).await
    }

    /// Starts a streaming chat completion request.
    pub async fn stream_chat_completion(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<crate::ChatCompletionStream, Error> {
        let prepared = self.prepare_streaming(request)?;
        self.send_streaming(&prepared).await
    }

    /// Lists models currently exposed by the configured endpoint.
    pub async fn list_models(&self) -> Result<ListModelsResponse, Error> {
        self.request_json(Method::GET, "/models", Option::<&()>::None)
            .await
    }

    /// Sends a pre-serialized JSON body with extra headers and deserializes the response.
    ///
    /// Used internally by [`send`](Self::send).
    pub(crate) async fn send_raw_json<T: DeserializeOwned>(
        &self,
        method: Method,
        path: &str,
        body: &Value,
        extra_headers: &HeaderMap,
    ) -> Result<T, Error> {
        let response = http::request(
            &self.http,
            &self.base_url,
            method,
            path,
            Some(body),
            Some(extra_headers),
        )
        .await?;
        Ok(http::parse_json::<T>(response).await?)
    }

    /// Sends a pre-serialized JSON body with extra headers, returning the raw response for SSE.
    ///
    /// Used internally by [`send_streaming`](Self::send_streaming).
    pub(crate) async fn stream_raw_json(
        &self,
        method: Method,
        path: &str,
        body: &Value,
        extra_headers: &HeaderMap,
    ) -> Result<reqwest::Response, Error> {
        Ok(http::request::<Value>(
            &self.http,
            &self.base_url,
            method,
            path,
            Some(body),
            Some(extra_headers),
        )
        .await?)
    }

    async fn request_json<Req, Resp>(
        &self,
        method: Method,
        path: &str,
        body: Option<&Req>,
    ) -> Result<Resp, Error>
    where
        Req: Serialize + ?Sized,
        Resp: DeserializeOwned,
    {
        Ok(
            http::request_json::<Req, Resp>(&self.http, &self.base_url, method, path, body, None)
                .await?,
        )
    }
}
