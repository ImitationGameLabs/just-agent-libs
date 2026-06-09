use std::time::Duration;

use just_common::error::TransportError;
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

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// Async client for OpenAI-compatible APIs.
#[derive(Clone, Debug)]
pub struct OpenAiCompatClient {
    http: reqwest::Client,
    base_url: String,
}

/// Builder for [`OpenAiCompatClient`].
pub struct OpenAiCompatClientBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    http_builder: Option<reqwest::ClientBuilder>,
}

impl OpenAiCompatClient {
    /// Returns a builder for constructing a new client.
    pub fn builder() -> OpenAiCompatClientBuilder {
        OpenAiCompatClientBuilder { api_key: None, base_url: None, http_builder: None }
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

impl OpenAiCompatClientBuilder {
    /// Sets the API key (required).
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the base URL (required).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Provides a custom `reqwest::ClientBuilder`.
    ///
    /// Defaults to `reqwest::Client::builder().timeout(60s).use_rustls_tls()`.
    /// The library injects Bearer auth headers before building.
    pub fn http_client(mut self, builder: reqwest::ClientBuilder) -> Self {
        self.http_builder = Some(builder);
        self
    }

    /// Builds the client, validating required fields.
    pub fn build(self) -> Result<OpenAiCompatClient, Error> {
        let api_key = self.api_key.ok_or_else(|| {
            Error::Transport(TransportError::InvalidConfig("api key is required"))
        })?;

        if api_key.trim().is_empty() {
            return Err(Error::Transport(TransportError::InvalidConfig(
                "api key cannot be empty",
            )));
        }

        let base_url = self.base_url.ok_or_else(|| {
            Error::Transport(TransportError::InvalidConfig("base url is required"))
        })?;

        if base_url.trim().is_empty() {
            return Err(Error::Transport(TransportError::InvalidConfig(
                "base url cannot be empty",
            )));
        }

        let builder = self.http_builder.unwrap_or_else(|| {
            reqwest::Client::builder()
                .timeout(DEFAULT_TIMEOUT)
                .use_rustls_tls()
        });

        let http = http::build_client(builder, &api_key)?;

        Ok(OpenAiCompatClient { http, base_url: base_url.trim_end_matches('/').to_owned() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_api_key() {
        let error = OpenAiCompatClient::builder().build().unwrap_err();
        assert!(matches!(
            error,
            Error::Transport(TransportError::InvalidConfig("api key is required"))
        ));
    }

    #[test]
    fn rejects_empty_api_key() {
        let error = OpenAiCompatClient::builder()
            .api_key("   ")
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            Error::Transport(TransportError::InvalidConfig("api key cannot be empty"))
        ));
    }

    #[test]
    fn rejects_missing_base_url() {
        let error = OpenAiCompatClient::builder()
            .api_key("key")
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            Error::Transport(TransportError::InvalidConfig("base url is required"))
        ));
    }
}
