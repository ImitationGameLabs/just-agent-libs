use std::time::Duration;

use just_common::error::TransportError;
use just_common::transport::http;
use reqwest::Method;
use serde::{Serialize, de::DeserializeOwned};

use crate::{
    ChatCompletionStream, Error,
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

    /// Executes a non-streaming chat completion request.
    pub async fn create_chat_completion(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<ChatCompletion, Error> {
        if request.stream.unwrap_or(false) {
            return Err(Error::InvalidRequest(
                "stream=true is not supported by create_chat_completion; use stream_chat_completion instead",
            ));
        }

        self.request_json(Method::POST, "/chat/completions", Some(&request))
            .await
    }

    /// Starts a streaming chat completion request.
    pub async fn stream_chat_completion(
        &self,
        mut request: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionStream, Error> {
        request.stream = Some(true);

        let response = self
            .request(Method::POST, "/chat/completions", Some(&request))
            .await?;

        Ok(ChatCompletionStream::from_response(response)?)
    }

    /// Lists models currently exposed by the configured endpoint.
    pub async fn list_models(&self) -> Result<ListModelsResponse, Error> {
        self.request_json(Method::GET, "/models", Option::<&()>::None)
            .await
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
        Ok(http::request_json::<Req, Resp>(&self.http, &self.base_url, method, path, body).await?)
    }

    async fn request<Req>(
        &self,
        method: Method,
        path: &str,
        body: Option<&Req>,
    ) -> Result<reqwest::Response, Error>
    where
        Req: Serialize + ?Sized,
    {
        Ok(http::request::<Req>(&self.http, &self.base_url, method, path, body).await?)
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
