use just_common::transport::http;
use reqwest::Method;
use serde::{Serialize, de::DeserializeOwned};

use crate::{
    ChatCompletionStream, Error, OpenAiCompatConfig,
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
    /// Creates a client with an explicit API key and base URL.
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Result<Self, Error> {
        Self::with_config(OpenAiCompatConfig::new(api_key, base_url))
    }

    /// Creates a client from a validated configuration value.
    pub fn with_config(config: OpenAiCompatConfig) -> Result<Self, Error> {
        config.validate()?;
        let http =
            http::build_http_client(config.api_key(), config.timeout(), config.user_agent())?;

        Ok(Self { http, base_url: config.base_url().trim_end_matches('/').to_owned() })
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
