use just_common::prepared::PreparedChatRequest;
use just_common::transport::http;
use reqwest::Method;
use serde::{Serialize, de::DeserializeOwned};

use crate::{
    Error,
    types::{
        balance::GetUserBalanceResponse,
        chat::{ChatCompletion, ChatCompletionRequest},
        models::ListModelsResponse,
    },
};

/// Async DeepSeek API client.
#[derive(Clone, Debug)]
pub struct DeepSeekClient {
    http: reqwest::Client,
    base_url: String,
}

impl DeepSeekClient {
    /// Creates a new client from pre-built components.
    pub(crate) fn new(http: reqwest::Client, base_url: String) -> Self {
        Self { http, base_url }
    }

    /// Returns a builder for constructing a new client.
    pub fn builder() -> crate::client_builder::DeepSeekClientBuilder {
        crate::client_builder::DeepSeekClientBuilder::new()
    }

    /// Prepares a non-streaming chat completion request for later execution.
    ///
    /// Serializes the request to JSON and validates that required fields are present.
    /// This is a synchronous operation (no IO).
    pub fn prepare(&self, request: ChatCompletionRequest) -> Result<PreparedChatRequest, Error> {
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
        let response = http::request(
            &self.http,
            &self.base_url,
            Method::POST,
            "/chat/completions",
            Some(prepared.request_body()),
            Some(prepared.headers()),
        )
        .await?;
        Ok(http::parse_json::<ChatCompletion>(response).await?)
    }

    /// Prepares a streaming chat completion request for later execution.
    ///
    /// Forces `stream = true` on the request, then serializes and validates.
    /// This is a synchronous operation (no IO).
    pub fn prepare_streaming(
        &self,
        mut request: ChatCompletionRequest,
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
        let response = http::request(
            &self.http,
            &self.base_url,
            Method::POST,
            "/chat/completions",
            Some(prepared.request_body()),
            Some(prepared.headers()),
        )
        .await?;
        crate::ChatCompletionStream::from_response(response).map_err(Error::Transport)
    }

    /// Executes a non-streaming chat completion request.
    pub async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletion, Error> {
        let prepared = self.prepare(request)?;
        self.send(&prepared).await
    }

    /// Starts a streaming chat completion request.
    pub async fn stream_chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<crate::ChatCompletionStream, Error> {
        let prepared = self.prepare_streaming(request)?;
        self.send_streaming(&prepared).await
    }

    /// Lists models currently exposed by the configured endpoint.
    pub async fn list_models(&self) -> Result<ListModelsResponse, Error> {
        self.request_json(Method::GET, "/models", Option::<&()>::None)
            .await
    }

    /// Returns the current user balance state.
    pub async fn get_user_balance(&self) -> Result<GetUserBalanceResponse, Error> {
        self.request_json(Method::GET, "/user/balance", Option::<&()>::None)
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
        Ok(
            http::request_json::<Req, Resp>(&self.http, &self.base_url, method, path, body, None)
                .await?,
        )
    }
}
