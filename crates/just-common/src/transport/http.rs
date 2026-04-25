//! Shared HTTP transport helpers for OpenAI-like providers.

use std::time::Duration;

use reqwest::{
    Method, Response,
    header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue},
};
use serde::{Serialize, de::DeserializeOwned};

use crate::error::TransportError;

/// Builds a `reqwest` client with Bearer auth and common defaults.
pub fn build_http_client(
    api_key: &str,
    timeout: Duration,
    user_agent: Option<&str>,
) -> Result<reqwest::Client, TransportError> {
    let mut default_headers = HeaderMap::new();
    let auth_value = HeaderValue::from_str(&format!("Bearer {api_key}"))
        .map_err(|_| TransportError::InvalidConfig("api key contains invalid header characters"))?;
    default_headers.insert(AUTHORIZATION, auth_value);
    default_headers.insert(ACCEPT, HeaderValue::from_static("application/json"));

    let mut builder = reqwest::Client::builder()
        .default_headers(default_headers)
        .timeout(timeout)
        .use_rustls_tls();

    if let Some(user_agent) = user_agent {
        builder = builder.user_agent(user_agent);
    }

    builder.build().map_err(TransportError::BuildClient)
}

/// Executes a JSON request and deserializes the response body.
pub async fn request_json<Req, Resp>(
    http: &reqwest::Client,
    base_url: &str,
    method: Method,
    path: &str,
    body: Option<&Req>,
) -> Result<Resp, TransportError>
where
    Req: Serialize + ?Sized,
    Resp: DeserializeOwned,
{
    let response = request::<Req>(http, base_url, method, path, body).await?;
    parse_json::<Resp>(response).await
}

/// Executes a request and returns the successful raw response.
pub async fn request<Req>(
    http: &reqwest::Client,
    base_url: &str,
    method: Method,
    path: &str,
    body: Option<&Req>,
) -> Result<Response, TransportError>
where
    Req: Serialize + ?Sized,
{
    let url = endpoint_url(base_url, path);
    let mut request = http.request(method, url);

    if let Some(body) = body {
        let payload = serde_json::to_vec(body).map_err(TransportError::Serialize)?;
        request = request
            .header(CONTENT_TYPE, "application/json")
            .body(payload);
    }

    let response = request.send().await.map_err(TransportError::Transport)?;
    ensure_success(response).await
}

/// Converts a successful HTTP response into JSON.
pub async fn parse_json<T>(response: Response) -> Result<T, TransportError>
where
    T: DeserializeOwned,
{
    let body = response.text().await.map_err(TransportError::Transport)?;
    serde_json::from_str(&body).map_err(|source| TransportError::Deserialize { source, body })
}

/// Returns an error for non-success HTTP responses while preserving the raw response body.
pub async fn ensure_success(response: Response) -> Result<Response, TransportError> {
    let status = response.status();

    if status.is_success() {
        return Ok(response);
    }

    let body = response.text().await.map_err(TransportError::Transport)?;
    Err(TransportError::HttpStatus { status, body })
}

/// Joins a base URL and endpoint path without duplicating slashes.
pub fn endpoint_url(base_url: &str, path: &str) -> String {
    format!("{}/{}", base_url, path.trim_start_matches('/'))
}
