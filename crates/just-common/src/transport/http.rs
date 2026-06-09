//! Shared HTTP transport helpers for OpenAI-like providers.

use reqwest::{
    Method, Response,
    header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue},
};
use serde::{Serialize, de::DeserializeOwned};

use crate::error::TransportError;

/// Applies Bearer auth and JSON accept headers to a caller-provided builder, then builds.
///
/// Use this when you need to customise TLS, proxy, connection pool, or other transport
/// settings but still want the library to manage authentication headers.
pub fn build_client(
    builder: reqwest::ClientBuilder,
    api_key: &str,
) -> Result<reqwest::Client, TransportError> {
    let mut default_headers = HeaderMap::new();
    let auth_value = HeaderValue::from_str(&format!("Bearer {api_key}"))
        .map_err(|_| TransportError::InvalidConfig("api key contains invalid header characters"))?;
    default_headers.insert(AUTHORIZATION, auth_value);
    default_headers.insert(ACCEPT, HeaderValue::from_static("application/json"));

    builder
        .default_headers(default_headers)
        .build()
        .map_err(TransportError::BuildClient)
}

/// Executes a JSON request and deserializes the response body.
pub async fn request_json<Req, Resp>(
    http: &reqwest::Client,
    base_url: &str,
    method: Method,
    path: &str,
    body: Option<&Req>,
    extra_headers: Option<&HeaderMap>,
) -> Result<Resp, TransportError>
where
    Req: Serialize + ?Sized,
    Resp: DeserializeOwned,
{
    let response = request::<Req>(http, base_url, method, path, body, extra_headers).await?;
    parse_json::<Resp>(response).await
}

/// Executes a request and returns the successful raw response.
///
/// `extra_headers` are merged last (after body/Content-Type), overriding any matching
/// default or per-request headers. Pass `None` to rely solely on the client's defaults.
pub async fn request<Req>(
    http: &reqwest::Client,
    base_url: &str,
    method: Method,
    path: &str,
    body: Option<&Req>,
    extra_headers: Option<&HeaderMap>,
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

    // Apply extra headers last so caller overrides take precedence.
    if let Some(headers) = extra_headers {
        request = request.headers(headers.clone());
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
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}
