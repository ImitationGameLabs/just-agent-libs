//! Shared HTTP transport helpers for OpenAI-like providers.
//!
//! # Status-checking invariant
//!
//! Provider clients never hand a raw `reqwest::Response` to a body consumer without its HTTP
//! status checked first. The check lives at the consume-the-response boundary: [`parse_json`]
//! validates status for JSON bodies; callers consuming a response any other way (e.g. an SSE
//! stream) must call [`ensure_success`] explicitly first.

use reqwest::{
    Response,
    header::{ACCEPT, AUTHORIZATION, HeaderMap, HeaderValue},
};
use serde::de::DeserializeOwned;

use crate::error::{ProviderError, TransportError};

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

/// Returns an error for non-success HTTP responses while preserving the raw response body.
pub async fn ensure_success(response: Response) -> Result<Response, TransportError> {
    let status = response.status();

    if status.is_success() {
        return Ok(response);
    }

    let body = response.text().await.map_err(TransportError::Transport)?;
    Err(TransportError::HttpStatus { status, body })
}

/// Joins a base URL and endpoint path using standards-compliant URL resolution.
///
/// Uses [`reqwest::Url::join`] (WHATWG URL Standard) to correctly handle trailing slashes,
/// query strings, and path segments.
///
/// # Errors
///
/// Returns [`TransportError::InvalidConfig`] if `base_url` is not a valid absolute URL.
pub fn endpoint_url(base_url: &str, path: &str) -> Result<String, TransportError> {
    let mut base = reqwest::Url::parse(base_url)
        .map_err(|_| TransportError::InvalidConfig("invalid base URL"))?;
    // Ensure the base URL ends with '/' so that WHATWG resolution preserves all
    // existing path segments.  Without this, `https://api.example.com/v1` joined
    // with `chat/completions` would yield `https://api.example.com/chat/completions`
    // (dropping `v1`) because WHATWG treats the last segment as a "file".
    if !base.path().ends_with('/') {
        let mut p = base.path().to_owned();
        p.push('/');
        base.set_path(&p);
    }
    let full = base
        .join(path.trim_start_matches('/'))
        .map_err(|_| TransportError::InvalidConfig("invalid endpoint path"))?;
    Ok(full.into())
}

/// Checks HTTP status, then reads the response body and deserializes it as JSON.
///
/// A non-success status is returned as a [`TransportError::HttpStatus`] (with the raw body
/// preserved) before deserialization is attempted; on deserialization failure the raw body text
/// is likewise preserved in the error for diagnostics.
pub async fn parse_json<T: DeserializeOwned>(response: Response) -> Result<T, ProviderError> {
    let response = ensure_success(response).await?;
    let body = response.text().await.map_err(TransportError::Transport)?;
    serde_json::from_str(&body).map_err(|source| ProviderError::Deserialize { source, body })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_url_resolves_paths_correctly() {
        // WHATWG URL resolution treats the last segment as a "file" and replaces it
        // when joining a relative path.  Our wrapper must preserve all path segments.
        let cases = [
            // (base_url, path, expected)
            (
                "https://api.example.com/v1",
                "chat/completions",
                "https://api.example.com/v1/chat/completions",
            ),
            (
                "https://api.example.com/v1/",
                "chat/completions",
                "https://api.example.com/v1/chat/completions",
            ),
            (
                "https://api.example.com/v1",
                "/chat/completions",
                "https://api.example.com/v1/chat/completions",
            ),
            (
                "https://api.deepseek.com",
                "chat/completions",
                "https://api.deepseek.com/chat/completions",
            ),
            (
                "https://proxy.example.com/openai/v1",
                "chat/completions",
                "https://proxy.example.com/openai/v1/chat/completions",
            ),
        ];

        for (base, path, expected) in cases {
            assert_eq!(
                endpoint_url(base, path).unwrap(),
                expected,
                "endpoint_url({base:?}, {path:?})"
            );
        }
    }

    #[test]
    fn endpoint_url_rejects_invalid_base() {
        assert!(endpoint_url("not a url", "chat/completions").is_err());
    }
}
