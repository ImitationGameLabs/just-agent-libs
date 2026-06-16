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

/// Maximum response-body size, in bytes, read into memory by the shared transport helpers.
///
/// Bounds every non-streaming body read (`read_body_text`) — both diagnostic error bodies
/// ([`TransportError::HttpStatus`]) and full response bodies fed to the deserializer
/// ([`parse_json`]) — so a malicious or broken server cannot exhaust memory with an arbitrarily
/// large body. Generous for any realistic chat-completion, model-catalog, or balance response.
///
/// This caps non-streaming reads only; the SSE streaming path is intentionally uncapped today
/// (see the `sse` module's "Known limitation" note).
pub(crate) const MAX_BODY_BYTES: usize = 8 * 1024 * 1024; // 8 MiB

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

/// Appends `chunk` to `buf` unless doing so would exceed `limit` bytes.
///
/// Pure core of the body-size cap, split out so it is unit-testable without a live HTTP server.
/// On overflow it returns [`TransportError::BodyTooLarge`] *without* appending the chunk.
fn append_capped(buf: &mut Vec<u8>, chunk: &[u8], limit: usize) -> Result<(), TransportError> {
    if buf.len().saturating_add(chunk.len()) > limit {
        return Err(TransportError::BodyTooLarge { limit });
    }
    buf.extend_from_slice(chunk);
    Ok(())
}

/// Reads the full response body as UTF-8 text, capped at `MAX_BODY_BYTES`.
///
/// This is the single place a non-streaming response body is consumed. Draining the chunk stream
/// incrementally (rather than `Response::text`) lets the cap reject an oversized body before it is
/// fully buffered. Reuses existing [`TransportError`] variants: a stream error becomes
/// [`Transport`](TransportError::Transport), an invalid-UTF-8 body becomes
/// [`Utf8`](TransportError::Utf8), and an oversized body becomes
/// [`BodyTooLarge`](TransportError::BodyTooLarge).
async fn read_body_text(response: Response) -> Result<String, TransportError> {
    use futures_util::StreamExt;
    let mut buf = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(TransportError::Transport)?;
        append_capped(&mut buf, &chunk, MAX_BODY_BYTES)?;
    }
    String::from_utf8(buf).map_err(TransportError::Utf8)
}

/// Returns an error for non-success HTTP responses, preserving up to `MAX_BODY_BYTES` of body.
///
/// On a non-success status the body is read via the capped `read_body_text`; an oversized error
/// body yields [`TransportError::BodyTooLarge`] rather than [`TransportError::HttpStatus`] (the
/// status is then not carried — an accepted trade-off for an extreme edge case).
pub async fn ensure_success(response: Response) -> Result<Response, TransportError> {
    let status = response.status();

    if status.is_success() {
        return Ok(response);
    }

    let body = read_body_text(response).await?;
    Err(TransportError::HttpStatus { status, body })
}

/// Joins a base URL and endpoint path using standards-compliant URL resolution.
///
/// Uses [`reqwest::Url::join`] (WHATWG URL Standard) to correctly handle trailing slashes,
/// query strings, and path segments. Any leading slashes on `path` are stripped before
/// resolution, so `"chat/completions"` and `"/chat/completions"` resolve identically against
/// the base URL.
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
/// Both the error body (on a non-success status) and the success body are read through the capped
/// `read_body_text`. An empty response body is returned as a [`TransportError::InvalidResponse`]
/// rather than reaching the deserializer, since the serde "EOF while parsing a value" message
/// would otherwise mislead readers into suspecting truncation. On any other deserialization
/// failure the raw body text is likewise preserved in the error for diagnostics.
///
/// # Errors
///
/// - Non-success status → [`TransportError::HttpStatus`] (full body, up to `MAX_BODY_BYTES`;
///   larger error bodies surface as `BodyTooLarge` instead).
/// - Oversized body (error or success) → [`TransportError::BodyTooLarge`].
/// - Invalid-UTF-8 body → [`TransportError::Utf8`].
/// - Stream/transport failure → [`TransportError::Transport`].
/// - Malformed JSON → [`ProviderError::Deserialize`].
pub async fn parse_json<T: DeserializeOwned>(response: Response) -> Result<T, ProviderError> {
    let response = ensure_success(response).await?;
    let body = read_body_text(response).await?;
    parse_body(body)
}

/// Deserializes a full response body, distinguishing empty bodies from malformed ones.
///
/// An empty (or whitespace-only) body is reported as an invalid response — a server returning
/// no content is almost always a misconfigured endpoint or proxy, and surfacing that explicitly
/// is clearer than the serde "EOF while parsing a value" message. Malformed non-empty bodies
/// surface as [`ProviderError::Deserialize`] with the raw body preserved.
///
/// Split out from [`parse_json`] so the classification logic is unit-testable without spinning
/// up a mock HTTP server.
fn parse_body<T: DeserializeOwned>(body: String) -> Result<T, ProviderError> {
    if body.trim().is_empty() {
        return Err(TransportError::InvalidResponse("empty response body".to_owned()).into());
    }
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

    #[test]
    fn parse_body_rejects_empty_body() {
        // Empty body → InvalidResponse, not Deserialize.
        let err = parse_body::<serde_json::Value>(String::new()).unwrap_err();
        assert!(
            matches!(
                err,
                ProviderError::Transport(TransportError::InvalidResponse(_))
            ),
            "empty body must surface as InvalidResponse, got {err:?}"
        );

        // Whitespace-only is also treated as empty.
        let err = parse_body::<serde_json::Value>("   \n\t ".to_owned()).unwrap_err();
        assert!(
            matches!(
                err,
                ProviderError::Transport(TransportError::InvalidResponse(_))
            ),
            "whitespace-only body must surface as InvalidResponse, got {err:?}"
        );
    }

    #[test]
    fn parse_body_maps_malformed_to_deserialize() {
        // Non-empty but malformed → Deserialize (the original behavior, unchanged).
        let err = parse_body::<serde_json::Value>("not valid json".to_owned()).unwrap_err();
        assert!(
            matches!(err, ProviderError::Deserialize { .. }),
            "malformed body must surface as Deserialize, got {err:?}"
        );
    }

    #[test]
    fn parse_body_deserializes_valid_json() {
        let value = parse_body::<serde_json::Value>(r#"{"x":7}"#.to_owned()).unwrap();
        assert_eq!(value["x"], 7);
    }

    #[test]
    fn append_capped_accepts_up_to_limit() {
        let mut buf = Vec::new();
        // Exactly at the limit is allowed: the bound is strict-greater.
        let fill = vec![0u8; 16];
        append_capped(&mut buf, &fill, 16).unwrap();
        assert_eq!(buf.len(), 16);
    }

    #[test]
    fn append_capped_rejects_overflow_without_appending() {
        let mut buf = Vec::new();
        let limit = 16;
        // Fill exactly to the limit.
        let fill = vec![0u8; limit];
        append_capped(&mut buf, &fill, limit).unwrap();
        // One more byte overflows.
        let err = append_capped(&mut buf, &[1], limit).unwrap_err();
        assert!(
            matches!(err, TransportError::BodyTooLarge { limit: 16 }),
            "overflow must surface as BodyTooLarge, got {err:?}"
        );
        // The overflowing chunk must NOT have been appended.
        assert_eq!(buf.len(), limit);
        assert!(!buf.contains(&1));
    }

    #[test]
    fn append_capped_rejects_single_oversized_chunk() {
        let mut buf = Vec::new();
        let oversized = vec![0u8; 17];
        let err = append_capped(&mut buf, &oversized, 16).unwrap_err();
        assert!(
            matches!(err, TransportError::BodyTooLarge { limit: 16 }),
            "first-chunk overflow must surface as BodyTooLarge, got {err:?}"
        );
        assert!(
            buf.is_empty(),
            "buffer must stay empty when the first chunk overflows"
        );
    }
}
