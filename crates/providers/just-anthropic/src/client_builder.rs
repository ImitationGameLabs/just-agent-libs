//! Builder for [`AnthropicClient`].

use just_common::ApiKey;
use just_common::error::TransportError;
use just_common::transport::http;
use reqwest::header::{HeaderMap, HeaderValue};

use crate::{AnthropicClient, Error};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const DEFAULT_API_VERSION: &str = "2023-06-01";

/// Builder for [`AnthropicClient`].
pub struct AnthropicClientBuilder {
    api_key: Option<ApiKey>,
    api_version: Option<String>,
    base_url: Option<String>,
    http_builder: Option<reqwest::ClientBuilder>,
}

impl AnthropicClientBuilder {
    /// Creates a new builder with default (empty) state.
    pub(crate) fn new() -> Self {
        Self {
            api_key: None,
            api_version: None,
            base_url: None,
            http_builder: None,
        }
    }

    /// Sets the API key (required).
    ///
    /// An unusable key (empty after trimming, or with characters that cannot go in an HTTP
    /// header) is rejected when [`build`](Self::build) runs, like a missing one.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        // An invalid key collapses to none here and is reported as missing at build time; the
        // discriminated reasons live in the ApiKey constructor.
        self.api_key = ApiKey::new(key.into()).ok();
        self
    }

    /// Sets the `anthropic-version` header. Defaults to `2023-06-01`.
    pub fn api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = Some(version.into());
        self
    }

    /// Sets a custom base URL. Defaults to `https://api.anthropic.com/v1`.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Provides a custom `reqwest::ClientBuilder`.
    ///
    /// Defaults to streaming-safe [`http::DEFAULT_CONNECT_TIMEOUT`] +
    /// [`http::DEFAULT_READ_TIMEOUT`] with `use_rustls_tls()`. The library injects
    /// the `x-api-key`, `anthropic-version`, and JSON `Accept` headers before
    /// building.
    pub fn http_client(mut self, builder: reqwest::ClientBuilder) -> Self {
        self.http_builder = Some(builder);
        self
    }

    /// Builds the client, validating required fields.
    pub fn build(self) -> Result<AnthropicClient, Error> {
        let api_key = self.api_key.ok_or_else(|| {
            Error::Transport(TransportError::InvalidConfig("api key is required"))
        })?;

        let base_url = self.base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_owned());

        if base_url.trim().is_empty() {
            return Err(Error::Transport(TransportError::InvalidConfig(
                "base url cannot be empty",
            )));
        }

        let api_version = self
            .api_version
            .unwrap_or_else(|| DEFAULT_API_VERSION.to_owned());

        let builder = self.http_builder.unwrap_or_else(|| {
            reqwest::Client::builder()
                .connect_timeout(http::DEFAULT_CONNECT_TIMEOUT)
                .read_timeout(http::DEFAULT_READ_TIMEOUT)
                .use_rustls_tls()
        });

        let mut headers = HeaderMap::new();
        let key_value = HeaderValue::from_str(api_key.expose()).map_err(|_| {
            Error::Transport(TransportError::InvalidConfig(
                "api key contains invalid header characters",
            ))
        })?;
        headers.insert("x-api-key", key_value);
        let version_value = HeaderValue::from_str(&api_version).map_err(|_| {
            Error::Transport(TransportError::InvalidConfig(
                "api version contains invalid header characters",
            ))
        })?;
        headers.insert("anthropic-version", version_value);

        let http = http::build_client_with_headers(builder, headers)?;

        Ok(AnthropicClient::new(http, base_url))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_api_key() {
        let error = AnthropicClient::builder().build().unwrap_err();
        assert!(matches!(
            error,
            Error::Transport(TransportError::InvalidConfig("api key is required"))
        ));
    }

    #[test]
    fn rejects_empty_api_key() {
        let error = AnthropicClient::builder()
            .api_key("   ")
            .build()
            .unwrap_err();
        // The key collapses to an unusable value at the boundary, so build rejects it like a
        // missing one; the discriminated reason is asserted by the ApiKey constructor tests.
        assert!(matches!(
            error,
            Error::Transport(TransportError::InvalidConfig(_))
        ));
    }

    #[test]
    fn rejects_empty_base_url() {
        let error = AnthropicClient::builder()
            .api_key("key")
            .base_url("   ")
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            Error::Transport(TransportError::InvalidConfig("base url cannot be empty"))
        ));
    }

    #[test]
    fn rejects_key_with_invalid_header_chars() {
        let error = AnthropicClient::builder()
            .api_key("bad\nkey")
            .build()
            .unwrap_err();
        // The key collapses to an unusable value at the boundary, so build rejects it like a
        // missing one; the discriminated reason is asserted by the ApiKey constructor tests.
        assert!(matches!(
            error,
            Error::Transport(TransportError::InvalidConfig(_))
        ));
    }

    #[test]
    fn rejects_version_with_invalid_header_chars() {
        let error = AnthropicClient::builder()
            .api_key("key")
            .api_version("bad\nversion")
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            Error::Transport(TransportError::InvalidConfig(
                "api version contains invalid header characters"
            ))
        ));
    }
}
