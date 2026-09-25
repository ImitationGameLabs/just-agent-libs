//! Builder for [`OpenAiCompatClient`].

use just_common::ApiKey;
use just_common::error::TransportError;
use just_common::transport::http;

use crate::{Error, OpenAiCompatClient};

/// Builder for [`OpenAiCompatClient`].
pub struct OpenAiCompatClientBuilder {
    api_key: Option<ApiKey>,
    base_url: Option<String>,
    http_builder: Option<reqwest::ClientBuilder>,
}

impl OpenAiCompatClientBuilder {
    /// Creates a new builder with default (empty) state.
    pub(crate) fn new() -> Self {
        Self {
            api_key: None,
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

    /// Sets the base URL (required).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Provides a custom `reqwest::ClientBuilder`.
    ///
    /// Defaults to streaming-safe [`http::DEFAULT_CONNECT_TIMEOUT`] +
    /// [`http::DEFAULT_READ_TIMEOUT`] with `use_rustls_tls()`. The library injects
    /// Bearer auth headers before building.
    pub fn http_client(mut self, builder: reqwest::ClientBuilder) -> Self {
        self.http_builder = Some(builder);
        self
    }

    /// Builds the client, validating required fields.
    pub fn build(self) -> Result<OpenAiCompatClient, Error> {
        let api_key = self.api_key.ok_or_else(|| {
            Error::Transport(TransportError::InvalidConfig("api key is required"))
        })?;

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
                .connect_timeout(http::DEFAULT_CONNECT_TIMEOUT)
                .read_timeout(http::DEFAULT_READ_TIMEOUT)
                .use_rustls_tls()
        });

        let http = http::build_client(builder, api_key.expose())?;

        Ok(OpenAiCompatClient::new(http, base_url))
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
        // The key collapses to an unusable value at the boundary, so build rejects it like a
        // missing one; the discriminated reason is asserted by the ApiKey constructor tests.
        assert!(matches!(
            error,
            Error::Transport(TransportError::InvalidConfig(_))
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
