use std::time::Duration;

use just_common::error::TransportError;
use just_common::transport::http;

use crate::{DeepSeekClient, Error};

const DEFAULT_BASE_URL: &str = "https://api.deepseek.com";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// Builder for [`DeepSeekClient`].
pub struct DeepSeekClientBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    http_builder: Option<reqwest::ClientBuilder>,
}

impl DeepSeekClientBuilder {
    /// Creates a new builder with default (empty) state.
    pub(crate) fn new() -> Self {
        Self {
            api_key: None,
            base_url: None,
            http_builder: None,
        }
    }

    /// Sets the API key (required).
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets a custom base URL. Defaults to `https://api.deepseek.com`.
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
    pub fn build(self) -> Result<DeepSeekClient, Error> {
        let api_key = self.api_key.ok_or_else(|| {
            Error::Transport(TransportError::InvalidConfig("api key is required"))
        })?;

        if api_key.trim().is_empty() {
            return Err(Error::Transport(TransportError::InvalidConfig(
                "api key cannot be empty",
            )));
        }

        let base_url = self.base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_owned());

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

        Ok(DeepSeekClient::new(
            http,
            base_url.trim_end_matches('/').to_owned(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_api_key() {
        let error = DeepSeekClient::builder().build().unwrap_err();
        assert!(matches!(
            error,
            Error::Transport(TransportError::InvalidConfig("api key is required"))
        ));
    }

    #[test]
    fn rejects_empty_api_key() {
        let error = DeepSeekClient::builder()
            .api_key("   ")
            .build()
            .unwrap_err();
        assert!(matches!(
            error,
            Error::Transport(TransportError::InvalidConfig("api key cannot be empty"))
        ));
    }
}
