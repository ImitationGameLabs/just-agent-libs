use std::time::Duration;

use just_common::error::TransportError;

const DEFAULT_BASE_URL: &str = "https://api.deepseek.com/v1";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// Configuration for [`crate::DeepSeekClient`].
#[derive(Clone, Debug)]
pub struct DeepSeekConfig {
    api_key: String,
    base_url: String,
    timeout: Duration,
    user_agent: Option<String>,
}

impl DeepSeekConfig {
    /// Creates a configuration with DeepSeek defaults.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_owned(),
            timeout: DEFAULT_TIMEOUT,
            user_agent: Some(format!("just-deepseek/{}", env!("CARGO_PKG_VERSION"))),
        }
    }

    /// Overrides the API base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Overrides the request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Overrides the default user-agent string.
    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = Some(user_agent.into());
        self
    }

    /// Removes the user-agent header override entirely.
    pub fn without_user_agent(mut self) -> Self {
        self.user_agent = None;
        self
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), TransportError> {
        if self.api_key.trim().is_empty() {
            return Err(TransportError::InvalidConfig("api key cannot be empty"));
        }

        if self.base_url.trim().is_empty() {
            return Err(TransportError::InvalidConfig("base url cannot be empty"));
        }

        Ok(())
    }

    /// Returns the configured API key.
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Returns the configured base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Returns the configured request timeout.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Returns the configured user-agent string.
    pub fn user_agent(&self) -> Option<&str> {
        self.user_agent.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::DeepSeekConfig;
    use just_common::error::TransportError;

    #[test]
    fn rejects_empty_api_key() {
        let config = DeepSeekConfig::new("   ");
        let error = config.validate().unwrap_err();

        assert!(matches!(
            error,
            TransportError::InvalidConfig("api key cannot be empty")
        ));
    }
}
