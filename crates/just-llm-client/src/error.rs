use std::{error::Error as StdError, fmt};

use just_common::error::PreparedRequestError;
use thiserror::Error;

/// Boxed provider error source carried by [`LlmError::Backend`].
pub type BoxError = Box<dyn StdError + Send + Sync>;

/// Capability names used in client-level unsupported or unavailable errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capability {
    /// One-shot chat completion execution.
    ChatCompletion,
    /// Incremental chat completion streaming.
    StreamingChatCompletion,
    /// Model catalog listing.
    ModelCatalog,
    /// Balance or quota inspection.
    Balance,
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::ChatCompletion => "chat completion",
            Self::StreamingChatCompletion => "streaming chat completion",
            Self::ModelCatalog => "model catalog",
            Self::Balance => "balance",
        };

        f.write_str(label)
    }
}

/// LLM client-layer error taxonomy.
///
/// This keeps capability-level failures distinct from provider transport or protocol failures.
#[derive(Debug, Error)]
pub enum LlmError {
    /// The request was invalid before it reached the provider.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// The backend never offers the requested capability.
    #[error("{backend} does not support {capability}")]
    UnsupportedCapability {
        /// Backend identifier.
        backend: &'static str,
        /// Capability that was requested.
        capability: Capability,
    },

    /// The backend is expected to offer the capability but has not implemented it yet.
    #[error("{backend} has not implemented {capability}")]
    UnimplementedCapability {
        /// Backend identifier.
        backend: &'static str,
        /// Capability that was expected.
        capability: Capability,
    },

    /// The backend can offer the capability in principle, but not in the current state.
    #[error("{backend} cannot currently provide {capability}: {message}")]
    UnavailableCapability {
        /// Backend identifier.
        backend: &'static str,
        /// Capability that is temporarily unavailable.
        capability: Capability,
        /// Additional explanation from the backend adapter.
        message: String,
    },

    /// The underlying provider SDK returned an error.
    #[error("{backend} backend error: {source}")]
    Backend {
        /// Backend identifier.
        backend: &'static str,
        #[source]
        /// Provider-specific source error.
        source: BoxError,
    },
}

impl LlmError {
    /// Creates an invalid-request error with a stable, user-facing message.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest(message.into())
    }

    /// Creates an unsupported-capability error for the given backend.
    pub fn unsupported(backend: &'static str, capability: Capability) -> Self {
        Self::UnsupportedCapability {
            backend,
            capability,
        }
    }

    /// Creates an unimplemented-capability error for the given backend.
    pub fn unimplemented(backend: &'static str, capability: Capability) -> Self {
        Self::UnimplementedCapability {
            backend,
            capability,
        }
    }

    /// Creates an unavailable-capability error for the given backend.
    pub fn unavailable(
        backend: &'static str,
        capability: Capability,
        message: impl Into<String>,
    ) -> Self {
        Self::UnavailableCapability {
            backend,
            capability,
            message: message.into(),
        }
    }

    /// Wraps a provider-specific source error.
    pub fn backend<E>(backend: &'static str, source: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self::Backend {
            backend,
            source: Box::new(source),
        }
    }
}

impl From<PreparedRequestError> for LlmError {
    fn from(e: PreparedRequestError) -> Self {
        LlmError::invalid_request(e.to_string())
    }
}
