//! Error taxonomy for the LLM client layer, partitioned by **failure nature**.
//!
//! Three distinct failure natures get three distinct types, so a caller can never conflate them
//! at the type level:
//!
//! - [`BackendConstructError`] — *constructing* a backend failed (`LlmBackend::new`,
//!   `BackendFactory::create`). Precondition/setup failures only; never a chat call.
//! - [`CapabilityError`] — a backend *statically* does not offer a capability
//!   (`CapabilityNegotiation`). Decided without IO; never a provider call.
//! - [`BackendError`] — *operating* an already-constructed backend failed (chat completion,
//!   streaming, prepare/send/parse, rendering, model catalog, balance). Runtime execution only.

use std::{error::Error as StdError, fmt};

use thiserror::Error;

/// Boxed provider error source carried by [`BackendError::Provider`] and
/// [`BackendConstructError::Provider`].
///
/// Kept as `BoxError` rather than a concrete `ProviderError` so that custom backends wrapping an
/// arbitrary provider SDK (not necessarily one built on `just-common`) can carry their own error
/// type. Callers that need the structured `ProviderError` produced by the built-in backends
/// downcast it (as the tests do).
pub type BoxError = Box<dyn StdError + Send + Sync>;

/// Capability names used in client-level unsupported or unavailable errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
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

/// Constructing a backend failed: a precondition or setup failure.
///
/// Produced only by [`LlmBackend::new`](crate::LlmBackend::new) and
/// [`BackendFactory::create`](crate::BackendFactory::create). Distinct from [`BackendError`]
/// (operating a backend) and [`CapabilityError`] (static capability gating): construction never
/// performs a chat call, so its failure model is isolated.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BackendConstructError {
    /// No constructor is registered for the requested family (factory dispatch).
    #[error("no backend registered for family '{family}'")]
    UnknownFamily {
        /// The family string that had no registered constructor.
        family: String,
    },

    /// The provider client could not be built.
    ///
    /// Carries the backend family for attribution and the provider build failure as a boxed
    /// source, so callers can downcast to inspect `TransportError::InvalidConfig` /
    /// `TransportError::BuildClient` for the built-in backends, while custom backends may box any
    /// SDK-specific error.
    #[error("failed to build {family} backend: {source}")]
    Provider {
        /// Backend family that failed to build.
        family: &'static str,
        /// Provider-specific source error.
        #[source]
        source: BoxError,
    },
}

impl BackendConstructError {
    /// Creates an unknown-family error (factory dispatch miss).
    pub fn unknown_family(family: impl Into<String>) -> Self {
        Self::UnknownFamily {
            family: family.into(),
        }
    }

    /// Wraps a provider build failure for the given backend family.
    pub fn provider<E>(family: &'static str, source: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self::Provider {
            family,
            source: Box::new(source),
        }
    }
}

/// Static capability gating failure, returned by
/// [`CapabilityNegotiation`](crate::CapabilityNegotiation).
///
/// Decided without IO: it is a fact about whether a backend *offers* a capability, not whether a
/// live call to it failed. The latter is [`BackendError`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CapabilityError {
    /// The backend never offers the requested capability.
    #[error("{family} does not support {capability}")]
    Unsupported {
        /// Backend family.
        family: &'static str,
        /// Capability that was requested.
        capability: Capability,
    },

    /// The backend is expected to offer the capability but has not implemented it yet.
    #[error("{family} has not implemented {capability}")]
    Unimplemented {
        /// Backend family.
        family: &'static str,
        /// Capability that was expected.
        capability: Capability,
    },

    /// The backend can offer the capability in principle, but not in the current state.
    #[error("{family} cannot currently provide {capability}: {message}")]
    Unavailable {
        /// Backend family.
        family: &'static str,
        /// Capability that is temporarily unavailable.
        capability: Capability,
        /// Additional explanation from the backend adapter.
        message: String,
    },
}

impl CapabilityError {
    /// Creates an unsupported-capability error for the given backend.
    pub fn unsupported(family: &'static str, capability: Capability) -> Self {
        Self::Unsupported { family, capability }
    }

    /// Creates an unimplemented-capability error for the given backend.
    pub fn unimplemented(family: &'static str, capability: Capability) -> Self {
        Self::Unimplemented { family, capability }
    }

    /// Creates an unavailable-capability error for the given backend.
    pub fn unavailable(
        family: &'static str,
        capability: Capability,
        message: impl Into<String>,
    ) -> Self {
        Self::Unavailable {
            family,
            capability,
            message: message.into(),
        }
    }
}

/// Operating an already-constructed backend failed: a runtime execution failure.
///
/// Returned by chat completion, streaming, prepare/send/parse, rendering, model catalog, and
/// balance calls — everything that drives a live backend. Construction failures are
/// [`BackendConstructError`]; static capability gating is [`CapabilityError`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BackendError {
    /// The request was invalid before it reached the provider.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// Failed to serialize a request payload or render provider-specific types.
    #[error("serialization error: {source}")]
    Serialization {
        /// Underlying `serde_json` serialization failure.
        #[source]
        source: serde_json::Error,
    },

    /// The underlying provider SDK returned an error.
    ///
    /// Carries the backend family for attribution and the provider error as a boxed source (see
    /// [`BoxError`] for why this is not a concrete `ProviderError`).
    #[error("{family} backend error: {source}")]
    Provider {
        /// Backend family.
        family: &'static str,
        /// Provider-specific source error.
        #[source]
        source: BoxError,
    },
}

impl BackendError {
    /// Creates an invalid-request error with a stable, user-facing message.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest(message.into())
    }

    /// Creates a serialization error wrapping a `serde_json` failure.
    pub fn serialization(source: serde_json::Error) -> Self {
        Self::Serialization { source }
    }

    /// Wraps a provider-specific source error for the given backend family.
    pub fn provider<E>(family: &'static str, source: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self::Provider {
            family,
            source: Box::new(source),
        }
    }
}
