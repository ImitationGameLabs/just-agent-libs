use std::{collections::HashMap, sync::Arc};

use crate::{error::LlmError, provider::LlmBackend};

/// Function pointer that builds a shared backend from raw inputs.
type BackendBuilder =
    fn(reqwest::ClientBuilder, &str, Option<&str>) -> Result<Arc<dyn LlmBackend>, LlmError>;

/// Dispatch table from backend family to its constructor function.
///
/// A composable primitive: `family -> constructor`, with no held configuration and no caching.
/// Each [`create`](Self::create) call builds a fresh shared backend; downstream that wants
/// sharing caches the returned [`Arc`] or clones the [`crate::ChatClient`] built from it.
///
/// Constructors and family names come from the [`LlmBackend`] trait itself
/// ([`LlmBackend::new`] / [`LlmBackend::family`]) — every backend type carries them. The common
/// entry point is [`new`](Self::new), which pre-seeds every compiled-in built-in backend; use
/// [`empty`](Self::empty) for full control over registration.
pub struct BackendFactory {
    builders: HashMap<&'static str, BackendBuilder>,
}

impl BackendFactory {
    /// A factory pre-seeded with every compiled-in built-in backend.
    ///
    /// The common path: under default features both the DeepSeek and OpenAI-compatible backends
    /// are registered automatically. With no backend features enabled this yields an empty factory
    /// (equivalent to [`empty`](Self::empty)).
    pub fn new() -> Self {
        #[cfg(any(feature = "deepseek", feature = "openai-compat"))]
        {
            let mut factory = Self::empty();
            #[cfg(feature = "deepseek")]
            factory.register::<crate::provider::DeepSeekBackend>();
            #[cfg(feature = "openai-compat")]
            factory.register::<crate::provider::OpenAiCompatBackend>();
            factory
        }
        #[cfg(not(any(feature = "deepseek", feature = "openai-compat")))]
        {
            Self::empty()
        }
    }

    /// An empty factory; the caller registers backends explicitly via [`register`](Self::register).
    pub fn empty() -> Self {
        Self {
            builders: HashMap::new(),
        }
    }

    /// Register (or replace) a backend, keyed on its [`LlmBackend::family`].
    ///
    /// Captures [`LlmBackend::new`] as the constructor. Takes only a type parameter, so the call
    /// requires turbofish: `factory.register::<DeepSeekBackend>()`. Registering a family that is
    /// already registered replaces the previous constructor.
    pub fn register<C: LlmBackend>(&mut self) -> &mut Self {
        // Fully qualified: `family` exists on both `LlmBackend` (static) and `Identifiable`
        // (instance); only the static one is callable without a receiver, but Rust still requires
        // disambiguation here.
        self.builders.insert(<C as LlmBackend>::family(), C::new);
        self
    }

    /// Build a shared backend for `family` from raw inputs.
    ///
    /// Returns [`LlmError::invalid_request`](crate::LlmError::invalid_request) when no constructor
    /// is registered for `family`.
    pub fn create(
        &self,
        family: &str,
        http: reqwest::ClientBuilder,
        api_key: &str,
        base_url: Option<&str>,
    ) -> Result<Arc<dyn LlmBackend>, LlmError> {
        let build = self
            .builders
            .get(family)
            .copied()
            .ok_or_else(|| LlmError::invalid_request(format!("unknown family: {family}")))?;
        build(http, api_key, base_url)
    }

    /// The family names of every registered constructor.
    pub fn families(&self) -> impl Iterator<Item = &str> {
        self.builders.keys().copied()
    }
}

impl Default for BackendFactory {
    fn default() -> Self {
        Self::new()
    }
}
