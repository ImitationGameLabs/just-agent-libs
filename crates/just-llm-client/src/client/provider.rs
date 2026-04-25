use std::sync::Arc;

use crate::error::LlmError;
use crate::provider::LlmBackend;

/// Programmatically registered, fully configured provider entry.
///
/// The registry looks providers up by [`ProviderEntry::id`], reuses the backend family identifier
/// exposed by [`ProviderEntry::provider`], and lazily connects a shared backend the first time a
/// [`crate::ChatClient`] is requested for a given entry.
pub trait ProviderEntry: Send + Sync + 'static {
    /// Stable lookup identifier for this configured provider entry.
    fn id(&self) -> &str;

    /// Provider family identifier used for backend selection and diagnostics.
    fn provider(&self) -> &str;

    /// Connect a shared backend using the provider's programmatically supplied configuration.
    fn connect(&self) -> Result<Arc<dyn LlmBackend>, LlmError>;
}
