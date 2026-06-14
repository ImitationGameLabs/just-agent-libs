use std::sync::Arc;

use crate::error::LlmError;
use crate::provider::LlmBackend;

/// Programmatically registered, fully configured provider entry.
///
/// The registry looks providers up by [`ProviderEntry::instance_id`] and lazily connects a shared backend
/// the first time a [`crate::ChatClient`] is requested for a given entry. The connected backend
/// exposes its family identifier via [`crate::Identifiable::family`].
pub trait ProviderEntry: Send + Sync + 'static {
    /// Stable instance identifier for this configured provider entry (the registry lookup key).
    fn instance_id(&self) -> &str;

    /// Connect a shared backend using the provider's programmatically supplied configuration.
    fn connect(&self) -> Result<Arc<dyn LlmBackend>, LlmError>;
}
