use std::sync::Arc;

use crate::error::LlmError;
use crate::provider::LlmBackend;

/// Programmatically registered, fully configured provider entry.
///
/// The registry looks providers up by [`ProviderEntry::instance_id`] and lazily connects a shared backend
/// the first time a [`crate::ChatClient`] is requested for a given entry. Each entry declares its target
/// [`family`](Self::family) — the backend family [`connect`](Self::connect) will produce — used to attribute
/// errors before the backend exists and matching the [`Identifiable::family`](crate::Identifiable::family)
/// the connected backend reports.
pub trait ProviderEntry: Send + Sync + 'static {
    /// Stable instance identifier for this configured provider entry (the registry lookup key).
    fn instance_id(&self) -> &str;

    /// The backend family this entry's [`connect`](Self::connect) will produce (e.g. `"deepseek"`,
    /// `"openai-compatible"`).
    ///
    /// This is the entry's *target* family, used to attribute errors in `connect` — which runs before
    /// the backend exists. The *runtime* family lives on the connected backend via
    /// [`Identifiable::family`](crate::Identifiable::family); the two must match (the built-in entries
    /// guarantee this by returning the backend's `FAMILY` const).
    fn family(&self) -> &'static str;

    /// Connect a shared backend using the provider's programmatically supplied configuration.
    fn connect(&self) -> Result<Arc<dyn LlmBackend>, LlmError>;
}
