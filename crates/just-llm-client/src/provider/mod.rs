#[cfg(feature = "deepseek")]
mod deepseek;
#[cfg(feature = "openai-compat")]
mod openai_compat;
/// Request validation helpers for building custom backends.
pub mod validation;

use crate::{CapabilityNegotiation, ChatCompletion, Identifiable, StreamingChatCompletion};

#[cfg(feature = "deepseek")]
pub use deepseek::DeepSeekBackend;
#[cfg(feature = "openai-compat")]
pub use openai_compat::OpenAiCompatBackend;

/// Convenience trait object for the runtime-selected LLM provider surface.
///
/// This intentionally stays smaller than the full extension-trait set exposed by the crate.
/// Runtime-selected providers optimize for the stable common path. Direct and prepared chat
/// execution for both non-streaming and streaming requests stay on the trait itself, while
/// optional capabilities are reached through
/// [`CapabilityNegotiation`].
pub trait LlmBackend:
    Identifiable + ChatCompletion + StreamingChatCompletion + CapabilityNegotiation
{
}

impl<T> LlmBackend for T where
    T: Identifiable + ChatCompletion + StreamingChatCompletion + CapabilityNegotiation
{
}
