//! Built-in backend family identifiers — the single source for family strings.
//!
//! Every backend is identified by its [`family`](crate::Identifiable::family): the stable
//! string used for error attribution, [`BackendFactory`](crate::BackendFactory) lookup, and
//! diagnostics. The constants here are the canonical values for the built-in backends, kept
//! in one place so application code references a named constant instead of a magic string.
//! Backends and the factory both read these constants, so the literal never drifts.

/// DeepSeek backend family.
///
/// Matches [`Identifiable::family`](crate::Identifiable::family) for the DeepSeek backend and
/// the key under which a [`BackendFactory`](crate::BackendFactory) registers/looks it up.
pub const DEEPSEEK: &str = "deepseek";

/// OpenAI-compatible backend family.
///
/// Matches [`Identifiable::family`](crate::Identifiable::family) for the OpenAI-compatible
/// backend and the key under which a [`BackendFactory`](crate::BackendFactory) registers/looks
/// it up.
pub const OPENAI_COMPATIBLE: &str = "openai-compatible";
