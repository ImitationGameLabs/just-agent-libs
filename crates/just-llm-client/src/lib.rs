//! Provider-neutral LLM client abstractions built on top of the provider type crates in this
//! workspace.
//!
//! # Architecture
//!
//! Two layers work together:
//!
//! - **Backend adapters** ([`crate::provider::DeepSeekBackend`], [`crate::provider::OpenAiCompatBackend`]): fully-constructed
//!   LLM adapters that hold a `reqwest::Client` and base URL directly, expose always-on operations,
//!   and negotiate optional capabilities such as model catalogs or balance inspection. Construct these
//!   directly when you know which provider you want at compile time.
//!
//! - **Programmatic provider entries** ([`DeepSeekProvider`], [`OpenAiCompatProvider`]): hold
//!   application-supplied provider configuration (entry id, API key, base URL) and produce a
//!   shared backend on demand via [`ProviderEntry::connect`]. Used by [`crate::ProviderRegistry`] to
//!   lazily create [`ChatClient`] values with per-call defaults such as model and system prompt.
//!
//! The split is required because the registry stores configured provider entries, but LLM
//! operations go through shared [`crate::LlmBackend`] trait objects. [`ProviderEntry::connect`]
//! materializes the shared backend the first time a registry entry is used, and later
//! [`ChatClient`] values reuse that backend while carrying their own request defaults.
//!
//! # Prepare-send pattern
//!
//! [`LlmBackend`] exposes a prepare-send pattern that gives callers full access to the HTTP
//! response, including headers like `retry-after` and `x-ratelimit-*`:
//!
//! ```ignore
//! let builder = backend.prepare(request)?;
//! let response = backend.send(builder).await?;
//! let retry_after = response.headers().get("retry-after");
//! ```
//!
//! # Capability traits
//!
//! Every backend adapter implements [`LlmBackend`], which provides `prepare`/`send`/`chat_completion`
//! and their streaming counterparts with concrete types (no associated types). [`ChatClient`]
//! derefs to `dyn LlmBackend` so all methods are available without importing the trait explicitly.
//! Optional operations are negotiated through [`CapabilityNegotiation`] before use so
//! `UnsupportedCapability` is reported at the negotiation boundary instead of from the
//! capability method itself.
//!
//! # DTO layering
//!
//! Some request and response types across provider type crates and this crate are intentionally
//! similar. That repetition is a trade-off so provider crates can evolve as independent
//! wire-level units — merging the code would not automatically make that boundary easier
//! to maintain.
//!
//! The shared types in this crate aim to be conservative abstractions. When a provider
//! cannot supply a piece of data faithfully, the normalized type should represent that
//! absence explicitly instead of fabricating certainty.
#![warn(missing_docs)]

/// Capability traits exposed by the LLM client layer.
pub mod capability;
/// Unified chat client and programmatic provider registry.
pub mod client;
/// LLM client error taxonomy.
pub mod error;
/// Provider adapters and runtime provider selection.
pub mod provider;
/// Local tool runtime helpers.
pub mod tools;
/// Shared normalized value types.
pub mod types;

pub use capability::{
    Balance, CapabilityNegotiation, ChatCompletionStream, Identifiable, ModelCatalog,
};
pub use error::{Capability, LlmError};
pub use just_common::error::{ProviderError, TransportError};
pub use just_common::transport::http::{build_client, endpoint_url, ensure_success, parse_json};
pub use just_common::transport::sse::JsonEventStream;
pub use provider::LlmBackend;
pub use provider::validation::{
    into_validated_streaming_request, validate_common_request, validate_non_streaming_request,
};
pub use tools::{LlmTool, ToolCallError, ToolDispatcher, ToolRegistrationError};

#[cfg(feature = "deepseek")]
pub use client::DeepSeekProvider;
#[cfg(feature = "openai-compat")]
pub use client::OpenAiCompatProvider;
pub use client::{ChatClient, ChatClientOptions, ProviderEntry, ProviderRegistry};
