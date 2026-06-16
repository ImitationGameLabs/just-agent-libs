//! Provider-neutral LLM client abstractions built on top of the provider type crates in this
//! workspace.
//!
//! # Architecture
//!
//! Two building blocks on top of the provider type crates:
//!
//! - **Backend adapters** ([`crate::provider::DeepSeekBackend`], [`crate::provider::OpenAiCompatBackend`]):
//!   fully-constructed LLM adapters that hold a `reqwest::Client` and base URL directly, expose
//!   always-on operations, and negotiate optional capabilities such as model catalogs or balance
//!   inspection. Every backend implements [`LlmBackend`], which also carries the uniform
//!   [`new`](LlmBackend::new) constructor and [`family`](LlmBackend::family) identifier used for
//!   construction and error attribution.
//!
//! - **[`BackendFactory`]**: a composable `family -> constructor` dispatch table. It maps a
//!   backend family string (see the [`family`] module) to that backend's
//!   [`LlmBackend::new`], building a fresh shared backend on each
//!   [`create`](BackendFactory::create). It holds no configuration and caches nothing — downstream
//!   composes any registry or sharing policy on top.
//!
//! [`ChatClient`] pairs per-call defaults (model, system prompt) with a shared [`LlmBackend`] and
//! derefs to `dyn LlmBackend`, so chat and capability methods are reachable directly. Construct a
//! backend via [`BackendFactory`] or [`LlmBackend::new`], then wrap it in a [`ChatClient`].
//!
//! # Prepare-send-parse pattern
//!
//! [`LlmBackend`] exposes a prepare-send-parse pattern that gives callers full access to the HTTP
//! response, including headers like `retry-after` and `x-ratelimit-*`, before deserializing:
//!
//! ```ignore
//! let prepared = backend.prepare(request)?;          // reqwest::Request (Send + Sync; try_clone-able)
//! let response = backend.send(prepared).await?;       // raw reqwest::Response, status unchecked
//! // Inspect status / headers (retry-after, x-ratelimit-*) here, or re-send a clone of `prepared` (via try_clone) for retry.
//! let retry_after = response.headers().get("retry-after");
//! let completion = backend.parse(response).await?;    // deserialized via dyn dispatch
//! ```
//!
//! # Capability traits
//!
//! Every backend adapter implements [`LlmBackend`], which provides the prepare/send/parse
//! primitives (with streaming variants) and the `chat_completion`/`stream_chat_completion`
//! convenience methods, all with concrete types (no associated types). [`ChatClient`]
//! derefs to `dyn LlmBackend` so all methods are available without importing the trait explicitly.
//! Optional operations are negotiated through [`CapabilityNegotiation`] before use so
//! [`CapabilityError::Unsupported`] is reported at the negotiation boundary instead of from the
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
/// Unified chat client and backend factory.
pub mod client;
/// LLM client error taxonomy.
pub mod error;
/// Built-in backend family identifiers.
pub mod family;
/// Provider adapters and runtime provider selection.
pub mod provider;
/// Local tool runtime helpers.
pub mod tools;
/// Shared normalized value types.
pub mod types;

pub use capability::{
    Balance, CapabilityNegotiation, ChatCompletionStream, Identifiable, ModelCatalog,
};
pub use error::{BackendConstructError, BackendError, Capability, CapabilityError};
pub use just_common::error::{ProviderError, TransportError};
pub use just_common::transport::http::{build_client, endpoint_url, ensure_success, parse_json};
pub use just_common::transport::sse::JsonEventStream;
pub use provider::LlmBackend;
pub use provider::validation::{
    into_validated_streaming_request, validate_common_request, validate_non_streaming_request,
};
pub use tools::{LlmTool, ToolCallError, ToolDispatcher, ToolRegistrationError};

pub use client::{BackendFactory, ChatClient, ChatClientOptions};
