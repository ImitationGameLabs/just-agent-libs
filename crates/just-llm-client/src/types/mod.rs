//! Shared LLM client-level types grouped by concern.

/// Normalized balance and quota types.
pub mod balance;
/// Normalized chat-completion request, response, and tool-calling types.
pub mod chat;
/// Best-effort context metrics request and response types.
pub mod context;
/// Normalized model catalog types.
pub mod model;
/// Backend-bound prepared request types.
pub mod prepared;
/// Token estimation types.
pub mod token;
