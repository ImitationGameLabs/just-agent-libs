//! Shared internals for OpenAI-like provider crates in this workspace.
//!
//! This crate is intentionally small and focused on protocol-family concerns:
//!
//! - authenticated HTTP transport construction
//! - JSON request/response helpers
//! - SSE stream parsing for JSON chunk payloads
//!
//! It is meant to reduce drift between sibling provider crates without collapsing their public
//! APIs into one shared provider crate.

pub mod error;
pub mod transport;
