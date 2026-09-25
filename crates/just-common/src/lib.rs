//! Shared internals for the just-agent ecosystem.
//!
//! This crate provides:
//!
//! - authenticated HTTP client construction and endpoint URL helpers
//! - HTTP response status checking
//! - SSE stream parsing for JSON chunk payloads
//!
//! It is meant to reduce drift between sibling provider crates without collapsing their public
//! APIs into one shared provider crate.

pub mod api_key;
pub mod error;
pub mod transport;

pub use api_key::{ApiKey, InvalidApiKey};
pub use error::{ProviderError, captured_body};
