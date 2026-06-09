//! Shared internals for the just-agent ecosystem.
//!
//! This crate provides:
//!
//! - authenticated HTTP transport construction and JSON request/response helpers
//! - SSE stream parsing for JSON chunk payloads
//! - prepared-request types and validation
//!
//! It is meant to reduce drift between sibling provider crates without collapsing their public
//! APIs into one shared provider crate.

pub mod error;
pub mod prepared;
pub mod transport;
