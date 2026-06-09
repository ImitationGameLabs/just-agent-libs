//! DeepSeek provider SDK.
//!
//! This crate exposes a thin Rust client over the DeepSeek HTTP API. Public request and
//! response types under [`types`] are wire-level DTOs that closely mirror the upstream
//! protocol shape so callers can reason about provider-specific fields directly.
#![warn(missing_docs)]

mod client;
mod error;
mod stream;
pub mod types;

pub use client::DeepSeekClient;
pub use client::DeepSeekClientBuilder;
pub use error::Error;
pub use just_common as common;
pub use just_common::prepared::PreparedChatRequest;
pub use stream::ChatCompletionStream;
