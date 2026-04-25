//! OpenAI-compatible provider SDK.
//!
//! This crate exposes a thin Rust client for services that implement an OpenAI-like chat
//! completion and model-listing surface. Public request and response types under [`types`]
//! are wire-level DTOs; compatibility depends on the target service actually supporting the
//! documented fields.
#![warn(missing_docs)]

mod client;
mod config;
mod error;
mod stream;
pub mod types;

pub use client::OpenAiCompatClient;
pub use config::OpenAiCompatConfig;
pub use error::Error;
pub use just_common as common;
pub use stream::ChatCompletionStream;
