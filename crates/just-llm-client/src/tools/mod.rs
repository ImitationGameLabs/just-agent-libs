//! Optional local tool runtime helpers and reusable tool implementations.
//!
//! The core `just-llm-client` crate only normalizes tool-call request and response DTOs.
//! Enabling the `tools` feature adds a small application-side runtime for composing local
//! executable tools, converting them into [`ToolDefinition`](crate::types::chat::ToolDefinition)
//! values, and dispatching model-emitted tool calls by name.

mod dispatch;
mod error;
mod llm_tool;
mod renamed_tool;

pub mod shell;

pub use dispatch::ToolDispatcher;
pub use error::{ToolCallError, ToolRegistrationError};
pub use llm_tool::LlmTool;
pub use renamed_tool::RenamedTool;
