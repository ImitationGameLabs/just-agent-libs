//! Local tool runtime helpers.
//!
//! Application-side runtime for composing local
//! executable tools, converting them into [`ToolDefinition`](crate::types::chat::ToolDefinition)
//! values, and dispatching model-emitted tool calls by name.

mod dispatch;
mod error;
mod llm_tool;
mod renamed_tool;

pub use dispatch::ToolDispatcher;
pub use error::{ToolCallError, ToolRegistrationError};
pub use llm_tool::LlmTool;
pub use renamed_tool::RenamedTool;
