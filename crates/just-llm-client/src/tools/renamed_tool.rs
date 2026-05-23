use async_trait::async_trait;
use serde_json::Value;

use crate::tools::LlmTool;

/// Adapter that wraps an [`LlmTool`] with an overridden name and optional description.
///
/// Useful when composing tool sets from shared implementations but exposing them under
/// application-specific names. For example:
///
/// ```rust,ignore
/// use just_llm_client::tools::{LlmTool, RenamedTool};
///
/// let adapted = RenamedTool::new(
///     original_tool,
///     "my_tool",
///     Some("A custom description."),
/// );
/// ```
///
/// Everything except [`name()`](LlmTool::name) and
/// [`description()`](LlmTool::description) delegates to the inner tool.
pub struct RenamedTool {
    inner: Box<dyn LlmTool>,
    name: String,
    description: Option<String>,
}

impl RenamedTool {
    /// Creates a renamed adapter.
    ///
    /// - `inner` — the underlying tool implementation.
    /// - `name` — the tool name exposed to the model.
    /// - `description` — optional override. When `None`, the inner tool's description is used.
    pub fn new(inner: Box<dyn LlmTool>, name: &str, description: Option<&str>) -> Self {
        Self { inner, name: name.to_owned(), description: description.map(str::to_owned) }
    }
}

#[async_trait]
impl LlmTool for RenamedTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        self.description
            .as_deref()
            .unwrap_or_else(|| self.inner.description())
    }

    fn parameters_schema(&self) -> Value {
        self.inner.parameters_schema()
    }

    async fn call(&self, args_json: &str) -> anyhow::Result<String> {
        self.inner.call(args_json).await
    }
}
