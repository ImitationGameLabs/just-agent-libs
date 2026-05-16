use async_trait::async_trait;
use serde_json::Value;

use crate::types::chat::{FunctionDefinition, ToolDefinition, ToolType};

/// Object-safe application-side tool contract for local function calling.
#[async_trait]
pub trait LlmTool: Send + Sync {
    /// Returns the tool name exposed to the model.
    fn name(&self) -> &str;

    /// Returns the human-readable tool description exposed to the model.
    fn description(&self) -> &str;

    /// Returns the JSON Schema describing accepted arguments.
    fn parameters_schema(&self) -> Value;

    /// Converts this runtime tool into a normalized tool definition.
    #[must_use]
    fn to_tool_definition(&self) -> ToolDefinition {
        ToolDefinition {
            kind: ToolType::Function,
            function: FunctionDefinition {
                name: self.name().to_owned(),
                description: Some(self.description().to_owned()),
                parameters: Some(self.parameters_schema()),
                strict: None,
            },
        }
    }

    /// Executes the tool with JSON-serialized arguments and returns a JSON-serializable result.
    ///
    /// Return `Ok(...)` for normal business outcomes, even if the tool-level operation itself
    /// reports failure in-band. Return `Err(...)` only for abnormal runtime failures such as
    /// argument deserialization bugs, transport failures, or unexpected backend errors.
    async fn call(&self, args_json: &str) -> anyhow::Result<String>;
}
