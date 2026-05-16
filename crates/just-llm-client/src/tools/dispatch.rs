use std::{collections::BTreeMap, sync::Arc};

use crate::types::chat::ToolDefinition;

use super::{LlmTool, ToolCallError, ToolRegistrationError};

/// Registry and dispatcher for locally executable tools.
#[derive(Default)]
pub struct ToolDispatcher {
    tools: BTreeMap<String, Arc<dyn LlmTool>>,
}

impl ToolDispatcher {
    /// Creates an empty tool registry.
    pub fn new() -> Self {
        Self { tools: BTreeMap::new() }
    }

    /// Returns the number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Returns whether no tools have been registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Registers a single tool.
    pub fn add_tool(&mut self, tool: Box<dyn LlmTool>) -> Result<(), ToolRegistrationError> {
        let name = tool.name().to_owned();
        if self.tools.contains_key(&name) {
            return Err(ToolRegistrationError::duplicate_tool(name));
        }
        self.tools.insert(name, Arc::from(tool));
        Ok(())
    }

    /// Registers multiple tools.
    pub fn add_tools(&mut self, tools: Vec<Box<dyn LlmTool>>) -> Result<(), ToolRegistrationError> {
        for tool in tools {
            self.add_tool(tool)?;
        }
        Ok(())
    }

    /// Returns the registered tool names in deterministic order.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(String::as_str).collect()
    }

    /// Calls a registered tool by name with JSON-serialized arguments.
    pub async fn call_tool(&self, name: &str, args_json: &str) -> Result<String, ToolCallError> {
        let tool = self.tools.get(name).ok_or_else(|| {
            ToolCallError::unknown_tool(name, self.tools.keys().map(String::as_str))
        })?;

        tool.call(args_json)
            .await
            .map_err(|source| ToolCallError::execution(name, source))
    }

    /// Converts all registered tools into normalized tool definitions.
    #[must_use]
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|tool| tool.to_tool_definition())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use serde_json::{Value, json};

    use super::*;

    struct EchoTool;

    #[async_trait]
    impl LlmTool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echo the provided message."
        }

        fn parameters_schema(&self) -> Value {
            json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string" }
                },
                "required": ["message"]
            })
        }

        async fn call(&self, args_json: &str) -> anyhow::Result<String> {
            let args: Value = serde_json::from_str(args_json)?;
            Ok(args["message"].to_string())
        }
    }

    struct FailingTool;

    #[async_trait]
    impl LlmTool for FailingTool {
        fn name(&self) -> &str {
            "fail"
        }

        fn description(&self) -> &str {
            "Always fails."
        }

        fn parameters_schema(&self) -> Value {
            json!({ "type": "object", "properties": {}, "required": [] })
        }

        async fn call(&self, _args_json: &str) -> anyhow::Result<String> {
            Err(anyhow::anyhow!("boom"))
        }
    }

    #[tokio::test]
    async fn dispatch_routes_registered_tools() {
        let mut dispatch = ToolDispatcher::new();
        dispatch.add_tool(Box::new(EchoTool)).unwrap();

        let result = dispatch
            .call_tool("echo", r#"{"message":"hello"}"#)
            .await
            .unwrap();

        assert_eq!(result, "\"hello\"");
    }

    #[test]
    fn rejects_duplicate_tool_names() {
        let mut dispatch = ToolDispatcher::new();
        dispatch.add_tool(Box::new(EchoTool)).unwrap();

        let error = dispatch.add_tool(Box::new(EchoTool)).unwrap_err();

        assert!(matches!(error, ToolRegistrationError::DuplicateTool { .. }));
    }

    #[tokio::test]
    async fn unknown_tool_error_lists_available_names() {
        let mut dispatch = ToolDispatcher::new();
        dispatch.add_tool(Box::new(EchoTool)).unwrap();

        let error = dispatch.call_tool("missing", "{}").await.unwrap_err();

        match error {
            ToolCallError::UnknownTool { name, available } => {
                assert_eq!(name, "missing");
                assert_eq!(available, "echo");
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[tokio::test]
    async fn execution_errors_preserve_tool_name() {
        let mut dispatch = ToolDispatcher::new();
        dispatch.add_tool(Box::new(FailingTool)).unwrap();

        let error = dispatch.call_tool("fail", "{}").await.unwrap_err();

        assert!(matches!(error, ToolCallError::Execution { name, .. } if name == "fail"));
    }

    #[test]
    fn tool_definitions_follow_normalized_shape() {
        let mut dispatch = ToolDispatcher::new();
        dispatch.add_tool(Box::new(EchoTool)).unwrap();

        let definitions = dispatch.tool_definitions();

        assert_eq!(definitions.len(), 1);
        assert_eq!(definitions[0].function.name, "echo");
        assert_eq!(
            definitions[0].function.description.as_deref(),
            Some("Echo the provided message.")
        );
    }
}
