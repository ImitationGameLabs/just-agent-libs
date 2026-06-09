use thiserror::Error;

/// Registration errors emitted by [`ToolDispatcher`](super::ToolDispatcher).
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum ToolRegistrationError {
    /// Attempted to register two tools with the same name.
    #[error("duplicate tool name '{name}'")]
    DuplicateTool { name: String },
}

impl ToolRegistrationError {
    /// Creates a duplicate-tool registration error.
    pub fn duplicate_tool(name: impl Into<String>) -> Self {
        Self::DuplicateTool { name: name.into() }
    }
}

/// Call-time errors emitted by [`ToolDispatcher`](super::ToolDispatcher).
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum ToolCallError {
    /// Attempted to execute an unknown tool.
    #[error("unknown tool '{name}'. available tools: {available}")]
    UnknownTool { name: String, available: String },

    /// Registered tool execution failed abnormally.
    #[error("tool '{name}' execution failed: {source:#}")]
    Execution {
        name: String,
        #[source]
        source: anyhow::Error,
    },
}

impl ToolCallError {
    /// Creates an unknown-tool call error.
    pub fn unknown_tool(
        name: impl Into<String>,
        available: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Self {
        let available = available
            .into_iter()
            .map(|name| name.as_ref().to_owned())
            .collect::<Vec<_>>();
        let available = if available.is_empty() {
            "(none)".to_owned()
        } else {
            available.join(", ")
        };

        Self::UnknownTool {
            name: name.into(),
            available,
        }
    }

    /// Creates a tool-execution error.
    pub fn execution(name: impl Into<String>, source: anyhow::Error) -> Self {
        Self::Execution {
            name: name.into(),
            source,
        }
    }
}
