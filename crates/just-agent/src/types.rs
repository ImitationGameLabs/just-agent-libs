//! Shared types used across the agent crate.
/// Events sent from the agent task to the UI.
#[derive(Debug)]
pub enum AgentEvent {
    Reasoning(String),
    AssistantContent(String),
    ToolCall {
        name: String,
        args: String,
    },
    ToolResult(String),
    Finished(String),
    MaxRoundsExceeded,
    Error(String),
    Status(String),
    Busy,
}

/// Outcome of running the agent round loop.
pub enum AgentOutcome {
    Finished { content: String },
    MaxRoundsExceeded,
}
