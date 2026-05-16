//! Authorized tool executor with approval caching.

use std::{
    collections::BTreeSet,
    io::{self, IsTerminal, Write},
};

use anyhow::{Context, Result};
use serde_json::{Value, json};

use just_llm_client::{ToolCallError, ToolDispatcher, types::chat::ToolDefinition};

use super::ToolDecision;

/// Executes tools behind a policy and an approval cache.
pub struct AuthorizedToolExecutor {
    dispatch: ToolDispatcher,
    policy: super::AgentPolicy,
    approval: ApprovalController,
}

impl AuthorizedToolExecutor {
    pub fn new(dispatch: ToolDispatcher, policy: super::AgentPolicy) -> Self {
        Self { dispatch, policy, approval: ApprovalController::new() }
    }

    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.dispatch.tool_definitions()
    }

    pub async fn execute(&mut self, tool_name: &str, args_json: &str) -> String {
        let key = approval_key(tool_name, args_json);

        let decision = match self.policy.evaluate(tool_name, args_json) {
            Ok(decision) => decision,
            Err(error) => {
                return error_result(tool_name, format!("policy evaluation failed: {error:#}"));
            }
        };

        match decision {
            ToolDecision::Allow => {}
            ToolDecision::Deny { reason } => {
                return error_result(tool_name, format!("tool denied: {reason}"));
            }
            ToolDecision::Ask { reason } => match self
                .approval
                .approve(&key, tool_name, args_json, &reason)
            {
                Ok(true) => {}
                Ok(false) => return error_result(tool_name, format!("approval denied: {reason}")),
                Err(error) => {
                    return error_result(tool_name, format!("approval failed: {error:#}"));
                }
            },
        }

        match self.dispatch.call_tool(tool_name, args_json).await {
            Ok(output) => success_result(tool_name, output),
            Err(error) => match error {
                ToolCallError::UnknownTool { .. } => error_result(tool_name, error.to_string()),
                ToolCallError::Execution { .. } => error_result(tool_name, error.to_string()),
            },
        }
    }
}

struct ApprovalController {
    approved: BTreeSet<String>,
    interactive: bool,
    auto_approve: bool,
}

impl ApprovalController {
    fn new() -> Self {
        let result = Self {
            approved: BTreeSet::new(),
            interactive: io::stdin().is_terminal() && io::stderr().is_terminal(),
            auto_approve: env_truthy("JUST_AGENT_AUTO_APPROVE"),
        };
        if result.auto_approve {
            eprintln!(
                "[warning] JUST_AGENT_AUTO_APPROVE is enabled — all tool calls bypass approval"
            );
        }
        result
    }

    fn approve(
        &mut self,
        key: &str,
        tool_name: &str,
        args_json: &str,
        reason: &str,
    ) -> Result<bool> {
        if self.auto_approve || self.approved.contains(key) {
            return Ok(true);
        }

        if !self.interactive {
            return Ok(false);
        }

        eprintln!("[approval] tool: {tool_name}");
        eprintln!("[approval] reason: {reason}");
        eprintln!("[approval] args: {args_json}");
        eprint!("[approval] allow once? [y/N/a(always)]: ");
        io::stderr()
            .flush()
            .context("failed to flush approval prompt")?;

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .context("failed to read approval response")?;

        match input.trim().to_ascii_lowercase().as_str() {
            "y" | "yes" => Ok(true),
            "a" | "always" => {
                self.approved.insert(key.to_owned());
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}

fn approval_key(tool_name: &str, args_json: &str) -> String {
    format!("{tool_name}:{args_json}")
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn success_result(tool_name: &str, output: String) -> String {
    let parsed = serde_json::from_str::<Value>(&output).unwrap_or(Value::String(output));
    json!({
        "ok": true,
        "tool_name": tool_name,
        "result": parsed,
    })
    .to_string()
}

fn error_result(tool_name: &str, error: String) -> String {
    json!({
        "ok": false,
        "tool_name": tool_name,
        "error": error,
    })
    .to_string()
}
