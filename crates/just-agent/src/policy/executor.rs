//! Authorized tool executor with approval provider.

use std::collections::BTreeSet;

use just_llm_client::{ToolCallError, ToolDispatcher, types::chat::ToolDefinition};
use serde_json::{Value, json};

use super::{ApprovalDecision, ApprovalProvider, ToolDecision};

/// Executes tools behind a policy and an approval provider.
pub struct AuthorizedToolExecutor {
    dispatch: ToolDispatcher,
    policy: super::AgentPolicy,
    approval: ApprovalController,
}

impl AuthorizedToolExecutor {
    pub fn new(
        dispatch: ToolDispatcher,
        policy: super::AgentPolicy,
        provider: Box<dyn ApprovalProvider>,
    ) -> Self {
        Self { dispatch, policy, approval: ApprovalController::new(provider) }
    }

    /// Replace the approval provider (used to switch to channel-based for TUI).
    pub fn set_approval_provider(&mut self, provider: Box<dyn ApprovalProvider>) {
        self.approval = ApprovalController::new(provider);
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
            ToolDecision::Ask { reason, dangerous } => {
                let decision = self
                    .approval
                    .approve(&key, tool_name, args_json, &reason, dangerous)
                    .await;
                match decision {
                    ApprovalDecision::Allow | ApprovalDecision::AlwaysAllow => {}
                    ApprovalDecision::Deny => {
                        return error_result(tool_name, format!("approval denied: {reason}"));
                    }
                }
            }
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
    provider: Box<dyn ApprovalProvider>,
    approved: BTreeSet<String>,
}

impl ApprovalController {
    fn new(provider: Box<dyn ApprovalProvider>) -> Self {
        Self { provider, approved: BTreeSet::new() }
    }

    async fn approve(
        &mut self,
        key: &str,
        tool_name: &str,
        args_json: &str,
        reason: &str,
        dangerous: bool,
    ) -> ApprovalDecision {
        if self.approved.contains(key) {
            return ApprovalDecision::Allow;
        }
        let decision = self
            .provider
            .request_approval(tool_name, args_json, reason, dangerous)
            .await;
        if matches!(decision, ApprovalDecision::AlwaysAllow) {
            self.approved.insert(key.to_owned());
        }
        decision
    }
}

fn approval_key(tool_name: &str, args_json: &str) -> String {
    format!("{tool_name}:{args_json}")
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
