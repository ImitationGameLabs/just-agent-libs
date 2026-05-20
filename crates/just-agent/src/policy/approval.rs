//! Approval provider trait and implementations for tool authorization.

use std::io::{self, IsTerminal, Write, stdout};

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};
use tracing::warn;

use super::ApprovalDecision;

/// A request for tool approval, sent from the executor to the UI.
pub struct ApprovalRequest {
    pub tool_name: String,
    pub args: String,
    pub reason: String,
    pub dangerous: bool,
    pub response_tx: oneshot::Sender<ApprovalDecision>,
}

/// Provider for tool approval decisions.
#[async_trait]
pub trait ApprovalProvider: Send + Sync {
    async fn request_approval(
        &mut self,
        tool_name: &str,
        args: &str,
        reason: &str,
        dangerous: bool,
    ) -> ApprovalDecision;
}

/// Stdin-based approval for non-TUI mode.
pub struct StdinApprovalProvider {
    interactive: bool,
    auto_approve: bool,
}

impl StdinApprovalProvider {
    pub fn new() -> Self {
        let interactive = io::stdin().is_terminal() && io::stderr().is_terminal();
        let auto_approve = env_truthy("JUST_AGENT_AUTO_APPROVE");
        if auto_approve {
            warn!("JUST_AGENT_AUTO_APPROVE enabled — all tool calls bypass approval");
        }
        Self { interactive, auto_approve }
    }
}

#[async_trait]
impl ApprovalProvider for StdinApprovalProvider {
    async fn request_approval(
        &mut self,
        tool_name: &str,
        args: &str,
        reason: &str,
        dangerous: bool,
    ) -> ApprovalDecision {
        if self.auto_approve {
            return ApprovalDecision::Allow;
        }
        if !self.interactive {
            return ApprovalDecision::Deny;
        }

        if dangerous {
            println!("[DANGER] tool: {tool_name}");
            println!("[DANGER] reason: {reason}");
        } else {
            println!("[approval] tool: {tool_name}");
            println!("[approval] reason: {reason}");
        }
        println!("[approval] args: {args}");
        if dangerous {
            print!("[DANGER] [1] Allow  [3] Deny: ");
        } else {
            print!("[approval] [1] Allow  [2] Always allow  [3] Deny: ");
        }
        stdout().flush().ok();

        let mut input = String::new();
        io::stdin().read_line(&mut input).ok();

        match input.trim() {
            "1" => ApprovalDecision::Allow,
            "2" => ApprovalDecision::AlwaysAllow,
            _ => ApprovalDecision::Deny,
        }
    }
}

/// Channel-based approval for TUI mode.
///
/// Sends an [`ApprovalRequest`] through a channel; the TUI renders a
/// widget and the user's choice is sent back via oneshot.
pub struct ChannelApprovalProvider {
    tx: mpsc::Sender<ApprovalRequest>,
}

impl ChannelApprovalProvider {
    pub fn new(tx: mpsc::Sender<ApprovalRequest>) -> Self {
        Self { tx }
    }
}

#[async_trait]
impl ApprovalProvider for ChannelApprovalProvider {
    async fn request_approval(
        &mut self,
        tool_name: &str,
        args: &str,
        reason: &str,
        dangerous: bool,
    ) -> ApprovalDecision {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.tx
            .send(ApprovalRequest {
                tool_name: tool_name.to_owned(),
                args: args.to_owned(),
                reason: reason.to_owned(),
                response_tx: resp_tx,
                dangerous,
            })
            .await
            .ok();
        resp_rx.await.unwrap_or(ApprovalDecision::Deny)
    }
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
