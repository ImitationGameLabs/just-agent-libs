//! Agent policy: per-tool authorization decisions.

use std::path::{Path, PathBuf};

use anyhow::Result;

use just_llm_client::tools::shell::{CreateArgs, ExecArgs, KillArgs, RestartArgs};

use super::ToolDecision;
use super::classifier;

/// A minimal policy layer inspired by Codex-style pre-execution gating.
#[derive(Clone, Debug)]
pub struct AgentPolicy {
    workspace_root: PathBuf,
    test_mode: bool,
}

impl AgentPolicy {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self { workspace_root, test_mode: false }
    }

    /// Create a policy for test / non-interactive use.
    ///
    /// In test mode the deny list is still enforced but every `Ask` decision
    /// is promoted to `Allow` so tool calls proceed without interactive
    /// approval prompts.
    #[allow(dead_code)]
    pub fn new_test(workspace_root: PathBuf) -> Self {
        Self { workspace_root, test_mode: true }
    }

    pub fn evaluate(&self, tool_name: &str, args_json: &str) -> Result<ToolDecision> {
        match tool_name {
            "shell_session_list" | "shell_session_capture" => Ok(ToolDecision::Allow),
            "shell_session_create" => self.evaluate_session_create(args_json),
            "shell_session_restart" => self.evaluate_session_restart(args_json),
            "shell_session_kill" => self.evaluate_session_kill(args_json),
            "shell_session_exec" => self.evaluate_session_exec(args_json),
            _ => Ok(ToolDecision::Deny {
                reason: format!("tool '{tool_name}' is not allowed by the current policy"),
            }),
        }
    }

    fn evaluate_session_create(&self, args_json: &str) -> Result<ToolDecision> {
        let args: CreateArgs = serde_json::from_str(args_json)?;
        let cwd = resolve_requested_path(args.cwd.as_deref(), &self.workspace_root);

        if !cwd.starts_with(&self.workspace_root) {
            return Ok(ToolDecision::Deny {
                reason: format!(
                    "session cwd {} is outside the workspace root {}",
                    cwd.display(),
                    self.workspace_root.display()
                ),
            });
        }

        Ok(ToolDecision::Allow)
    }

    fn evaluate_session_restart(&self, args_json: &str) -> Result<ToolDecision> {
        let args: RestartArgs = serde_json::from_str(args_json)?;
        Ok(ToolDecision::Ask {
            reason: format!(
                "restarting shell session '{}' discards its current state",
                args.name
            ),
        })
    }

    fn evaluate_session_kill(&self, args_json: &str) -> Result<ToolDecision> {
        let args: KillArgs = serde_json::from_str(args_json)?;
        if args.name == "main" {
            return Ok(ToolDecision::Deny {
                reason: "killing the main session is not allowed".into(),
            });
        }

        Ok(ToolDecision::Ask {
            reason: format!(
                "killing shell session '{}' terminates running processes",
                args.name
            ),
        })
    }

    fn evaluate_session_exec(&self, args_json: &str) -> Result<ToolDecision> {
        let args: ExecArgs = serde_json::from_str(args_json)?;
        let decision = classifier::classify_command(&args.command);

        if self.test_mode {
            match decision {
                ToolDecision::Deny { .. } => Ok(decision),
                _ => Ok(ToolDecision::Allow),
            }
        } else {
            Ok(decision)
        }
    }
}

fn resolve_requested_path(path: Option<&Path>, workspace_root: &Path) -> PathBuf {
    match path {
        Some(path) if path.is_absolute() => path.to_path_buf(),
        Some(path) => workspace_root.join(path),
        None => workspace_root.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denies_killing_main_session() {
        let policy = AgentPolicy::new(PathBuf::from("/tmp/workspace"));
        let decision = policy
            .evaluate("shell_session_kill", r#"{"name":"main"}"#)
            .unwrap();
        assert!(matches!(decision, ToolDecision::Deny { .. }));
    }

    #[test]
    fn denies_sessions_outside_workspace() {
        let policy = AgentPolicy::new(PathBuf::from("/tmp/workspace"));
        let decision = policy
            .evaluate("shell_session_create", r#"{"name":"tmp","cwd":"/etc"}"#)
            .unwrap();
        assert!(matches!(decision, ToolDecision::Deny { .. }));
    }
}
