mod agent;
mod classifier;
mod executor;

/// Authorization decision for a tool invocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolDecision {
    Allow,
    Ask { reason: String },
    Deny { reason: String },
}

pub use agent::AgentPolicy;
pub use executor::AuthorizedToolExecutor;

/// Commands allowed in test mode for exercising shell tools.
///
/// These supplement the allowlist with commands that
/// change lightweight state (cwd, temp dirs) needed by the integration-test
/// example.  Deny-list checks are still enforced even in test mode.
#[allow(dead_code)]
pub(crate) const TEST_SAFE_COMMANDS: &[&str] = &[
    "cd",    // test session cwd changes
    "mkdir", // test directory creation
    "touch", // test file creation
    "rmdir", // cleanup empty dirs
    "sleep", // test timeout / background mode
];
