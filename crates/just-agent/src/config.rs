use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use crate::args::Args;

const DEFAULT_SYSTEM_PROMPT: &str = "You are a minimal coding agent. Use shell_session_exec for shell commands. Use shell_session_create to create persistent shell sessions, shell_session_list to inspect them, shell_session_capture to inspect recent output, and shell_session_restart or shell_session_kill when session lifecycle control is necessary. Keep answers concise and prefer the least risky tool that can accomplish the task.";
const DEFAULT_MAX_TOOL_ROUNDS: usize = 32;
const DEFAULT_COMPACT_TRIGGER_TOKENS: usize = 12_000;
const DEFAULT_COMPACT_KEEP_RECENT_TOKENS: usize = 4_000;
const DEFAULT_COMPACT_MAX_TOKENS: u32 = 1_200;

/// Runtime configuration for `just-agent`.
#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub prompt: String,
    pub system_prompt: String,
    pub max_tool_rounds: usize,
    pub workspace_root: PathBuf,
    pub compact_trigger_tokens: usize,
    pub compact_keep_recent_tokens: usize,
    pub compact_max_tokens: u32,
}

impl AgentConfig {
    /// Loads configuration from CLI arguments and environment variables.
    pub fn load(args: &Args) -> Result<Self> {
        let prompt = args.prompt.clone();
        let system_prompt = std::env::var("JUST_AGENT_SYSTEM_PROMPT")
            .unwrap_or_else(|_| DEFAULT_SYSTEM_PROMPT.into());
        let max_tool_rounds = std::env::var("JUST_AGENT_MAX_TOOL_ROUNDS")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()
            .context("JUST_AGENT_MAX_TOOL_ROUNDS must be a positive integer")?
            .unwrap_or(DEFAULT_MAX_TOOL_ROUNDS);
        let workspace_root = std::env::var("JUST_AGENT_WORKSPACE_ROOT")
            .map(PathBuf::from)
            .unwrap_or(std::env::current_dir().context("failed to determine current directory")?);
        let compact_trigger_tokens = std::env::var("JUST_AGENT_COMPACT_TRIGGER_TOKENS")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()
            .context("JUST_AGENT_COMPACT_TRIGGER_TOKENS must be a positive integer")?
            .unwrap_or(DEFAULT_COMPACT_TRIGGER_TOKENS);
        let compact_keep_recent_tokens = std::env::var("JUST_AGENT_COMPACT_KEEP_RECENT_TOKENS")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()
            .context("JUST_AGENT_COMPACT_KEEP_RECENT_TOKENS must be a positive integer")?
            .unwrap_or(DEFAULT_COMPACT_KEEP_RECENT_TOKENS);
        let compact_max_tokens = std::env::var("JUST_AGENT_COMPACT_MAX_TOKENS")
            .ok()
            .map(|value| value.parse::<u32>())
            .transpose()
            .context("JUST_AGENT_COMPACT_MAX_TOKENS must be a positive integer")?
            .unwrap_or(DEFAULT_COMPACT_MAX_TOKENS);

        let workspace_root = workspace_root.canonicalize().with_context(|| {
            format!(
                "failed to resolve workspace root {}",
                workspace_root.display()
            )
        })?;

        if compact_trigger_tokens == 0 || compact_keep_recent_tokens == 0 || compact_max_tokens == 0
        {
            bail!("JUST_AGENT_COMPACT_*_TOKENS must be greater than zero");
        }
        if max_tool_rounds == 0 {
            bail!("JUST_AGENT_MAX_TOOL_ROUNDS must be greater than zero");
        }

        Ok(Self {
            prompt,
            system_prompt,
            max_tool_rounds,
            workspace_root,
            compact_trigger_tokens,
            compact_keep_recent_tokens,
            compact_max_tokens,
        })
    }
}
