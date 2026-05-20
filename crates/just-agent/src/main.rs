//! just-agent: a minimal coding agent.
//!
//! Usage:
//!   cargo run -p just-agent -- --prompt "fix the login bug"
//!   cargo run -p just-agent -- --interactive
//!   JUST_AGENT_PROMPT="fix the login bug" cargo run -p just-agent
//!
//! For sandboxed execution, run the run-agent-with-prompt example:
//!   cargo run -p just-agent --example run-agent-with-prompt --workspace \<dir\> \[args...\]

mod args;
mod command;
mod config;
mod context;
mod interactive;
mod policy;
mod provider;
mod runner;
mod session;
mod tools;
mod tui;
mod types;

use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::Mutex;
use tracing::info;

use args::Args;
use clap::Parser;
use config::AgentConfig;
use context::{AgenticContext, ContextStore, strategy_from_name};
use just_llm_client::types::chat::ChatMessage;
use policy::{AgentPolicy, AuthorizedToolExecutor, StdinApprovalProvider};
use provider::client_from_env;
use session::AgentContext;
use tools::{build_tool_dispatch, ensure_meta_skill, load_skill};

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let mut config = AgentConfig::load(&args)?;

    // In TUI mode, write logs to file to avoid corrupting the display
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    if args.interactive {
        let log_path = std::env::var("JUST_AGENT_DATA_DIR")
            .map(|d| std::path::PathBuf::from(d).join("agent.log"))
            .unwrap_or_else(|_| {
                dirs::data_dir()
                    .unwrap_or_else(|| std::path::PathBuf::from("."))
                    .join("just-agent")
                    .join("agent.log")
            });
        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        if let Ok(file) = std::fs::File::create(&log_path) {
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_writer(file)
                .with_ansi(false)
                .init();
        }
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }

    // Load meta-skill into the system prompt (protected from agent modification).
    let client = {
        let meta = ensure_meta_skill()?;
        let mut system_prompt = config.system_prompt.clone();
        system_prompt.push_str("\n\n");
        system_prompt.push_str(&meta);
        client_from_env(&system_prompt)?
    };
    info!("loaded meta-skill into system prompt");

    std::env::set_current_dir(&config.workspace_root).with_context(|| {
        format!(
            "failed to change directory to {}",
            config.workspace_root.display()
        )
    })?;
    let store = Arc::new(Mutex::new(ContextStore::new()));

    // Load user-requested skills (pinned into context, agent can manage them).
    for skill_name in &config.skills {
        let content = load_skill(skill_name)?;
        store.lock().await.pin(
            &format!("skill:{skill_name}"),
            ChatMessage::user(format!("[skill: {skill_name}]\n{content}")),
        )?;
        info!(skill = skill_name, "loaded skill");
    }

    // Build tool dispatch (shell + context tools sharing the store).
    let dispatch = build_tool_dispatch(store.clone()).await?;

    let executor = AuthorizedToolExecutor::new(
        dispatch,
        AgentPolicy::new(config.workspace_root.clone()),
        Box::new(StdinApprovalProvider::new()),
    );
    let tool_definitions = executor.tool_definitions();

    store
        .lock()
        .await
        .set_tool_definitions(tool_definitions.clone());
    let strategy = strategy_from_name(&config.compaction_strategy, config.compact_max_tokens);

    let prompt = config.prompt.take();

    let ctx = AgentContext { client, store, executor, strategy, config };

    if ctx.config.interactive {
        interactive::run_tui(ctx, prompt).await
    } else {
        interactive::run_noninteractive(ctx, prompt).await
    }
}
