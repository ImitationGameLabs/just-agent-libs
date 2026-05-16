//! just-agent: a minimal coding agent.
//!
//! Usage:
//!   cargo run -p just-agent -- --prompt "fix the login bug"
//!   JUST_AGENT_PROMPT="fix the login bug" cargo run -p just-agent
//!
//! For sandboxed execution, run the run-agent-with-prompt example:
//!   cargo run -p just-agent --example run-agent-with-prompt --workspace \u003Cdir\u003E [args...]

mod args;
mod compact;
mod config;
mod policy;
mod provider;
mod tools;

use anyhow::{Context, Result, bail};

use args::Args;
use clap::Parser;
use compact::{CompactionConfig, ContextCompactor};
use config::AgentConfig;
use just_llm_client::types::chat::{ChatMessage, ToolCallsMessage, ToolChoice, ToolChoiceMode};
use policy::{AgentPolicy, AuthorizedToolExecutor};
use provider::client_from_env;
use tools::build_tool_dispatch;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let config = AgentConfig::load(&args)?;

    let client = client_from_env(&config.system_prompt)?;
    std::env::set_current_dir(&config.workspace_root).with_context(|| {
        format!(
            "failed to change directory to {}",
            config.workspace_root.display()
        )
    })?;
    let dispatch = build_tool_dispatch().await?;

    let mut executor =
        AuthorizedToolExecutor::new(dispatch, AgentPolicy::new(config.workspace_root.clone()));
    let tool_definitions = executor.tool_definitions();
    let mut compactor = ContextCompactor::new(CompactionConfig {
        trigger_tokens: config.compact_trigger_tokens,
        keep_recent_tokens: config.compact_keep_recent_tokens,
        summary_max_tokens: config.compact_max_tokens,
    });

    let mut messages = vec![ChatMessage::user(config.prompt)];

    for _round in 0..config.max_tool_rounds {
        compactor.maybe_compact(&client, &mut messages).await?;

        let request = client
            .request(messages.clone())
            .with_tools(tool_definitions.clone())
            .with_tool_choice(ToolChoice::Mode(ToolChoiceMode::Auto));

        let response = client.create_chat_completion(request).await?;
        let message = response
            .first_message()
            .cloned()
            .context("provider returned no completion choices")?;

        if let Some(reasoning) = message.reasoning_content.as_deref() {
            eprintln!("[reasoning] {reasoning}");
        }

        let tool_calls = message.tool_calls.clone().unwrap_or_default();
        if tool_calls.is_empty() {
            if let Some(content) = message.content {
                println!("{content}");
                return Ok(());
            }

            bail!("assistant returned neither tool calls nor final content");
        }

        if let Some(content) = message.content.as_deref() {
            eprintln!("[assistant] {content}");
        }

        messages.push(ChatMessage::ToolCalls(ToolCallsMessage {
            role: "assistant".into(),
            content: message.content,
            name: None,
            tool_calls: tool_calls.clone(),
            reasoning_content: message.reasoning_content,
        }));

        for call in tool_calls {
            eprintln!(
                "[tool call] {}({})",
                call.function.name, call.function.arguments
            );
            let result = executor
                .execute(&call.function.name, &call.function.arguments)
                .await;
            eprintln!("[tool result] {result}");
            messages.push(ChatMessage::tool_result(result, call.id));
        }
    }

    bail!(
        "agent exceeded the maximum number of tool rounds ({})",
        config.max_tool_rounds
    );
}
