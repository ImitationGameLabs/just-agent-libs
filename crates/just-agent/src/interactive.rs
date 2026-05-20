//! Interactive TUI and non-interactive agent modes.

use std::time::Duration;

use anyhow::{Context, Result, bail};

use crate::policy::{ApprovalRequest, ChannelApprovalProvider};
use crate::runner;
use crate::session::{self, AgentContext};
use crate::tui;
use crate::types::{AgentEvent, AgentOutcome};
use just_llm_client::types::chat::ChatMessage;

/// Non-interactive: single prompt, stdout output, exit.
pub async fn run_noninteractive(mut ctx: AgentContext, prompt: Option<String>) -> Result<()> {
    let prompt = prompt.context("--prompt is required in non-interactive mode")?;
    ctx.store
        .lock()
        .await
        .push_turn(vec![ChatMessage::user(&prompt)]);

    let (tx, mut rx) = tokio::sync::mpsc::channel(256);

    let printer = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            print_agent_event(&event);
        }
    });

    let outcome = runner::run_agent_rounds(&mut ctx, &tx).await;

    drop(tx);
    printer.await.ok();

    match outcome {
        Ok(AgentOutcome::Finished { content }) => println!("{content}"),
        Ok(AgentOutcome::MaxRoundsExceeded) => {
            bail!(
                "agent exceeded the maximum number of tool rounds ({})",
                ctx.config.max_tool_rounds
            )
        }
        Err(e) => return Err(e),
    }

    Ok(())
}

/// Interactive TUI mode.
pub async fn run_tui(mut ctx: AgentContext, initial_prompt: Option<String>) -> Result<()> {
    let (agent_tx, mut agent_rx) = tokio::sync::mpsc::channel::<AgentEvent>(256);
    let (prompt_tx, prompt_rx) = tokio::sync::mpsc::channel(16);
    let (approval_tx, mut approval_rx) = tokio::sync::mpsc::channel::<ApprovalRequest>(16);

    ctx.executor
        .set_approval_provider(Box::new(ChannelApprovalProvider::new(approval_tx)));

    let agent_handle = tokio::spawn(session::agent_task(
        ctx,
        initial_prompt,
        prompt_rx,
        agent_tx.clone(),
    ));

    // Crossterm event bridge (crossterm::event::read is blocking).
    let (key_tx, mut key_rx) = tokio::sync::mpsc::channel::<ratatui::crossterm::event::Event>(64);
    std::thread::spawn(move || {
        while let Ok(event) = ratatui::crossterm::event::read() {
            if key_tx.blocking_send(event).is_err() {
                break;
            }
        }
    });

    ratatui::try_init()?;
    let mut terminal = ratatui::init();
    ratatui::crossterm::execute!(
        std::io::stdout(),
        ratatui::crossterm::event::EnableMouseCapture
    )?;
    let mut app = tui::App::new();

    loop {
        terminal.draw(|frame| app.render(frame))?;

        tokio::select! {
            // Crossterm events
            Some(event) = key_rx.recv() => {
                match event {
                    ratatui::crossterm::event::Event::Key(key) => {
                        app.handle_key_event(key, &prompt_tx);
                        if app.should_quit {
                            break;
                        }
                    }
                    ratatui::crossterm::event::Event::Mouse(mouse) => {
                        let chat_height = terminal.get_frame().area().height.saturating_sub(7);
                        app.handle_mouse_event(mouse, chat_height);
                    }
                    _ => {}
                }
            }
            // Agent events
            Some(event) = agent_rx.recv() => {
                app.handle_agent_event(event);
            }
            // Approval requests from the executor
            Some(req) = approval_rx.recv() => {
                app.show_approval(req);
            }
            // Periodic redraw tick
            _ = tokio::time::sleep(Duration::from_millis(33)) => {}
        }
    }

    drop(prompt_tx); // signal agent task to stop
    agent_handle.await.ok();
    ratatui::crossterm::execute!(
        std::io::stdout(),
        ratatui::crossterm::event::DisableMouseCapture
    )
    .ok();
    ratatui::restore();
    Ok(())
}

/// Print an agent event to stdout (for non-interactive mode).
fn print_agent_event(event: &AgentEvent) {
    match event {
        AgentEvent::Reasoning(text) => println!("[reasoning] {text}"),
        AgentEvent::AssistantContent(text) => println!("[assistant] {text}"),
        AgentEvent::ToolCall { name, args } => println!("[tool call] {name}({args})"),
        AgentEvent::ToolResult(result) => println!("[tool result] {result}"),
        AgentEvent::Finished(_)
        | AgentEvent::MaxRoundsExceeded
        | AgentEvent::Error(_)
        | AgentEvent::Status(_)
        | AgentEvent::Busy => {}
    }
}
