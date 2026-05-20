//! Agent round execution loop and context compaction.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use tracing::{info, warn};

use crate::context::compose_context;
use crate::session::AgentContext;
use crate::types::{AgentEvent, AgentOutcome};
use just_llm_client::types::chat::{ChatMessage, ToolCallsMessage, ToolChoice, ToolChoiceMode};

/// Run the agent round loop until completion or max rounds.
pub async fn run_agent_rounds(
    ctx: &mut AgentContext,
    tx: &tokio::sync::mpsc::Sender<AgentEvent>,
) -> Result<AgentOutcome> {
    let tool_timeout = Duration::from_secs(ctx.config.tool_timeout_secs);
    let output_reserve = ctx.config.output_reserve_tokens;
    let context_window = ctx.config.context_window_tokens;

    for _round in 0..ctx.config.max_tool_rounds {
        let messages = compose_context(ctx.store.clone()).await;

        let request = ctx
            .client
            .request(messages)
            .with_tools(ctx.store.lock().await.tool_definitions().to_vec())
            .with_tool_choice(ToolChoice::Mode(ToolChoiceMode::Auto));

        let prompt_tokens = match estimate_prompt_tokens(&ctx.client, &request).await {
            Ok(tokens) => tokens,
            Err(e) => {
                warn!("token estimation failed, sending request anyway: {e:#}");
                0
            }
        };

        if prompt_tokens > 0 && prompt_tokens + output_reserve > context_window {
            info!(
                prompt_tokens,
                context_window, "context exceeds budget, triggering compaction"
            );
            match compact_context(ctx).await {
                Ok(true) => continue,
                Ok(false) => {} // nothing to compact, fall through
                Err(e) => warn!("compaction failed: {e:#}"),
            }
        }

        let response = ctx.client.create_chat_completion(request).await?;

        if let Some(usage) = &response.usage {
            ctx.store.lock().await.set_last_usage(usage.prompt_tokens);
        }

        let message = response
            .first_message()
            .cloned()
            .context("provider returned no completion choices")?;

        if let Some(reasoning) = message.reasoning_content.as_deref() {
            tx.send(AgentEvent::Reasoning(reasoning.to_owned()))
                .await
                .ok();
        }

        let tool_calls = message.tool_calls.clone().unwrap_or_default();
        if tool_calls.is_empty() {
            if let Some(content) = message.content {
                return Ok(AgentOutcome::Finished { content });
            }

            bail!("assistant returned neither tool calls nor final content");
        }

        if let Some(content) = message.content.as_deref() {
            tx.send(AgentEvent::AssistantContent(content.to_owned()))
                .await
                .ok();
        }

        let mut turn_messages = vec![ChatMessage::ToolCalls(ToolCallsMessage {
            role: "assistant".into(),
            content: message.content,
            name: None,
            tool_calls: tool_calls.clone(),
            reasoning_content: message.reasoning_content,
        })];

        for call in tool_calls {
            tx.send(AgentEvent::ToolCall {
                name: call.function.name.clone(),
                args: call.function.arguments.clone(),
            })
            .await
            .ok();
            let result = match tokio::time::timeout(
                tool_timeout,
                ctx.executor
                    .execute(&call.function.name, &call.function.arguments),
            )
            .await
            {
                Ok(output) => output,
                Err(_) => format!(
                    "tool '{}' timed out after {}s",
                    call.function.name,
                    tool_timeout.as_secs()
                ),
            };
            tx.send(AgentEvent::ToolResult(result.clone())).await.ok();
            turn_messages.push(ChatMessage::tool_result(result, call.id));
        }

        ctx.store.lock().await.push_turn(turn_messages);
    }

    Ok(AgentOutcome::MaxRoundsExceeded)
}

/// Estimate prompt tokens via the ChatClient pipeline.
async fn estimate_prompt_tokens(
    client: &just_llm_client::ChatClient,
    request: &just_llm_client::types::chat::ChatCompletionRequest,
) -> Result<usize> {
    let estimator = client
        .token_estimation()
        .context("backend does not support token estimation")?;
    let prepared = client.prepared_request(request.clone()).await?;
    let estimate = estimator.estimate_tokens(&prepared).await?;
    Ok(estimate.prompt_tokens as usize)
}

/// Drain old turns, run compaction strategy, write back results.
///
/// Returns `Ok(true)` if compaction was performed, `Ok(false)` if
/// there were no turns to compact.
pub async fn compact_context(ctx: &AgentContext) -> Result<bool> {
    let (drained, existing_summary) = {
        let mut guard = ctx.store.lock().await;
        let turn_count = guard.turn_count();
        if turn_count == 0 {
            return Ok(false);
        }
        let drained = guard.drain_turns(0..turn_count);
        let summary = guard.summary().map(|s| s.to_owned());
        (drained, summary)
    };

    let available = ctx.config.context_window_tokens;
    let result = match ctx
        .strategy
        .compact(
            &drained,
            existing_summary.as_deref(),
            available,
            &ctx.client,
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            ctx.store.lock().await.prepend_turns(drained);
            return Err(e.context("compaction failed; drained turns restored"));
        }
    };

    let mut guard = ctx.store.lock().await;
    if result.summary.is_none() && result.modified_turns.is_none() {
        guard.prepend_turns(drained);
        bail!(
            "compaction strategy '{}' produced no summary and no modified turns",
            ctx.strategy.name()
        );
    }

    if let Some(modified) = result.modified_turns {
        guard.prepend_turns(modified);
    }
    if let Some(summary) = result.summary {
        guard.set_summary(summary);
    }

    info!(
        strategy = ctx.strategy.name(),
        turns_compacted = result.turns_compacted,
        summary_tokens = result.summary_tokens,
        "compacted turns"
    );

    Ok(true)
}
