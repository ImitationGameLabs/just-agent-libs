//! Pluggable compaction strategies for context management.
//!
//! When the composed context exceeds the token budget, a [`CompactionStrategy`]
//! is invoked to reduce the stored turns. Three strategies are provided:
//!
//! - [`EvictStrategy`] — drops turns outright
//! - [`SummarizeStrategy`] — uses the LLM to summarize old turns
//! - [`TruncateStrategy`] — truncates long messages within turns

use anyhow::Result;
use async_trait::async_trait;
use just_llm_client::ChatClient;
use just_llm_client::types::chat::{ChatMessage, ToolCallsMessage};
use tracing::warn;

use super::turn::Turn;

/// Result of a compaction operation.
#[derive(Clone, Debug)]
pub struct CompactionResult {
    /// Summary text that replaces compacted turns (if any).
    pub summary: Option<String>,
    /// Estimated tokens in the summary.
    pub summary_tokens: usize,
    /// Number of turns that were compacted.
    pub turns_compacted: usize,
    /// Modified turns to re-insert instead of discarding (used by TruncateStrategy).
    pub modified_turns: Option<Vec<Turn>>,
}

/// Pluggable compaction strategy.
#[async_trait]
pub trait CompactionStrategy: Send + Sync {
    /// Human-readable name for diagnostics.
    fn name(&self) -> &str;

    /// Compact the given turns into a replacement.
    ///
    /// The caller provides the current summary (if any) so the strategy
    /// can incorporate it.
    async fn compact(
        &self,
        turns: &[Turn],
        existing_summary: Option<&str>,
        available: usize,
        client: &ChatClient,
    ) -> Result<CompactionResult>;
}

// ---------------------------------------------------------------------------
// EvictStrategy
// ---------------------------------------------------------------------------

/// Drops all provided turns outright. No LLM call needed.
///
/// If an existing summary is present it is preserved. Use this strategy
/// when you want to free context aggressively without spending tokens
/// on summarization.
pub struct EvictStrategy;

#[async_trait]
impl CompactionStrategy for EvictStrategy {
    fn name(&self) -> &str {
        "evict"
    }

    async fn compact(
        &self,
        _turns: &[Turn],
        existing_summary: Option<&str>,
        _available: usize,
        _client: &ChatClient,
    ) -> Result<CompactionResult> {
        let (summary, summary_tokens) = match existing_summary {
            Some(s) => {
                let tokens = s.chars().count() / 4 + 16;
                (Some(s.to_owned()), tokens)
            }
            None => (None, 0),
        };
        Ok(CompactionResult {
            summary,
            summary_tokens,
            turns_compacted: _turns.len(),
            modified_turns: None,
        })
    }
}

// ---------------------------------------------------------------------------
// SummarizeStrategy
// ---------------------------------------------------------------------------

const COMPACT_PROMPT: &str = "Summarize the key facts from our conversation so far: \
    user goals, decisions made, important outcomes, and the current state of work. \
    Be concise.";

/// LLM-powered summarization of old turns.
///
/// Incorporates any existing summary so summaries accumulate across
/// multiple compaction rounds rather than being replaced wholesale.
pub struct SummarizeStrategy {
    /// Maximum tokens for the generated summary.
    pub max_summary_tokens: u32,
    /// Prompt to use when requesting summarization.
    pub prompt: String,
}

impl SummarizeStrategy {
    pub fn new(max_summary_tokens: u32) -> Self {
        Self { max_summary_tokens, prompt: COMPACT_PROMPT.to_owned() }
    }
}

#[async_trait]
impl CompactionStrategy for SummarizeStrategy {
    fn name(&self) -> &str {
        "summarize"
    }

    async fn compact(
        &self,
        turns: &[Turn],
        existing_summary: Option<&str>,
        available: usize,
        client: &ChatClient,
    ) -> Result<CompactionResult> {
        let mut summary_messages: Vec<ChatMessage> = Vec::new();
        let mut input_budget = available.saturating_sub(self.max_summary_tokens as usize);

        if let Some(existing) = existing_summary {
            let msg = ChatMessage::assistant(format!("[Previous context summary]\n{existing}"));
            input_budget = input_budget.saturating_sub(super::turn::estimate_message_tokens(&msg));
            summary_messages.push(msg);
        }

        // Fill from oldest turns forward, stopping when the input budget is exhausted.
        let mut turns_used = 0;
        for turn in turns.iter() {
            if turn.estimated_tokens > input_budget {
                break;
            }
            input_budget -= turn.estimated_tokens;
            turns_used += 1;
        }
        for turn in turns.iter().take(turns_used) {
            summary_messages.extend(turn.messages.iter().cloned());
        }

        summary_messages.push(ChatMessage::user(&self.prompt));

        let request = client
            .request(summary_messages)
            .with_max_tokens(self.max_summary_tokens);

        let response = client.create_chat_completion(request).await?;

        let (summary, summary_tokens) = match response
            .first_choice_content()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            Some(s) => {
                let tokens = s.chars().count() / 4 + 16;
                (Some(s.to_owned()), tokens)
            }
            None => {
                warn!("compaction: LLM returned empty summary");
                (None, 0)
            }
        };

        Ok(CompactionResult {
            summary,
            summary_tokens,
            turns_compacted: turns_used,
            modified_turns: None,
        })
    }
}

// ---------------------------------------------------------------------------
// TruncateStrategy
// ---------------------------------------------------------------------------

/// Truncates individual long messages within turns.
///
/// Preserves turn structure but clips oversized tool results or
/// assistant messages to a maximum token budget per message.
pub struct TruncateStrategy {
    /// Maximum estimated tokens per individual message.
    pub max_message_tokens: usize,
    /// Notice appended to truncated messages.
    pub truncation_notice: String,
}

impl TruncateStrategy {
    pub fn new(max_message_tokens: usize) -> Self {
        Self { max_message_tokens, truncation_notice: "\n[truncated]".to_owned() }
    }
}

#[async_trait]
impl CompactionStrategy for TruncateStrategy {
    fn name(&self) -> &str {
        "truncate"
    }

    async fn compact(
        &self,
        turns: &[Turn],
        existing_summary: Option<&str>,
        _available: usize,
        _client: &ChatClient,
    ) -> Result<CompactionResult> {
        let (summary, summary_tokens) = match existing_summary {
            Some(s) => {
                let tokens = s.chars().count() / 4 + 16;
                (Some(s.to_owned()), tokens)
            }
            None => (None, 0),
        };

        let max_chars = self.max_message_tokens.saturating_sub(16) * 4;
        let mut modified = Vec::with_capacity(turns.len());

        for turn in turns {
            let truncated_messages: Vec<ChatMessage> = turn
                .messages
                .iter()
                .map(|msg| truncate_message(msg, max_chars, &self.truncation_notice))
                .collect();

            let estimated_tokens = Turn::estimate_tokens(&truncated_messages);
            modified.push(Turn { id: turn.id, messages: truncated_messages, estimated_tokens });
        }

        Ok(CompactionResult {
            summary,
            summary_tokens,
            turns_compacted: turns.len(),
            modified_turns: Some(modified),
        })
    }
}

/// Creates the default compaction strategy from config values.
pub fn strategy_from_name(name: &str, max_summary_tokens: u32) -> Box<dyn CompactionStrategy> {
    match name {
        "evict" => Box::new(EvictStrategy),
        "truncate" => Box::new(TruncateStrategy::new(2_000)),
        _ => Box::new(SummarizeStrategy::new(max_summary_tokens)),
    }
}

/// Truncate a single message's text content to `max_chars` characters.
/// Messages without text content (e.g., tool-call-only messages) pass through unchanged.
fn truncate_message(msg: &ChatMessage, max_chars: usize, notice: &str) -> ChatMessage {
    let content = match msg.content() {
        Some(c) => c,
        None => return msg.clone(),
    };

    if content.chars().count() <= max_chars {
        return msg.clone();
    }

    let truncated: String = content
        .chars()
        .take(max_chars)
        .chain(notice.chars())
        .collect();

    match msg {
        ChatMessage::ToolResult(tr) => ChatMessage::tool_result(truncated, &tr.tool_call_id),
        ChatMessage::ToolCalls(tc) => {
            ChatMessage::ToolCalls(ToolCallsMessage { content: Some(truncated), ..(*tc).clone() })
        }
        ChatMessage::Message(m) => ChatMessage::new(&m.role, truncated),
    }
}
