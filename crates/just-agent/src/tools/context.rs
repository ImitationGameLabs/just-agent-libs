//! Context management tools for just-agent.
//!
//! Provides tools for pinning and unpinning content in the agent's
//! persistent context layer, checking token usage, and evicting old turns.

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use just_llm_client::tools::LlmTool;
use just_llm_client::types::chat::ChatMessage;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::context::AgenticContext;

// --- context_pin ---

#[derive(Debug, Deserialize, Serialize)]
struct PinArgs {
    label: String,
    content: String,
}

/// Pins arbitrary content into the agent's persistent context.
pub struct ContextPinTool {
    ctx: Arc<Mutex<dyn AgenticContext>>,
}

impl ContextPinTool {
    pub fn new(ctx: Arc<Mutex<dyn AgenticContext>>) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl LlmTool for ContextPinTool {
    fn name(&self) -> &str {
        "context_pin"
    }

    fn description(&self) -> &str {
        "Pin content into the agent's persistent context. Pinned content is \
         included in every LLM request until explicitly removed with context_unpin. \
         Use this to keep important instructions, constraints, or reference material \
         available throughout the conversation."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Unique identifier for this pinned item."
                },
                "content": {
                    "type": "string",
                    "description": "The content to pin."
                }
            },
            "required": ["label", "content"]
        })
    }

    async fn call(&self, args_json: &str) -> Result<String> {
        let args: PinArgs =
            serde_json::from_str(args_json).context("context_pin: invalid arguments")?;
        let mut ctx = self.ctx.lock().await;
        ctx.pin(&args.label, ChatMessage::user(&args.content))?;
        let labels = ctx.pinned_labels();
        Ok(serde_json::to_string(&json!({
            "pinned": args.label,
            "pinned_labels": labels,
        }))?)
    }
}

// --- context_unpin ---

#[derive(Debug, Deserialize, Serialize)]
struct UnpinArgs {
    label: String,
}

/// Removes a pinned item by label.
pub struct ContextUnpinTool {
    ctx: Arc<Mutex<dyn AgenticContext>>,
}

impl ContextUnpinTool {
    pub fn new(ctx: Arc<Mutex<dyn AgenticContext>>) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl LlmTool for ContextUnpinTool {
    fn name(&self) -> &str {
        "context_unpin"
    }

    fn description(&self) -> &str {
        "Remove a pinned item from the agent's context by label. \
         The content will no longer be included in future LLM requests."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "The label of the pinned item to remove."
                }
            },
            "required": ["label"]
        })
    }

    async fn call(&self, args_json: &str) -> Result<String> {
        let args: UnpinArgs =
            serde_json::from_str(args_json).context("context_unpin: invalid arguments")?;
        let mut ctx = self.ctx.lock().await;
        ctx.unpin(&args.label)?;
        let labels = ctx.pinned_labels();
        Ok(serde_json::to_string(&json!({
            "unpinned": args.label,
            "pinned_labels": labels,
        }))?)
    }
}

// --- context_status ---

/// Reports the agent's current token budget and usage.
pub struct ContextStatusTool {
    ctx: Arc<Mutex<dyn AgenticContext>>,
}

impl ContextStatusTool {
    pub fn new(ctx: Arc<Mutex<dyn AgenticContext>>) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl LlmTool for ContextStatusTool {
    fn name(&self) -> &str {
        "context_status"
    }

    fn description(&self) -> &str {
        "Report the agent's current context window usage: how many tokens are \
         consumed by pinned items, summary, and conversation turns, and how many \
         remain. Use this to decide whether to evict old turns with context_evict \
         before the automatic compaction triggers."
    }

    fn parameters_schema(&self) -> Value {
        json!({ "type": "object", "properties": {}, "required": [] })
    }

    async fn call(&self, _args_json: &str) -> Result<String> {
        let ctx = self.ctx.lock().await;
        let usage = ctx.usage_snapshot();
        let pinned_tokens: usize = usage.pinned_items.iter().map(|(_, t)| *t).sum();
        Ok(serde_json::to_string(&json!({
            "last_prompt_tokens": usage.last_prompt_tokens,
            "usage": {
                "pinned_tokens": pinned_tokens,
                "summary_tokens": usage.summary_tokens,
                "turn_tokens": usage.turn_tokens,
            },
            "pinned_items": usage.pinned_items,
            "turn_count": usage.turn_count,
            "has_summary": usage.summary_tokens > 0,
        }))?)
    }
}

// --- context_evict ---

#[derive(Debug, Deserialize, Serialize)]
struct EvictArgs {
    count: usize,
}

/// Evicts the oldest conversation turns to free context capacity.
pub struct ContextEvictTool {
    ctx: Arc<Mutex<dyn AgenticContext>>,
}

impl ContextEvictTool {
    pub fn new(ctx: Arc<Mutex<dyn AgenticContext>>) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl LlmTool for ContextEvictTool {
    fn name(&self) -> &str {
        "context_evict"
    }

    fn description(&self) -> &str {
        "Evict the oldest conversation turns to free context capacity. Evicted turns \
         are permanently discarded (not summarized). Pin important content with \
         context_pin before evicting if you need to preserve it. Use context_status \
         to check current usage before deciding how many turns to evict."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of oldest turns to evict."
                }
            },
            "required": ["count"]
        })
    }

    async fn call(&self, args_json: &str) -> Result<String> {
        let args: EvictArgs =
            serde_json::from_str(args_json).context("context_evict: invalid arguments")?;
        let mut ctx = self.ctx.lock().await;
        let result = ctx.evict_turns(args.count);
        Ok(serde_json::to_string(&json!({
            "evicted": result.evicted,
            "remaining_turns": result.remaining_turns,
            "freed_tokens": result.freed_tokens,
        }))?)
    }
}

/// Creates context management tools sharing the same backing store.
pub fn context_tool_set(ctx: Arc<Mutex<dyn AgenticContext>>) -> Vec<Box<dyn LlmTool>> {
    vec![
        Box::new(ContextPinTool::new(ctx.clone())),
        Box::new(ContextUnpinTool::new(ctx.clone())),
        Box::new(ContextStatusTool::new(ctx.clone())),
        Box::new(ContextEvictTool::new(ctx)),
    ]
}
