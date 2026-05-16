use anyhow::{Context, Result};
use just_llm_client::{ChatClient, types::chat::ChatMessage};

const COMPACTION_PREFIX: &str = "[COMPRESSED CONTEXT SUMMARY]";
const SUMMARY_SUFFIX: &str = "[/COMPRESSED CONTEXT SUMMARY]";
const COMPACT_PROMPT: &str = "Summarize the key facts from our conversation so far: user goals, decisions made, important outcomes, and the current state of work. Be concise.";

#[derive(Clone, Debug)]
pub struct CompactionConfig {
    pub trigger_tokens: usize,
    pub keep_recent_tokens: usize,
    pub summary_max_tokens: u32,
}

pub struct ContextCompactor {
    config: CompactionConfig,
}

impl ContextCompactor {
    pub fn new(config: CompactionConfig) -> Self {
        Self { config }
    }

    pub async fn maybe_compact(
        &mut self,
        client: &ChatClient,
        messages: &mut Vec<ChatMessage>,
    ) -> Result<()> {
        if estimate_messages_tokens(messages) < self.config.trigger_tokens {
            return Ok(());
        }

        let Some(split_index) = find_split_index(messages, self.config.keep_recent_tokens) else {
            return Ok(());
        };

        messages.push(ChatMessage::user(COMPACT_PROMPT));

        let request = client
            .request(messages.clone())
            .with_max_tokens(self.config.summary_max_tokens);

        let response = match client.create_chat_completion(request).await {
            Ok(r) => r,
            Err(e) => {
                messages.pop();
                return Err(e.into());
            }
        };

        let summary = response
            .first_choice_content()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .context("compaction summary response was empty")?;

        // Keep: summary prefix + recent suffix (excluding the compact prompt)
        let suffix = messages[split_index..messages.len() - 1].to_vec();
        *messages = std::iter::once(ChatMessage::assistant(format!(
            "{COMPACTION_PREFIX}\n{summary}\n{SUMMARY_SUFFIX}"
        )))
        .chain(suffix)
        .collect();

        eprintln!("[compaction] compacted earlier context into a synthetic summary message");
        Ok(())
    }
}

fn estimate_messages_tokens(messages: &[ChatMessage]) -> usize {
    messages.iter().map(estimate_message_tokens).sum()
}

fn estimate_message_tokens(message: &ChatMessage) -> usize {
    let content_tokens = message
        .content()
        .map(|content| content.chars().count() / 4)
        .unwrap_or_default();
    let tool_tokens = message
        .tool_calls()
        .map(|tool_calls| {
            tool_calls
                .iter()
                .map(|tool_call| tool_call.function.arguments.chars().count() / 4 + 24)
                .sum::<usize>()
        })
        .unwrap_or_default();
    content_tokens + tool_tokens + 16
}

fn find_split_index(messages: &[ChatMessage], keep_recent_tokens: usize) -> Option<usize> {
    let mut kept_tokens = 0usize;
    let mut keep_from = messages.len();

    for index in (0..messages.len()).rev() {
        kept_tokens += estimate_message_tokens(&messages[index]);
        if kept_tokens > keep_recent_tokens {
            keep_from = index;
            break;
        }
    }

    (keep_from > 0 && keep_from < messages.len()).then_some(keep_from)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user_message(content: &str) -> ChatMessage {
        ChatMessage::user(content)
    }

    #[test]
    fn estimate_tokens_scales_with_content() {
        let short = estimate_messages_tokens(&[user_message("hello")]);
        let long = estimate_messages_tokens(&[user_message(&"x".repeat(400))]);
        assert!(long > short);
    }

    #[test]
    fn split_index_keeps_recent_budget() {
        let messages = (0..10)
            .map(|index| user_message(&format!("message {index} {}", "x".repeat(300))))
            .collect::<Vec<_>>();

        let split = find_split_index(&messages, 300).unwrap();
        assert!(split < messages.len());
        let suffix_tokens = estimate_messages_tokens(&messages[split..]);
        assert!(suffix_tokens > 300);
    }
}
