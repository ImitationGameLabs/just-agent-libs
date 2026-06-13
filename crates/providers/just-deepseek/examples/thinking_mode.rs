//! DeepSeek thinking mode — enables reasoning and shows `reasoning_content`.
//!
//! ```bash
//! JUST_LLM_DEEPSEEK_API_KEY=your-key JUST_LLM_DEEPSEEK_MODEL=deepseek-reasoner \
//!   cargo run -p just-deepseek --example thinking_mode
//! ```

use just_deepseek::types::chat::{
    ChatCompletionRequest, ChatMessage, ReasoningEffort, ThinkingConfig, ThinkingMode,
};
use just_deepseek::{DeepSeekClient, Error};

#[tokio::main]
async fn main() -> Result<(), Error> {
    dotenvy::dotenv().ok();

    let api_key =
        std::env::var("JUST_LLM_DEEPSEEK_API_KEY").expect("JUST_LLM_DEEPSEEK_API_KEY must be set");
    let base_url = std::env::var("JUST_LLM_DEEPSEEK_BASE_URL").ok();
    let model =
        std::env::var("JUST_LLM_DEEPSEEK_MODEL").expect("JUST_LLM_DEEPSEEK_MODEL must be set");
    let prompt = "How many 'r's are in 'strawberry'? Think step by step.";

    let mut builder = DeepSeekClient::builder().api_key(&api_key);
    if let Some(url) = base_url {
        builder = builder.base_url(&url);
    }
    let client = builder.build()?;

    let mut request = ChatCompletionRequest::new(
        model,
        vec![
            ChatMessage::system("You are a concise assistant."),
            ChatMessage::user(prompt),
        ],
    );
    request.thinking = Some(ThinkingConfig {
        kind: ThinkingMode::Enabled,
    });
    request.reasoning_effort = Some(ReasoningEffort::High);

    println!("--- request ---");
    println!("  [system] You are a concise assistant.");
    println!("  [user] {prompt}");
    println!("  [thinking] enabled, effort=high");

    let completion = client.chat_completion(request).await?;

    println!("\n--- response ---");
    if let Some(choice) = completion.choices.first() {
        if let Some(reasoning) = &choice.message.reasoning_content {
            println!("  [reasoning] {reasoning}");
        }
        println!(
            "  [assistant] {}",
            choice.message.content.as_deref().unwrap_or_default()
        );
    }
    if let Some(usage) = &completion.usage {
        println!(
            "  [usage] prompt={} completion={} total={}",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        );
        if let Some(details) = &usage.completion_tokens_details {
            println!(
                "  [reasoning tokens] {}",
                details.reasoning_tokens.unwrap_or(0)
            );
        }
    }

    Ok(())
}
