//! Basic non-streaming chat completion using `OpenAiCompatClient`.
//!
//! ```bash
//! JUST_LLM_OPENAI_COMPAT_API_KEY=your-key \
//! JUST_LLM_OPENAI_COMPAT_BASE_URL=https://your-endpoint/v1 \
//! JUST_LLM_OPENAI_COMPAT_MODEL=gpt-4.1-mini \
//!   cargo run -p just-openai-compat --example openai_compat_chat_completion
//! ```

use just_openai_compat::OpenAiCompatClient;
use just_openai_compat::types::chat::{ChatCompletionRequest, ChatMessage};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().expect("failed to load .env file");

    let api_key = std::env::var("JUST_LLM_OPENAI_COMPAT_API_KEY")
        .expect("JUST_LLM_OPENAI_COMPAT_API_KEY must be set");
    let base_url = std::env::var("JUST_LLM_OPENAI_COMPAT_BASE_URL")
        .expect("JUST_LLM_OPENAI_COMPAT_BASE_URL must be set");
    let model = std::env::var("JUST_LLM_OPENAI_COMPAT_MODEL")
        .expect("JUST_LLM_OPENAI_COMPAT_MODEL must be set");
    let prompt = "Say hello in one sentence.";

    let client = OpenAiCompatClient::builder()
        .api_key(&api_key)
        .base_url(&base_url)
        .build()?;

    let request = ChatCompletionRequest::new(
        model,
        vec![
            ChatMessage::system("You are a concise assistant."),
            ChatMessage::user(prompt),
        ],
    );

    println!("--- request ---");
    println!("  [system] You are a concise assistant.");
    println!("  [user] {prompt}");

    let completion = client.chat_completion(request).await?;

    println!("\n--- response ---");
    if let Some(choice) = completion.choices.first() {
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
    }

    Ok(())
}
