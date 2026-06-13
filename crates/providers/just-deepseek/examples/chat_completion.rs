//! Basic non-streaming chat completion via [`DeepSeekClient`].
//!
//! ```bash
//! JUST_LLM_DEEPSEEK_API_KEY=your-key JUST_LLM_DEEPSEEK_MODEL=deepseek-chat \
//!   cargo run -p just-deepseek --example chat_completion
//! ```

use just_deepseek::types::chat::{ChatCompletionRequest, ChatMessage};
use just_deepseek::{DeepSeekClient, Error};

#[tokio::main]
async fn main() -> Result<(), Error> {
    dotenvy::dotenv().ok();

    let api_key =
        std::env::var("JUST_LLM_DEEPSEEK_API_KEY").expect("JUST_LLM_DEEPSEEK_API_KEY must be set");
    let base_url = std::env::var("JUST_LLM_DEEPSEEK_BASE_URL").ok();
    let model =
        std::env::var("JUST_LLM_DEEPSEEK_MODEL").expect("JUST_LLM_DEEPSEEK_MODEL must be set");
    let prompt = "Say hello in one sentence.";

    // Build the client. A custom base URL can be supplied for proxies or
    // self-hosted endpoints; otherwise it defaults to the DeepSeek API.
    let mut builder = DeepSeekClient::builder().api_key(&api_key);
    if let Some(url) = base_url {
        builder = builder.base_url(&url);
    }
    let client = builder.build()?;

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
