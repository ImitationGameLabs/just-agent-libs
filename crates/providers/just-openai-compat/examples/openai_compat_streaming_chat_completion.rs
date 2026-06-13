//! Streaming chat completion using `OpenAiCompatClient`.
//!
//! ```bash
//! JUST_LLM_OPENAI_COMPAT_API_KEY=your-key \
//! JUST_LLM_OPENAI_COMPAT_BASE_URL=https://your-endpoint/v1 \
//! JUST_LLM_OPENAI_COMPAT_MODEL=gpt-4.1-mini \
//!   cargo run -p just-openai-compat --example openai_compat_streaming_chat_completion
//! ```

use futures_util::StreamExt;
use just_openai_compat::OpenAiCompatClient;
use just_openai_compat::types::chat::{ChatCompletionRequest, ChatMessage, StreamOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().expect("failed to load .env file");

    let api_key = std::env::var("JUST_LLM_OPENAI_COMPAT_API_KEY")
        .expect("JUST_LLM_OPENAI_COMPAT_API_KEY must be set");
    let base_url = std::env::var("JUST_LLM_OPENAI_COMPAT_BASE_URL")
        .expect("JUST_LLM_OPENAI_COMPAT_BASE_URL must be set");
    let model = std::env::var("JUST_LLM_OPENAI_COMPAT_MODEL")
        .expect("JUST_LLM_OPENAI_COMPAT_MODEL must be set");
    let prompt = "Explain Rust ownership in two sentences.";

    let client = OpenAiCompatClient::builder()
        .api_key(&api_key)
        .base_url(&base_url)
        .build()?;

    let mut request = ChatCompletionRequest::new(
        model,
        vec![
            ChatMessage::system("You are a concise assistant."),
            ChatMessage::user(prompt),
        ],
    );
    request.stream_options = Some(StreamOptions {
        include_usage: Some(true),
    });

    println!("--- request ---");
    println!("  [system] You are a concise assistant.");
    println!("  [user] {prompt}");

    // stream_chat_completion forces stream=true internally.
    let mut stream = client.stream_chat_completion(request).await?;

    println!("\n--- response (streaming) ---");
    while let Some(result) = stream.next().await {
        let chunk = result?;
        for choice in &chunk.choices {
            if let Some(content) = &choice.delta.content {
                print!("{content}");
            }
        }
        if let Some(usage) = &chunk.usage {
            println!(
                "\n  [usage] prompt={} completion={} total={}",
                usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
            );
        }
    }
    println!();

    Ok(())
}
