use std::error::Error;

use futures_util::StreamExt;
use just_openai_compat::{
    OpenAiCompatClient,
    types::chat::{ChatMessage, CreateChatCompletionRequest},
};

fn expect_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set"))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenvy::dotenv().expect("failed to load .env file");

    let api_key = expect_env("JUST_LLM_OPENAI_COMPAT_API_KEY");
    let base_url = expect_env("JUST_LLM_OPENAI_COMPAT_BASE_URL");
    let model = expect_env("JUST_LLM_OPENAI_COMPAT_MODEL");

    let client = OpenAiCompatClient::builder()
        .api_key(api_key)
        .base_url(base_url)
        .build()?;

    let mut request = CreateChatCompletionRequest::new(
        model,
        vec![
            ChatMessage::system("You are a concise Rust tutor. Respond in English."),
            ChatMessage::user(
                "Explain the difference between Rust's String and &str, with examples for each.",
            ),
        ],
    );
    request.stream_options =
        Some(just_openai_compat::types::chat::StreamOptions { include_usage: Some(true) });

    let mut stream = client.stream_chat_completion(request).await?;
    let mut had_reasoning = false;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;

        for choice in &chunk.choices {
            if let Some(delta) = choice.delta.reasoning_content.as_deref() {
                if !had_reasoning {
                    println!("[reasoning] ");
                    had_reasoning = true;
                }
                print!("{delta}");
            }
            if let Some(delta) = choice.delta.content.as_deref() {
                if had_reasoning {
                    println!("\n[assistant] ");
                    had_reasoning = false;
                }
                print!("{delta}");
            }
        }

        if let Some(usage) = &chunk.usage {
            println!("\n[usage]");
            println!("  prompt tokens:       {}", usage.prompt_tokens);
            println!("  completion tokens:   {}", usage.completion_tokens);
            println!("  total tokens:        {}", usage.total_tokens);
        }
    }

    Ok(())
}
