use std::error::Error;

use futures_util::StreamExt;
use just_deepseek::{
    DeepSeekClient,
    types::chat::{ChatCompletionRequest, ChatMessage},
};

fn expect_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set"))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenvy::dotenv().expect("failed to load .env file");

    let api_key = expect_env("JUST_LLM_DEEPSEEK_API_KEY");
    let base_url = std::env::var("JUST_LLM_DEEPSEEK_BASE_URL").ok();
    let model = expect_env("JUST_LLM_DEEPSEEK_MODEL");

    let mut builder = DeepSeekClient::builder().api_key(api_key);
    if let Some(url) = base_url {
        builder = builder.base_url(url);
    }
    let client = builder.build()?;

    let mut request = ChatCompletionRequest::new(
        model,
        vec![
            ChatMessage::system("You are a concise Rust tutor. Respond in English."),
            ChatMessage::user(
                "Explain the difference between Rust's String and &str, with examples for each.",
            ),
        ],
    );
    request.stream_options = Some(just_deepseek::types::chat::StreamOptions {
        include_usage: Some(true),
    });

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
            println!("  cache hit tokens:    {}", usage.prompt_cache_hit_tokens);
            println!("  cache miss tokens:   {}", usage.prompt_cache_miss_tokens);
            println!("  total tokens:        {}", usage.total_tokens);
        }
    }

    Ok(())
}
