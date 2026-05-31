use std::error::Error;

use just_deepseek::{
    DeepSeekClient,
    types::chat::{
        ChatMessage, CreateChatCompletionRequest, ReasoningEffort, ThinkingConfig, ThinkingMode,
    },
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

    let mut request = CreateChatCompletionRequest::new(
        model,
        vec![
            ChatMessage::system("You are a helpful math tutor."),
            ChatMessage::user("What is 17 * 23? Show your reasoning."),
        ],
    );
    request.thinking = Some(ThinkingConfig { kind: ThinkingMode::Enabled });
    request.reasoning_effort = Some(ReasoningEffort::High);

    let response = client.create_chat_completion(request).await?;
    let choice = response
        .choices
        .first()
        .expect("expected at least one choice");

    if let Some(reasoning) = &choice.message.reasoning_content {
        println!("[reasoning] {reasoning}\n");
    }
    println!(
        "[assistant] {}",
        choice.message.content.as_deref().unwrap_or_default()
    );

    if let Some(usage) = &response.usage {
        println!("\n[usage]");
        if let Some(details) = &usage.completion_tokens_details
            && let Some(rt) = details.reasoning_tokens
        {
            println!("  reasoning tokens:    {rt}");
        }
        println!("  prompt tokens:       {}", usage.prompt_tokens);
        println!("  completion tokens:   {}", usage.completion_tokens);
        println!("  cache hit tokens:    {}", usage.prompt_cache_hit_tokens);
        println!("  cache miss tokens:   {}", usage.prompt_cache_miss_tokens);
        println!("  total tokens:        {}", usage.total_tokens);
    }

    Ok(())
}
