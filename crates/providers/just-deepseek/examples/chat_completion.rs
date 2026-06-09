use std::error::Error;

use just_deepseek::{
    DeepSeekClient,
    types::chat::{ChatMessage, CreateChatCompletionRequest},
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
    let prompt = "Say hello in one sentence.";

    let mut builder = DeepSeekClient::builder().api_key(api_key);
    if let Some(url) = base_url {
        builder = builder.base_url(url);
    }
    let client = builder.build()?;

    let request = CreateChatCompletionRequest::new(
        model,
        vec![
            ChatMessage::system("You are a concise assistant."),
            ChatMessage::user(prompt),
        ],
    );

    let response = client.create_chat_completion(request).await?;
    let choice = response
        .choices
        .first()
        .expect("expected at least one choice");

    if let Some(rc) = &choice.message.reasoning_content {
        println!("[reasoning] {rc}\n");
    }
    println!(
        "[assistant] {}",
        choice.message.content.as_deref().unwrap_or_default()
    );
    Ok(())
}
