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

    let api_key = expect_env("DEEPSEEK_API_KEY");
    let model = expect_env("DEEPSEEK_MODEL");
    let prompt = expect_env("DEEPSEEK_PROMPT");

    let client = DeepSeekClient::new(api_key)?;

    println!("--- request 1 ---");
    println!("  [system] You are a concise assistant.");
    println!("  [user] {prompt}");

    let request = CreateChatCompletionRequest::new(
        model,
        vec![ChatMessage::system("You are a concise assistant."), ChatMessage::user(prompt)],
    );

    let response = client.create_chat_completion(request).await?;
    let choice = response
        .choices
        .first()
        .expect("expected at least one choice");

    println!("\n--- response 1 ---");
    if let Some(rc) = &choice.message.reasoning_content {
        println!("  [reasoning] {rc}");
    }
    println!(
        "  [assistant] {}",
        choice.message.content.as_deref().unwrap_or_default()
    );
    Ok(())
}
