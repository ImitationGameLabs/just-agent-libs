use std::error::Error;

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
    let prompt = "Say hello in one sentence.";

    let client = OpenAiCompatClient::builder()
        .api_key(api_key)
        .base_url(base_url)
        .build()?;

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
