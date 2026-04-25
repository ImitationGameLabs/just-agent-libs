//! Runtime provider selection from environment variables.
//!
//! Requires both `deepseek` and `openai-compat` features.

mod common;

use just_llm_client::types::chat::ChatMessage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().expect("failed to load .env file");

    let client = common::client_from_env("You are a concise assistant.")?;
    let prompt = common::expect_env("JUST_LLM_PROMPT");

    println!("--- request 1 ---");
    println!("  [provider] {}", client.provider_id());
    println!("  [model] {}", client.model());
    println!("  [system] {}", client.system_prompt().unwrap_or(""),);
    println!("  [user] {prompt}");

    let prepared = client
        .prepared_request(client.request(vec![ChatMessage::user(prompt)]))
        .await?;
    let estimate = client
        .token_estimation()?
        .estimate_tokens(&prepared)
        .await?;

    eprintln!(
        "prepared payload snapshot: {}",
        prepared.request_body_text()
    );
    eprintln!("request preview: {:?}", prepared.preview());
    eprintln!("token estimate: {:?}", estimate);

    let response = client.send_prepared(&prepared).await?;

    println!("\n--- response 1 ---");
    if let Some(rc) = response.first_choice_reasoning_content() {
        println!("  [reasoning] {rc}");
    }
    println!(
        "  [assistant] {}",
        response.first_choice_content().unwrap_or_default()
    );
    Ok(())
}
