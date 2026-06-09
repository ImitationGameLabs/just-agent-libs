//! Runtime provider selection from environment variables.
//!
//! Requires both `deepseek` and `openai-compat` features.

mod common;

use just_llm_client::types::chat::ChatMessage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().expect("failed to load .env file");

    let client = common::client_from_env("You are a concise assistant.")?;
    let prompt = "Say hello in one sentence.";

    println!("--- request 1 ---");
    println!("  [provider] {}", client.provider_id());
    println!("  [model] {}", client.model());
    println!("  [system] {}", client.system_prompt().unwrap_or(""),);
    println!("  [user] {prompt}");

    let prepared = client.prepare(client.create_request(vec![ChatMessage::user(prompt)]))?;

    eprintln!(
        "prepared payload snapshot: {}",
        prepared.request_body_text()
    );

    let response = client.send(&prepared).await?;

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
