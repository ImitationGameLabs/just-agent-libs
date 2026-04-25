//! Programmatic registry-based initialization.
//!
//! Demonstrates building a [`ProviderRegistry`] by hand and obtaining a
//! [`ChatClient`] from it.

mod common;

use just_llm_client::{
    ChatClientOptions, OpenAiCompatProvider, ProviderRegistry, types::chat::ChatMessage,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().expect("failed to load .env file");

    let api_key = common::expect_env("JUST_LLM_OPENAI_COMPAT_API_KEY");
    let model = common::expect_env("JUST_LLM_OPENAI_COMPAT_MODEL");
    let prompt = common::expect_env("JUST_LLM_OPENAI_COMPAT_PROMPT");

    let provider = match std::env::var("JUST_LLM_OPENAI_COMPAT_BASE_URL") {
        Ok(base_url) => {
            OpenAiCompatProvider::from_api_key("openai-compatible", api_key).with_base_url(base_url)
        }
        Err(_) => OpenAiCompatProvider::from_api_key("openai-compatible", api_key),
    };
    let registry = ProviderRegistry::with_openai_compat(provider);
    let client = registry.chat(
        "openai-compatible",
        ChatClientOptions::new(model).with_system_prompt("You are a concise assistant."),
    )?;

    println!("--- request 1 ---");
    println!("  [provider] {}", client.provider_id());
    println!("  [model] {}", client.model());
    println!("  [system] You are a concise assistant.");
    println!("  [user] {prompt}");

    let prepared = client
        .prepared_request(client.request(vec![ChatMessage::user(prompt)]))
        .await?;
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
