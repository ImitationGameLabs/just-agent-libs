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
    let base_url = common::expect_env("JUST_LLM_OPENAI_COMPAT_BASE_URL");
    let model = common::expect_env("JUST_LLM_OPENAI_COMPAT_MODEL");
    let prompt = "Say hello in one sentence.";

    let provider = OpenAiCompatProvider::from_api_key("openai-compatible", api_key, base_url);
    let registry = ProviderRegistry::with_openai_compat(provider);
    let client = registry.chat(
        "openai-compatible",
        ChatClientOptions::new(model).with_system_prompt("You are a concise assistant."),
    )?;

    println!("--- request 1 ---");
    println!("  [instance] {}", client.instance_id());
    println!("  [model] {}", client.model());
    println!("  [system] You are a concise assistant.");
    println!("  [user] {prompt}");

    let request = client.create_request(vec![ChatMessage::user(prompt)]);
    let response = client.chat_completion(request).await?;

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
