mod common;

use just_llm_client::{
    LlmBackend,
    provider::DeepSeekBackend,
    types::chat::{ChatCompletionRequest, ChatMessage},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().expect("failed to load .env file");

    let api_key = common::expect_env("JUST_LLM_DEEPSEEK_API_KEY");
    let base_url = std::env::var("JUST_LLM_DEEPSEEK_BASE_URL").ok();
    let model = common::expect_env("JUST_LLM_DEEPSEEK_MODEL");
    let prompt = "Say hello in one sentence.";

    let backend = DeepSeekBackend::new(
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .use_rustls_tls(),
        &api_key,
        base_url.as_deref(),
    )?;

    println!("--- request 1 ---");
    println!("  [system] You are a concise assistant.");
    println!("  [user] {prompt}");

    let response = backend
        .chat_completion(
            ChatCompletionRequest::new(model, vec![ChatMessage::user(prompt)])
                .with_system_prompt("You are a concise assistant."),
        )
        .await?;

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
