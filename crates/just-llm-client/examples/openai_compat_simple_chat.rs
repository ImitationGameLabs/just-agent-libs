mod common;

use just_llm_client::{
    LlmBackend,
    provider::OpenAiCompatBackend,
    types::chat::{ChatCompletionRequest, ChatMessage},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().expect("failed to load .env file");

    let api_key = common::expect_env("JUST_LLM_OPENAI_COMPAT_API_KEY");
    let base_url = common::expect_env("JUST_LLM_OPENAI_COMPAT_BASE_URL");
    let model = common::expect_env("JUST_LLM_OPENAI_COMPAT_MODEL");
    let prompt = "Say hello in one sentence.";

    let backend = OpenAiCompatBackend::new(
        just_openai_compat::OpenAiCompatClient::builder()
            .api_key(api_key)
            .base_url(base_url)
            .build()?,
    );

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
