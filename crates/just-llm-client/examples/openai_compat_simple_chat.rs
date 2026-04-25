mod common;

use just_llm_client::{
    ChatCompletion,
    provider::OpenAiCompatBackend,
    types::chat::{ChatCompletionRequest, ChatMessage},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().expect("failed to load .env file");

    let api_key = common::expect_env("JUST_LLM_OPENAI_COMPAT_API_KEY");
    let base_url = std::env::var("JUST_LLM_OPENAI_COMPAT_BASE_URL").ok();
    let model = common::expect_env("JUST_LLM_OPENAI_COMPAT_MODEL");
    let prompt = common::expect_env("JUST_LLM_OPENAI_COMPAT_PROMPT");

    let backend = match base_url {
        Some(base_url) => OpenAiCompatBackend::with_base_url(api_key, base_url)?,
        None => {
            OpenAiCompatBackend::with_config(just_openai_compat::OpenAiCompatConfig::new(api_key))?
        }
    };

    println!("--- request 1 ---");
    println!("  [system] You are a concise assistant.");
    println!("  [user] {prompt}");

    let response = backend
        .create_chat_completion(
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
