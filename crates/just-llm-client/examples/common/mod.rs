#![allow(dead_code)]

use std::error::Error;

#[cfg(feature = "deepseek")]
use just_llm_client::DeepSeekProvider;
#[cfg(feature = "openai-compat")]
use just_llm_client::OpenAiCompatProvider;
#[cfg(any(feature = "deepseek", feature = "openai-compat"))]
use just_llm_client::{ChatClient, ChatClientOptions, ProviderRegistry};

pub fn expect_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set"))
}

#[cfg(any(feature = "deepseek", feature = "openai-compat"))]
pub fn client_from_env(system_prompt: impl Into<String>) -> Result<ChatClient, Box<dyn Error>> {
    let provider_id = expect_env("JUST_LLM_PROVIDER");
    let model = expect_env("JUST_LLM_MODEL");
    let mut registry = ProviderRegistry::new();

    match provider_id.as_str() {
        #[cfg(feature = "deepseek")]
        "deepseek" => {
            let api_key = expect_env("JUST_LLM_DEEPSEEK_API_KEY");
            let provider = match std::env::var("JUST_LLM_DEEPSEEK_BASE_URL") {
                Ok(base_url) => {
                    DeepSeekProvider::from_api_key("deepseek", api_key).with_base_url(base_url)
                }
                Err(_) => DeepSeekProvider::from_api_key("deepseek", api_key),
            };
            registry.register(provider);
        }
        #[cfg(feature = "openai-compat")]
        "openai-compatible" => {
            let api_key = expect_env("JUST_LLM_OPENAI_COMPAT_API_KEY");
            let provider = match std::env::var("JUST_LLM_OPENAI_COMPAT_BASE_URL") {
                Ok(base_url) => OpenAiCompatProvider::from_api_key("openai-compatible", api_key)
                    .with_base_url(base_url),
                Err(_) => OpenAiCompatProvider::from_api_key("openai-compatible", api_key),
            };
            registry.register(provider);
        }
        _ => return Err(format!("unsupported JUST_LLM_PROVIDER: {provider_id}").into()),
    }

    Ok(registry.chat(
        &provider_id,
        ChatClientOptions::new(model).with_system_prompt(system_prompt),
    )?)
}
