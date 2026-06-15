#![allow(dead_code)]

use std::error::Error;

#[cfg(any(feature = "deepseek", feature = "openai-compat"))]
use just_llm_client::{BackendFactory, ChatClient, ChatClientOptions};

pub fn expect_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set"))
}

/// Build a [`ChatClient`] for the provider named by `JUST_LLM_PROVIDER`, using a pre-seeded
/// [`BackendFactory`] to construct the backend from credentials in the environment.
#[cfg(any(feature = "deepseek", feature = "openai-compat"))]
pub fn client_from_env(system_prompt: impl Into<String>) -> Result<ChatClient, Box<dyn Error>> {
    let family = expect_env("JUST_LLM_PROVIDER");
    let model = expect_env("JUST_LLM_MODEL");

    // The match only gathers per-family credentials; the factory does the family dispatch.
    let (api_key, base_url) = match family.as_str() {
        #[cfg(feature = "deepseek")]
        "deepseek" => (
            expect_env("JUST_LLM_DEEPSEEK_API_KEY"),
            std::env::var("JUST_LLM_DEEPSEEK_BASE_URL").ok(),
        ),
        #[cfg(feature = "openai-compat")]
        "openai-compatible" => (
            expect_env("JUST_LLM_OPENAI_COMPAT_API_KEY"),
            Some(expect_env("JUST_LLM_OPENAI_COMPAT_BASE_URL")),
        ),
        _ => return Err(format!("unsupported JUST_LLM_PROVIDER: {family}").into()),
    };

    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .use_rustls_tls();
    let backend = BackendFactory::new().create(&family, http, &api_key, base_url.as_deref())?;
    Ok(ChatClient::new(
        backend,
        ChatClientOptions::new(model).with_system_prompt(system_prompt),
    ))
}
