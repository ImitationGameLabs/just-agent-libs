# just-llm-client

Just a lightweight, composable, and minimal LLM client — not an agent framework.

## Quick start

```toml
# Cargo.toml
[dependencies]
just-llm-client = "0.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust,no_run
use just_llm_client::{
    ChatCompletion,
    provider::DeepSeekBackend,
    types::chat::{ChatCompletionRequest, ChatMessage},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = just_deepseek::DeepSeekClient::builder()
        .api_key("your-api-key")
        .build()?;
    let backend = DeepSeekBackend::new(client);

    let response = backend
        .chat_completion(
            ChatCompletionRequest::new(
                "deepseek-chat",
                vec![ChatMessage::user("Say hello in one sentence.")],
            )
            .with_system_prompt("You are a concise assistant."),
        )
        .await?;

    println!("{}", response.first_choice_content().unwrap_or_default());
    Ok(())
}
```

## Feature flags

| Feature         | Default | Description                                |
| --------------- | ------- | ------------------------------------------ |
| `deepseek`      | yes     | Enables the [`just-deepseek`] backend      |
| `openai-compat` | yes     | Enables the [`just-openai-compat`] backend |

Both providers are enabled by default. Disable default features and enable only what you need:

```toml
[dependencies]
just-llm-client = { version = "0.1", default-features = false, features = ["deepseek"] }
```

## Ecosystem

| Crate                | Description                               |
| -------------------- | ----------------------------------------- |
| [just-deepseek]      | Rust client for the DeepSeek API          |
| [just-openai-compat] | Rust client for any OpenAI-compatible API |
| [just-common]        | Shared transport and error types          |

[just-deepseek]: https://crates.io/crates/just-deepseek
[just-openai-compat]: https://crates.io/crates/just-openai-compat
[just-common]: https://crates.io/crates/just-common
