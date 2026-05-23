# just-deepseek

Rust client for the DeepSeek API.

## Quick start

```toml
# Cargo.toml
[dependencies]
just-deepseek = "0.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust,no_run
use just_deepseek::{
    DeepSeekClient,
    types::chat::{ChatMessage, CreateChatCompletionRequest},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = DeepSeekClient::new("your-api-key")?;

    let request = CreateChatCompletionRequest::new(
        "deepseek-chat",
        vec![
            ChatMessage::system("You are a concise assistant."),
            ChatMessage::user("Say hello in one sentence."),
        ],
    );

    let response = client.create_chat_completion(request).await?;
    let choice = response.choices.first().expect("expected at least one choice");

    println!("{}", choice.message.content.as_deref().unwrap_or_default());
    Ok(())
}
```

## Highlights

- **Streaming** — `stream_chat_completion()` returns a `Stream` of typed chunks.
- **Thinking mode** — Enable extended reasoning with `ThinkingConfig` and `ReasoningEffort`.
- **Tool calling** — Full tool-calling loop with `ToolDefinition` and `ToolCallsMessage`.
- **Wire-level DTOs** — Request/response types mirror the upstream DeepSeek API shape.

> Looking for a provider-neutral interface?
> Check out [**just-llm-client**] for a unified LLM client with capability traits,
> runtime provider selection, and tool dispatch.

## Ecosystem

| Crate                 | Description                                           |
| --------------------- | ----------------------------------------------------- |
| [**just-llm-client**] | Provider-neutral LLM client — recommended entry point |
| [just-openai-compat]  | Rust client for any OpenAI-compatible API             |
| [just-common]         | Shared transport and error types                      |

[**just-llm-client**]: https://crates.io/crates/just-llm-client
[just-openai-compat]: https://crates.io/crates/just-openai-compat
[just-common]: https://crates.io/crates/just-common
