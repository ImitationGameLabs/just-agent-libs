# just-openai-compat

Rust client for any OpenAI-compatible API.

## Quick start

```toml
# Cargo.toml
[dependencies]
just-openai-compat = "0.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust,no_run
use just_openai_compat::{
    OpenAiCompatClient,
    types::chat::{ChatMessage, CreateChatCompletionRequest},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OpenAiCompatClient::new("your-api-key", "https://api.example.com/v1")?;

    let request = CreateChatCompletionRequest::new(
        "gpt-4o",
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
- **Tool calling** — Full tool-calling loop with `ToolDefinition` and convenience constructors.
- **Wire-level DTOs** — Request/response types mirror the OpenAI chat completion API shape.
- **Broad compatibility** — Works with any service that implements the OpenAI chat completion surface.

> Looking for a provider-neutral interface?
> Check out [**just-llm-client**] for a unified LLM client with capability traits,
> runtime provider selection, and tool dispatch.

## Ecosystem

| Crate                 | Description                                           |
| --------------------- | ----------------------------------------------------- |
| [**just-llm-client**] | Provider-neutral LLM client — recommended entry point |
| [just-deepseek]       | Rust client for the DeepSeek API                      |
| [just-common]         | Shared transport and error types                      |

[**just-llm-client**]: https://crates.io/crates/just-llm-client
[just-deepseek]: https://crates.io/crates/just-deepseek
[just-common]: https://crates.io/crates/just-common
