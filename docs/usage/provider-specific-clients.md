# Provider clients

The workspace includes two concrete provider SDK crates:

- `just-deepseek`
- `just-openai-compat`

Use these crates when you already know the target provider family and want direct access to its
wire-level request and response types.

If you want a provider-neutral abstraction instead, see [Provider-neutral client](provider-neutral-client.md).
If you want `just-llm-client`'s normalized types but do not need runtime provider selection, use the direct
backend examples in the `just-llm-client` crate instead of these provider clients.

## DeepSeek endpoints

- `POST /chat/completions`
- `GET /models`
- `GET /user/balance`

## Direct client example

```rust
use just_deepseek::{
    DeepSeekClient,
    types::chat::{ChatMessage, CreateChatCompletionRequest},
};

#[tokio::main]
async fn main() -> Result<(), just_deepseek::Error> {
    let client = DeepSeekClient::new("your-api-key")?;

    let request = CreateChatCompletionRequest::new(
        "deepseek-v4-pro",
        vec![
            ChatMessage::system("You are a concise assistant."),
            ChatMessage::user("Say hello in one sentence."),
        ],
    );

    let response = client.create_chat_completion(request).await?;
    println!("{}", response.choices[0].message.content.as_deref().unwrap_or_default());
    Ok(())
}
```

## Streaming example

```rust
use just_deepseek::{
    DeepSeekClient,
    types::chat::{ChatMessage, CreateChatCompletionRequest},
};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), just_deepseek::Error> {
    let client = DeepSeekClient::new("your-api-key")?;
    let request = CreateChatCompletionRequest::new(
        "deepseek-v4-pro",
        vec![ChatMessage::user("Stream a short answer.")],
    );

    let mut stream = client.stream_chat_completion(request).await?;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;

        for choice in chunk.choices {
            if let Some(content) = choice.delta.content {
                print!("{content}");
            }
        }
    }

    Ok(())
}
```

## Environment variables for examples

```bash
# DeepSeek examples
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_PROMPT="Say hello in one sentence."

# OpenAI-compatible examples
OPENAI_COMPATIBLE_API_KEY=your-openai-compatible-api-key
OPENAI_COMPATIBLE_BASE_URL=https://api.openai.com/v1
OPENAI_COMPATIBLE_MODEL=
OPENAI_COMPATIBLE_PROMPT="Say hello in one sentence."
```

## Runnable examples

```bash
cargo run -p just-deepseek --example deepseek_chat_completion
cargo run -p just-openai-compat --example openai_compat_chat_completion
```

The direct client-backend examples live alongside these provider examples:

```bash
cargo run -p just-llm-client --example deepseek_simple_chat
cargo run -p just-llm-client --example openai_compat_simple_chat
```

Those examples intentionally sit in the middle:

- provider clients: provider-native wire DTOs
- direct client backends: normalized `just-llm-client` types with a concrete backend known in code
- `ProviderRegistry`: normalized `just-llm-client` types with programmatic provider registration and runtime selection by id
