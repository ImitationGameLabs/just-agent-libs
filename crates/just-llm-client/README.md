# just-llm-client

Just a lightweight, composable, and minimal LLM client — not an agent framework.

## Quick start

```toml
# Cargo.toml
[dependencies]
just-llm-client = "0.3"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust,no_run
use just_llm_client::{
    LlmBackend,
    provider::DeepSeekBackend,
    types::generation::{GenerationRequest, Message},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = DeepSeekBackend::new(
        reqwest::Client::builder().use_rustls_tls(),
        "your-api-key",
        None,
    )?;

    let response = backend
        .generate(
            GenerationRequest::new(
                "deepseek-chat",
                vec![Message::user("Say hello in one sentence.")],
            )
            .with_system_prompt("You are a concise assistant."),
        )
        .await?;

    println!("{}", response.text().unwrap_or_default());
    Ok(())
}
```

## Stateful conversations

[`Conversation`] continues multi-turn conversations on Responses-family backends (OpenAI, xAI) by
sending only the new messages plus `previous_response_id`, instead of re-transmitting the stored
prefix on every turn. The caller owns the message list and passes the full logical context each
turn; the `Conversation` decides the wire payload:

```rust,no_run
use just_llm_client::{
    GenerationClient, GenerationClientOptions, LlmBackend,
    provider::OpenAiResponsesBackend,
    types::generation::Message,
};

let backend = OpenAiResponsesBackend::new(
    reqwest::Client::builder().use_rustls_tls(),
    "your-api-key",
    None,
)?;
let client = GenerationClient::new(
    backend,
    GenerationClientOptions::new("gpt-5.6").with_system_prompt("You are concise."),
);
let mut conv = client.conversation();

let first = conv.generate(client.create_request(vec![Message::user("Hi")])).await?;
let mirror = conv.last_message().expect("first turn produced a message");
let second = conv.generate(client.create_request(vec![
    Message::user("Hi"),
    mirror,
    Message::user("And now?"),
])).await?;
```

On stateless backends (chat completions, Anthropic) the same code degrades to a full-resend replay,
so one code path works across all families. Streaming turns assemble the assistant message from the
events via [`Conversation::last_message`] and adopt it as the next anchor on completion. See
`examples/conversation.rs` for a runnable example.

## Feature flags

| Feature         | Default | Description                                     |
| --------------- | ------- | ----------------------------------------------- |
| `deepseek`      | yes     | Enables the [`just-deepseek`] backend           |
| `openai-compat` | yes     | Enables the [`just-openai-compat`] backend      |
| `responses`     | no      | Enables the [`just-openai-responses`] backend   |
| `anthropic`     | no      | Enables the [`just-anthropic`] backend          |
| `http2`         | yes     | Enables HTTP/2 support in the transport         |

The DeepSeek and OpenAI-compatible backends are enabled by default, as is `http2`: a default
build negotiates h2 over TLS when the server does. Disable default features and enable only what you need:

```toml
[dependencies]
just-llm-client = { version = "0.2", default-features = false, features = ["deepseek"] }
```

## Ecosystem

| Crate                | Description                                     |
| -------------------- | ----------------------------------------------- |
| [just-deepseek]      | DeepSeek API client + wire-level types          |
| [just-openai-compat] | OpenAI-compatible API client + wire-level types |
| [just-openai-responses] | OpenAI Responses API client + wire-level types |
| [just-anthropic]     | Anthropic Messages API client + wire-level types |
| [just-common]        | Shared HTTP transport and error types           |

[just-deepseek]: https://crates.io/crates/just-deepseek
[just-openai-compat]: https://crates.io/crates/just-openai-compat
[just-openai-responses]: https://crates.io/crates/just-openai-responses
[just-anthropic]: https://crates.io/crates/just-anthropic
[just-common]: https://crates.io/crates/just-common
