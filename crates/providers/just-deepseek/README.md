# just-deepseek

Wire-level types, a thin async client, and HTTP transport helpers for the DeepSeek API.

## Overview

This crate provides serde-serializable request and response types that closely mirror the DeepSeek HTTP API shape, a thin async client (`DeepSeekClient`), and re-exported HTTP transport helpers for making requests with a raw `reqwest::Client`.

- **[`types`]** — Wire-level DTOs (request/response types with serde derives).
- **[`transport`]** — Re-exported HTTP helpers from `just-common` (`build_client`, `endpoint_url`, `ensure_success`, `parse_json`, `JsonEventStream`).

For a provider-neutral abstraction with capability traits and runtime provider selection, use [**just-llm-client**] with the `deepseek` feature.

[`types`]: https://docs.rs/just-deepseek/latest/just_deepseek/types
[`transport`]: https://docs.rs/just-deepseek/latest/just_deepseek/transport

## Quick start

### Type serialization

```rust
use just_deepseek::types::chat::{ChatCompletionRequest, ChatMessage};

let request = ChatCompletionRequest::new(
    "deepseek-chat",
    vec![
        ChatMessage::system("You are a concise assistant."),
        ChatMessage::user("Say hello in one sentence."),
    ],
);

let json = serde_json::to_string(&request)?;
```

### HTTP request with transport helpers

```rust
use just_deepseek::transport::{build_client, endpoint_url, ensure_success};
use just_deepseek::types::chat::{ChatCompletion, ChatCompletionRequest, ChatMessage};

let http = build_client(
    reqwest::Client::builder().use_rustls_tls(),
    "your-api-key",
)?;
let request = ChatCompletionRequest::new(
    "deepseek-chat",
    vec![
        ChatMessage::system("You are a concise assistant."),
        ChatMessage::user("Say hello in one sentence."),
    ],
);

let url = endpoint_url("https://api.deepseek.com", "/chat/completions")?;
let response = http
    .post(&url)
    .header(reqwest::header::CONTENT_TYPE, "application/json")
    .body(serde_json::to_vec(&request)?)
    .send()
    .await?;
let response = ensure_success(response).await?;
let completion: ChatCompletion = response.json().await?;
```

## Highlights

- **Wire-level DTOs** — Request/response types mirror the upstream DeepSeek API shape with serde derives.
- **Thinking mode** — `ThinkingConfig` and `ReasoningEffort` fields for extended reasoning.
- **Tool calling** — `ToolDefinition`, `ToolCallsMessage`, and related types for tool-calling workflows.
- **Transport helpers** — Re-exported HTTP utilities from `just-common` for convenience. Use `just-llm-client` for a full client abstraction.

> Looking for a provider-neutral interface with capability traits, runtime provider selection,
> and tool dispatch? Check out [**just-llm-client**].

## Examples

```bash
# Non-streaming chat completion
JUST_LLM_DEEPSEEK_API_KEY=your-key JUST_LLM_DEEPSEEK_MODEL=deepseek-chat \
  cargo run -p just-deepseek --example chat_completion

# Streaming
JUST_LLM_DEEPSEEK_API_KEY=your-key JUST_LLM_DEEPSEEK_MODEL=deepseek-chat \
  cargo run -p just-deepseek --example streaming_chat_completion

# Thinking mode (DeepSeek-specific)
JUST_LLM_DEEPSEEK_API_KEY=your-key JUST_LLM_DEEPSEEK_MODEL=deepseek-chat \
  cargo run -p just-deepseek --example thinking_mode

# Tool calling loop
cargo run -p just-deepseek --example tool_calling
```

## Ecosystem

| Crate                 | Description                                                      |
| --------------------- | ---------------------------------------------------------------- |
| [**just-llm-client**] | Provider-neutral LLM client — recommended entry point            |
| [just-openai-compat]  | Wire-level DTOs and transport helpers for OpenAI-compatible APIs |
| [just-common]         | Shared HTTP transport, SSE parsing, and error types              |

[**just-llm-client**]: https://crates.io/crates/just-llm-client
[just-openai-compat]: https://crates.io/crates/just-openai-compat
[just-common]: https://crates.io/crates/just-common
