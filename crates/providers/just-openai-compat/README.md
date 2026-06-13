# just-openai-compat

Wire-level types, a thin async client, and HTTP transport helpers for any OpenAI-compatible API.

## Overview

This crate provides serde-serializable request and response types that mirror the OpenAI chat completion API shape, a thin async client (`OpenAiCompatClient`), and re-exported HTTP transport helpers for making requests with a raw `reqwest::Client`.

- **[`types`]** — Wire-level DTOs (request/response types with serde derives).
- **[`transport`]** — Re-exported HTTP helpers from `just-common` (`build_client`, `endpoint_url`, `ensure_success`, `parse_json`, `JsonEventStream`).

For a provider-neutral abstraction with capability traits and runtime provider selection, use [**just-llm-client**] with the `openai-compat` feature.

[`types`]: https://docs.rs/just-openai-compat/latest/just_openai_compat/types
[`transport`]: https://docs.rs/just-openai-compat/latest/just_openai_compat/transport

## Quick start

### Type serialization

```rust
use just_openai_compat::types::chat::{ChatCompletionRequest, ChatMessage};

let request = ChatCompletionRequest::new(
    "gpt-4o",
    vec![
        ChatMessage::system("You are a concise assistant."),
        ChatMessage::user("Say hello in one sentence."),
    ],
);

let json = serde_json::to_string(&request)?;
```

### HTTP request with transport helpers

```rust
use just_openai_compat::transport::{build_client, endpoint_url, ensure_success};
use just_openai_compat::types::chat::{ChatCompletion, ChatCompletionRequest, ChatMessage};

let http = build_client(
    reqwest::Client::builder().use_rustls_tls(),
    "your-api-key",
)?;
let request = ChatCompletionRequest::new(
    "gpt-4o",
    vec![
        ChatMessage::system("You are a concise assistant."),
        ChatMessage::user("Say hello in one sentence."),
    ],
);

let url = endpoint_url("https://api.openai.com/v1", "/chat/completions")?;
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

- **Wire-level DTOs** — Request/response types mirror the OpenAI chat completion API shape with serde derives.
- **Tool calling** — `ToolDefinition` and convenience constructors for tool-calling workflows.
- **Broad compatibility** — Types for any service that implements the OpenAI chat completion surface.
- **Transport helpers** — Re-exported HTTP utilities from `just-common` for convenience. Use `just-llm-client` for a full client abstraction.

> Looking for a provider-neutral interface with capability traits, runtime provider selection,
> and tool dispatch? Check out [**just-llm-client**].

## Examples

```bash
# Non-streaming chat completion
JUST_LLM_OPENAI_COMPAT_API_KEY=your-key \
JUST_LLM_OPENAI_COMPAT_BASE_URL=https://your-endpoint/v1 \
JUST_LLM_OPENAI_COMPAT_MODEL=gpt-4.1-mini \
  cargo run -p just-openai-compat --example openai_compat_chat_completion

# Streaming
JUST_LLM_OPENAI_COMPAT_API_KEY=your-key \
JUST_LLM_OPENAI_COMPAT_BASE_URL=https://your-endpoint/v1 \
JUST_LLM_OPENAI_COMPAT_MODEL=gpt-4.1-mini \
  cargo run -p just-openai-compat --example openai_compat_streaming_chat_completion

# Tool calling loop
JUST_LLM_OPENAI_COMPAT_API_KEY=your-key \
JUST_LLM_OPENAI_COMPAT_BASE_URL=https://your-endpoint/v1 \
JUST_LLM_OPENAI_COMPAT_MODEL=gpt-4.1-mini \
  cargo run -p just-openai-compat --example openai_compat_tool_calling
```

## Ecosystem

| Crate                 | Description                                                |
| --------------------- | ---------------------------------------------------------- |
| [**just-llm-client**] | Provider-neutral LLM client — recommended entry point      |
| [just-deepseek]       | Wire-level DTOs and transport helpers for the DeepSeek API |
| [just-common]         | Shared HTTP transport, SSE parsing, and error types        |

[**just-llm-client**]: https://crates.io/crates/just-llm-client
[just-deepseek]: https://crates.io/crates/just-deepseek
[just-common]: https://crates.io/crates/just-common
