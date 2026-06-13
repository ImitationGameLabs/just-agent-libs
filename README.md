# just-agent-libs

Not an agent framework, not a platform — just the LLM client. Minimal, well-abstracted, and extensible.

## Architecture

### Provider-neutral client — `just-llm-client`

A lightweight, provider-neutral abstraction that sits on top of the provider type crates. Use it when you want one code path that can target multiple providers, or when you want prepare-send-parse patterns and capability negotiation.

- **Capability-oriented traits.** Each operation is its own trait — `ModelCatalog`, `Balance`. Backends implement only what they support, with chat completion provided by the unified `LlmBackend` trait.
- **Explicit capability negotiation.** Optional capabilities are requested upfront — unsupported backends fail immediately, not at call time.
- **Prepare-send-parse pattern.** Build a `reqwest::Request`, optionally inspect/modify it, then send, then parse. Callers get full access to the HTTP response including headers (`retry-after`, `x-ratelimit-*`) before deserializing.

```rust
use just_llm_client::{
    build_client,
    provider::{DeepSeekBackend, LlmBackend},
    types::chat::{ChatCompletionRequest, ChatMessage},
};

let http = build_client(
    reqwest::Client::builder().use_rustls_tls(),
    "your-api-key",
)?;
let backend = DeepSeekBackend::new(http, "https://api.deepseek.com".to_owned());
let response = backend.chat_completion(
    ChatCompletionRequest::new(
        "deepseek-v4-flash",
        vec![ChatMessage::user("Say hello.")],
    ),
).await?;
```

#### Bring your own backend

just-agent-libs aims to support more model providers over time. But if your provider is not yet covered, or you are a model provider with a custom API that does not follow any well-known protocol, you can easily build your own backend by implementing the `LlmBackend` trait. It requires `Identifiable + CapabilityNegotiation + Send + Sync` and seven methods: `prepare`, `prepare_streaming`, `send`, `parse`, `parse_streaming`, `render_messages`, `render_tools`. (`chat_completion` and `stream_chat_completion` have default implementations that compose `prepare` + `send` + `parse`, so override them only for non-HTTP backends.)

```rust
struct MyBackend { /* ... */ }

impl Identifiable for MyBackend {
    fn backend_id(&self) -> &'static str { "my-backend" }
}

impl CapabilityNegotiation for MyBackend {}

#[async_trait]
impl LlmBackend for MyBackend {
    fn prepare(&self, request: ChatCompletionRequest)
        -> Result<reqwest::Request, LlmError> { /* ... */ }

    fn prepare_streaming(&self, request: ChatCompletionRequest)
        -> Result<reqwest::Request, LlmError> { /* ... */ }

    async fn send(&self, prepared: reqwest::Request)
        -> Result<reqwest::Response, LlmError> { /* ... */ }

    async fn parse(&self, response: reqwest::Response)
        -> Result<ChatCompletionResponse, LlmError> { /* ... */ }

    async fn parse_streaming(&self, response: reqwest::Response)
        -> Result<ChatCompletionStream, LlmError> { /* ... */ }

    fn render_messages(&self, messages: &[ChatMessage])
        -> Result<String, LlmError> { /* ... */ }

    fn render_tools(&self, tools: &[ToolDefinition])
        -> Result<String, LlmError> { /* ... */ }

    // chat_completion and stream_chat_completion have default impls
    // (prepare + send + parse); override only for non-HTTP backends.
}
```

The validation module (`just_llm_client::provider::validation`) provides reusable helpers for building custom backends.

### Provider type crates

Wire-level request/response types with serde derives, plus a thin async client and re-exported
HTTP transport helpers. Use these when you need full control over a specific provider's wire
protocol, or as building blocks for your own client layer or agent framework.

| Crate                | Description                                                                |
| -------------------- | -------------------------------------------------------------------------- |
| `just-deepseek`      | DeepSeek API client + wire-level types — chat completions, models, balance |
| `just-openai-compat` | OpenAI-compatible API client + wire-level types — chat completions, models |

Bindings for OpenAI, Google, xAI, Anthropic, and others are planned but deferred until needed. If you urgently need a specific provider, feel free to open an issue so we can prioritize it.

## Documentation

- [just-llm-client](docs/usage/just-llm-client.md) — capability model, initialization styles, and prepare-send-parse pattern
- [Provider type crates](docs/usage/provider-type-crates.md) — wire-level DTOs and environment variables

## Quick start

```bash
# Provider-neutral client examples
cargo run -p just-llm-client --example deepseek_simple_chat
cargo run -p just-llm-client --example openai_compat_simple_chat
cargo run -p just-llm-client --example runtime_selected_provider
```
