# just-rs

Not an agent framework, not a platform — just the LLM client. Minimal, well-abstracted, and extensible.

## Architecture

### Provider-neutral client — `just-llm-client`

A lightweight, provider-neutral abstraction that sits on top of the provider SDKs. Use it when you want one code path that can target multiple providers, or when you want prepared requests, token estimation, and capability negotiation.

- **Capability-oriented traits.** Each operation is its own trait — `ChatCompletion`, `StreamingChatCompletion`, `ModelCatalog`, `Balance`, `TokenEstimation`. Backends implement only what they support.
- **Explicit capability negotiation.** Optional capabilities are requested upfront — unsupported backends fail immediately, not at call time.
- **Prepared requests.** Build, inspect, estimate token usage, then execute.
- **Optional tool runtime.** Enable `tools` for the local executable-tool runtime plus the built-in PTY-backed shell/session tools that compose into tool-calling loops.
```rust
use just_llm_client::{
    ChatCompletion, provider::DeepSeekBackend,
    types::chat::{ChatCompletionRequest, ChatMessage},
};

let backend = DeepSeekBackend::with_config(just_deepseek::DeepSeekConfig::new("your-api-key"))?;
let response = backend.create_chat_completion(
    ChatCompletionRequest::new(
        "deepseek-v4-flash",
        vec![ChatMessage::user("Say hello.")],
    ),
).await?;
```

When you want reusable local tools next to the client layer:

```toml
just-llm-client = { version = "...", features = ["openai-compat", "tools"] }
```

The workspace also includes a deliberately tiny reference binary, `just-agent`, that wires
`ProviderRegistry` to a modular shell-oriented tool set and runs a basic tool-calling loop with a
small built-in policy gate. It also performs conservative pre-turn context compaction, keeping
recent turns verbatim and summarizing older history when the conversation grows too large.

#### Bring your own backend

just-rs aims to support more model providers over time. But if your provider is not yet covered, or you are a model provider with a custom API that does not follow any well-known protocol, you can easily build your own backend by implementing the capability traits. `LlmBackend` is a convenience trait that requires `Identifiable + ChatCompletion + StreamingChatCompletion + CapabilityNegotiation`. Implement the core traits, then opt into optional capabilities by overriding the default `CapabilityNegotiation` methods:

```rust
struct MyBackend { /* ... */ }

#[async_trait]
impl Identifiable for MyBackend {
    fn backend_id(&self) -> &'static str { "my-backend" }
}

#[async_trait]
impl ChatCompletion for MyBackend {
    async fn create_chat_completion(
        &self, request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> { /* ... */ }

    async fn prepared_request(
        &self, request: ChatCompletionRequest,
    ) -> Result<PreparedChatRequest, LlmError> { /* ... */ }

    async fn send_prepared(
        &self, request: &PreparedChatRequest,
    ) -> Result<ChatCompletionResponse, LlmError> { /* ... */ }
}

// Implement StreamingChatCompletion similarly, then:
// MyBackend now satisfies LlmBackend automatically.
```

The validation module (`just_llm_client::provider::validation`) provides reusable helpers for building custom backends.

### Provider-specific SDKs — direct API bindings

Direct Rust bindings to model provider HTTP APIs — wire-level DTOs, no extra abstraction. Use these when you need full control over a specific provider's wire protocol, or as building blocks for your own neutral layer or agent framework.

| Crate                | Description                                                    |
| -------------------- | -------------------------------------------------------------- |
| `just-deepseek`      | DeepSeek API — chat completions, model listing, balance lookup |
| `just-openai-compat` | Any OpenAI-compatible API — chat completions, model listing    |

Bindings for OpenAI, Google, xAI, Anthropic, and others are planned but deferred until needed. If you urgently need a specific provider, feel free to open an issue so we can prioritize it.

```rust
use just_deepseek::{DeepSeekClient, types::chat::{ChatMessage, CreateChatCompletionRequest}};

let client = DeepSeekClient::new("your-api-key")?;
let response = client.create_chat_completion(
    CreateChatCompletionRequest::new(
        "deepseek-v4-flash",
        vec![ChatMessage::user("Say hello.")],
    ),
).await?;
```

## Documentation

- [just-llm-client](docs/usage/just-llm-client.md) — capability model, initialization styles, and prepared requests
- [Provider-specific clients](docs/usage/provider-specific-clients.md) — direct SDK usage, streaming, and environment variables

## Quick start

```bash
# Provider-neutral client examples
cargo run -p just-llm-client --example deepseek_simple_chat
cargo run -p just-llm-client --example openai_compat_simple_chat
cargo run -p just-llm-client --example runtime_selected_provider

# Minimal reference agent (requires JUST_LLM_* env vars)
# Exposes shell_session_* tools with lightweight approval gating
# Optional compaction tuning:
# JUST_AGENT_COMPACT_TRIGGER_TOKENS=12000
# JUST_AGENT_COMPACT_KEEP_RECENT_TOKENS=4000
# JUST_AGENT_COMPACT_MAX_TOKENS=1200
cargo run -p just-agent --example run-agent-with-prompt -- --workspace=. -- --prompt "Show the current working directory."

# Provider-specific SDK examples
cargo run -p just-deepseek --example deepseek_chat_completion
cargo run -p just-openai-compat --example openai_compat_chat_completion
```
