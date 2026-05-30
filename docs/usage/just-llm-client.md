# LLM client layer

The `just-llm-client` crate provides a provider-neutral surface on top of the concrete provider SDK crates.

## Choosing a layer

The workspace intentionally keeps three adjacent but different entry points:

| Entry point                                                            | Use it when                                                                                       | What you get                                                 |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Provider client crate (`just-deepseek`, `just-openai-compat`)          | You want direct access to provider wire DTOs and do not need a shared abstraction                 | Provider-native request/response types and behavior          |
| Direct backend construction (`DeepSeekBackend`, `OpenAiCompatBackend`) | You know the provider family in code but still want `just-llm-client` normalized types and traits | Lowest-noise path into the `just-llm-client` layer           |
| `ProviderRegistry`                                                     | You want to register a few configured provider entries and request `ChatClient`s by id at runtime | A shared-backend `ChatClient` surface with per-call defaults |

The direct backend path still belongs to the `just-llm-client` layer. It is "concrete backend, normalized
types", not "provider wire DTOs without the client layer".

## When to use it

- You want one code path that can target DeepSeek or an OpenAI-compatible endpoint.
- You want prepared requests or provider-neutral response types.
- You want optional capabilities to be negotiated explicitly at runtime instead of spread across
  broad generic bounds.

## Boundary trade-off

Some `just-llm-client` request and response types look very close to the provider crates' wire DTOs. That is
currently deliberate. The repository keeps those definitions independent so:

- provider crates stay free to evolve as wire-level SDKs
- `just-llm-client` stays free to evolve as a trait- and normalization-oriented layer
- future reviewers have an explicit record that this duplication is a boundary trade-off, not an
  accidentally missed shared abstraction

If that trade-off stops paying for itself, re-evaluate it when protocol changes begin to require
repeated cross-crate edits rather than just visually similar definitions.

## Initialization styles

The workspace supports two complementary initialization styles:

1. **Direct backend construction** for the clearest, least abstract setup when you already know the provider family.
2. **`ProviderRegistry` runtime selection** when your application builds a few configured provider entries and chooses among them by id.

| Style                       | Best when                                                                                                         | Tradeoff                                                                                                                 |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Direct backend construction | You already know the provider family in code and want the shortest path from config to normalized client requests | You write separate setup code per provider family                                                                        |
| `ProviderRegistry`          | Provider choice is configuration-driven and you want one runtime-selected entry point                             | You register configured providers programmatically, then derive `ChatClient`s with explicit model/system-prompt defaults |

That example defines:

- `build_direct_openai_backend()` for the concrete path
- `build_registry_client()` for the registry path

Example environment:

```bash
# Select which function path the example exercises
INIT_STYLE=direct
INIT_MODEL=gpt-4.1-mini

# Direct path
JUST_LLM_OPENAI_COMPAT_API_KEY=your-openai-compatible-api-key
JUST_LLM_OPENAI_COMPAT_BASE_URL=https://your-compatible-endpoint/v1

# Registry path
JUST_LLM_PROVIDER=openai-compatible
JUST_LLM_MODEL=gpt-4.1-mini
JUST_LLM_OPENAI_COMPAT_API_KEY=your-openai-compatible-api-key
JUST_LLM_OPENAI_COMPAT_BASE_URL=https://your-compatible-endpoint/v1
```

## Prepared request flow

The `just-llm-client` crate supports preparing, inspecting, and later executing a request:

```rust
use just_llm_client::{
    ChatCompletion,
    provider::OpenAiCompatBackend,
    types::chat::{ChatCompletionRequest, ChatMessage},
};

#[tokio::main]
async fn main() -> Result<(), just_llm_client::LlmError> {
    let backend = OpenAiCompatBackend::with_base_url(
        "your-api-key",
        "https://your-compatible-endpoint/v1",
    )?;

    let prepared = backend
        .prepared_request(
            ChatCompletionRequest::new(
                "gpt-4.1-mini",
                vec![ChatMessage::user("Say hello in one sentence.")],
            )
            .with_system_prompt("You are a concise assistant."),
        )
        .await?;

    println!("{}", prepared.request_body_text());
    println!("{:?}", prepared.preview());

    let response = backend.send_prepared(&prepared).await?;
    println!("{}", response.first_choice_content().unwrap_or_default());
    Ok(())
}
```

`PreparedChatRequest` stays as pure data. That keeps it easy to serialize, persist, or move across
threads/process boundaries; execution happens explicitly through a backend or client.

Streaming follows the same shape through `StreamingChatCompletion`:
direct streaming, `prepared_streaming_request(...)`, and `send_prepared_stream(...)`.

## Optional local tools

`just-llm-client` keeps request/response tool DTOs in the core crate, and exposes application-side
tool execution as an opt-in runtime:

- `tools` — object-safe `LlmTool` trait plus `ToolDispatcher`, together with the built-in
  PTY-backed shell/session tools, for turning local tools into `ToolDefinition` values and
  dispatching tool calls by name

Example dependency:

```toml
just-llm-client = { version = "...", features = ["openai-compat", "tools"] }
```

Example composition:

```rust,ignore
use std::sync::Arc;

use just_llm_client::{
    ToolDispatcher,
    tools::shell::{PtyBackend, shell_tool_set},
};
use tokio::sync::Mutex;

let backend = Arc::new(Mutex::new(PtyBackend::new("main").await?));
let mut tools = ToolDispatcher::new();
tools.add_tools(shell_tool_set(backend))?;

let tool_definitions = tools.tool_definitions();
```

That lets you keep the provider-neutral request/response flow in `just-llm-client`, while opting
into a small reusable tool runtime only when your application needs it.

## Runtime-selected provider example

The workspace also includes:

- `cargo run -p just-llm-client --example runtime_selected_provider`
- `cargo run -p just-llm-client --example deepseek_simple_chat`
- `cargo run -p just-llm-client --example openai_compat_simple_chat`

Those examples are intentionally complementary:

- `deepseek_simple_chat` / `openai_compat_simple_chat` show the lowest-noise concrete path into the `just-llm-client` layer
- `runtime_selected_provider` shows the shared registry-backed path

## Capability negotiation

The shared `LlmBackend` surface keeps always-on operations such as chat completion
directly callable, and routes optional operations through `CapabilityNegotiation`. A successful
negotiation returns a handle like `&dyn ModelCatalog`; unsupported
backends fail at negotiation time instead of inside the capability method.
