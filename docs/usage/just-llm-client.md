# LLM client layer

The `just-llm-client` crate provides a provider-neutral surface on top of the concrete provider type crates.

## Choosing a layer

The workspace intentionally keeps three adjacent but different entry points:

| Entry point                                                            | Use it when                                                                                       | What you get                                                 |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Provider type crate (`just-deepseek`, `just-openai-compat`)            | You want direct access to provider wire DTOs and do not need a shared abstraction                 | Serde-serializable request/response types                    |
| Direct backend construction (`DeepSeekBackend`, `OpenAiCompatBackend`) | You know the provider family in code but still want `just-llm-client` normalized types and traits | Lowest-noise path into the `just-llm-client` layer           |
| `BackendFactory`                                                      | You want to dispatch a family string to a backend constructor at runtime                          | A shared backend built from `(family, http, key, base_url)`  |

The direct backend path still belongs to the `just-llm-client` layer. It is "concrete backend, normalized
types", not "provider wire DTOs without the client layer".

## When to use it

- You want one code path that can target DeepSeek or an OpenAI-compatible endpoint.
- You want the prepare-send-parse pattern with full HTTP response access (headers like `retry-after`) before deserializing.
- You want optional capabilities to be negotiated explicitly at runtime instead of spread across
  broad generic bounds.

## Boundary trade-off

Some `just-llm-client` request and response types look very close to the provider crates' wire DTOs. That is
currently deliberate. The repository keeps those definitions independent so:

- provider crates stay free to evolve as wire-level type libraries
- `just-llm-client` stays free to evolve as a trait- and normalization-oriented layer
- future reviewers have an explicit record that this duplication is a boundary trade-off, not an
  accidentally missed shared abstraction

If that trade-off stops paying for itself, re-evaluate it when protocol changes begin to require
repeated cross-crate edits rather than just visually similar definitions.

## Initialization styles

The workspace supports two complementary initialization styles:

1. **Direct backend construction** for the clearest, least abstract setup when you already know the provider family.
2. **`BackendFactory` runtime dispatch** when your application selects a provider by family string at runtime.

| Style                       | Best when                                                                                                         | Tradeoff                                                                                                                 |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Direct backend construction | You already know the provider family in code and want the shortest path from config to normalized client requests | You write separate setup code per provider family                                                                        |
| `BackendFactory`            | Provider choice is configuration-driven and you want runtime dispatch from a family string                        | You construct a backend from `(family, http, key, base_url)`, then wrap it in a `ChatClient` with explicit model/system-prompt defaults |

Example environment:

```bash
# Direct path
JUST_LLM_DEEPSEEK_API_KEY=your-deepseek-api-key
JUST_LLM_DEEPSEEK_MODEL=deepseek-v4-flash

# Factory path
JUST_LLM_PROVIDER=deepseek
JUST_LLM_MODEL=deepseek-v4-flash
JUST_LLM_DEEPSEEK_API_KEY=your-deepseek-api-key
```

## Prepare-send-parse pattern

The `LlmBackend` trait supports preparing a request, sending it, and parsing it as separate steps,
giving callers full access to the HTTP response (including headers) before deserializing:

```rust
use just_llm_client::{
    LlmBackend,
    provider::OpenAiCompatBackend,
    types::chat::{ChatCompletionRequest, ChatMessage},
};

let backend = OpenAiCompatBackend::new(
    reqwest::Client::builder().use_rustls_tls(),
    "your-api-key",
    Some("https://your-compatible-endpoint/v1"),
)?;

let builder = backend.prepare(
    ChatCompletionRequest::new(
        "gpt-4.1-mini",
        vec![ChatMessage::user("Say hello in one sentence.")],
    )
    .with_system_prompt("You are a concise assistant."),
)?;

// Send and get the raw HTTP response with full header access.
let response = backend.send(builder).await?;
let retry_after = response.headers().get("retry-after");

// Deserialize into the normalized type via dyn dispatch on the backend.
let completion = backend.parse(response).await?;
```

For convenience, `chat_completion()` and `stream_chat_completion()` compose prepare + send + parse
into a single call.

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
