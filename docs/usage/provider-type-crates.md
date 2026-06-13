# Provider type crates

The workspace includes two concrete provider type crates:

- `just-deepseek`
- `just-openai-compat`

These crates provide wire-level DTOs, HTTP transport helpers, and thin client wrappers that closely
mirror the upstream provider API shapes.

If you want a provider-neutral abstraction, see [just-llm-client](just-llm-client.md).

## DeepSeek types

- `types::chat` — Chat completion request/response types, messages, tool definitions
- `types::models` — Model listing response types
- `types::balance` — Balance and quota response types

## OpenAI-compatible types

- `types::chat` — Chat completion request/response types, messages, tool definitions
- `types::models` — Model listing response types

## Usage

```rust
use just_deepseek::types::chat::{ChatCompletionRequest, ChatMessage};

let request = ChatCompletionRequest::new(
    "deepseek-v4-pro",
    vec![
        ChatMessage::system("You are a concise assistant."),
        ChatMessage::user("Say hello in one sentence."),
    ],
);

let json = serde_json::to_string_pretty(&request)?;
println!("{json}");
```

Pair with `just-llm-client` for HTTP transport and provider-neutral abstractions.

## Environment variables for examples

```bash
# DeepSeek examples
JUST_LLM_DEEPSEEK_API_KEY=your-deepseek-api-key
#JUST_LLM_DEEPSEEK_BASE_URL=https://api.deepseek.com
JUST_LLM_DEEPSEEK_MODEL=deepseek-v4-flash

# OpenAI-compatible examples
JUST_LLM_OPENAI_COMPAT_API_KEY=your-openai-compatible-api-key
JUST_LLM_OPENAI_COMPAT_BASE_URL=https://your-compatible-endpoint/v1
JUST_LLM_OPENAI_COMPAT_MODEL=gpt-4.1-mini
```

## Runnable examples

```bash
cargo run -p just-llm-client --example deepseek_simple_chat
cargo run -p just-llm-client --example openai_compat_simple_chat
```

Those examples use `just-llm-client` backends that internally serialize into the provider DTOs from these crates.
