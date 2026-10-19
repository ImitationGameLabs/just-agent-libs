# Provider type crates

The workspace includes four concrete provider type crates:

- `just-deepseek`
- `just-openai-compat`
- `just-openai-responses`
- `just-anthropic`

These crates provide wire-level DTOs, HTTP transport helpers, and thin client wrappers that closely
mirror the upstream provider API shapes.

If you want a provider-neutral abstraction, see [just-llm-client](just-llm-client.md).

Each crate enables HTTP/2 by default: `http2 = ["just-common/http2"]` together with
`default = ["http2"]`. That is a pass-through, not a decision of its own — a default build
negotiates h2 over TLS whenever the server does. The switch is build-level rather than per crate:
turning it on anywhere in one build turns it on for every provider in that build, so it cannot be
scoped to a single provider. To build without it, disable default features where you depend on the
crate, in a build where no other path turns it on; see [HTTP/2 in just-llm-client](just-llm-client.md#http2).

## DeepSeek types

- `types::chat` — Chat completion request/response types, messages, tool definitions
- `types::models` — Model listing response types
- `types::balance` — Balance and quota response types

## OpenAI-compatible types

- `types::chat` — Chat completion request/response types, messages, tool definitions
- `types::models` — Model listing response types

## OpenAI Responses types

- `types::request` — Response creation and compaction request types
- `types::response` — The `Response` object, usage, and error details
- `types::item` — Input and output items (messages, function calls, reasoning, ...)
- `types::event` — SSE streaming events
- `types::tool` — Tool definitions (function, custom, web search, file search)
- `types::models` — Model listing response types

## Anthropic types

- `types::message` — Message params/content blocks and the response `Message` object
- `types::request` — Message and token-counting request types
- `types::tool` — Tool definitions and tool-choice configuration
- `types::thinking` — Extended-thinking configuration
- `types::event` — SSE streaming events
- `types::usage` — Token accounting (including cache and thinking tokens)
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

## API key handling

Every builder takes the API key as a plain string and wraps it in `just_common::ApiKey` at the
boundary: the value is wiped from memory when dropped, `Debug` shows `ApiKey([REDACTED])`, and
there is no way to print or serialize it by accident. The only read is `ApiKey::expose`, called
inside the library exactly where the key is handed to the HTTP stack; treat the result as
transport input — never log, format or echo it. A key that is empty after trimming, or holds
characters that cannot go in an HTTP header, is rejected when the builder's `build` runs.

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

# OpenAI Responses examples
JUST_LLM_OPENAI_RESPONSES_API_KEY=your-openai-api-key
#JUST_LLM_OPENAI_RESPONSES_BASE_URL=https://api.openai.com/v1
JUST_LLM_OPENAI_RESPONSES_MODEL=gpt-5.6

# Anthropic examples
JUST_LLM_ANTHROPIC_API_KEY=your-anthropic-api-key
#JUST_LLM_ANTHROPIC_BASE_URL=https://api.anthropic.com/v1
JUST_LLM_ANTHROPIC_MODEL=claude-opus-5
```

## Runnable examples

```bash
cargo run -p just-llm-client --example deepseek_simple_chat
cargo run -p just-llm-client --example openai_compat_simple_chat
cargo run -p just-openai-responses --example responses_chat_completion
cargo run -p just-openai-responses --example responses_streaming
cargo run -p just-openai-responses --example responses_tool_calling
cargo run -p just-anthropic --example anthropic_chat
cargo run -p just-anthropic --example anthropic_streaming
cargo run -p just-anthropic --example anthropic_tool_calling
```

The `deepseek_simple_chat` and `openai_compat_simple_chat` examples use `just-llm-client`
backends that internally serialize into the provider DTOs from these crates; the
`responses_*` and `anthropic_*` examples use their provider client directly.
