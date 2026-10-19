# LLM client layer

The `just-llm-client` crate provides a provider-neutral surface on top of the concrete provider type crates.

## Choosing a layer

The workspace intentionally keeps three adjacent but different entry points:

| Entry point                                                            | Use it when                                                                                       | What you get                                                 |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Provider type crate (`just-deepseek`, `just-openai-compat`, `just-openai-responses`) | You want direct access to provider wire DTOs and do not need a shared abstraction | Serde-serializable request/response types |
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
| `BackendFactory`            | Provider choice is configuration-driven and you want runtime dispatch from a family string                        | You construct a backend from `(family, http, key, base_url)`, then wrap it in a `GenerationClient` with explicit model/system-prompt defaults |

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
    types::generation::{GenerationRequest, Message},
};

let backend = OpenAiCompatBackend::new(
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .read_timeout(std::time::Duration::from_secs(120))
        .use_rustls_tls(),
    "your-api-key",
    Some("https://your-compatible-endpoint/v1"),
)?;

let builder = backend.prepare(
    GenerationRequest::new(
        "gpt-4.1-mini",
        vec![Message::user("Say hello in one sentence.")],
    )
    .with_system_prompt("You are a concise assistant."),
)?;

// Send and get the raw HTTP response with full header access.
let response = backend.send(builder).await?;
let retry_after = response.headers().get("retry-after");

// Deserialize into the normalized type via dyn dispatch on the backend.
let generation = backend.parse(response).await?;
```

For convenience, `generate()` and `stream_generate()` compose prepare + send + parse
into a single call.

## Response metadata

Non-streaming responses may carry a `meta` field (`ResponseMeta`) with two optional,
best-effort observations filled in by the library:

- `request_id` is the provider's request identifier, extracted from the response headers when
  present (`x-request-id` on OpenAI-compatible and DeepSeek endpoints, `request-id` on Anthropic).
- `latency` is the measured end-to-end call duration, attached by `generate()` and by
  `Conversation::generate()`.

Providers may omit the headers, so both fields are optional and their absence is normal. The
manual prepare-send-parse path carries `request_id` but no `latency`; time the call yourself
there if you need it. A response is never rejected for missing metadata.

## Runtime-selected provider example

The workspace also includes:

- `cargo run -p just-llm-client --example runtime_selected_provider`
- `cargo run -p just-llm-client --example deepseek_simple_chat`
- `cargo run -p just-llm-client --example openai_compat_simple_chat`

Those examples are intentionally complementary:

- `deepseek_simple_chat` / `openai_compat_simple_chat` show the lowest-noise concrete path into the `just-llm-client` layer
- `runtime_selected_provider` shows the shared registry-backed path

## Capability negotiation

The shared `LlmBackend` surface keeps always-on operations such as generation
directly callable, and routes optional operations through `CapabilityNegotiation`. A successful
negotiation returns a handle like `&dyn ModelCatalog`; unsupported
backends fail at negotiation time instead of inside the capability method.

## HTTP/2

HTTP/2 is a Cargo feature named `http2`, and it is on by default: both the provider crates and
`just-llm-client` enable it, so a default build negotiates `h2` over TLS and falls back to
`http/1.1` when the server does not. `just-common` alone leaves it off, because it is the layer
that should not pick a protocol on your behalf.

Nothing in the API changes; the wire behavior does:

- **ALPN** — TLS requests advertise `h2` alongside `http/1.1`. A server that negotiates only
  `http/1.1` behaves exactly as before, drain included.
- **Multiplexing** — concurrent calls to one host share a connection instead of each opening its
  own, so the connection count is no longer a lever for concurrency and `pool_max_idle_per_host`
  counts idle connections. `read_timeout` applies per response body; `connect_timeout` covers
  the shared connection setup, which h2 reuses across requests. See [timeouts](timeouts.md).
- **Cancellation** — dropping a response resets that one stream (`RST_STREAM`) instead of closing
  the connection, so sibling calls keep running. The reset can reach the peer after your drop, so
  do not rely on cancellation to stop work elsewhere.
- **Errors** — connection-level shutdowns (`GOAWAY`) and stream resets surface as request errors
  of their own, not as the `http/1.1` incomplete-message family.
- **Proxies and gateways** — HTTP/2 is negotiated per hop. An intermediary that talks `http/1.1`
  to the origin leaves the connection h2 only up to that hop, and the terminator shortcut below
  applies only when the connection this client holds is h2.
- **Terminator** — over HTTP/2 the parser returns as soon as `[DONE]` arrives. Discarding the
  unread remainder resets this stream only and leaves the connection reusable, so waiting out
  the drain would buy nothing.
- **Build-level switch** — the feature is a property of the dependency graph, not of a client or a
  provider: enabling it anywhere in one build enables it for every provider in that build, and it
  cannot be narrowed to a single one.

To build without HTTP/2, disable it where you consume the crate:

```toml
just-llm-client = { version = "0.2", default-features = false, features = ["deepseek"] }
```

That also drops the default provider selection, so name the provider features you want. The
recipe works because the provider defaults are closed at the dependency declaration — the
workspace sets `default-features = false` on each provider entry — and only this crate's `http2`
feature re-enables them. The published manifest keeps that shape, and it is a promise this crate
makes: it turns on only the provider features it names, never a provider's whole default set.

Cleartext HTTP/2 (h2c) is not a toggle here: it belongs to the client you build, so opt in on the
builder you hand in — `http_client(..)` keeps the `Authorization` and `Accept` injection intact:

```rust
use std::time::Duration;

use just_openai_compat::OpenAiCompatClient;

let client = OpenAiCompatClient::builder()
    .api_key(api_key.clone())
    .base_url("http://your-endpoint.example/v1")
    .http_client(
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .read_timeout(Duration::from_secs(120))
            .http2_prior_knowledge(),
    )
    .build()?;
```

h2c assumes the endpoint speaks HTTP/2 in cleartext; servers generally need explicit configuration
for that, and a mismatch fails hard instead of falling back. An already-built `reqwest::Client`
passed to `OpenAiCompatClient::new(http, base_url)` carries no injected headers, so set your own.

Not covered by the test suite: public-CA endpoints and system trust stores, real provider endpoints
over HTTP/2, sandboxed (Nix) builds, and any latency or throughput benefit. The tests exercise the
switch against a local h2c server and a self-signed TLS server whose ALPN list decides between h2
and the HTTP/1.1 fallback.

## Stateful conversations

`Conversation` (via `GenerationClient::conversation`) continues multi-turn conversations on
Responses-family backends (OpenAI, xAI) by sending only the new messages plus
`previous_response_id`, instead of re-transmitting the stored prefix each turn. The caller owns the
message list and passes the full logical context every turn; the `Conversation` decides the wire
payload:

- **First turn** — full request, stored server-side so the chain can continue.
- **Pure append** — the caller's list is `last_input ++ [last_assistant] ++ delta` (mirroring the
  previous assistant turn via `Conversation::last_message`); only `delta` is sent, plus
  `previous_response_id`.
- **Anything else** — trimming, rewriting, a changed generation setting (model, tools, sampling,
  reasoning, response format); the full request is sent and re-anchors the chain.

On stateless backends (chat completions, Anthropic) the same code degrades to a full-resend replay,
so one code path works across all families. Stateful chaining requires the `responses` feature;
`Conversation::is_stateful()` reports whether the effective mode is stateful. Streaming turns are
covered too: `Conversation::stream_generate` returns a `ConversationStream` that assembles the
assistant message. Consume it until `next()` returns `None`, not merely until the `End` event:
stopping at `End` and dropping the stream leaves the response body unread, and over HTTP/1.1
the connection is then closed instead of reused.
Over HTTP/1.1, a `[DONE]`-terminated stream leaves the rest of the body unread, so the client
drains it for up to 100 ms as long as the consumer keeps polling; the drain is best-effort, so an
expired budget or a transport error just closes the connection instead of failing the completed
response. Over HTTP/2 the parser returns as soon as `[DONE]` arrives: discarding the unread
remainder resets this stream only, so the connection stays reusable without a drain.
Take the response with `into_response` and pass it to `Conversation::adopt` to anchor
the turn. `into_response` returns `None` for a stream that never reached `End`, or a turn that
carries no id; such a turn is not adopted and the previous anchor remains valid.
Configure the HTTP client with `connect_timeout` + `read_timeout` rather than a single `timeout`
so long streams are not aborted mid-flight; see [timeouts](timeouts.md).

The two request fields `previous_response_id`/`store` are Conversation-owned; callers do not set
them directly, and chat/anthropic backends reject them explicitly.
