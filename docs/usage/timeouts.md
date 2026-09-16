# HTTP timeouts and streaming

How to configure timeouts on the `reqwest::ClientBuilder` you hand to a provider
builder's `.http_client(..)` or a backend's `new(http, ..)`.

## The problem with a single `timeout`

reqwest's [`ClientBuilder::timeout`] is a **total deadline**: it starts when the
request begins (connection) and never resets. If the deadline fires at any point
up to and including the last byte of the response body, the request fails.

That is reasonable for a short request/response call, but it is wrong for
streaming. An SSE stream from an LLM can legitimately stay open for minutes while
frames keep arriving — a slow token rate, a long output, or a reasoning model
that pauses to think before each chunk. A never-resetting total deadline aborts
those streams mid-flight even though data is genuinely still flowing.

[`ClientBuilder::timeout`]: https://docs.rs/reqwest/~0.13/reqwest/struct.ClientBuilder.html#method.timeout

## The idiom: `connect_timeout` + `read_timeout`

Split the deadline into two phases:

```rust
reqwest::Client::builder()
    .connect_timeout(std::time::Duration::from_secs(30))
    .read_timeout(std::time::Duration::from_secs(120))
    .use_rustls_tls()
```

- [`connect_timeout`] — a deadline for establishing the connection (DNS, TCP,
  TLS handshake). Fires once, during connection setup.
- [`read_timeout`] — an **idle** timeout that **resets on every successful
  read**. As long as no single gap between frames exceeds it, the stream can run
  arbitrarily long. A genuinely stalled connection (no bytes for the whole
  window) still fails.

[`connect_timeout`]: https://docs.rs/reqwest/~0.13/reqwest/struct.ClientBuilder.html#method.connect_timeout
[`read_timeout`]: https://docs.rs/reqwest/~0.13/reqwest/struct.ClientBuilder.html#method.read_timeout

Because this split is required for streaming, the library default already applies
it (`stream_chat_completion`, `stream_generate`, `Conversation::stream_generate`,
and the provider-native streaming methods are all covered); supply your own
builder only to tune the values.

## Library default

Every provider builder defaults to `connect_timeout(30s) + read_timeout(120s)`
when you do not supply `.http_client(..)`. These values are the
`just_common::transport::http::DEFAULT_CONNECT_TIMEOUT` /
`DEFAULT_READ_TIMEOUT` constants. Changing either const does not update these
docs automatically; grep for `from_secs(30)` / `from_secs(120)` to refresh the
literals here and in the examples. The 30 s connect window is strict enough to
fail fast on unreachable endpoints; the 120 s read window tolerates
reasoning-model thinking pauses while still catching dead connections.

Supplying your own builder replaces the whole default client; there is no per-value
override:

```rust
use just_openai_compat::OpenAiCompatClient;

let client = OpenAiCompatClient::builder()
    .api_key(&api_key)
    .base_url("https://your-compatible-endpoint/v1")
    .http_client(
        reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .read_timeout(std::time::Duration::from_secs(300))
            .use_rustls_tls(),
    )
    .build()?;
```

## Which construction path gets the library defaults

The defaults above live in each provider's client builder and apply only when you
construct a provider directly and do not supply `.http_client(..)`. The rule is
either/or: defaults never stack with your builder; a supplied builder replaces the
default client wholesale.

Through the unified entry points, `LlmBackend::new(http, ..)` and
`BackendFactory::create(..)`, the builder you pass is forwarded untouched, so
`.http_client(..)` is always considered supplied and **no library transport defaults
apply on this path**. Timeouts and the TLS backend are then entirely yours to set;
the idiom at the top of this page is the recommended baseline.

## Request header precedence

Builder-preset headers and library-injected headers do not follow one rule;
precedence differs per header:

| Header | Who wins |
| --- | --- |
| `Authorization` | The library. `build_client` injects the `Bearer` credential and overwrites any preset. |
| `Accept` | The library's `application/json` default. A builder preset is overwritten; only an explicit entry in a provider's header map is respected. |
| `User-Agent` and everything else | You. The library never sets them, so presets (or reqwest's built-in default) survive. |

The mechanism: injected headers go through reqwest's `default_headers`, which merges
per key by overwriting, and `User-Agent` is not among the injected keys.

## Non-streaming requests

Non-streaming reads are unaffected in practice: the response body arrives within
the read window as long as data keeps flowing, so a slow-but-live JSON response
(say, a large model list) completes normally instead of being cut off by a fixed
total deadline. The library applies an 8 MiB body cap (`just_common`) on the
non-streaming path, which bounds memory independently of timeouts. The trade-off
of dropping the total deadline is that a pathological server emitting one byte
just inside the read window forever would never time out — an accepted
consequence of using `read_timeout`, and the reason the default keeps the window
bounded rather than removing timeouts entirely.

See [just-llm-client.md](just-llm-client.md) for the broader client layer.
