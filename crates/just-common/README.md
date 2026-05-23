# just-common

Shared HTTP transport, SSE stream parsing, and error types for the [just-agent] ecosystem.

[just-agent]: https://github.com/ImitationGameLabs/just-agent-libs

## Overview

This crate provides the internal building blocks used by the just-agent provider crates:

- **Authenticated HTTP transport** — reqwest-based transport construction with JSON helpers.
- **SSE stream parsing** — `JsonEventStream` for parsing server-sent event streams into typed JSON chunks.
- **Error types** — shared `TransportError` for protocol-level error handling.

> **Note:** This is an implementation dependency. You typically won't depend on it directly.
> Instead, use one of the crates below.

## Ecosystem

| Crate                 | Description                                               |
| --------------------- | --------------------------------------------------------- |
| [**just-llm-client**] | Provider-neutral LLM client — the recommended entry point |
| [just-deepseek]       | Rust client for the DeepSeek API                          |
| [just-openai-compat]  | Rust client for any OpenAI-compatible API                 |

[**just-llm-client**]: https://crates.io/crates/just-llm-client
[just-deepseek]: https://crates.io/crates/just-deepseek
[just-openai-compat]: https://crates.io/crates/just-openai-compat
