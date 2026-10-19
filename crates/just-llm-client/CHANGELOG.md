# Changelog

## [Unreleased]

### Changed
- Default HTTP/2: the `http2` feature is on by default here and in every provider crate, so a
  default build negotiates h2 over TLS (falling back to HTTP/1.1 through ALPN) and multiplexes
  calls onto shared connections. Over h2 the post-`[DONE]` drain is skipped: discarding the
  unread remainder resets that stream only and leaves the connection reusable. Opt out with
  `default-features = false` plus the provider features you want.
- `Conversation::stream_generate` no longer adopts the turn implicitly; consume the stream until
  it ends, then take the assembled response via `into_response()` and pass it to `adopt()`.
- `Conversation`: failed requests no longer clear the anchor. The previous successful turn's
  stored response remains valid server-side; the next turn keeps delta continuation.

### Fixed
- Streaming responses now drain the rest of the body after `[DONE]` so the connection
  can be reused; the drain is capped at 100 ms, never surfaces as an error, and consumers
  must consume to `None`.
