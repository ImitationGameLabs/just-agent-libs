# Changelog

## [Unreleased]

### Changed
- `Conversation::stream_generate` no longer adopts the turn implicitly; consume the stream until
  it ends, then take the assembled response via `into_response()` and pass it to `adopt()`.
- `Conversation`: failed requests no longer clear the anchor. The previous successful turn's
  stored response remains valid server-side; the next turn keeps delta continuation.

### Fixed
- Streaming responses now drain the rest of the body after `[DONE]` so the connection
  can be reused; the drain is capped at 100 ms, never surfaces as an error, and consumers
  must consume to `None`.
