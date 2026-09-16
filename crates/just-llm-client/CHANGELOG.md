# Changelog

## [Unreleased]

### Changed
- `Conversation::stream_generate` no longer adopts the turn implicitly; consume the stream to
  `End`, take the assembled response via `into_response()`, and pass it to `adopt()`.
- `Conversation`: failed requests no longer clear the anchor. The previous successful turn's
  stored response remains valid server-side; the next turn keeps delta continuation.
