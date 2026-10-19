//! Shared SSE parsing for OpenAI-like JSON chunk streams.
//!
//! # Known limitation: unbounded buffering
//!
//! The pending byte buffer grows without limit until a blank line terminates an event, and each
//! event's `data:` payload is materialized in full before deserialization. Unlike the
//! non-streaming path (capped by `crate::transport::http::MAX_BODY_BYTES`), there is no size cap
//! here, so a malicious or broken server emitting very large events — or withholding the
//! terminating blank line — can drive unbounded memory growth. Acceptable today because providers
//! bound event size server-side; tighten if this is ever exposed to untrusted endpoints.
//!
//! # Terminator handling: a bounded drain after `[DONE]` over HTTP/1.1
//!
//! Over HTTP/1.1, `[DONE]` ends the events, not the response: the body still carries its framing end,
//! and a consumer that stops reading at `[DONE]` drops the body mid-response, which forces the
//! HTTP client to close the connection instead of returning it to the pool. The parser therefore
//! keeps reading and discarding bytes for up to `DRAIN_BUDGET` (100 ms) after `[DONE]`, so a
//! shared client can reuse the connection.
//!
//! The budget bounds the wait the consumer sees on its final poll, and letting it expire is
//! not an error: a peer that holds the response open without sending anything ends the drain and
//! the connection closes, so no consumer waits on a peer that will never finish. Bytes read
//! during the drain are discarded, never decoded or yielded, and a transport error during the
//! drain stops it. Abandoning the drain — by expiry or by error — is reported at `trace` level
//! and never surfaces as an error: the stream has already delivered its last chunk, and a
//! connection that cannot be reused is not worth turning a completed response into an error.
//!
//! Over HTTP/2 the parser skips the drain and returns as soon as `[DONE]` arrives: discarding the
//! unread remainder resets only this stream, and the connection stays reusable, so waiting for a
//! framing end would buy nothing.
//!
//! Over HTTP/1.1 the drain waits on a tokio timer, so the stream must be driven on a tokio runtime
//! with the time driver enabled; the HTTP/2 path never waits on one.

use std::{
    fmt,
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

use async_stream::try_stream;
use futures_core::Stream;
use futures_util::StreamExt;
use reqwest::header::CONTENT_TYPE;
use serde::de::DeserializeOwned;
use tracing::trace;

use crate::error::TransportError;

type BoxedJsonStream<T> = Pin<Box<dyn Stream<Item = Result<T, TransportError>> + Send>>;

/// How long the parser keeps draining the response body after a `[DONE]` event.
///
/// Matches the drain timeout the OpenAI Python client uses for the same purpose: long enough
/// for the framing end of a live response to arrive, short enough that a peer withholding it
/// cannot delay the consumer noticeably.
const DRAIN_BUDGET: Duration = Duration::from_millis(100);

/// Generic JSON-over-SSE stream.
///
/// Over HTTP/1.1 the bounded drain this stream performs after the terminator event waits on a
/// tokio timer, so the stream must be driven on a tokio runtime with the time driver enabled;
/// the drain is capped at `DRAIN_BUDGET` (100 ms), which bounds the extra wait the consumer
/// sees on its final poll. Over HTTP/2 the drain is skipped.
pub struct JsonEventStream<T> {
    inner: BoxedJsonStream<T>,
}

impl<T> JsonEventStream<T>
where
    T: DeserializeOwned + Send + 'static,
{
    /// Creates a stream from a successful SSE response.
    pub fn from_response(response: reqwest::Response) -> Result<Self, TransportError> {
        ensure_event_stream(&response)?;
        let version = response.version();

        let stream = try_stream! {
            let mut bytes = response.bytes_stream();
            let mut buffer = Vec::new();
            let mut done = false;

            while let Some(chunk) = bytes.next().await {
                let chunk = chunk.map_err(TransportError::Transport)?;
                buffer.extend_from_slice(&chunk);

                while let Some((event_end, consumed)) = split_event(&buffer) {
                    let event = buffer[..event_end].to_vec();
                    buffer.drain(..consumed);

                    match parse_event::<T>(&event)? {
                        ParsedEvent::Done => {
                            done = true;
                            break;
                        }
                        ParsedEvent::Skip => {}
                        ParsedEvent::Chunk(chunk) => yield chunk,
                    }
                }

                if done {
                    break;
                }
            }

            if done && version == reqwest::Version::HTTP_2 {
                // The drain exists to finish an HTTP/1.1 body and hand its connection back:
                // over HTTP/2, dropping the unread remainder resets this stream alone and the
                // connection stays reusable, so waiting would buy nothing.
                trace!("SSE stream ended at [DONE] over HTTP/2; no drain needed");
            } else if done {
                // Keep reading to the framing end so the client can return the connection to
                // its pool. The drain is best-effort: an expired budget and a transport error
                // both just stop it, because the stream already delivered its last chunk.
                let drain = tokio::time::timeout(DRAIN_BUDGET, async {
                    while let Some(next) = bytes.next().await {
                        next?;
                    }

                    Ok::<(), reqwest::Error>(())
                })
                .await;

                match drain {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => trace!(
                        error = %error,
                        "SSE stream ended at [DONE]; the drain stopped on a transport error \
                         with the response body unfinished, closing the connection"
                    ),
                    Err(_) => trace!(
                        "SSE stream ended at [DONE]; the drain budget expired with the \
                         response body unfinished, closing the connection"
                    ),
                }
            }

            if !done && !buffer.iter().all(u8::is_ascii_whitespace) {
                match parse_event::<T>(&buffer)? {
                    ParsedEvent::Done | ParsedEvent::Skip => {}
                    ParsedEvent::Chunk(chunk) => yield chunk,
                }
            }
        };

        Ok(Self {
            inner: Box::pin(stream),
        })
    }
}

impl<T> fmt::Debug for JsonEventStream<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JsonEventStream").finish_non_exhaustive()
    }
}

impl<T> Stream for JsonEventStream<T> {
    type Item = Result<T, TransportError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // `inner` is already pinned in a `Pin<Box<...>>`, so projecting to it is safe.
        unsafe { self.map_unchecked_mut(|stream| &mut stream.inner) }.poll_next(cx)
    }
}

#[derive(Debug)]
enum ParsedEvent<T> {
    Done,
    Skip,
    Chunk(T),
}

fn ensure_event_stream(response: &reqwest::Response) -> Result<(), TransportError> {
    let Some(content_type) = response.headers().get(CONTENT_TYPE) else {
        return Err(TransportError::InvalidResponse(
            "streaming response was missing content-type".to_owned(),
        ));
    };

    let content_type = content_type.to_str().map_err(|_| {
        TransportError::InvalidResponse(
            "streaming response content-type was not valid UTF-8".to_owned(),
        )
    })?;

    if !content_type.starts_with("text/event-stream") {
        return Err(TransportError::InvalidResponse(format!(
            "expected text/event-stream response, got {content_type}"
        )));
    }

    Ok(())
}

fn split_event(buffer: &[u8]) -> Option<(usize, usize)> {
    let mut index = 0;

    while index < buffer.len() {
        if buffer[index..].starts_with(b"\r\n\r\n") {
            return Some((index, index + 4));
        }

        if buffer[index..].starts_with(b"\n\n") {
            return Some((index, index + 2));
        }

        index += 1;
    }

    None
}

fn parse_event<T>(raw_event: &[u8]) -> Result<ParsedEvent<T>, TransportError>
where
    T: DeserializeOwned,
{
    if raw_event.is_empty() || raw_event.iter().all(u8::is_ascii_whitespace) {
        return Ok(ParsedEvent::Skip);
    }

    let event = String::from_utf8(raw_event.to_vec()).map_err(TransportError::Utf8)?;
    let mut data_lines = Vec::new();

    for line in event.lines() {
        let line = line.trim_end_matches('\r');

        if line.starts_with(':') {
            continue;
        }

        if let Some(data) = line.strip_prefix("data:") {
            data_lines.push(data.trim_start());
        }
    }

    if data_lines.is_empty() {
        return Ok(ParsedEvent::Skip);
    }

    let payload = data_lines.join("\n");

    if payload == "[DONE]" {
        return Ok(ParsedEvent::Done);
    }

    let chunk = serde_json::from_str(&payload).map_err(|source| TransportError::Deserialize {
        source,
        body: payload,
    })?;

    Ok(ParsedEvent::Chunk(chunk))
}
