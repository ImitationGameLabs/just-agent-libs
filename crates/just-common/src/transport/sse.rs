//! Shared SSE parsing for OpenAI-like JSON chunk streams.

use std::{
    fmt,
    pin::Pin,
    task::{Context, Poll},
};

use async_stream::try_stream;
use futures_core::Stream;
use futures_util::StreamExt;
use reqwest::header::CONTENT_TYPE;
use serde::de::DeserializeOwned;

use crate::error::TransportError;

type BoxedJsonStream<T> = Pin<Box<dyn Stream<Item = Result<T, TransportError>> + Send>>;

/// Generic JSON-over-SSE stream.
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

            if !done && !buffer.iter().all(u8::is_ascii_whitespace) {
                match parse_event::<T>(&buffer)? {
                    ParsedEvent::Done | ParsedEvent::Skip => {}
                    ParsedEvent::Chunk(chunk) => yield chunk,
                }
            }
        };

        Ok(Self { inner: Box::pin(stream) })
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

    let chunk = serde_json::from_str(&payload)
        .map_err(|source| TransportError::Deserialize { source, body: payload })?;

    Ok(ParsedEvent::Chunk(chunk))
}
