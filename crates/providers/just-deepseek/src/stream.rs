//! Stream type for DeepSeek chat completion SSE chunks.

use std::{
    fmt,
    pin::Pin,
    task::{Context, Poll},
};

use futures_core::Stream;
use just_common::error::TransportError;
use just_common::transport::sse::JsonEventStream;

use crate::types::chat::ChatCompletionChunk;

/// Stream of DeepSeek chat-completion SSE chunks.
pub struct ChatCompletionStream {
    inner: JsonEventStream<ChatCompletionChunk>,
}

impl ChatCompletionStream {
    /// Creates a stream from an SSE HTTP response.
    pub fn from_response(response: reqwest::Response) -> Result<Self, TransportError> {
        Ok(Self {
            inner: JsonEventStream::from_response(response)?,
        })
    }
}

impl fmt::Debug for ChatCompletionStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatCompletionStream")
            .finish_non_exhaustive()
    }
}

impl Stream for ChatCompletionStream {
    type Item = Result<ChatCompletionChunk, TransportError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}
