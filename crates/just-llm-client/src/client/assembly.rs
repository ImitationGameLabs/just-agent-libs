//! Stream assembly for stateful conversation continuation.
//!
//! [`AssemblingStream`] wraps the backend generation event stream, forwards every event
//! unchanged, and accumulates the assistant message from text, reasoning, and tool-call deltas.
//! Once the terminal `End` event has been consumed,
//! [`into_response`](AssemblingStream::into_response) yields the assembled response for the
//! caller to record with `Conversation::adopt`.

use std::{
    fmt,
    pin::Pin,
    task::{Context, Poll},
};

use futures_core::Stream;
use futures_util::StreamExt;

use super::conversation::Settings;
use crate::{
    capability::GenerationStream,
    types::generation::{
        AssistantMessage, FinishReason, GenerationEvent, GenerationResponse, Reasoning, ToolCall,
        Usage,
    },
};

/// Stream-side accumulation of the assistant message from normalized events.
struct StreamState {
    settings: Option<Settings>,
    content: String,
    reasoning: String,
    tool_calls: Vec<AccumulatingCall>,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
    response_id: Option<String>,
}

/// A single in-flight tool call being assembled from `ToolCall` deltas (keyed by `index`).
#[derive(Default)]
struct AccumulatingCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

/// Assembling wrapper around a backend generation event stream.
///
/// Forwards every event unchanged while assembling the assistant message from the deltas.
/// Consume the stream through its terminal `End` event, then take the assembled response with
/// [`into_response`](AssemblingStream::into_response): `None` means there is nothing to record,
/// either because the stream never reached `End`, or because the backend is stateless and the
/// `End` event carried no response id. Both share the meaning "no adoptable response".
#[must_use = "a turn is only recorded by consuming the stream to `End` and adopting the response obtained via `into_response`"]
pub struct AssemblingStream {
    inner: GenerationStream,
    state: StreamState,
}

/// Streaming turn returned by
/// [`Conversation::stream_generate`](crate::client::conversation::Conversation::stream_generate).
pub type ConversationStream = AssemblingStream;

impl AssemblingStream {
    pub(super) fn new(inner: GenerationStream, settings: Settings) -> Self {
        Self {
            inner,
            state: StreamState {
                settings: Some(settings),
                content: String::new(),
                reasoning: String::new(),
                tool_calls: Vec::new(),
                finish_reason: None,
                usage: None,
                response_id: None,
            },
        }
    }

    /// The assembled response, if the stream reached `End` carrying a response id.
    ///
    /// `None` before `End`, and `None` when a stateless backend's `End` event carried no id:
    /// without an id there is nothing to adopt, so the caller skips `adopt` for that turn.
    pub fn into_response(self) -> Option<GenerationResponse> {
        let message = build_assistant(&self.state)?;
        let response_id = self.state.response_id.clone()?;
        let model = self.state.settings.as_ref()?.model.clone();
        Some(GenerationResponse {
            id: response_id,
            model,
            message,
            finish_reason: self.state.finish_reason.clone(),
            usage: self.state.usage.clone(),
            meta: None,
        })
    }
}

impl Stream for AssemblingStream {
    type Item = Result<GenerationEvent, just_common::error::TransportError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let item = this.inner.poll_next_unpin(cx);

        match &item {
            Poll::Ready(Some(Ok(GenerationEvent::Text { delta }))) => {
                this.state.content.push_str(delta);
            }
            Poll::Ready(Some(Ok(GenerationEvent::Reasoning { delta }))) => {
                this.state.reasoning.push_str(delta);
            }
            Poll::Ready(Some(Ok(GenerationEvent::ToolCall { delta }))) => {
                let state = &mut this.state;
                let index = delta.index.unwrap_or(0) as usize;
                while state.tool_calls.len() <= index {
                    state.tool_calls.push(AccumulatingCall::default());
                }
                if let Some(id) = &delta.id {
                    state.tool_calls[index].id = Some(id.clone());
                }
                if let Some(name) = &delta.name {
                    state.tool_calls[index].name = Some(name.clone());
                }
                if let Some(arguments) = &delta.arguments {
                    state.tool_calls[index].arguments.push_str(arguments);
                }
            }
            Poll::Ready(Some(Ok(GenerationEvent::Usage { usage }))) => {
                // Usage reports are last-wins: a final report supersedes any earlier one.
                this.state.usage = Some(usage.clone());
            }
            Poll::Ready(Some(Ok(GenerationEvent::End {
                finish_reason,
                response_id,
            }))) => {
                let state = &mut this.state;
                state.finish_reason = finish_reason.clone();
                state.response_id = response_id.clone();
            }
            _ => {}
        }

        item
    }
}

impl fmt::Debug for AssemblingStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AssemblingStream").finish_non_exhaustive()
    }
}

/// Assembles the assistant message from the accumulated stream state.
fn build_assistant(state: &StreamState) -> Option<AssistantMessage> {
    let tool_calls = state
        .tool_calls
        .iter()
        .filter_map(|call| {
            let id = call.id.as_ref()?;
            let name = call.name.as_ref()?;
            Some(ToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: call.arguments.clone(),
            })
        })
        .collect();
    let reasoning = if state.reasoning.is_empty() {
        None
    } else {
        Some(Reasoning {
            text: Some(state.reasoning.clone()),
            id: None,
            encrypted: None,
            signature: None,
            redacted: None,
        })
    };

    Some(AssistantMessage {
        content: if state.content.is_empty() {
            None
        } else {
            Some(state.content.clone())
        },
        tool_calls,
        reasoning,
    })
}
