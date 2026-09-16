//! Stream assembly for stateful conversation continuation.
//!
//! [`ConversationStream`] forwards the backend event stream unchanged while accumulating the
//! assistant message from text, reasoning, and tool-call deltas. On the terminal `End` event it
//! writes the assembled capture into the slot shared with the conversation, which adopts it as
//! the next turn's anchor. A stream that never reaches `End` writes nothing, leaving the
//! previous anchor valid.

use std::{
    fmt,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
};

use futures_core::Stream;
use futures_util::StreamExt;

use super::conversation::{Settings, StreamCapture};
use crate::{
    capability::GenerationStream,
    types::generation::{AssistantMessage, GenerationEvent, Message, Reasoning, ToolCall},
};

/// Stream-side accumulation of the assistant message from normalized events.
#[derive(Default)]
pub(super) struct StreamState {
    pub(super) settings: Option<Settings>,
    pub(super) messages: Vec<Message>,
    pub(super) content: String,
    pub(super) reasoning: String,
    pub(super) tool_calls: Vec<AccumulatingCall>,
}

/// A single in-flight tool call being assembled from `ToolCall` deltas (keyed by `index`).
#[derive(Default)]
pub(super) struct AccumulatingCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

/// Streaming wrapper returned by
/// [`Conversation::stream_generate`](crate::client::conversation::Conversation::stream_generate).
///
/// Forwards every event unchanged while assembling the assistant message from the deltas; on the
/// terminal `End` event the assembled message and response id are recorded for the next turn.
pub struct ConversationStream {
    pub(super) inner: GenerationStream,
    pub(super) state: Arc<Mutex<StreamState>>,
    pub(super) slot: Arc<Mutex<Option<StreamCapture>>>,
}

impl Stream for ConversationStream {
    type Item = Result<GenerationEvent, just_common::error::TransportError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let item = this.inner.poll_next_unpin(cx);

        match &item {
            Poll::Ready(Some(Ok(GenerationEvent::Text { delta }))) => {
                this.state.lock().unwrap().content.push_str(delta);
            }
            Poll::Ready(Some(Ok(GenerationEvent::Reasoning { delta }))) => {
                this.state.lock().unwrap().reasoning.push_str(delta);
            }
            Poll::Ready(Some(Ok(GenerationEvent::ToolCall { delta }))) => {
                let mut state = this.state.lock().unwrap();
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
            Poll::Ready(Some(Ok(GenerationEvent::End { response_id, .. }))) => {
                let state = this.state.lock().unwrap();
                if let (Some(settings), Some(assistant)) =
                    (state.settings.clone(), build_assistant(&state))
                {
                    let capture = StreamCapture {
                        settings,
                        messages: state.messages.clone(),
                        assistant,
                        response_id: response_id.clone(),
                    };
                    if let Ok(mut slot) = this.slot.lock() {
                        *slot = Some(capture);
                    }
                }
            }
            _ => {}
        }

        item
    }
}

impl fmt::Debug for ConversationStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConversationStream").finish_non_exhaustive()
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
