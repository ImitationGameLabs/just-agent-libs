use crate::{
    LlmError,
    types::{
        chat::{ChatCompletionRequest, ToolChoice},
        prepared::PreparedChatRequest,
    },
};

/// Validates that a request is suitable for a non-streaming endpoint.
pub fn validate_non_streaming_request(
    request: &ChatCompletionRequest,
    method_name: &'static str,
    streaming_method_name: &'static str,
) -> Result<(), LlmError> {
    validate_common_request(request)?;

    if request.stream.unwrap_or(false) {
        return Err(LlmError::invalid_request(format!(
            "stream=true is not supported by {method_name}; use {streaming_method_name} instead"
        )));
    }

    if request.stream_options.is_some() {
        return Err(LlmError::invalid_request(
            "stream_options require stream=true",
        ));
    }

    Ok(())
}

/// Validates a request and ensures streaming is enabled, returning the modified request.
pub fn into_validated_streaming_request(
    mut request: ChatCompletionRequest,
    method_name: &'static str,
) -> Result<ChatCompletionRequest, LlmError> {
    validate_common_request(&request)?;

    if request.stream == Some(false) {
        return Err(LlmError::invalid_request(format!(
            "stream=false is not supported by {method_name}"
        )));
    }

    request.stream = Some(true);
    Ok(request)
}

/// Validates that a prepared request is suitable for a non-streaming endpoint.
pub fn validate_prepared_non_streaming_request(
    request: &PreparedChatRequest,
    method_name: &'static str,
    streaming_method_name: &'static str,
) -> Result<(), LlmError> {
    if request.has_stream() {
        return Err(LlmError::invalid_request(format!(
            "prepared request enables stream=true and cannot be sent via {method_name}; use {streaming_method_name} instead"
        )));
    }

    Ok(())
}

/// Validates that a prepared request is suitable for a streaming endpoint.
pub fn validate_prepared_streaming_request(
    request: &PreparedChatRequest,
    method_name: &'static str,
    non_streaming_method_name: &'static str,
) -> Result<(), LlmError> {
    if !request.has_stream() {
        return Err(LlmError::invalid_request(format!(
            "prepared request does not enable stream=true and cannot be sent via {method_name}; use {non_streaming_method_name} instead"
        )));
    }

    Ok(())
}

/// Validates request fields shared by streaming and non-streaming paths.
pub fn validate_common_request(request: &ChatCompletionRequest) -> Result<(), LlmError> {
    if request.tool_choice.is_some() && request.tools.as_ref().is_none_or(Vec::is_empty) {
        return Err(LlmError::invalid_request(
            "tool_choice requires at least one configured tool",
        ));
    }

    if let Some(ToolChoice::Named(choice)) = &request.tool_choice {
        let tool_exists = request.tools.as_ref().is_some_and(|tools| {
            tools
                .iter()
                .any(|tool| tool.function.name == choice.function.name)
        });

        if !tool_exists {
            return Err(LlmError::invalid_request(format!(
                "tool_choice references unknown tool '{}'",
                choice.function.name
            )));
        }
    }

    if request.top_logprobs.is_some() && request.logprobs != Some(true) {
        return Err(LlmError::invalid_request(
            "top_logprobs requires logprobs=true",
        ));
    }

    Ok(())
}
