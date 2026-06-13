//! Full tool-calling loop using `OpenAiCompatClient`.
//!
//! Demonstrates: define a tool -> model calls it -> execute locally -> send result back -> final answer.
//!
//! ```bash
//! JUST_LLM_OPENAI_COMPAT_API_KEY=your-key \
//! JUST_LLM_OPENAI_COMPAT_BASE_URL=https://your-endpoint/v1 \
//! JUST_LLM_OPENAI_COMPAT_MODEL=gpt-4.1-mini \
//!   cargo run -p just-openai-compat --example openai_compat_tool_calling
//! ```

use just_openai_compat::OpenAiCompatClient;
use just_openai_compat::types::chat::{
    ChatCompletionRequest, ChatMessage, FunctionDefinition, ToolDefinition, ToolType,
};

/// A mock tool implementation.
fn add(args: &serde_json::Value) -> serde_json::Value {
    let x = args["x"].as_f64().expect("x is not a number");
    let y = args["y"].as_f64().expect("y is not a number");
    serde_json::json!({ "result": x + y })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().expect("failed to load .env file");

    let api_key = std::env::var("JUST_LLM_OPENAI_COMPAT_API_KEY")
        .expect("JUST_LLM_OPENAI_COMPAT_API_KEY must be set");
    let base_url = std::env::var("JUST_LLM_OPENAI_COMPAT_BASE_URL")
        .expect("JUST_LLM_OPENAI_COMPAT_BASE_URL must be set");
    let model = std::env::var("JUST_LLM_OPENAI_COMPAT_MODEL")
        .expect("JUST_LLM_OPENAI_COMPAT_MODEL must be set");

    let client = OpenAiCompatClient::builder()
        .api_key(&api_key)
        .base_url(&base_url)
        .build()?;

    let add_tool = ToolDefinition {
        kind: ToolType::Function,
        function: FunctionDefinition {
            name: "add".to_owned(),
            description: Some("Add two numbers together.".to_owned()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "x": { "type": "number", "description": "The first number." },
                    "y": { "type": "number", "description": "The second number." }
                },
                "required": ["x", "y"]
            })),
            strict: None,
        },
    };

    let system_prompt = "You are a helpful math assistant. Use the provided tools.";
    let user_prompt = "What is 12345 + 67890?";

    // --- Request 1: ask with tools ---
    let mut request = ChatCompletionRequest::new(
        model,
        vec![
            ChatMessage::system(system_prompt),
            ChatMessage::user(user_prompt),
        ],
    );
    request.tools = Some(vec![add_tool.clone()]);

    println!("--- request 1 ---");
    println!("  [system] {system_prompt}");
    println!("  [user] {user_prompt}");

    let completion = client.chat_completion(request).await?;

    let choice = completion
        .choices
        .first()
        .expect("expected at least one choice");
    let tool_calls = choice
        .message
        .tool_calls
        .as_ref()
        .expect("expected tool calls in response");

    println!("\n--- response 1 ---");
    let call = &tool_calls[0];
    println!(
        "  [tool call] {}({})",
        call.function.name, call.function.arguments
    );

    // --- Execute the tool locally ---
    let args: serde_json::Value = serde_json::from_str(&call.function.arguments)?;
    let tool_result = add(&args);
    println!("\n--- tool result ---");
    println!("  {tool_result}");

    // --- Request 2: replay conversation with tool result ---
    let request2 = ChatCompletionRequest::new(
        completion.model,
        vec![
            ChatMessage::system(system_prompt),
            ChatMessage::user(user_prompt),
            ChatMessage::assistant_tool_calls(tool_calls.clone()),
            ChatMessage::tool_result(tool_result.to_string(), &call.id),
        ],
    );

    let final_completion = client.chat_completion(request2).await?;

    println!("\n--- response 2 ---");
    if let Some(choice) = final_completion.choices.first() {
        println!(
            "  [assistant] {}",
            choice.message.content.as_deref().unwrap_or_default()
        );
    }
    if let Some(usage) = &final_completion.usage {
        println!(
            "  [usage] prompt={} completion={} total={}",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        );
    }

    Ok(())
}
