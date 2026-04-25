mod common;

use just_llm_client::{
    ChatCompletion,
    provider::DeepSeekBackend,
    types::chat::{
        ChatCompletionRequest, ChatMessage, FunctionDefinition, ToolDefinition, ToolType,
    },
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().expect("failed to load .env file");

    let api_key = common::expect_env("JUST_LLM_DEEPSEEK_API_KEY");
    let base_url = std::env::var("JUST_LLM_DEEPSEEK_BASE_URL").ok();
    let model = common::expect_env("JUST_LLM_DEEPSEEK_MODEL");

    let backend = match base_url {
        Some(base_url) => DeepSeekBackend::with_base_url(api_key, base_url)?,
        None => DeepSeekBackend::with_config(just_deepseek::DeepSeekConfig::new(api_key))?,
    };

    let tools = vec![ToolDefinition {
        kind: ToolType::Function,
        function: FunctionDefinition {
            name: "sum".to_owned(),
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
    }];

    let request =
        ChatCompletionRequest::new(model, vec![ChatMessage::user("What is 12345 + 67890?")])
            .with_system_prompt("You are a helpful math assistant. Use the provided tools.")
            .with_tools(tools);

    println!("--- request 1 ---");
    println!("  [system] You are a helpful math assistant. Use the provided tools.");
    println!("  [user] What is 12345 + 67890?");

    let response = backend.create_chat_completion(request).await?;
    println!("\n--- response 1 ---");
    if let Some(rc) = response.first_choice_reasoning_content() {
        println!("  [reasoning] {rc}");
    }
    let response_model = response.model.clone();
    let reasoning = response.first_choice_reasoning_content().map(String::from);

    let tool_calls = response
        .first_choice_tool_calls()
        .expect("expected tool calls in response");

    let call = &tool_calls[0];
    println!(
        "  [tool call] {}({})",
        call.function.name, call.function.arguments
    );

    // Execute the tool locally.
    let args: serde_json::Value = serde_json::from_str(&call.function.arguments)?;
    let x: f64 = args["x"].as_f64().expect("x is not a number");
    let y: f64 = args["y"].as_f64().expect("y is not a number");
    let result = x + y;
    println!("\n--- request 2 ---");
    println!("  [tool result] {x} + {y} = {result}");

    // Build the assistant message, preserving reasoning content for DeepSeek thinking mode.
    let assistant_msg = match reasoning {
        Some(rc) => ChatMessage::assistant_tool_calls_with_reasoning(tool_calls.to_vec(), rc),
        None => ChatMessage::assistant_tool_calls(tool_calls.to_vec()),
    };

    // Send the tool result back for a final answer.
    let follow_up = ChatCompletionRequest::new(
        response_model,
        vec![
            ChatMessage::user("What is 12345 + 67890?"),
            assistant_msg,
            ChatMessage::tool_result(serde_json::json!({"result": result}).to_string(), &call.id),
        ],
    )
    .with_system_prompt("You are a helpful math assistant. Use the provided tools.")
    .with_tools(vec![ToolDefinition {
        kind: ToolType::Function,
        function: FunctionDefinition {
            name: "sum".to_owned(),
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
    }]);

    println!("\n--- response 2 ---");
    let final_response = backend.create_chat_completion(follow_up).await?;
    if let Some(rc) = final_response.first_choice_reasoning_content() {
        println!("  [reasoning] {rc}");
    }
    println!(
        "  [assistant] {}",
        final_response.first_choice_content().unwrap_or_default()
    );

    Ok(())
}
