use std::error::Error;

use just_openai_compat::{
    OpenAiCompatClient,
    types::chat::{
        ChatCompletionRequest, ChatCompletionToolCall as ToolCall, ChatMessage, FunctionCall,
        FunctionDefinition, ToolChoice, ToolChoiceMode, ToolDefinition, ToolType,
    },
};
use serde_json::json;

fn expect_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set"))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenvy::dotenv().expect("failed to load .env file");

    let api_key = expect_env("JUST_LLM_OPENAI_COMPAT_API_KEY");
    let base_url = expect_env("JUST_LLM_OPENAI_COMPAT_BASE_URL");
    let model = expect_env("JUST_LLM_OPENAI_COMPAT_MODEL");

    let client = OpenAiCompatClient::builder()
        .api_key(api_key)
        .base_url(base_url)
        .build()?;

    let weather_tool = ToolDefinition {
        kind: ToolType::Function,
        function: FunctionDefinition {
            name: "get_weather".to_owned(),
            description: Some("Get the current weather for a city.".to_owned()),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            })),
            strict: None,
        },
    };

    // --- request 1 ---
    println!("--- request 1 ---");
    println!("  [system] You are a weather assistant. Use the provided tool.");
    println!("  [user] What's the weather in Shanghai?");

    let mut request = ChatCompletionRequest::new(
        model.clone(),
        vec![
            ChatMessage::system("You are a weather assistant. Use the provided tool."),
            ChatMessage::user("What's the weather in Shanghai?"),
        ],
    );
    request.tools = Some(vec![weather_tool]);
    request.tool_choice = Some(ToolChoice::Mode(ToolChoiceMode::Auto));

    let response = client.chat_completion(request).await?;
    let choice = response
        .choices
        .first()
        .expect("expected at least one choice");

    // --- response 1 ---
    println!("\n--- response 1 ---");
    if let Some(reasoning) = &choice.message.reasoning_content {
        println!("  [reasoning] {reasoning}\n");
    }
    let tool_calls: Vec<ToolCall> = choice
        .message
        .tool_calls
        .clone()
        .expect("expected tool calls");
    for call in &tool_calls {
        println!(
            "  [tool call] {}({}) [id={}]",
            call.function.name, call.function.arguments, call.id
        );
    }

    // --- request 2 ---
    println!("\n--- request 2 ---");
    let weather_result = json!({
        "city": "Shanghai",
        "temperature": "26C",
        "condition": "sunny"
    })
    .to_string();
    println!("  [tool result] {weather_result}");

    let messages = vec![
        ChatMessage::system("You are a weather assistant. Use the provided tool."),
        ChatMessage::user("What's the weather in Shanghai?"),
        ChatMessage::assistant_tool_calls(
            tool_calls
                .iter()
                .map(|c| ToolCall {
                    id: c.id.clone(),
                    kind: c.kind.clone(),
                    function: FunctionCall {
                        name: c.function.name.clone(),
                        arguments: c.function.arguments.clone(),
                    },
                })
                .collect(),
        ),
        ChatMessage::tool_result(&weather_result, &tool_calls[0].id),
    ];

    let request2 = ChatCompletionRequest::new(model, messages);
    let response2 = client.chat_completion(request2).await?;
    let choice2 = response2
        .choices
        .first()
        .expect("expected at least one choice");

    // --- response 2 ---
    println!("\n--- response 2 ---");
    if let Some(reasoning) = &choice2.message.reasoning_content {
        println!("  [reasoning] {reasoning}\n");
    }
    println!(
        "  [assistant] {}",
        choice2.message.content.as_deref().unwrap_or_default()
    );

    if let Some(usage) = &response2.usage {
        println!("\n[usage]");
        println!("  prompt tokens:       {}", usage.prompt_tokens);
        println!("  completion tokens:   {}", usage.completion_tokens);
        println!("  total tokens:        {}", usage.total_tokens);
    }

    Ok(())
}
