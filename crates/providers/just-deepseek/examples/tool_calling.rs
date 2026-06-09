use std::error::Error;

use just_deepseek::{
    DeepSeekClient,
    types::chat::{
        ChatCompletionRequest, ChatCompletionToolCall as ToolCall, ChatMessage, FunctionCall,
        FunctionDefinition, ReasoningEffort, ThinkingConfig, ThinkingMode, ToolCallsMessage,
        ToolChoice, ToolChoiceMode, ToolDefinition, ToolType,
    },
};
use serde_json::json;

fn expect_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set"))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenvy::dotenv().expect("failed to load .env file");

    let api_key = expect_env("JUST_LLM_DEEPSEEK_API_KEY");
    let base_url = std::env::var("JUST_LLM_DEEPSEEK_BASE_URL").ok();
    let model = expect_env("JUST_LLM_DEEPSEEK_MODEL");

    let mut builder = DeepSeekClient::builder().api_key(api_key);
    if let Some(url) = base_url {
        builder = builder.base_url(url);
    }
    let client = builder.build()?;

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
    println!("[system] You are a weather assistant. Use the provided tool.");
    println!("[user] What's the weather in Shanghai?");

    let mut request = ChatCompletionRequest::new(
        model.clone(),
        vec![
            ChatMessage::system("You are a weather assistant. Use the provided tool."),
            ChatMessage::user("What's the weather in Shanghai?"),
        ],
    );
    request.thinking = Some(ThinkingConfig {
        kind: ThinkingMode::Enabled,
    });
    request.reasoning_effort = Some(ReasoningEffort::High);
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
        println!("[reasoning] {reasoning}\n");
    }
    let tool_calls: Vec<ToolCall> = choice
        .message
        .tool_calls
        .clone()
        .expect("expected tool calls");
    for call in &tool_calls {
        println!(
            "[tool call] {}({}) [id={}]",
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
    println!("[tool result] {weather_result}");

    let mut messages = vec![
        ChatMessage::system("You are a weather assistant. Use the provided tool."),
        ChatMessage::user("What's the weather in Shanghai?"),
        ChatMessage::ToolCalls(ToolCallsMessage {
            role: "assistant".to_owned(),
            content: choice.message.content.clone(),
            name: None,
            tool_calls: tool_calls
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
            reasoning_content: choice.message.reasoning_content.clone(),
        }),
    ];
    for call in &tool_calls {
        messages.push(ChatMessage::tool_result(&weather_result, &call.id));
    }

    let mut request2 = ChatCompletionRequest::new(model, messages);
    request2.thinking = Some(ThinkingConfig {
        kind: ThinkingMode::Enabled,
    });
    request2.reasoning_effort = Some(ReasoningEffort::High);

    let response2 = client.chat_completion(request2).await?;
    let choice2 = response2
        .choices
        .first()
        .expect("expected at least one choice");

    // --- response 2 ---
    println!("\n--- response 2 ---");
    if let Some(reasoning) = &choice2.message.reasoning_content {
        println!("[reasoning] {reasoning}\n");
    }
    println!(
        "[assistant] {}",
        choice2.message.content.as_deref().unwrap_or_default()
    );

    if let Some(usage) = &response2.usage {
        println!("\n[usage]");
        if let Some(details) = &usage.completion_tokens_details
            && let Some(rt) = details.reasoning_tokens
        {
            println!("  reasoning tokens:    {rt}");
        }
        println!("  prompt tokens:       {}", usage.prompt_tokens);
        println!("  completion tokens:   {}", usage.completion_tokens);
        println!("  cache hit tokens:    {}", usage.prompt_cache_hit_tokens);
        println!("  cache miss tokens:   {}", usage.prompt_cache_miss_tokens);
        println!("  total tokens:        {}", usage.total_tokens);
    }

    Ok(())
}
