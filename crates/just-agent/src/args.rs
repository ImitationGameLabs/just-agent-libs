use clap::Parser;

/// CLI arguments for just-agent.
#[derive(Parser)]
#[command(name = "just-agent", about = "A minimal coding agent")]
pub struct Args {
    /// The prompt to send to the agent
    #[arg(long, env = "JUST_AGENT_PROMPT")]
    pub prompt: String,
}
