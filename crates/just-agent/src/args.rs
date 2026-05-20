use clap::Parser;

/// CLI arguments for just-agent.
#[derive(Parser)]
#[command(name = "just-agent", about = "A minimal coding agent")]
pub struct Args {
    /// The prompt to send to the agent (required unless --interactive)
    #[arg(long, env = "JUST_AGENT_PROMPT")]
    pub prompt: Option<String>,

    /// Enable interactive mode: after completing a task, prompt for the next input.
    #[arg(long, short = 'i')]
    pub interactive: bool,

    /// Activate a skill by name (repeatable).
    /// Skills are loaded from .just-agent/skills/\<name\>/SKILL.md
    #[arg(long = "skill", env = "JUST_AGENT_SKILLS", value_delimiter = ',')]
    pub skills: Vec<String>,
}
