//! Sandbox wrapper for just-agent.
//!
//! Spawns just-agent inside a strict platform sandbox:
//! - Linux: bubblewrap (bwrap) — minimal read-only filesystem,
//!   namespace isolation, network allowed
//! - macOS: sandbox-exec (Seatbelt) — deny-default with explicit allowlist
//!
//! The ONLY writable path is the workspace directory. Everything else is
//! read-only and limited to what a shell needs to function.
//!
//! Usage:
//!   cargo run --example run-agent-with-prompt -- --workspace=<dir> [--no-sandbox] [-- <just-agent args...>]
//!
//! Example:
//!   cargo run --example run-agent-with-prompt -- --workspace=. -- --prompt="hello, try print the current dir"

#[path = "common/sandbox.rs"]
mod sandbox;

#[path = "common/agent.rs"]
mod agent;

use std::path::PathBuf;

use clap::Parser;

/// Sandbox wrapper for just-agent.
///
/// Spawns just-agent inside a platform sandbox (bwrap on Linux,
/// Seatbelt on macOS). The workspace directory is the only writable
/// path; everything else is read-only.
#[derive(Parser)]
#[command(name = "sandbox")]
struct Args {
    /// Run without sandbox isolation
    #[arg(long)]
    no_sandbox: bool,

    /// Workspace directory to bind into the sandbox
    #[arg(long)]
    workspace: PathBuf,

    /// Arguments passed through to just-agent
    #[arg(trailing_var_arg = true)]
    args: Vec<String>,
}

fn main() {
    dotenvy::dotenv().ok();
    let cli = Args::parse();

    let workspace = cli
        .workspace
        .canonicalize()
        .unwrap_or_else(|e| sandbox::die(&format!("invalid workspace directory: {e}")));

    if !workspace.is_dir() {
        sandbox::die("workspace path is not a directory");
    }

    let just_agent_bin = agent::ensure_agent_bin();

    if cli.no_sandbox {
        let data_dir = workspace.join(".draft");
        std::fs::create_dir_all(&data_dir).ok();

        let status = std::process::Command::new(&just_agent_bin)
            .current_dir(&workspace)
            .env("JUST_AGENT_DATA_DIR", &data_dir)
            .args(&cli.args)
            .status()
            .unwrap_or_else(|e| sandbox::die(&format!("failed to spawn binary: {e}")));
        std::process::exit(status.code().unwrap_or(1));
    }

    sandbox::run(&just_agent_bin, &workspace, &cli.args);
}
