//! Agent binary discovery and build helpers.

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Build just-agent and return the binary path.
///
/// Always runs `cargo build` to ensure the binary is up-to-date.
pub fn ensure_agent_bin() -> PathBuf {
    let exe =
        env::current_exe().unwrap_or_else(|e| fail(&format!("cannot determine current exe: {e}")));

    let bin_path = exe
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("just-agent"))
        .unwrap_or_else(|| fail("unexpected binary layout"));

    let status = Command::new("cargo")
        .args(["build", "-p", "just-agent"])
        .status()
        .unwrap_or_else(|e| fail(&format!("failed to spawn cargo: {e}")));
    if !status.success() {
        fail("cargo build -p just-agent failed");
    }

    if bin_path.is_file() { bin_path } else { fail("just-agent binary not found after build") }
}

fn fail(msg: &str) -> ! {
    eprintln!("[sandbox] error: {msg}");
    std::process::exit(1)
}
