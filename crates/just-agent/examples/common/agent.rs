//! Agent binary discovery and build helpers.

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Ensure the just-agent binary exists, building it with cargo if necessary.
pub fn ensure_agent_bin() -> PathBuf {
    let exe =
        env::current_exe().unwrap_or_else(|e| fail(&format!("cannot determine current exe: {e}")));

    let bin_path = exe
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("just-agent"));

    if let Some(path) = bin_path.as_ref().filter(|p| p.is_file()) {
        return path.clone();
    }

    eprintln!("[sandbox] just-agent binary not found, building with cargo...");
    let status = Command::new("cargo")
        .args(["build", "-p", "just-agent"])
        .status()
        .unwrap_or_else(|e| fail(&format!("failed to spawn cargo: {e}")));
    if !status.success() {
        fail("cargo build -p just-agent failed");
    }

    bin_path
        .filter(|p| p.is_file())
        .unwrap_or_else(|| fail("just-agent binary still not found after build"))
}

fn fail(msg: &str) -> ! {
    eprintln!("[sandbox] error: {msg}");
    std::process::exit(1)
}
