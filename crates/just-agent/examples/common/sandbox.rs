//! Platform sandboxing primitives for examples.
//!
//! Provides sandboxed process execution using:
//! - Linux: bubblewrap (bwrap)
//! - macOS: sandbox-exec (Seatbelt)

use std::path::Path;
use std::process::Command;

/// Run a binary inside a platform sandbox.
///
/// Exits the process with the child's exit code.
pub fn run(bin: &Path, workspace: &Path, extra_args: &[String]) -> ! {
    if cfg!(target_os = "linux") {
        if which("bwrap") {
            run_bwrap(bin, workspace, extra_args);
        } else {
            die("bwrap not found — install bubblewrap or pass --no-sandbox to skip sandboxing");
        }
    } else if cfg!(target_os = "macos") {
        if which("sandbox-exec") {
            run_seatbelt(bin, workspace, extra_args);
        } else {
            die("sandbox-exec (Seatbelt) not found — pass --no-sandbox to skip sandboxing");
        }
    } else {
        die("unsupported platform — pass --no-sandbox to skip sandboxing");
    }
}

/// Print an error message to stderr and exit with code 1.
pub fn die(msg: &str) -> ! {
    eprintln!("[sandbox] error: {msg}");
    std::process::exit(1);
}

fn which(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn run_bwrap(bin: &Path, workspace: &Path, extra_args: &[String]) -> ! {
    let ws_str = workspace
        .to_str()
        .unwrap_or_else(|| die("workspace path is not valid UTF-8"));

    let status = Command::new("bwrap")
        .args([
            "--unshare-all",
            "--share-net",
            "--die-with-parent",
            "--new-session",
            // Read-only root filesystem
            "--ro-bind",
            "/",
            "/",
            // Virtual filesystems
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--size",
            "536870912",
            "--tmpfs",
            "/tmp",
            // Writable workspace (the only write path)
            "--bind",
            ws_str,
            ws_str,
            "--chdir",
            ws_str,
        ])
        .arg(bin)
        .args(extra_args)
        .status()
        .unwrap_or_else(|e| die(&format!("failed to spawn bwrap: {e}")));

    std::process::exit(status.code().unwrap_or(1))
}

fn run_seatbelt(bin: &Path, workspace: &Path, extra_args: &[String]) -> ! {
    let ws_str = workspace
        .to_str()
        .unwrap_or_else(|| die("workspace path is not valid UTF-8"));

    let policy = format!(
        "(version 1)\
        (deny default)\
        (allow process-exec)\
        (allow process-fork)\
        (allow signal (target same-sandbox))\
        (allow pseudo-tty)\
        (allow file-read* file-write* file-ioctl (literal \"/dev/ptmx\"))\
        (allow file-read* file-write* (regex \"^/dev/ttys[0-9]+\"))\
        (allow file-read* file-write* (subpath \"{ws_str}\"))\
        (allow file-read* (subpath \"/tmp\"))\
        (allow file-read* (subpath \"/private/tmp\"))\
        (allow file-read* (subpath \"/usr\"))\
        (allow file-read* (subpath \"/bin\"))\
        (allow file-read* (subpath \"/sbin\"))\
        (allow file-read* (subpath \"/lib\"))\
        (allow file-read* (subpath \"/System\"))\
        (allow file-read* (subpath \"/etc\"))\
        (allow file-read* (subpath \"/var\"))\
        (allow network*)"
    );

    let status = Command::new("sandbox-exec")
        .args(["-p", &policy, "--"])
        .arg(bin)
        .args(extra_args)
        .status()
        .unwrap_or_else(|e| die(&format!("failed to spawn sandbox-exec: {e}")));
    std::process::exit(status.code().unwrap_or(1))
}
