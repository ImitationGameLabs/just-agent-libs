//! PTY-backed shell backend using `portable-pty`.
//!
//! Cross-platform persistent shell backend that requires no external
//! binary. Each session is an independent PTY pair with its own shell process.

use std::collections::HashMap;
use std::ffi::OsString;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use portable_pty::{Child, CommandBuilder, MasterPty, PtySize, native_pty_system};
use strip_ansi_escapes::strip as strip_ansi;
use tokio::time::{sleep, timeout};

use super::super::compat::strip_common_prefix;
use super::super::error::ShellError;
use super::{SessionInfo, ShellBackend, ShellOutput};

// ---------------------------------------------------------------------------
// ScrollbackBuffer
// ---------------------------------------------------------------------------

/// Line-oriented ring buffer that accumulates PTY output.
struct ScrollbackBuffer {
    lines: Vec<String>,
    max_lines: usize,
}

impl ScrollbackBuffer {
    fn new(max_lines: usize) -> Self {
        Self { lines: Vec::with_capacity(1024), max_lines }
    }

    fn append_line(&mut self, line: &str) {
        self.lines.push(line.to_owned());
        if self.lines.len() > self.max_lines {
            let excess = self.lines.len() - self.max_lines;
            self.lines.drain(..excess);
        }
    }

    /// Returns the last `n` lines joined with `\n`.
    fn last_n(&self, n: usize) -> String {
        let start = self.lines.len().saturating_sub(n);
        self.lines[start..].join("\n")
    }

    /// Returns the full buffer joined with `\n`.
    fn snapshot(&self) -> String {
        self.lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// PtySession
// ---------------------------------------------------------------------------

/// State for a single PTY-backed shell session.
///
/// Each session owns its own PTY master/slave pair, a background reader thread
/// that fills the scrollback buffer, and a handle to the child shell process.
/// Call [`shutdown`](PtySession::shutdown) to cleanly tear down these resources.
struct PtySession {
    master: Mutex<Option<Box<dyn MasterPty + Send>>>,
    writer: Mutex<Box<dyn std::io::Write + Send>>,
    child: Mutex<Box<dyn Child + Send + Sync>>,
    scrollback: Arc<Mutex<ScrollbackBuffer>>,
    reader_handle: Option<std::thread::JoinHandle<()>>,
    cwd: PathBuf,
}

impl PtySession {
    /// Kill the child process, drop the master (which terminates the reader
    /// thread), and join the reader thread.
    fn shutdown(&mut self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.try_wait();
        }
        // Dropping master causes the reader's `read()` to error out.
        self.master.lock().unwrap().take();
        if let Some(handle) = self.reader_handle.take() {
            let _ = handle.join();
        }
    }
}

// ---------------------------------------------------------------------------
// CommandConfig
// ---------------------------------------------------------------------------

/// Configuration for the command [`PtyBackend`] spawns inside the PTY.
///
/// By default PtyBackend spawns `bash --login`. A custom `CommandConfig` lets
/// callers wrap the shell in a sandbox (e.g. `bwrap ... bash` on Linux or
/// `sandbox-exec ... bash` on macOS) without modifying PtyBackend itself.
#[derive(Clone, Debug)]
pub struct CommandConfig {
    /// Full argv vector. `argv[0]` is the program to execute.
    pub argv: Vec<OsString>,
    /// When `true` (default), PtyBackend appends `--login` (or `--noprofile
    /// --norc` in clean-env mode) to the argv. Set to `false` when the wrapper
    /// already includes shell flags (e.g. `bwrap ... -- bash --login`).
    pub login_shell: bool,
}

impl Default for CommandConfig {
    fn default() -> Self {
        Self { argv: vec![OsString::from("bash")], login_shell: true }
    }
}

// ---------------------------------------------------------------------------
// PtyBackend
// ---------------------------------------------------------------------------

/// Persistent shell backend implemented on top of native PTY sessions.
///
/// This backend requires no external terminal multiplexer and works on Linux,
/// macOS, and Windows.
pub struct PtyBackend {
    sessions: HashMap<String, PtySession>,
    current_session: String,
    command_config: CommandConfig,
    next_sentinel: u64,
}

impl PtyBackend {
    /// Exit-code marker appended after every command.
    const EC_PREFIX: &'static str = "__JUST_EC__:";
    /// Output-start sentinel prepended before every command.
    const START_PREFIX: &'static str = "__JUST_OUT_S__:";

    fn generate_sentinel(&mut self) -> String {
        let s = format!("{:08x}", self.next_sentinel);
        self.next_sentinel += 1;
        s
    }

    /// Creates a backend and ensures the default session exists.
    pub async fn new(default_name: &str) -> Result<Self, ShellError> {
        Self::with_command(default_name, CommandConfig::default()).await
    }

    /// Creates a backend with a custom [`CommandConfig`] for the spawned process.
    ///
    /// Use this to wrap the shell in a sandbox. The `command_config.argv`
    /// becomes the full argv passed to `CommandBuilder::from_argv`.
    pub async fn with_command(
        default_name: &str,
        command_config: CommandConfig,
    ) -> Result<Self, ShellError> {
        let mut backend = Self {
            sessions: HashMap::new(),
            current_session: default_name.to_owned(),
            command_config,
            next_sentinel: 0,
        };

        backend
            .create_session_internal(default_name, None, false)
            .await?;

        Ok(backend)
    }

    // -- private helpers ----------------------------------------------------

    fn get_session(&self, name: &str) -> Result<&PtySession, ShellError> {
        self.sessions
            .get(name)
            .ok_or_else(|| ShellError::session_not_found(name))
    }

    async fn create_session_internal(
        &mut self,
        name: &str,
        cwd: Option<&Path>,
        clean_env: bool,
    ) -> Result<(), ShellError> {
        let cwd = cwd
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/tmp")));

        let mut cmd = CommandBuilder::from_argv(self.command_config.argv.clone());
        if clean_env {
            let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_owned());
            if self.command_config.login_shell {
                cmd.arg("--noprofile");
                cmd.arg("--norc");
            }
            cmd.env_clear();
            if let Ok(home) = std::env::var("HOME") {
                cmd.env("HOME", &home);
            }
            if let Ok(path) = std::env::var("PATH") {
                cmd.env("PATH", &path);
            }
            cmd.env("SHELL", &shell);
        } else if self.command_config.login_shell {
            cmd.arg("--login");
        }

        // Disable color output from all programs.
        cmd.env("TERM", "dumb");
        cmd.env("NO_COLOR", "1");
        cmd.env("LS_COLORS", "");
        cmd.env("CLICOLOR", "0");

        cmd.cwd(&cwd);

        let pty_system = native_pty_system();
        let pair = pty_system
            .openpty(PtySize { rows: 24, cols: 500, pixel_width: 0, pixel_height: 0 })
            .map_err(|e| ShellError::session_create_failed(name, e.to_string()))?;

        let child = pair
            .slave
            .spawn_command(cmd)
            .map_err(|e| ShellError::session_create_failed(name, e.to_string()))?;

        // Take writer immediately — can only be called once.
        let writer = pair
            .master
            .take_writer()
            .map_err(|e| ShellError::session_create_failed(name, e.to_string()))?;

        let reader = pair
            .master
            .try_clone_reader()
            .map_err(|e| ShellError::session_create_failed(name, e.to_string()))?;

        let scrollback = Arc::new(Mutex::new(ScrollbackBuffer::new(10_000)));
        let reader_handle = spawn_reader(reader, scrollback.clone());

        let session = PtySession {
            master: Mutex::new(Some(pair.master)),
            writer: Mutex::new(writer),
            child: Mutex::new(child),
            scrollback,
            reader_handle: Some(reader_handle),
            cwd: cwd.clone(),
        };

        self.sessions.insert(name.to_owned(), session);
        sleep(Duration::from_millis(100)).await;

        // Suppress prompt and input echo for clean command output.
        let session = self.get_session(name)?;
        self.write_to_session(session, b"export PS1='' PS2=''; stty -echo 2>/dev/null\n")?;
        sleep(Duration::from_millis(100)).await;

        Ok(())
    }

    /// Send a command to the current session's PTY writer.
    fn write_to_session(&self, session: &PtySession, data: &[u8]) -> Result<(), ShellError> {
        let mut writer = session
            .writer
            .lock()
            .map_err(|e| ShellError::Io(e.to_string()))?;
        writer
            .write_all(data)
            .map_err(|e| ShellError::Io(e.to_string()))?;
        writer.flush().map_err(|e| ShellError::Io(e.to_string()))
    }

    /// Poll the scrollback buffer until the exit-code marker appears and the
    /// output has been stable for 3 consecutive reads.
    async fn wait_for_completion(
        &self,
        session: &PtySession,
        timeout_duration: Duration,
    ) -> Result<String, ShellError> {
        let mut last_output = String::new();
        let mut stable_checks = 0usize;

        let wait = async {
            loop {
                let output = session.scrollback.lock().unwrap().snapshot();
                let has_marker = output.contains(Self::EC_PREFIX);

                if has_marker && output == last_output {
                    stable_checks += 1;
                    if stable_checks >= 3 {
                        return Ok(output);
                    }
                } else {
                    stable_checks = 0;
                }

                last_output = output;
                sleep(Duration::from_millis(100)).await;
            }
        };

        match timeout(timeout_duration, wait).await {
            Ok(result) => result,
            Err(_) => Err(ShellError::timeout(timeout_duration.as_secs())),
        }
    }

    /// Extract command output between start and end sentinel markers.
    fn extract_output(output: &str, sentinel: &str) -> (String, Option<i32>) {
        let start_marker = format!("{}{}", Self::START_PREFIX, sentinel);
        let mut exit_code = None;
        let mut in_range = false;
        let mut result_lines: Vec<&str> = Vec::new();

        for line in output.lines() {
            if line == start_marker {
                in_range = true;
                continue;
            }
            if let Some(rest) = line.strip_prefix(Self::EC_PREFIX) {
                exit_code = rest.trim().parse::<i32>().ok();
                break;
            }
            if in_range {
                // Fallback: strip echo artifacts when stty -echo is unavailable.
                if line.contains(&format!("echo {}", Self::START_PREFIX)) {
                    continue;
                }
                if line.contains(&format!("echo {}", Self::EC_PREFIX)) {
                    continue;
                }
                result_lines.push(line);
            }
        }

        (result_lines.join("\n"), exit_code)
    }
}

// ---------------------------------------------------------------------------
// Background reader thread
// ---------------------------------------------------------------------------

fn spawn_reader(
    reader: Box<dyn std::io::Read + Send>,
    scrollback: Arc<Mutex<ScrollbackBuffer>>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut reader = std::io::BufReader::new(reader);
        let mut pending = String::new();

        loop {
            pending.clear();
            match reader.read_line(&mut pending) {
                Ok(0) => break, // EOF
                Ok(_) => {
                    let line = pending.strip_suffix('\n').unwrap_or(&pending);
                    let line = line.strip_suffix('\r').unwrap_or(line);
                    let clean = strip_ansi(line.as_bytes());
                    let line = String::from_utf8_lossy(&clean);
                    scrollback.lock().unwrap().append_line(&line);
                }
                Err(_) => break, // PTY closed
            }
        }
    })
}

// ---------------------------------------------------------------------------
// ShellBackend implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl ShellBackend for PtyBackend {
    async fn execute(
        &mut self,
        command: &str,
        timeout_duration: Duration,
        background: bool,
    ) -> Result<ShellOutput, ShellError> {
        let session_name = self.current_session.clone();
        let sentinel = self.generate_sentinel();
        let session = self.get_session(&session_name)?;

        let before = session.scrollback.lock().unwrap().snapshot();

        // Send start sentinel + command + exit-code marker.
        let payload = format!(
            "echo {}{}\n{}\necho {}$?\n",
            Self::START_PREFIX,
            sentinel,
            command,
            Self::EC_PREFIX,
        );
        self.write_to_session(session, payload.as_bytes())?;

        if background {
            return Ok(ShellOutput { output: String::new(), exit_code: None, timed_out: false });
        }

        let full_output = match self.wait_for_completion(session, timeout_duration).await {
            Ok(output) => output,
            Err(ShellError::Timeout { .. }) => {
                return Ok(ShellOutput { output: String::new(), exit_code: None, timed_out: true });
            }
            Err(error) => return Err(error),
        };

        let diffed = strip_common_prefix(&before, &full_output);
        let (output, exit_code) = Self::extract_output(&diffed, &sentinel);

        Ok(ShellOutput { output, exit_code, timed_out: false })
    }
    async fn capture_output(&mut self, lines: usize) -> Result<String, ShellError> {
        let session_name = self.current_session.clone();
        let session = self.get_session(&session_name)?;

        Ok(session.scrollback.lock().unwrap().last_n(lines))
    }

    async fn list_sessions(&self) -> Result<Vec<SessionInfo>, ShellError> {
        Ok(self
            .sessions
            .iter()
            .map(|(name, session)| SessionInfo {
                name: name.clone(),
                cwd: session.cwd.to_string_lossy().into_owned(),
                is_current: name == &self.current_session,
                window_count: 1,
            })
            .collect())
    }

    async fn create_session(&mut self, name: &str, cwd: Option<&Path>) -> Result<(), ShellError> {
        if self.sessions.contains_key(name) {
            return Err(ShellError::session_exists(name));
        }
        self.create_session_internal(name, cwd, false).await
    }

    async fn switch_session(&mut self, name: &str) -> Result<(), ShellError> {
        if !self.sessions.contains_key(name) {
            return Err(ShellError::session_not_found(name));
        }
        self.current_session = name.to_owned();
        Ok(())
    }

    async fn kill_session(&mut self, name: &str) -> Result<(), ShellError> {
        if !self.sessions.contains_key(name) {
            return Err(ShellError::session_not_found(name));
        }

        let mut session = self.sessions.remove(name).unwrap();
        session.shutdown();

        if self.current_session == name {
            self.current_session = self
                .sessions
                .keys()
                .next()
                .cloned()
                .unwrap_or_else(|| "main".to_owned());
        }

        Ok(())
    }

    async fn restart_session(&mut self, name: &str, clean_env: bool) -> Result<(), ShellError> {
        if !self.sessions.contains_key(name) {
            return Err(ShellError::session_not_found(name));
        }

        let was_current = self.current_session == name;
        // Remember the cwd before killing.
        let cwd = self.sessions.get(name).unwrap().cwd.clone();

        self.kill_session(name).await?;
        self.create_session_internal(name, Some(&cwd), clean_env)
            .await?;

        if was_current {
            self.current_session = name.to_owned();
        }

        Ok(())
    }

    fn current_session(&self) -> &str {
        &self.current_session
    }
}

impl Drop for PtyBackend {
    fn drop(&mut self) {
        for (_, mut session) in self.sessions.drain() {
            session.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PtyBackend>();
    }

    #[test]
    fn scrollback_last_n_returns_recent_lines() {
        let mut buf = ScrollbackBuffer::new(100);
        for i in 0..5 {
            buf.append_line(&format!("line{i}"));
        }
        assert_eq!(buf.last_n(3), "line2\nline3\nline4");
    }

    #[test]
    fn scrollback_trims_oldest_on_overflow() {
        let mut buf = ScrollbackBuffer::new(3);
        for i in 0..5 {
            buf.append_line(&format!("line{i}"));
        }
        assert_eq!(buf.snapshot(), "line2\nline3\nline4");
    }

    #[test]
    fn extract_output_returns_content_between_sentinels() {
        let output = "__JUST_OUT_S__:00000001\nhello\nworld\n__JUST_EC__:0\n";
        let (cleaned, code) = PtyBackend::extract_output(output, "00000001");
        assert_eq!(code, Some(0));
        assert_eq!(cleaned, "hello\nworld");
    }

    #[test]
    fn extract_output_returns_empty_when_markers_missing() {
        let output = "no markers here\n";
        let (cleaned, code) = PtyBackend::extract_output(output, "00000001");
        assert_eq!(code, None);
        assert_eq!(cleaned, "");
    }

    #[test]
    fn extract_output_strips_echo_artifacts_as_fallback() {
        let output = "bash-5.3$ echo __JUST_OUT_S__:00000001\n__JUST_OUT_S__:00000001\nbash-5.3$ mkdir foo\nfoo created\nbash-5.3$ echo __JUST_EC__:0\n__JUST_EC__:0\n";
        let (cleaned, code) = PtyBackend::extract_output(output, "00000001");
        assert_eq!(code, Some(0));
        assert_eq!(cleaned, "bash-5.3$ mkdir foo\nfoo created");
    }

    #[test]
    fn extract_output_with_nonzero_exit_code() {
        let output = "__JUST_OUT_S__:00000042\nerror: something failed\n__JUST_EC__:1\n";
        let (cleaned, code) = PtyBackend::extract_output(output, "00000042");
        assert_eq!(code, Some(1));
        assert_eq!(cleaned, "error: something failed");
    }
}
