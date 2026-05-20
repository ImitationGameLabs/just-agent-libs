use std::sync::Arc;

use anyhow::Result;
use just_llm_client::ToolDispatcher;
use just_llm_client::tools::shell::{PtyBackend, shell_tool_set};
use tokio::sync::Mutex;

use crate::context::{AgenticContext, ContextStore};
pub mod context;
pub mod skill;

pub use skill::{ensure_meta_skill, load_skill, pin_skill};

/// Builds the tool registry exposed by `just-agent`.
///
/// Spawns bash via [`PtyBackend`], preserving full shell session state.
/// The shell's working directory is the process current directory (set by
/// the caller via `std::env::set_current_dir`).
///
/// Context tools share the same `ContextStore` as the main loop.
pub async fn build_tool_dispatch(ctx: Arc<Mutex<ContextStore>>) -> Result<ToolDispatcher> {
    let backend = PtyBackend::new("main").await?;
    let backend = Arc::new(Mutex::new(backend));

    let mut dispatch = ToolDispatcher::new();
    dispatch.add_tools(shell_tool_set(backend))?;
    let ctx_dyn: Arc<Mutex<dyn AgenticContext>> = ctx;
    dispatch.add_tools(context::context_tool_set(ctx_dyn.clone()))?;
    dispatch.add_tools(skill::skill_tool_set(ctx_dyn))?;

    Ok(dispatch)
}
