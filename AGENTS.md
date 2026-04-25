# AGENTS.md

AI Agent working guide. This document provides code structure and decision rules for AI agents.

## Directory Structure

```
.
├── flake.nix              # Flake entry point
├── crates/                # Rust workspace members
│   ├── just-common/      # Shared HTTP transport, SSE parsing, and error types
│   ├── just-llm-client/  # Provider-neutral LLM client traits, types, and adapters
│   └── providers/        # Backend-specific provider crates
│       ├── just-deepseek/         # DeepSeek provider SDK
│       └── just-openai-compat/    # OpenAI-compatible provider SDK
├── docs/                  # Project documentation
└── nix/
      ├── common.nix       # Core config (crate paths, dependencies)
      └── dev/
            ├── shell.nix  # Development environment
            └── checks.nix # CI checks
```

## Common Tasks

For adding workspace members, see [add-workspace-member.md](docs/agent-wizards/add-workspace-member.md).

## Dependency Management

When adding dependencies to any crate:
1. Look up the latest version: `cargo search <crate-name> --registry crates-io`
2. Add to `[workspace.dependencies]` in root `Cargo.toml`
3. Reference in crate's `Cargo.toml` with `workspace = true`

Example:
```toml
# Root Cargo.toml
[workspace.dependencies]
serde = { version = "1.0", features = ["derive"] }

# crates/my-app/Cargo.toml
[dependencies]
serde = { workspace = true }
```

## Verification Checklist

After modifying Nix files:
- `nixfmt <nix file>` - Format single file
- `nixfmt $(find nix/ -name "*.nix") flake.nix` - Format all Nix files at once
- `statix check .` - Static analysis (run from project root)

After modifying TOML files:
- `taplo fmt <toml file>` - Format specific file (never use bare `taplo fmt` — it ignores .gitignore and formats everything)

After modifying Rust code:
- `cargo fmt` - Format check
- `cargo clippy --workspace --all-targets --all-features` - Lint check
- `cargo test --workspace --all-targets --all-features` - Run tests
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps` - Build docs and fail on rustdoc warnings
