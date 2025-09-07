# Repository Guidelines

## Project Structure & Module Organization

- Rust 2021 workspace. Main crate at `src/` with Axum HTTP API and llama.cpp backend.
- Key paths: `src/api.rs` (routes), `src/model/` (LLM backend), `src/validate.rs` (JSON Schema), `src/config.rs` (CLI/env), `gbnf/` (grammar), `schema/` (JSON Schema), `build.rs` (build hints).
- Example assets: `models` symlink (local GGUFs), `.env.example` for configuration.

## Build, Test, and Development Commands

- Build service: `cargo build` (use `--release` for perf).
- Run service: `cargo run --release -- --MODEL_PATH /path/to/model.gguf --bind-addr 0.0.0.0:8080`.
- Load test (if xtask is available): `cargo run -p xtask --release --bin loadtest -- http://127.0.0.1:8080/v1/word`.
- Build llama.cpp (macOS): `cmake -B build -S vendor/llama.cpp -DLLAMA_METAL=1 && cmake --build build -j` (Linux: use `-DLLAMA_CUBLAS=1`).

## Coding Style & Naming Conventions

- Follow Rust conventions: `snake_case` for modules/functions, `PascalCase` for types, `SCREAMING_SNAKE_CASE` for consts.
- Format with `cargo fmt` and lint with `cargo clippy --all-targets --all-features -D warnings`.
- Keep modules focused; place public APIs in `src/api.rs` and backend-specific code under `src/model/`.

## Testing Guidelines

- Unit tests: co-locate with modules using `#[cfg(test)]` in `src/*.rs`.
- Integration tests: create `tests/` and use `cargo test` to run all tests.
- Prefer integration tests; the project only works if tests pass with real LLM inference, never mock anything.
- Always verify tests by running `cargo test`

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject (e.g., `add validator for synonyms dedupe`). Group related changes.
- PRs: include description, rationale, and how to test (commands and sample `curl`). Link issues when applicable.
- Include screenshots or sample JSON when changing API responses. Note any schema/GBNF updates explicitly.

## Security & Configuration Tips
- Use `.env` (see `.env.example`); required: `MODEL_PATH`. Do not commit secrets.
- Prefer minimal prompts; rely on `gbnf/word_contract.gbnf` and `schema/word_contract.schema.json` for structure and validation.
- Ensure llama.cpp is built with the correct accelerator for your host.

