# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a Rust Axum service that provides a linguistics API using llama.cpp for LLM inference with GBNF-constrained JSON generation. The service analyzes words and returns structured linguistic data.

**Key Components:**

- `src/main.rs` - Entry point with tokio runtime, configuration parsing, and service setup
- `src/api.rs` - HTTP routes (`/v1/word`, `/v1/words`) with concurrent batch processing
- `src/model/llama.rs` - llama.cpp backend integration via `llama-cpp-2` crate
- `src/validate.rs` - JSON Schema validation using embedded schema file
- `src/config.rs` - CLI argument parsing and environment variable handling
- `gbnf/word_contract.gbnf` - Grammar constraints for structured LLM output
- `schema/word_contract.schema.json` - JSON Schema for response validation
- `xtask/` - Load testing utility workspace member

## Development Commands

**Build and Run:**

```bash
# Development build
cargo build

# Production build
cargo build --release

# Run service (uses .env configuration)
cargo run --release

# Or with explicit CLI arguments
cargo run --release -- --bind-addr 0.0.0.0:8080 --MODEL_PATH /path/to/model.gguf

# Copy configuration template
cp .env.example .env
# Edit MODEL_PATH in .env file
```

**Testing:**

```bash
# Run all tests (requires real model file - tests use actual inference)
cargo test

# Run tests with output
cargo test -- --nocapture

# Override default model path for tests
export MODEL_PATH=/path/to/model.gguf && cargo test
```

**Code Quality:**

```bash
# Format code
cargo fmt

# Run lints
cargo clippy --all-targets --all-features -- -D warnings
```

**Load Testing:**

```bash
# Run load test against running service
cargo run -p xtask --release -- http://127.0.0.1:8080/v1/word
```

## llama.cpp Setup

The service requires llama.cpp to be built separately with platform-appropriate acceleration:

**macOS (Apple Silicon):**

```bash
brew install cmake
cmake -B build -S vendor/llama.cpp -DLLAMA_METAL=1
cmake --build build -j
```

**Linux with NVIDIA:**

```bash
cmake -B build -S vendor/llama.cpp -DLLAMA_CUBLAS=1
cmake --build build -j
```

## Configuration

Configuration is handled via environment variables and CLI arguments. Key settings in `.env`:

- `MODEL_PATH` - Path to GGUF model file (required)
- `N_CTX` - Context size (default: 4096)
- `N_GPU_LAYERS` - GPU layer count for acceleration
- `TEMP`, `TOP_P`, `MIN_P`, `REPEAT_PENALTY` - Sampling parameters

## API Usage

**Single word analysis:**

```bash
curl -X POST http://127.0.0.1:8080/v1/word \
  -H 'content-type: application/json' \
  -d '{"word":"communicated"}'
```

**Batch processing:**

```bash
curl -X POST http://127.0.0.1:8080/v1/words \
  -H 'content-type: application/json' \
  -d '{"words":["communicated","run","magnificent"]}'
```

Batch responses include per-item status to handle partial failures gracefully.

## Important Notes

- Tests require actual model inference - never mock the LLM backend
- Batch processing uses controlled concurrency (max 8 parallel requests)
- GBNF grammar constrains LLM output structure, JSON Schema validates results
- The service loads schema at startup from embedded `schema/word_contract.schema.json`
- Default model path for tests: `./models/granite/granite-3.3-2b-instruct-Q4_K_M.gguf`
