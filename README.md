# lingua-fast

🚀 **Fast linguistics API powered by llama.cpp**

Analyze words and get structured linguistic data through a high-performance Rust service. Uses GBNF-constrained generation and JSON Schema validation to ensure reliable, structured responses.

## Quick Start

### 1. Prerequisites

You'll need a GGUF model file (like Granite 3.3B) and llama.cpp built with acceleration:

**macOS (Apple Silicon):**

```bash
brew install cmake
cmake -B build -S vendor/llama.cpp -DLLAMA_METAL=1
cmake --build build -j
```

**Linux with NVIDIA GPU:**

```bash
cmake -B build -S vendor/llama.cpp -DLLAMA_CUBLAS=1
cmake --build build -j
```

### 2. Configure and Run

```bash
# Copy config template and set your model path
cp .env.example .env
# Edit MODEL_PATH in .env to point to your GGUF file

# Start the service
cargo run --release
```

### 3. Try it out

**Analyze a single word:**

```bash
curl -X POST http://127.0.0.1:8080/v1/word \
  -H 'content-type: application/json' \
  -d '{"word":"beautiful"}' | jq
```

**Batch processing:**

```bash
curl -X POST http://127.0.0.1:8080/v1/words \
  -H 'content-type: application/json' \
  -d '{"words":["happy","running","analysis"]}' | jq
```

## Features

✨ **Fast & Reliable**

- Concurrent batch processing (up to 8 parallel requests)
- GBNF grammar constraints ensure valid JSON structure
- JSON Schema validation for data quality
- Built-in error handling and graceful degradation

🔧 **Production Ready**

- Configurable via environment variables or CLI
- Built-in load testing tools
- Optimized for GPU acceleration
- Real inference testing (no mocks)

## API Response Format

Single word responses return linguistic analysis directly. Batch responses include per-item status to handle partial failures:

```json
[
  { "word": "beautiful", "ok": true, "data": { ... linguistic analysis ... } },
  { "word": "invalid", "ok": false, "error": "validation failed" },
  { "word": "happy", "ok": true, "data": { ... } }
]
```

## Performance Testing

```bash
# Load test with 200 concurrent requests
cargo run -p xtask --release -- http://127.0.0.1:8080/v1/word
```

## Configuration

Key settings (see `.env.example`):

- `MODEL_PATH` - Path to your GGUF model file *(required)*
- `N_GPU_LAYERS` - Number of layers to run on GPU (higher = faster)
- `TEMP` - Sampling temperature (0.3-0.5 recommended)
- `N_CTX` - Context window size

## Development

```bash
# Run tests (requires model file)
cargo test

# Format and lint
cargo fmt && cargo clippy --all-features -- -D warnings

# Build optimized release
cargo build --release
```

## Architecture

Built with modern Rust tooling:

- **Axum** - Fast async web framework
- **llama-cpp-2** - Safe Rust bindings to llama.cpp
- **tokio** - Async runtime with multi-threading
- **GBNF + JSON Schema** - Structured output validation

The service loads grammar constraints and JSON schemas at startup, ensuring consistent and validated linguistic analysis for every request.
