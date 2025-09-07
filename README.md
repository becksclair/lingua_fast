# lingua-fast

Axum service that generates linguistics JSON via llama.cpp with GBNF-constrained decoding and JSON Schema validation.


## Build llama.cpp


**macOS (Apple Silicon):**
```bash
brew install cmake
cmake -B build -S vendor/llama.cpp -DLLAMA_METAL=1
cmake --build build -j
```
Set `LLAMA_METAL=1` env when using a crate that links it, or ensure bindings can find the dynamic lib.


**Linux:**
```bash
cmake -B build -S vendor/llama.cpp -DLLAMA_CUBLAS=1 # if NVIDIA
cmake --build build -j
```


## Run service
```bash
cp .env.example .env
# edit MODEL_PATH to your Granite GGUF
cargo run --release -- \
--bind-addr 0.0.0.0:8080 \
--MODEL_PATH "$MODEL_PATH" \
--n-ctx 4096 --n-batch 256 --n-gpu-layers 28
```


### Request
```bash
curl -s http://127.0.0.1:8080/v1/word -X POST \
-H 'content-type: application/json' \
-d '{"word":"communicated"}' | jq
```


## Load test
```bash
cargo run -p xtask --release -- http://127.0.0.1:8080/v1/word
```


## Tuning
- Prefer `temp 0.3–0.5`, `top_p 0.9`, `min_p 0.05`, `repeat_penalty 1.1`.
- Increase `n_gpu_layers` on Apple Silicon / CUDA hosts until VRAM hits limits.
- Keep the system prompt minimal; rely on GBNF for structure.


### What you’ll likely tweak next

- Replace the minimal llama sys calls with the exact functions exposed by your `llama-cpp-sys` version (names drift slightly across releases).
- Consider pinning a specific llama.cpp commit and embedding as a `vendor/llama.cpp` submodule.
- Add a one-retry path with slightly safer sampling if validation fails.
- Add Prometheus exporter and a `/metrics` route.
- Add per-model KV cache warming for the system prompt.

## Features
- By default, the build uses the real llama.cpp backend via the `llama-cpp-2` crate.
- A `mock-llama` feature exists only for local development convenience if you want to skip compiling llama.cpp, but it is not enabled by default.

### Run (real inference)
```bash
cargo run --release -- \
  --bind-addr 0.0.0.0:8080 \
  --MODEL_PATH "$MODEL_PATH" \
  --n-ctx 4096 --n-batch 256 --n-gpu-layers 28
```

### Tests (real inference)
An integration test runs real inference when `MODEL_PATH` is set and otherwise skips.
```bash
export MODEL_PATH=/path/to/model.gguf
cargo test --test inference_llama -- --nocapture
```

If you previously built with Ninja as the CMake generator, you may need a clean build to switch to Makefiles:
```bash
cargo clean && cargo build
```
