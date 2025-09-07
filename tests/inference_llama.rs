//! Integration test for real llama.cpp inference.
//! Requires MODEL_PATH env var pointing to a local GGUF.

#[tokio::test]
async fn real_inference_produces_json() -> anyhow::Result<()> {
    use lingua_fast::model::{llama::LlamaBackend, InferParams, LlmBackend, PromptParts};
    use std::{env, fs, path::PathBuf};
    use walkdir::WalkDir;

    // Resolve model path: prefer $MODEL_PATH, else search ./models for any .gguf
    let model_path: PathBuf = env::var("MODEL_PATH")
        .ok()
        .map(PathBuf::from)
        .filter(|p| fs::metadata(p).is_ok())
        .or_else(|| {
            let root = PathBuf::from("./models");
            if fs::metadata(&root).is_ok() {
                for entry in WalkDir::new(&root).into_iter().filter_map(Result::ok) {
                    let p = entry.into_path();
                    if p.extension().and_then(|s| s.to_str()) == Some("gguf") {
                        return Some(p);
                    }
                }
            }
            None
        })
        .expect("No model found. Set MODEL_PATH or place a .gguf under ./models");

    // Conservative params to keep the test reasonably fast and deterministic
    let backend = LlamaBackend::new(model_path, 2048, 256, 0)?;
    let params = InferParams {
        max_tokens: 512,
        temp: 0.4,
        top_p: 0.9,
        min_p: 0.05,
        repeat_penalty: 1.1,
    };
    let prompt = PromptParts {
        system: "You are a linguistic annotator.".to_string(),
        user_word: "communicated".to_string(),
    };

    let bytes = backend.infer_json(prompt, &params).await?;
    let v: serde_json::Value = serde_json::from_slice(&bytes)?;

    // Minimal sanity checks on expected shape
    assert!(v.get("word").is_some(), "missing word field");
    assert!(v.get("meanings").is_some(), "missing meanings array");

    Ok(())
}
