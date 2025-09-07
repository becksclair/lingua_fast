//! Integration test for real llama.cpp inference.
//! Requires MODEL_PATH env var pointing to a local GGUF.

#[cfg(feature = "llama")]
#[tokio::test]
async fn real_inference_produces_json() -> anyhow::Result<()> {
    use lingua_fast::model::{llama::LlamaBackend, InferParams, PromptParts, LlmBackend};

    let model_path = match std::env::var("MODEL_PATH") {
        Ok(p) if !p.is_empty() => p,
        _ => {
            eprintln!("skipping: MODEL_PATH not set");
            return Ok(());
        }
    };

    // Conservative params to keep the test reasonably fast and deterministic
    let backend = LlamaBackend::new(model_path.into(), 2048, 256, 0)?;
    let params = InferParams { max_tokens: 512, temp: 0.4, top_p: 0.9, min_p: 0.05, repeat_penalty: 1.1 };

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

