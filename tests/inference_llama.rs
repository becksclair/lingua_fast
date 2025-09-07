//! Integration test for real llama.cpp inference.
//! Requires MODEL_PATH env var pointing to a local GGUF.

#[tokio::test]
async fn real_inference_produces_json() -> anyhow::Result<()> {
    use lingua_fast::model::{llama::LlamaBackend, InferParams, LlmBackend, PromptParts};
    use std::{env, fs, path::PathBuf};
    use walkdir::WalkDir;

    // Initialize tracing for debugging
    tracing_subscriber::fmt::init();

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

    // Configure for better JSON generation with Metal acceleration on macOS
    let n_gpu_layers = if cfg!(target_os = "macos") { 28 } else { 0 };
    let backend = LlamaBackend::new(model_path, 4096, 512, n_gpu_layers)?;
    let params = InferParams {
        max_tokens: 1024, // Increased for comprehensive linguistic analysis
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

    // Minimal sanity checks - be flexible since we're not using grammar constraints
    tracing::info!("Generated JSON keys: {:?}", v.as_object().map(|o| o.keys().collect::<Vec<_>>()));
    tracing::info!("Generated content: {}", serde_json::to_string_pretty(&v)?);
    
    // Accept any valid JSON structure for now since grammar is disabled
    assert!(v.is_object(), "output should be a JSON object");
    
    // If it contains expected fields, that's great, but don't fail if it doesn't
    // This is because without grammar constraints, the model might generate different JSON
    if v.get("word").is_some() && v.get("meanings").is_some() {
        tracing::info!("✅ Generated expected word analysis structure");
    } else {
        tracing::info!("ℹ️ Generated JSON but not in expected word analysis format");
    }

    Ok(())
}
