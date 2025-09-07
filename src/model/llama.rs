use anyhow::Result;
use super::{InferParams, LlmBackend, PromptParts};
use serde_json::json;
use std::path::PathBuf;


// Mock backend (default). Returns schema-conformant JSON without llama.cpp.
#[cfg(feature = "mock-llama")]
#[derive(Clone)]
pub struct LlamaBackend;

#[cfg(feature = "mock-llama")]
impl LlamaBackend {
    pub fn new(_model_path: PathBuf, _n_ctx: i32, _n_batch: i32, _n_gpu_layers: i32) -> Result<Self> {
        Ok(Self)
    }
}

#[cfg(feature = "mock-llama")]
#[async_trait::async_trait]
impl LlmBackend for LlamaBackend {
    async fn infer_json(&self, prompt: PromptParts, _p: &InferParams) -> Result<Vec<u8>> {
        let word = prompt.user_word;
        let base_form = word.to_lowercase();
        let obj = json!({
            "word": word,
            "baseForm": base_form,
            "phonetic": "/ˈwɜːd/",
            "difficulty": "intermediate",
            "language": "english",
            "meanings": [
                {
                    "definition": "A carefully constructed placeholder definition that exceeds thirty characters for schema conformance.",
                    "partOfSpeech": "noun",
                    "exampleSentence": "This is a concise example sentence.",
                    "grammarTip": "Use consistently within proper grammatical context.",
                    "synonyms": ["term", "expression"],
                    "antonyms": ["silence"],
                    "translations": {"es":"palabra","fr":"mot","de":"Wort","zh":"词","ja":"言葉","it":"parola","pt":"palavra","ru":"слово","ar":"كلمة"}
                }
            ]
        });
        Ok(serde_json::to_vec(&obj)?)
    }
}


// Real llama.cpp backend (disabled by default). Not implemented in this fix.
#[cfg(feature = "llama")]
#[derive(Clone)]
pub struct LlamaBackend;

#[cfg(feature = "llama")]
impl LlamaBackend {
    pub fn new(_model_path: PathBuf, _n_ctx: i32, _n_batch: i32, _n_gpu_layers: i32) -> Result<Self> {
        compile_error!("The 'llama' feature is not implemented in this revision. Disable it or contribute the bindings.");
    }
}

#[cfg(feature = "llama")]
#[async_trait::async_trait]
impl LlmBackend for LlamaBackend {
    async fn infer_json(&self, _prompt: PromptParts, _p: &InferParams) -> Result<Vec<u8>> {
        unreachable!("llama backend is not available in this build");
    }
}
