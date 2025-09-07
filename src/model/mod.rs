use anyhow::Result;


#[derive(Clone, Debug)]
pub struct InferParams {
pub max_tokens: i32,
pub temp: f32,
pub top_p: f32,
pub min_p: f32,
pub repeat_penalty: f32,
}


#[derive(Clone)]
pub struct PromptParts {
pub system: String,
pub user_word: String,
}


#[async_trait::async_trait]
pub trait LlmBackend: Send + Sync + 'static {
async fn infer_json(&self, prompt: PromptParts, params: &InferParams) -> Result<Vec<u8>>;
}


pub mod llama;
