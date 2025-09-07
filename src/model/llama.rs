use crate::util::read_to_string;
use anyhow::{anyhow, Result};
use parking_lot::Mutex;
use std::{ffi::CString, path::PathBuf, sync::Arc};
use super::{InferParams, LlmBackend, PromptParts};


// Minimal llama.cpp sys imports (names may vary slightly by crate version)
use llama_cpp_sys as ll;


pub struct LlamaBackend {
_ctx: Arc<Mutex<*mut ll::llama_context>>,
_model: Arc<Mutex<*mut ll::llama_model>>,
grammar: String,
}


impl LlamaBackend {
pub fn new(model_path: PathBuf, n_ctx: i32, n_batch: i32, n_gpu_layers: i32) -> Result<Self> {
unsafe {
let mut params = ll::llama_model_default_params();
params.n_gpu_layers = n_gpu_layers;
let cpath = CString::new(model_path.to_string_lossy().to_string())?;
let model = ll::llama_load_model_from_file(cpath.as_ptr(), params);
if model.is_null() { return Err(anyhow!("failed to load model")); }


let mut cparams = ll::llama_context_default_params();
cparams.n_ctx = n_ctx;
cparams.n_batch = n_batch;
let ctx = ll::llama_new_context_with_model(model, cparams);
if ctx.is_null() { return Err(anyhow!("failed to create context")); }


let grammar = read_to_string("gbnf/word_contract.gbnf")?;
Ok(Self { _ctx: Arc::new(Mutex::new(ctx)), _model: Arc::new(Mutex::new(model)), grammar })
}
}
}


#[async_trait::async_trait]
impl LlmBackend for LlamaBackend {
async fn infer_json(&self, prompt: PromptParts, p: &InferParams) -> Result<Vec<u8>> {
// Build a compact system+user prompt. Keep it tiny; grammar does heavy lifting.
let sys = prompt.system;
let user = format!("Word: {}\nRespond with one JSON object only.", prompt.user_word);
let full = format!("<|system|>\n{}\n<|user|>\n{}\n<|assistant|>", sys, user);


let cfull = CString::new(full)?;
let cgrammar = CString::new(self.grammar.clone())?;


unsafe {
let ctx = *self._ctx.lock();


// Tokenize prompt
let tokens = ll::llama_tokenize(ctx, cfull.as_ptr(), true);
if tokens.data.is_null() { return Err(anyhow!("tokenize failed")); }


// Apply grammar
let g = ll::llama_grammar_init(cgrammar.as_ptr());


// Sampling params
let mut sparams = ll::llama_sampling_default_params();
sparams.temp = p.temp;
sparams.top_p = p.top_p;
sparams.min_p = p.min_p;
sparams.penalty_last_n = 64;
}
