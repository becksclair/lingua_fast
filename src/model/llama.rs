use super::{InferParams, LlmBackend, PromptParts};

use anyhow::{anyhow, Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend as LLBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{ggml_time_us, send_logs_to_tracing, LogOptions};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Semaphore;

pub struct Inner {
    backend: LLBackend,
    model: LlamaModel,
    n_ctx: i32,
    _n_batch: i32,
    limiter: Arc<Semaphore>,
}

#[derive(Clone)]
pub struct LlamaBackend {
    inner: Arc<Inner>,
}

impl LlamaBackend {
    pub fn new(model_path: PathBuf, n_ctx: i32, n_batch: i32, n_gpu_layers: i32) -> Result<Self> {
        send_logs_to_tracing(LogOptions::default());

        let backend = LLBackend::init().context("init llama backend")?;

        let mut model_params = LlamaModelParams::default();
        if n_gpu_layers > 0 {
            model_params = model_params.with_n_gpu_layers(n_gpu_layers as u32);
        }

        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .context("load GGUF model")?;

        let permits = usize::min(8, usize::max(1, num_cpus::get()));

        Ok(Self {
            inner: Arc::new(Inner {
                backend,
                model,
                n_ctx,
                _n_batch: n_batch,
                limiter: Arc::new(Semaphore::new(permits)),
            }),
        })
    }

    fn build_prompt(prompt: PromptParts) -> String {
        format!(
            "{sys}\n\nTask: Given the word, produce a single valid JSON object matching the expected schema. Do not include any preface, only JSON.\nWord: {word}\nOutput strictly as JSON only.",
            sys = prompt.system,
            word = prompt.user_word
        )
    }

    fn extract_json_bytes(s: &str) -> Option<Vec<u8>> {
        let mut depth = 0i32;
        let mut start = None;
        for (i, ch) in s.char_indices() {
            if ch == '{' {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            } else if ch == '}' {
                depth -= 1;
                if depth == 0 {
                    if let Some(st) = start {
                        return Some(s.as_bytes()[st..=i].to_vec());
                    }
                }
            }
        }
        None
    }
}

#[async_trait::async_trait]
impl LlmBackend for LlamaBackend {
    async fn infer_json(&self, prompt: PromptParts, p: &InferParams) -> Result<Vec<u8>> {
        let _permit = self
            .inner
            .limiter
            .acquire()
            .await
            .expect("semaphore not closed");

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(self.inner.n_ctx as u32).unwrap()))
            .with_n_threads(num_cpus::get() as i32)
            .with_n_threads_batch(num_cpus::get() as i32);
        let mut ctx = self
            .inner
            .model
            .new_context(&self.inner.backend, ctx_params)
            .context("create llama context")?;

        let prompt_text = Self::build_prompt(prompt);
        let tokens_list = self
            .inner
            .model
            .str_to_token(&prompt_text, AddBos::Always)
            .with_context(|| format!("tokenize prompt: {}", prompt_text))?;

        let n_ctx = ctx.n_ctx() as i32;
        let max_new = p
            .max_tokens
            .min((n_ctx - 8).saturating_sub(tokens_list.len() as i32));
        if max_new <= 0 {
            return Err(anyhow!("prompt too long for context"));
        }

        let mut batch = LlamaBatch::new(512, 1);
        let last_index: i32 = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last)?;
        }
        ctx.decode(&mut batch).context("decode prompt")?;

        let mut samplers: Vec<LlamaSampler> = vec![
            LlamaSampler::temp(p.temp),
            LlamaSampler::top_p(p.top_p, 1),
            LlamaSampler::min_p(p.min_p, 1),
            LlamaSampler::penalties(64, p.repeat_penalty, 0.0, 0.0),
        ];

        let grammar = LlamaSampler::grammar(
            &self.inner.model,
            include_str!("../../gbnf/word_contract.gbnf"),
            "root",
        )
        .ok_or_else(|| anyhow!("failed to parse GBNF grammar"))?;
        samplers.push(grammar);

        samplers.push(LlamaSampler::greedy());
        let mut sampler = LlamaSampler::chain_simple(samplers);

        let mut n_cur = batch.n_tokens();
        let mut n_decode = 0;
        let _t_main_start = ggml_time_us();

        let mut out = String::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        while n_decode < max_new {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.inner.model.is_eog_token(token) {
                break;
            }

            let output_bytes = self.inner.model.token_to_bytes(token, Special::Tokenize)?;
            let mut output_string = String::with_capacity(16);
            let _ = decoder.decode_to_string(&output_bytes, &mut output_string, false);
            out.push_str(&output_string);

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;
            ctx.decode(&mut batch).context("decode step")?;
            n_decode += 1;
        }

        if let Some(bytes) = Self::extract_json_bytes(&out) {
            return Ok(bytes);
        }

        Ok(out.into_bytes())
    }
}
