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
    n_batch: i32,
    threads: i32,
    limiter: Arc<Semaphore>,
}

#[derive(Clone)]
pub struct LlamaBackend {
    inner: Arc<Inner>,
}

impl LlamaBackend {
    pub fn new(
        model_path: PathBuf,
        n_ctx: i32,
        n_batch: i32,
        n_gpu_layers: i32,
        threads: i32,
        infer_concurrency: i32,
    ) -> Result<Self> {
        tracing::info!("Initializing LlamaBackend with model_path={:?}, n_ctx={}, n_batch={}, n_gpu_layers={}",
                      model_path, n_ctx, n_batch, n_gpu_layers);

        send_logs_to_tracing(LogOptions::default());

        tracing::debug!("Initializing llama backend...");
        let backend = LLBackend::init().context("init llama backend")?;
        tracing::debug!("Llama backend initialized successfully");

        let mut model_params = LlamaModelParams::default();
        if n_gpu_layers > 0 {
            tracing::info!("Enabling {} GPU layers", n_gpu_layers);
            model_params = model_params.with_n_gpu_layers(n_gpu_layers as u32);
        }

        tracing::info!("Loading model from file: {:?}", model_path);
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .context("load GGUF model")?;
        tracing::info!("Model loaded successfully");

        let permits = if infer_concurrency > 0 {
            usize::max(1, infer_concurrency as usize)
        } else {
            usize::min(8, usize::max(1, num_cpus::get()))
        };

        Ok(Self {
            inner: Arc::new(Inner {
                backend,
                model,
                n_ctx,
                n_batch,
                threads,
                limiter: Arc::new(Semaphore::new(permits)),
            }),
        })
    }

    fn build_prompt(prompt: PromptParts) -> String {
        format!(
            "{sys}\n\nYou are an expert linguist and lexicographer. Your only job is to produce a single valid JSON object describing an English word.\n\n## OUTPUT CONTRACT — ABSOLUTE RULES\n\n1) Output must be a single JSON object only. No explanations, no code fences, no comments, no trailing commas, no nulls, no placeholders like \"<...>\", no markdown.\n2) All required fields must be present and non-empty strings or arrays (arrays may be empty but must exist).\n3) Use straight quotes (\") only. Escape any internal quotes per JSON.\n4) Use UTF-8. IPA must be valid IPA characters.\n\n## CONTENT REQUIREMENTS\n\n- \"word\": the surface/inflected form exactly as given by the user (case-preserve).\n- \"baseForm\": the lemma/root form in lowercase.\n- \"phonetic\": the IPA transcription in slashes, e.g., \"/kəˈmjuːnɪkeɪt/\". Use a standard, contemporary pronunciation (General American or widely accepted international), not a regional outlier.\n- \"difficulty\": one of \"beginner\", \"intermediate\", \"advanced\" based on typical frequency and morphology; choose conservatively.\n- \"language\": always \"english\".\n- \"meanings\": an array of 1-4 sense objects. Each sense MUST have a unique \"partOfSpeech\" value across the array.\n  • \"definition\": 30-80 words, clear, neutral, and sense-specific; do not repeat the headword mechanically.\n  • \"partOfSpeech\": one of [\"noun\",\"verb\",\"adjective\",\"adverb\",\"pronoun\",\"preposition\",\"conjunction\",\"interjection\",\"article\",\"determiner\",\"numeral\",\"participle\",\"gerund\"].\n  • \"exampleSentence\": natural, contemporary usage; keep under 25 words; do not quote famous works.\n  • \"grammarTip\": short usage guidance (morphology, typical complements, common errors, or register).\n  • \"synonyms\": 2-8 near-synonyms as single tokens or short phrases; none may duplicate the headword; keep sense-appropriate.\n  • \"antonyms\": 0-6 reasonable opposites; empty array allowed if none fit.\n  • \"translations\": object with keys [\"es\",\"fr\",\"de\",\"zh\",\"ja\",\"it\",\"pt\",\"ru\",\"ar\"]; each value a common single-word or brief phrase capturing THIS sense.\n\n## QUALITY & CONSISTENCY CHECKS (perform before finalizing):\n\n- Valid JSON when parsed strictly.\n- \"meanings\" present with 1-4 items and all \"partOfSpeech\" values unique.\n- No hallucinated morphology (e.g., correct lemma and typical inflections).\n- No repetitive or circular definitions.\n- Translations match each individual sense, not copied across blindly.\n- Arrays contain unique, lower-case items unless proper-case is standard.\n- No extra keys beyond the schema.\n\nWord: {word}\nRespond with the JSON object only.",
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
        tracing::info!("Starting inference for word: {}", prompt.user_word);
        let _permit = self
            .inner
            .limiter
            .acquire()
            .await
            .expect("semaphore not closed");

        let threads = if self.inner.threads > 0 {
            self.inner.threads as i32
        } else {
            num_cpus::get() as i32
        };
        tracing::debug!("Creating context with n_ctx={}, n_threads={}",
                       self.inner.n_ctx, threads);
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(self.inner.n_ctx as u32).unwrap()))
            .with_n_threads(threads)
            .with_n_threads_batch(threads);
        let mut ctx = self
            .inner
            .model
            .new_context(&self.inner.backend, ctx_params)
            .context("create llama context")?;
        tracing::debug!("Context created successfully");

        let prompt_text = Self::build_prompt(prompt);
        tracing::debug!("Built prompt (length={}): {}", prompt_text.len(), &prompt_text[..prompt_text.len().min(200)]);

        let tokens_list = self
            .inner
            .model
            .str_to_token(&prompt_text, AddBos::Always)
            .with_context(|| format!("tokenize prompt: {}", prompt_text))?;
        tracing::debug!("Tokenized prompt into {} tokens", tokens_list.len());

        let n_ctx = ctx.n_ctx() as i32;
        let max_new = p
            .max_tokens
            .min((n_ctx - 8).saturating_sub(tokens_list.len() as i32));
        tracing::info!("Context size: {}, prompt tokens: {}, max new tokens: {}",
                      n_ctx, tokens_list.len(), max_new);
        if max_new <= 0 {
            return Err(anyhow!("prompt too long for context: {} tokens exceeds {} context size",
                              tokens_list.len(), n_ctx));
        }

        tracing::debug!("Creating batch and decoding prompt...");
        let mut batch = LlamaBatch::new(self.inner.n_batch as usize, 1);
        let last_index: i32 = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last)
                .with_context(|| format!("failed to add token {} to batch at position {}", token, i))?;
        }
        ctx.decode(&mut batch)
            .context("decode prompt - this may indicate model compatibility issues")?;
        tracing::debug!("Prompt decoded successfully");

        let mut samplers: Vec<LlamaSampler> = vec![
            LlamaSampler::temp(p.temp),
            LlamaSampler::top_p(p.top_p, 1),
            LlamaSampler::min_p(p.min_p, 1),
            LlamaSampler::penalties(64, p.repeat_penalty, 0.0, 0.0),
        ];

        // Skip GBNF grammar due to inference crashes - use JSON extraction instead
        tracing::info!("Using unconstrained generation with JSON extraction (GBNF disabled due to stability issues)");
        // Note: GBNF grammar constraints cause SIGABRT during inference with this model/setup
        // The extract_json_bytes function will extract valid JSON from the free-form output

        samplers.push(LlamaSampler::greedy());
        let mut sampler = LlamaSampler::chain_simple(samplers);

        let mut n_cur = batch.n_tokens();
        let mut n_decode = 0;
        let _t_main_start = ggml_time_us();

        let mut out = String::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        tracing::info!("Starting generation loop, max_new={}", max_new);
        while n_decode < max_new {
            tracing::trace!("Sampling token {} of {}", n_decode + 1, max_new);

            // Sample next token with error handling
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.inner.model.is_eog_token(token) {
                tracing::debug!("Encountered end-of-generation token at position {}", n_decode);
                break;
            }

            // Convert token to string with error handling
            let output_bytes = self.inner.model.token_to_bytes(token, Special::Tokenize)
                .with_context(|| format!("failed to convert token {} to bytes", token))?;
            let mut output_string = String::with_capacity(16);
            let _ = decoder.decode_to_string(&output_bytes, &mut output_string, false);
            out.push_str(&output_string);

            // Prepare for next iteration
            batch.clear();
            batch.add(token, n_cur, &[0], true)
                .with_context(|| format!("failed to add generated token {} to batch", token))?;
            n_cur += 1;
            ctx.decode(&mut batch)
                .with_context(|| format!("decode step failed at token {}", n_decode + 1))?;
            n_decode += 1;
        }

        tracing::info!("Generation completed after {} tokens, output length: {}",
                      n_decode, out.len());
        tracing::debug!("Raw output: {}", &out[..out.len().min(500)]);

        if let Some(bytes) = Self::extract_json_bytes(&out) {
            return Ok(bytes);
        }

        Ok(out.into_bytes())
    }
}
