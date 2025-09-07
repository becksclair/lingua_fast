use super::{InferParams, LlmBackend, PromptParts};
use anyhow::Result;
#[cfg(feature = "mock-llama")]
use serde_json::json;
use std::path::PathBuf;

// Mock backend (default). Returns schema-conformant JSON without llama.cpp.
#[cfg(feature = "mock-llama")]
#[derive(Clone)]
pub struct LlamaBackend;

#[cfg(feature = "mock-llama")]
impl LlamaBackend {
    pub fn new(
        _model_path: PathBuf,
        _n_ctx: i32,
        _n_batch: i32,
        _n_gpu_layers: i32,
    ) -> Result<Self> {
        Ok(Self)
    }
}

#[cfg(feature = "mock-llama")]
#[async_trait::async_trait]
impl LlmBackend for LlamaBackend {
    async fn infer_json(&self, prompt: PromptParts, p: &InferParams) -> Result<Vec<u8>> {
        // Touch otherwise-unused fields to keep default build warning-free
        let _ = &prompt.system;
        let _ = (p.max_tokens, p.temp, p.top_p, p.min_p, p.repeat_penalty);
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

// Real llama.cpp backend using `llama-cpp-2` (enabled with feature `llama`).
#[cfg(feature = "llama")]
mod real_backend {
    use super::*;
    use anyhow::{anyhow, Context};
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_backend::LlamaBackend as LLBackend;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::params::LlamaModelParams;
    use llama_cpp_2::model::{AddBos, LlamaModel, Special};
    use llama_cpp_2::sampling::LlamaSampler;
    use llama_cpp_2::{ggml_time_us, send_logs_to_tracing, LogOptions};
    use std::num::NonZeroU32;
    use std::sync::Arc;
    use std::time::Duration;

    pub struct Inner {
        backend: LLBackend,
        model: LlamaModel,
        n_ctx: i32,
        _n_batch: i32,
    }

    #[derive(Clone)]
    pub struct LlamaBackend {
        inner: Arc<Inner>,
    }

    impl LlamaBackend {
        pub fn new(model_path: PathBuf, n_ctx: i32, n_batch: i32, n_gpu_layers: i32) -> Result<Self> {
            // route llama.cpp logs to tracing so they appear in server logs when RUST_LOG is set
            send_logs_to_tracing(LogOptions::default());

            let backend = LLBackend::init().context("init llama backend")?;

            let mut model_params = LlamaModelParams::default();
            if n_gpu_layers > 0 {
                // llama-cpp-2 expects u32
                model_params = model_params.with_n_gpu_layers(n_gpu_layers as u32);
            }

            let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
                .context("load GGUF model")?;

            Ok(Self {
                inner: Arc::new(Inner { backend, model, n_ctx, _n_batch: n_batch }),
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
            // Try to extract the first balanced {...} block to guard against stray tokens
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
                            return Some(s[st..=i].as_bytes().to_vec());
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
            // Build a fresh context per request to keep state simple and avoid cross-request leakage.
            // For higher throughput, this could be pooled, but we optimize for correctness here.
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(Some(NonZeroU32::new(self.inner.n_ctx as u32).unwrap()))
                .with_n_threads(num_cpus::get() as i32)
                .with_n_threads_batch(num_cpus::get() as i32);

            let mut ctx = self
                .inner
                .model
                .new_context(&self.inner.backend, ctx_params)
                .context("create llama context")?;

            // Tokenize prompt
            let prompt_text = Self::build_prompt(prompt);
            let tokens_list = self
                .inner
                .model
                .str_to_token(&prompt_text, AddBos::Always)
                .with_context(|| format!("tokenize prompt: {}", prompt_text))?;

            // Safety margin: ensure we don't exceed context window
            let n_ctx = ctx.n_ctx() as i32;
            let max_new = p.max_tokens.min((n_ctx - 8).saturating_sub(tokens_list.len() as i32));
            if max_new <= 0 {
                return Err(anyhow!("prompt too long for context"));
            }

            // Prepare batch
            let mut batch = LlamaBatch::new(512, 1);
            let last_index: i32 = (tokens_list.len() - 1) as i32;
            for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
                let is_last = i == last_index;
                batch.add(token, i, &[0], is_last)?;
            }
            ctx.decode(&mut batch).context("decode prompt")?;

            // Build sampler chain with grammar + sampling params
            let mut samplers: Vec<LlamaSampler> = vec![
                LlamaSampler::temp(p.temp),
                LlamaSampler::top_p(p.top_p, 1),
                LlamaSampler::min_p(p.min_p, 1),
                LlamaSampler::penalties(64, p.repeat_penalty, 0.0, 0.0),
            ];

            if let Some(grammar) = LlamaSampler::grammar(
                &self.inner.model,
                include_str!("../../gbnf/word_contract.gbnf"),
                "root",
            ) {
                samplers.push(grammar);
            }

            // Always terminate with a selection sampler
            samplers.push(LlamaSampler::greedy());
            let mut sampler = LlamaSampler::chain_simple(samplers);

            // Main generation loop
            let mut n_cur = batch.n_tokens();
            let mut n_decode = 0;
            let t_main_start = ggml_time_us();

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

            let _duration = Duration::from_micros((ggml_time_us() - t_main_start) as u64);

            // If we used grammar, the output should be JSON; still extract defensively.
            if let Some(bytes) = Self::extract_json_bytes(&out) {
                return Ok(bytes);
            }

            // Fallback: try entire string
            Ok(out.into_bytes())
        }
    }

    pub use LlamaBackend as RealLlamaBackend;
}

#[cfg(feature = "llama")]
pub use real_backend::RealLlamaBackend as LlamaBackend;
