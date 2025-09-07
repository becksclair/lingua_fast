use clap::Parser;

#[derive(Parser, Debug, Clone)]
pub struct Config {
    #[arg(long, env, default_value = "0.0.0.0:8080")]
    pub bind_addr: String,
    #[arg(long = "MODEL_PATH", env = "MODEL_PATH")]
    pub model_path: String,
    // Must be >= 1 to satisfy NonZeroU32 context requirement
    #[arg(long, env, default_value_t = 4096, value_parser = clap::value_parser!(i32).range(1..))]
    pub n_ctx: i32,
    #[arg(long, env, default_value_t = 256)]
    pub n_batch: i32,
    // Disallow negatives; 0 means CPU-only inference
    #[arg(long, env, default_value_t = 28, value_parser = clap::value_parser!(i32).range(0..))]
    pub n_gpu_layers: i32,
    // 0 means auto-detect (use all available logical CPUs)
    #[arg(long, env = "THREADS", default_value_t = 0, value_parser = clap::value_parser!(i32).range(0..))]
    pub threads: i32,
    // 0 means default (min(8, num_cpus)) per-process inference concurrency
    #[arg(long = "INFER_CONCURRENCY", env = "INFER_CONCURRENCY", default_value_t = 0, value_parser = clap::value_parser!(i32).range(0..))]
    pub infer_concurrency: i32,
    #[arg(long, env, default_value_t = 1024)]
    pub max_tokens: i32,
    #[arg(long, env, default_value_t = 0.4)]
    pub temp: f32,
    #[arg(long, env, default_value_t = 0.9)]
    pub top_p: f32,
    #[arg(long, env, default_value_t = 0.05)]
    pub min_p: f32,
    #[arg(long, env, default_value_t = 1.1)]
    pub repeat_penalty: f32,
}
