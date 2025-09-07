use clap::Parser;

#[derive(Parser, Debug, Clone)]
pub struct Config {
    #[arg(long, env, default_value = "0.0.0.0:8080")]
    pub bind_addr: String,
    #[arg(long = "MODEL_PATH", env = "MODEL_PATH")]
    pub model_path: String,
    #[arg(long, env, default_value_t = 4096)]
    pub n_ctx: i32,
    #[arg(long, env, default_value_t = 256)]
    pub n_batch: i32,
    #[arg(long, env, default_value_t = 28)]
    pub n_gpu_layers: i32,
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
