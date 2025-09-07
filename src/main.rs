mod api; mod config; mod model; mod validate; mod util;
use axum::{Router};
use config::Config;
use dotenvy::dotenv;
use tracing_subscriber::{fmt, EnvFilter};
use crate::model::llama::LlamaBackend;
use crate::model::{InferParams};
use crate::validate::Validator;
use std::net::SocketAddr;


#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
dotenv().ok();
let cfg = <Config as clap::Parser>::parse();


// logs
let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
fmt().with_env_filter(filter).init();


// load schema & validator
let schema: serde_json::Value = serde_json::from_str(include_str!("../schema/word_contract.schema.json"))?;
let validator = Validator::new(&schema)?;


// llama backend
let backend = LlamaBackend::new(cfg.model_path.into(), cfg.n_ctx, cfg.n_batch, cfg.n_gpu_layers)?;


let params = InferParams {
max_tokens: cfg.max_tokens, temp: cfg.temp, top_p: cfg.top_p, min_p: cfg.min_p, repeat_penalty: cfg.repeat_penalty,
};


let app = api::routes(backend, validator, params);
let addr: SocketAddr = cfg.bind_addr.parse()?;


tracing::info!(%addr, "listening");
axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
Ok(())
}
