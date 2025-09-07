mod api;
mod config;
mod model;
mod util;
mod validate;
use crate::model::llama::LlamaBackend;
use crate::model::InferParams;
use crate::validate::Validator;
use config::Config;
use dotenvy::dotenv;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing_subscriber::{fmt, EnvFilter};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();
    let cfg = <Config as clap::Parser>::parse();

    // logs
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).init();

    // load schema & validator
    let schema_src: &str = include_str!("../schema/word_contract.schema.json");
    let validator = Arc::new(Validator::new(schema_src)?);

    // llama backend
    let backend = LlamaBackend::new(
        cfg.model_path.into(),
        cfg.n_ctx,
        cfg.n_batch,
        cfg.n_gpu_layers,
        cfg.threads,
        cfg.infer_concurrency,
    )?;

    let params = InferParams {
        max_tokens: cfg.max_tokens,
        temp: cfg.temp,
        top_p: cfg.top_p,
        min_p: cfg.min_p,
        repeat_penalty: cfg.repeat_penalty,
    };

    let app = api::routes(backend, validator, params);
    let addr: SocketAddr = cfg.bind_addr.parse()?;

    tracing::info!(%addr, "listening");
    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
    Ok(())
}
