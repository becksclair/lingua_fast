use crate::{
    model::{InferParams, LlmBackend, PromptParts},
    validate::Validator,
};
use anyhow::Result;
use axum::{http::StatusCode, response::IntoResponse, routing::post, Json, Router};
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;

#[derive(Deserialize)]
pub struct WordReq {
    pub word: String,
}

pub fn routes<B: LlmBackend + Clone + 'static>(
    backend: B,
    validator: Arc<Validator>,
    params: InferParams,
) -> Router {
    Router::new().route("/v1/word", post(move |Json(req): Json<WordReq>| {
let backend = backend.clone();
let validator = validator.clone();
let params = params.clone();
 async move {
let system = "You are an expert linguist and lexicographer. Produce a single valid JSON object only.".to_string();
let prompt = PromptParts { system, user_word: req.word.clone() };
// First attempt
let result: Result<Json<Value>, anyhow::Error> = async {
    let bytes = backend.infer_json(prompt, &params).await?;
    let v: Value = serde_json::from_slice(&bytes)?;
    let v = validator.validate_and_fix(v, &req.word)?;
    Ok(Json(v))
}.await;
match result {
    Ok(json) => json.into_response(),
    Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
}
}
}))
}
