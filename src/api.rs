use crate::{
    model::{InferParams, LlmBackend, PromptParts},
    validate::Validator,
};
use anyhow::Result;
use axum::{http::StatusCode, response::IntoResponse, routing::post, Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;

#[derive(Deserialize)]
pub struct WordReq {
    pub word: String,
}

#[derive(Deserialize)]
pub struct BatchReq {
    pub words: Vec<String>,
}

pub fn routes<B: LlmBackend + Clone + 'static>(
    backend: B,
    validator: Arc<Validator>,
    params: InferParams,
) -> Router {
    let backend_single = backend.clone();
    let validator_single = validator.clone();
    let params_single = params.clone();
    let backend_batch = backend.clone();
    let validator_batch = validator.clone();
    let params_batch = params.clone();

    Router::new()
        .route("/v1/word", post(move |Json(req): Json<WordReq>| {
let backend = backend_single.clone();
let validator = validator_single.clone();
let params = params_single.clone();
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
        .route("/v1/words", post(move |Json(req): Json<BatchReq>| {
            let backend = backend_batch.clone();
            let validator = validator_batch.clone();
            let params = params_batch.clone();
            async move {
                let n = req.words.len();
                let mut results: Vec<Option<Value>> = vec![None; n];

                // Concurrency with order preservation
                let mut set = tokio::task::JoinSet::new();
                let concurrency_limit = usize::min(8, num_cpus::get());
                for (idx, word) in req.words.iter().cloned().enumerate() {
                    let backend = backend.clone();
                    let validator = validator.clone();
                    let params = params.clone();
                    set.spawn(async move {
                        let system = "You are an expert linguist and lexicographer. Produce a single valid JSON object only.".to_string();
                        let prompt = PromptParts { system, user_word: word.clone() };
                        let result: anyhow::Result<Value> = async {
                            let bytes = backend.infer_json(prompt, &params).await?;
                            let v: Value = serde_json::from_slice(&bytes)?;
                            let v = validator.validate_and_fix(v, &word)?;
                            Ok(v)
                        }
                        .await;
                        Ok::<(usize, anyhow::Result<Value>), anyhow::Error>((idx, result))
                    });

                    // Backpressure to cap concurrency
                    if set.len() >= concurrency_limit {
                        if let Some(res) = set.join_next().await {
                            match res {
                                Ok(Ok((idx, inner))) => {
                                    match inner {
                                        Ok(v) => {
                                            results[idx] = Some(json!({
                                                "word": req.words[idx].clone(),
                                                "ok": true,
                                                "data": v,
                                            }));
                                        }
                                        Err(e) => {
                                            results[idx] = Some(json!({
                                                "word": req.words[idx].clone(),
                                                "ok": false,
                                                "error": e.to_string(),
                                            }));
                                        }
                                    }
                                }
                                Ok(Err(e)) => {
                                    if let Some(i) = results.iter().position(|x| x.is_none()) {
                                        results[i] = Some(json!({
                                            "word": req.words[i].clone(),
                                            "ok": false,
                                            "error": e.to_string(),
                                        }));
                                    }
                                }
                                Err(join_err) => {
                                    if let Some(i) = results.iter().position(|x| x.is_none()) {
                                        results[i] = Some(json!({
                                            "word": req.words[i].clone(),
                                            "ok": false,
                                            "error": join_err.to_string(),
                                        }));
                                    }
                                }
                            }
                        }
                    }
                }

                while let Some(res) = set.join_next().await {
                    match res {
                        Ok(Ok((idx, inner))) => {
                            match inner {
                                Ok(v) => {
                                    results[idx] = Some(json!({
                                        "word": req.words[idx].clone(),
                                        "ok": true,
                                        "data": v,
                                    }));
                                }
                                Err(e) => {
                                    results[idx] = Some(json!({
                                        "word": req.words[idx].clone(),
                                        "ok": false,
                                        "error": e.to_string(),
                                    }));
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            if let Some(i) = results.iter().position(|x| x.is_none()) {
                                results[i] = Some(json!({
                                    "word": req.words[i].clone(),
                                    "ok": false,
                                    "error": e.to_string(),
                                }));
                            }
                        }
                        Err(join_err) => {
                            // Task join error; include message
                            if let Some(i) = results.iter().position(|x| x.is_none()) {
                                results[i] = Some(json!({
                                    "word": req.words[i].clone(),
                                    "ok": false,
                                    "error": join_err.to_string(),
                                }));
                            }
                        }
                    }
                }

                // Convert to Vec<Value>, safe to unwrap as all Some on success
                let out: Vec<Value> = results
                    .into_iter()
                    .map(|v| v.expect("batch item missing"))
                    .collect();
                Json(out).into_response()
            }
        }))
}
