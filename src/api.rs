use crate::{
    model::{InferParams, LlmBackend, PromptParts},
    validate::Validator,
};
use anyhow::{Context, Result};
use axum::{http::StatusCode, response::IntoResponse, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{sync::Arc, time::Duration};
use tracing::{debug, error, info, warn};

#[derive(Debug, Deserialize)]
pub struct WordReq {
    pub word: String,
}

#[derive(Debug, Deserialize)]
pub struct BatchReq {
    pub words: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub error_type: String,
    pub word: Option<String>,
    pub retry_suggested: bool,
}

#[derive(Debug, Clone)]
enum ApiErrorType {
    Validation(String),
    Inference(String),
    JsonParse(String),
    Internal(String),
}

impl ApiErrorType {
    fn should_retry(&self) -> bool {
        matches!(self, Self::Inference(_) | Self::Internal(_))
    }

    fn status_code(&self) -> StatusCode {
        match self {
            Self::Validation(_) => StatusCode::UNPROCESSABLE_ENTITY,
            Self::JsonParse(_) => StatusCode::UNPROCESSABLE_ENTITY,
            Self::Inference(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn error_type_str(&self) -> &'static str {
        match self {
            Self::Validation(_) => "validation_error",
            Self::JsonParse(_) => "json_parse_error",
            Self::Inference(_) => "inference_error",
            Self::Internal(_) => "internal_error",
        }
    }

    fn message(&self) -> &str {
        match self {
            Self::Validation(msg) | Self::JsonParse(msg) |
            Self::Inference(msg) | Self::Internal(msg) => msg,
        }
    }
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
                info!("Processing single word request: {}", req.word);

                // Input validation
                if req.word.trim().is_empty() {
                    let error_response = ErrorResponse {
                        error: "Word cannot be empty".to_string(),
                        error_type: "validation_error".to_string(),
                        word: Some(req.word.clone()),
                        retry_suggested: false,
                    };
                    return (StatusCode::BAD_REQUEST, Json(error_response)).into_response();
                }

                if req.word.len() > 100 {
                    let error_response = ErrorResponse {
                        error: "Word too long (max 100 characters)".to_string(),
                        error_type: "validation_error".to_string(),
                        word: Some(req.word.clone()),
                        retry_suggested: false,
                    };
                    return (StatusCode::BAD_REQUEST, Json(error_response)).into_response();
                }

                // Attempt inference with retry logic
                let result = attempt_word_inference(backend, validator, params, &req.word).await;

                match result {
                    Ok(json_value) => {
                        info!("Successfully processed word: {}", req.word);
                        Json(json_value).into_response()
                    }
                    Err(api_error) => {
                        error!("Failed to process word '{}': {}", req.word, api_error.message());
                        let error_response = ErrorResponse {
                            error: api_error.message().to_string(),
                            error_type: api_error.error_type_str().to_string(),
                            word: Some(req.word.clone()),
                            retry_suggested: api_error.should_retry(),
                        };
                        (api_error.status_code(), Json(error_response)).into_response()
                    }
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
                // Allow overriding batch concurrency via INFER_CONCURRENCY to avoid GPU thrash
                let concurrency_limit = std::env::var("INFER_CONCURRENCY")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .filter(|&v| v > 0)
                    .unwrap_or_else(|| usize::min(8, num_cpus::get()));
                for (idx, word) in req.words.iter().cloned().enumerate() {
                    let backend = backend.clone();
                    let validator = validator.clone();
                    let params = params.clone();
                    set.spawn(async move {
                        let result = attempt_word_inference(backend.clone(), validator.clone(), params.clone(), &word).await;
                        Ok::<(usize, Result<Value, ApiErrorType>), anyhow::Error>((idx, result))
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
                                        Err(api_error) => {
                                            results[idx] = Some(json!({
                                                "word": req.words[idx].clone(),
                                                "ok": false,
                                                "error": api_error.message(),
                                                "error_type": api_error.error_type_str(),
                                                "retry_suggested": api_error.should_retry(),
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
                                Err(api_error) => {
                                    results[idx] = Some(json!({
                                        "word": req.words[idx].clone(),
                                        "ok": false,
                                        "error": api_error.message(),
                                        "error_type": api_error.error_type_str(),
                                        "retry_suggested": api_error.should_retry(),
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

/// Attempt word inference with retry logic and enhanced error handling
async fn attempt_word_inference<B: LlmBackend>(
    backend: B,
    validator: Arc<Validator>,
    params: InferParams,
    word: &str,
) -> Result<Value, ApiErrorType> {
    const MAX_RETRIES: usize = 2;
    const RETRY_DELAY: Duration = Duration::from_millis(500);

    let system = "You are an expert linguist and lexicographer. Produce a single valid JSON object only.".to_string();
    let prompt = PromptParts {
        system,
        user_word: word.to_string()
    };

    for attempt in 0..=MAX_RETRIES {
        debug!("Inference attempt {} for word: {}", attempt + 1, word);

        let inference_result = async {
            let bytes = backend.infer_json(prompt.clone(), &params).await
                .context("LLM inference failed")?;
            Ok::<Vec<u8>, anyhow::Error>(bytes)
        }.await;

        let bytes = match inference_result {
            Ok(bytes) => bytes,
            Err(e) => {
                warn!("Inference attempt {} failed for '{}': {}", attempt + 1, word, e);
                if attempt < MAX_RETRIES {
                    tokio::time::sleep(RETRY_DELAY).await;
                    continue;
                }
                return Err(ApiErrorType::Inference(
                    format!("LLM inference failed after {} attempts: {}", MAX_RETRIES + 1, e)
                ));
            }
        };

        // Parse JSON
        let json_value = match serde_json::from_slice::<Value>(&bytes) {
            Ok(v) => v,
            Err(e) => {
                warn!("JSON parsing failed for '{}' on attempt {}: {}", word, attempt + 1, e);
                if attempt < MAX_RETRIES {
                    tokio::time::sleep(RETRY_DELAY).await;
                    continue;
                }
                return Err(ApiErrorType::JsonParse(
                    format!("Failed to parse JSON response: {}", e)
                ));
            }
        };

        // Validate and fix
        match validator.validate_and_fix(json_value, word) {
            Ok(validated) => {
                debug!("Successfully processed '{}' on attempt {}", word, attempt + 1);
                return Ok(validated);
            }
            Err(e) => {
                // Check if it's a validation error we shouldn't retry
                let error_msg = e.to_string();
                if error_msg.contains("Missing required field") ||
                   error_msg.contains("Invalid value") ||
                   error_msg.contains("duplicate partOfSpeech") {
                    warn!("Validation failed for '{}': {}", word, e);
                    return Err(ApiErrorType::Validation(error_msg));
                }

                warn!("Validation attempt {} failed for '{}': {}", attempt + 1, word, e);
                if attempt < MAX_RETRIES {
                    tokio::time::sleep(RETRY_DELAY).await;
                    continue;
                }
                return Err(ApiErrorType::Validation(
                    format!("Validation failed after {} attempts: {}", MAX_RETRIES + 1, e)
                ));
            }
        }
    }

    Err(ApiErrorType::Internal("Unexpected end of retry loop".to_string()))
}
