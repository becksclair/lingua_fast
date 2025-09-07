use axum::{body::Body, http, response::Response, Router};
use lingua_fast::api::routes;
use lingua_fast::model::{InferParams, LlmBackend, PromptParts};
use lingua_fast::validate::Validator;
use serde_json::{json, Value};
use std::sync::Arc;
use tower::util::ServiceExt;

#[derive(Clone)]
struct FakeBackend;

#[async_trait::async_trait]
impl LlmBackend for FakeBackend {
    async fn infer_json(&self, _prompt: PromptParts, _p: &InferParams) -> anyhow::Result<Vec<u8>> {
        // Simulate a backend error for specific input to exercise error handling
        if _prompt.user_word == "fail" {
            anyhow::bail!("backend failure for test word");
        }
        let out = serde_json::json!({
            "word": _prompt.user_word,
            "baseForm": _prompt.user_word.to_lowercase(),
            "phonetic": "tɛst",
            "difficulty": "beginner",
            "language": "english",
            "meanings": [
                {
                    "partOfSpeech": "noun",
                    "definition": "This is a long enough definition to satisfy schema.",
                    "exampleSentence": "A valid example sentence.",
                    "grammarTip": "A short useful tip.",
                    "synonyms": ["Alpha", "alpha", "BETA"],
                    "antonyms": ["Opposite", "opposite"],
                    "translations": {
                        "es": "x", "fr": "x", "de": "x", "zh": "x", "ja": "x",
                        "it": "x", "pt": "x", "ru": "x", "ar": "x"
                    }
                }
            ]
        });
        Ok(serde_json::to_vec(&out)?)
    }
}

fn test_router() -> Router {
    let backend = FakeBackend;
    let validator =
        Arc::new(Validator::new(include_str!("../schema/word_contract.schema.json")).unwrap());
    let params = InferParams {
        max_tokens: 64,
        temp: 0.4,
        top_p: 0.9,
        min_p: 0.05,
        repeat_penalty: 1.1,
    };
    routes(backend, validator, params)
}

#[tokio::test]
async fn single_word_ok() {
    let app = test_router();
    let body = serde_json::to_vec(&json!({"word":"Test"})).unwrap();
    let req = http::Request::builder()
        .method(http::Method::POST)
        .uri("/v1/word")
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap();

    let res: Response = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), http::StatusCode::OK);
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(v.get("word").is_some());
    assert!(v.get("meanings").is_some());
}

#[tokio::test]
async fn batch_mixed_results() {
    let app = test_router();
    let body = serde_json::to_vec(&json!({"words":["ok1","fail","ok2"]})).unwrap();
    let req = http::Request::builder()
        .method(http::Method::POST)
        .uri("/v1/words")
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap();

    let res: Response = app.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), http::StatusCode::OK);
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .unwrap();
    let arr: Vec<Value> = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[0]["word"], "ok1");
    assert!(arr[0]["ok"].as_bool().unwrap());
    assert!(arr[0]["data"].is_object());

    // The fake backend returns an error for "fail"
    assert!(!arr[1]["ok"].as_bool().unwrap());

    assert_eq!(arr[2]["word"], "ok2");
    assert!(arr[2]["ok"].as_bool().unwrap());
}

#[tokio::test]
async fn single_word_backend_error() {
    let app = test_router();
    let body = serde_json::to_vec(&json!({"word":"fail"})).unwrap();
    let req = http::Request::builder()
        .method(http::Method::POST)
        .uri("/v1/word")
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap();

    let res: Response = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
}

#[tokio::test]
async fn single_word_bad_request() {
    let app = test_router();
    // missing required field "word"
    let body = serde_json::to_vec(&json!({"not_word":"x"})).unwrap();
    let req = http::Request::builder()
        .method(http::Method::POST)
        .uri("/v1/word")
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap();

    let res: Response = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), http::StatusCode::UNPROCESSABLE_ENTITY);
}
