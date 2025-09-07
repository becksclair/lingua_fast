use anyhow::{anyhow, bail, Result};
use jsonschema::{Draft, JSONSchema};
use once_cell::sync::Lazy;
use serde_json::Value;
use std::collections::HashSet;

pub struct Validator;

impl Validator {
    pub fn new(_schema_src: &str) -> Result<Self> {
        Ok(Self)
    }

    pub fn validate_and_fix(&self, mut v: Value, surface_word: &str) -> Result<Value> {
        // quick invariants not expressible in schema
        if let Some(obj) = v.as_object_mut() {
            if let Some(w) = obj.get_mut("word") {
                *w = Value::String(surface_word.to_string());
            }
        }

        // unique partOfSpeech across senses + lower-case/dedupe arrays
        if let Some(meanings) = v.get_mut("meanings").and_then(|m| m.as_array_mut()) {
            let mut seen = HashSet::new();
            for sense in meanings.iter_mut() {
                if let Some(o) = sense.as_object_mut() {
                    if let Some(pos) = o.get("partOfSpeech").and_then(|x| x.as_str()) {
                        if !seen.insert(pos.to_string()) {
                            bail!("duplicate partOfSpeech: {}", pos);
                        }
                    }
                    for key in ["synonyms", "antonyms"] {
                        if let Some(arr) = o.get_mut(key).and_then(|x| x.as_array_mut()) {
                            let mut uniq = HashSet::new();
                            let mut out = vec![];
                            for s in arr.iter() {
                                if let Some(t) = s.as_str() {
                                    let lc = t.to_lowercase();
                                    if uniq.insert(lc.clone()) {
                                        out.push(Value::String(lc));
                                    }
                                }
                            }
                            *arr = out;
                        }
                    }
                }
            }
        }

        static SCHEMA_VALUE: Lazy<Value> = Lazy::new(|| {
            serde_json::from_str(include_str!("../schema/word_contract.schema.json"))
                .expect("valid schema JSON")
        });
        let compiled: JSONSchema = JSONSchema::options()
            .with_draft(Draft::Draft202012)
            .compile(&SCHEMA_VALUE)?;
        compiled.validate(&v).map_err(|mut errors| {
            let first = errors.next().unwrap();
            anyhow!("schema error at {}: {:?}", first.instance_path, first.kind)
        })?;
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_json() -> Value {
        serde_json::json!({
            "word": "ignored",
            "baseForm": "ignore",
            "phonetic": "ɪgˈnɔːd",
            "difficulty": "beginner",
            "language": "english",
            "meanings": [
                {
                    "partOfSpeech": "noun",
                    "definition": "This is a sufficiently long definition string for schema.",
                    "exampleSentence": "An example sentence that is valid.",
                    "grammarTip": "A short grammar tip.",
                    "synonyms": ["Alpha", "alpha", "BETA"],
                    "antonyms": ["Opposite", "opposite"],
                    "translations": {
                        "es": "x", "fr": "x", "de": "x", "zh": "x", "ja": "x",
                        "it": "x", "pt": "x", "ru": "x", "ar": "x"
                    }
                }
            ]
        })
    }

    #[test]
    fn sets_surface_word_and_dedupes() {
        let v = base_json();
        let out = Validator::new("")
            .unwrap()
            .validate_and_fix(v, "Surface")
            .unwrap();
        assert_eq!(out["word"], "Surface");
        let syn = out["meanings"][0]["synonyms"].as_array().unwrap();
        assert_eq!(
            syn,
            &vec![Value::String("alpha".into()), Value::String("beta".into())]
        );
        let ant = out["meanings"][0]["antonyms"].as_array().unwrap();
        assert_eq!(ant, &vec![Value::String("opposite".into())]);
    }

    #[test]
    fn duplicate_pos_errors() {
        let mut v = base_json();
        if let Some(arr) = v.get_mut("meanings").and_then(|m| m.as_array_mut()) {
            arr.push(serde_json::json!({
                "partOfSpeech": "noun",
                "definition": "Another sufficiently long definition string for schema validity.",
                "exampleSentence": "Another example.",
                "grammarTip": "Another tip.",
                "synonyms": [],
                "antonyms": [],
                "translations": {
                    "es": "x", "fr": "x", "de": "x", "zh": "x", "ja": "x",
                    "it": "x", "pt": "x", "ru": "x", "ar": "x"
                }
            }));
        }
        let res = Validator::new("").unwrap().validate_and_fix(v, "Surface");
        assert!(res.is_err(), "expected error on duplicate partOfSpeech");
    }
}
