use anyhow::{anyhow, Context, Result};
use jsonschema::{Draft, JSONSchema};
use once_cell::sync::Lazy;
use serde_json::Value;
use std::collections::HashSet;
use tracing::{debug, warn};

#[derive(Debug, Clone)]
pub enum ValidationErrorType {
    SchemaValidation(String),
    MissingRequiredField(String),
    InvalidFieldValue { field: String, reason: String },
    DuplicatePartOfSpeech(String),
    InsufficientMeanings,
    InvalidPhonetic(String),
}

impl std::fmt::Display for ValidationErrorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SchemaValidation(msg) => write!(f, "Schema validation failed: {}", msg),
            Self::MissingRequiredField(field) => write!(f, "Missing required field: {}", field),
            Self::InvalidFieldValue { field, reason } => write!(f, "Invalid value for {}: {}", field, reason),
            Self::DuplicatePartOfSpeech(pos) => write!(f, "Duplicate part of speech: {}", pos),
            Self::InsufficientMeanings => write!(f, "At least one meaning is required"),
            Self::InvalidPhonetic(reason) => write!(f, "Invalid phonetic transcription: {}", reason),
        }
    }
}

pub struct Validator;

impl Validator {
    pub fn new(_schema_src: &str) -> Result<Self> {
        Ok(Self)
    }

    /// Enhanced validation with detailed error reporting and automatic fixes
    pub fn validate_and_fix(&self, mut v: Value, surface_word: &str) -> Result<Value> {
        debug!("Starting validation for word: {}", surface_word);

        // Step 1: Basic structure fixes
        self.fix_basic_structure(&mut v, surface_word)?;

        // Step 2: Validate and fix meanings structure
        self.validate_and_fix_meanings(&mut v)?;

        // Step 3: Apply schema validation with detailed error reporting
        self.apply_schema_validation(&v)?;

        debug!("Validation completed successfully for word: {}", surface_word);
        Ok(v)
    }

    /// Fix basic structural issues and ensure required top-level fields
    fn fix_basic_structure(&self, v: &mut Value, surface_word: &str) -> Result<()> {
        let obj = v.as_object_mut()
            .ok_or_else(|| anyhow!("Expected JSON object at root"))?;

        // Ensure word matches surface word
        obj.insert("word".to_string(), Value::String(surface_word.to_string()));

        // Validate required top-level fields exist
        let required_fields = ["baseForm", "phonetic", "difficulty", "language", "meanings"];
        for field in &required_fields {
            if !obj.contains_key(*field) {
                return Err(anyhow!(ValidationErrorType::MissingRequiredField(field.to_string())));
            }
        }

        // Validate language is "english"
        if let Some(lang) = obj.get("language").and_then(|l| l.as_str()) {
            if lang != "english" {
                warn!("Language was '{}', correcting to 'english'", lang);
                obj.insert("language".to_string(), Value::String("english".to_string()));
            }
        }

        // Validate difficulty is one of the accepted values
        if let Some(diff) = obj.get("difficulty").and_then(|d| d.as_str()) {
            if !["beginner", "intermediate", "advanced"].contains(&diff) {
                warn!("Invalid difficulty '{}', setting to 'intermediate'", diff);
                obj.insert("difficulty".to_string(), Value::String("intermediate".to_string()));
            }
        }

        // Basic phonetic validation (should start and end with /)
        if let Some(phonetic_val) = obj.get("phonetic") {
            if let Some(phonetic) = phonetic_val.as_str() {
                let trimmed = phonetic.trim();
                // If not wrapped with slashes, auto-wrap instead of erroring.
                let normalized = if trimmed.starts_with('/') && trimmed.ends_with('/') && trimmed.len() >= 2 {
                    trimmed.to_string()
                } else {
                    // Normalize by trimming and wrapping
                    let inner = trimmed.trim_matches('/');
                    format!("/{}/", inner)
                };
                obj.insert("phonetic".to_string(), Value::String(normalized));
            } else {
                return Err(anyhow!(ValidationErrorType::InvalidPhonetic(
                    "phonetic must be a string".to_string()
                )));
            }
        }

        Ok(())
    }

    /// Validate and fix meanings array structure
    fn validate_and_fix_meanings(&self, v: &mut Value) -> Result<()> {
        let meanings = v.get_mut("meanings").and_then(|m| m.as_array_mut())
            .ok_or_else(|| anyhow!(ValidationErrorType::MissingRequiredField("meanings".to_string())))?;

        if meanings.is_empty() {
            return Err(anyhow!(ValidationErrorType::InsufficientMeanings));
        }

        // Validate unique partOfSpeech across meanings
        let mut seen_pos = HashSet::new();
        let valid_pos = [
            "noun", "verb", "adjective", "adverb", "pronoun", "preposition",
            "conjunction", "interjection", "article", "determiner", "numeral",
            "participle", "gerund"
        ];

        for (idx, meaning) in meanings.iter_mut().enumerate() {
            let meaning_obj = meaning.as_object_mut()
                .ok_or_else(|| anyhow!("Meaning {} must be an object", idx))?;

            // Validate and normalize partOfSpeech
            if let Some(pos) = meaning_obj.get("partOfSpeech").and_then(|p| p.as_str()) {
                let pos_lower = pos.to_lowercase();
                if !valid_pos.contains(&pos_lower.as_str()) {
                    return Err(anyhow!(ValidationErrorType::InvalidFieldValue {
                        field: "partOfSpeech".to_string(),
                        reason: format!("'{}' is not a valid part of speech", pos)
                    }));
                }

                if !seen_pos.insert(pos_lower.clone()) {
                    return Err(anyhow!(ValidationErrorType::DuplicatePartOfSpeech(pos.to_string())));
                }

                // Normalize to lowercase
                meaning_obj.insert("partOfSpeech".to_string(), Value::String(pos_lower));
            } else {
                return Err(anyhow!(ValidationErrorType::MissingRequiredField(
                    format!("partOfSpeech in meaning {}", idx)
                )));
            }

            // Validate and fix synonyms/antonyms arrays
            for key in ["synonyms", "antonyms"] {
                if let Some(arr) = meaning_obj.get_mut(key).and_then(|x| x.as_array_mut()) {
                    let mut unique_items = HashSet::new();
                    let mut cleaned = vec![];

                    for item in arr.iter() {
                        if let Some(text) = item.as_str() {
                            let normalized = text.trim().to_lowercase();
                            if !normalized.is_empty() && unique_items.insert(normalized.clone()) {
                                cleaned.push(Value::String(normalized));
                            }
                        }
                    }

                    *arr = cleaned;
                } else {
                    // Ensure arrays exist even if empty
                    meaning_obj.insert(key.to_string(), Value::Array(vec![]));
                }
            }

            // Validate required meaning fields
            let required_meaning_fields = ["definition", "exampleSentence", "grammarTip", "translations"];
            for field in &required_meaning_fields {
                if !meaning_obj.contains_key(*field) {
                    return Err(anyhow!(ValidationErrorType::MissingRequiredField(
                        format!("{} in meaning {}", field, idx)
                    )));
                }
            }

            // Validate translations object
            if let Some(translations) = meaning_obj.get("translations").and_then(|t| t.as_object()) {
                let required_langs = ["es", "fr", "de", "zh", "ja", "it", "pt", "ru", "ar"];
                for lang in &required_langs {
                    if !translations.contains_key(*lang) {
                        return Err(anyhow!(ValidationErrorType::MissingRequiredField(
                            format!("translation for '{}' in meaning {}", lang, idx)
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply JSON Schema validation with enhanced error reporting
    fn apply_schema_validation(&self, v: &Value) -> Result<()> {
        static SCHEMA_VALUE: Lazy<Value> = Lazy::new(|| {
            serde_json::from_str(include_str!("../schema/word_contract.schema.json"))
                .expect("valid schema JSON")
        });

        let compiled: JSONSchema = JSONSchema::options()
            .with_draft(Draft::Draft202012)
            .compile(&SCHEMA_VALUE)
            .context("Failed to compile JSON schema")?;

        let validation_result = compiled.validate(v);
        if let Err(errors) = validation_result {
            let error_messages: Vec<String> = errors
                .take(5) // Limit to first 5 errors to avoid overwhelming output
                .map(|error| format!("at {}: {:?}", error.instance_path, error.kind))
                .collect();

            return Err(anyhow!(ValidationErrorType::SchemaValidation(
                error_messages.join("; ")
            )));
        }

        Ok(())
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
