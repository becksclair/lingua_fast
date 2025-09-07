use jsonschema::{Draft, JSONSchema};
use serde_json::{json, Value};
use anyhow::{Result, bail};
use std::collections::HashSet;


pub struct Validator {
compiled: JSONSchema,
}


impl Validator {
pub fn new(schema: &serde_json::Value) -> Result<Self> {
let compiled = JSONSchema::options().with_draft(Draft::Draft202012).compile(schema)?;
Ok(Self { compiled })
}


pub fn validate_and_fix(&self, mut v: Value, surface_word: &str) -> Result<Value> {
// quick invariants not expressible in schema
if let Some(obj) = v.as_object_mut() {
if let Some(w) = obj.get_mut("word") { *w = Value::String(surface_word.to_string()); }
}


// unique partOfSpeech across senses + lower-case/dedupe arrays
if let Some(meanings) = v.get_mut("meanings").and_then(|m| m.as_array_mut()) {
let mut seen = HashSet::new();
for sense in meanings.iter_mut() {
if let Some(o) = sense.as_object_mut() {
if let Some(pos) = o.get("partOfSpeech").and_then(|x| x.as_str()) {
if !seen.insert(pos.to_string()) { bail!("duplicate partOfSpeech: {}", pos); }
}
for key in ["synonyms", "antonyms"] {
if let Some(arr) = o.get_mut(key).and_then(|x| x.as_array_mut()) {
let mut uniq = HashSet::new();
let mut out = vec![];
for s in arr.iter() {
if let Some(t) = s.as_str() {
let lc = t.to_lowercase();
if uniq.insert(lc.clone()) { out.push(Value::String(lc)); }
}
}
*arr = out;
}
}
}
}
}


let result = self.compiled.validate(&v);
if let Err(errors) = result {
let first = errors.into_iter().next().unwrap();
bail!("schema error at {}: {}", first.instance_path, first.error);
}
Ok(v)
}
}
