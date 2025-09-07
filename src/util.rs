use std::{fs, path::Path};
use anyhow::Context;


pub fn read_to_string<P: AsRef<Path>>(p: P) -> anyhow::Result<String> {
Ok(fs::read_to_string(&p).with_context(|| format!("read file {:?}", p.as_ref()))?)
}
