//! Build and search parameters: `KEY=VALUE` pairs with int, float, or string values.

use serde::Serialize;
use std::collections::BTreeMap;

/// One parameter value. Serializes as a bare JSON number or string.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ParamValue {
    Int(i64),
    Float(f64),
    Str(String),
}

impl ParamValue {
    /// Parses a command-line value: int first, then float, else string.
    pub fn parse(text: &str) -> ParamValue {
        if let Ok(i) = text.parse::<i64>() {
            ParamValue::Int(i)
        } else if let Ok(f) = text.parse::<f64>() {
            ParamValue::Float(f)
        } else {
            ParamValue::Str(text.to_string())
        }
    }
}

/// A parameter set. Keys are sorted, so the JSON output is stable.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Params(pub BTreeMap<String, ParamValue>);

impl Params {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds or replaces a key. Returns `self` so defaults read as one chain.
    pub fn with(mut self, key: &str, value: ParamValue) -> Self {
        self.0.insert(key.to_string(), value);
        self
    }

    pub fn insert(&mut self, key: &str, value: ParamValue) {
        self.0.insert(key.to_string(), value);
    }

    pub fn contains(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }

    fn get(&self, key: &str) -> Result<&ParamValue, String> {
        self.0
            .get(key)
            .ok_or_else(|| format!("missing parameter: {key}"))
    }

    pub fn get_int(&self, key: &str) -> Result<i64, String> {
        match self.get(key)? {
            ParamValue::Int(i) => Ok(*i),
            other => Err(format!("parameter {key} must be an integer, got {other:?}")),
        }
    }

    /// Reads a non-negative integer as `usize`.
    pub fn get_usize(&self, key: &str) -> Result<usize, String> {
        let v = self.get_int(key)?;
        usize::try_from(v).map_err(|_| format!("parameter {key} must be >= 0, got {v}"))
    }

    /// Reads a float. An integer value is accepted and converted.
    pub fn get_float(&self, key: &str) -> Result<f64, String> {
        match self.get(key)? {
            ParamValue::Float(f) => Ok(*f),
            ParamValue::Int(i) => Ok(*i as f64),
            other => Err(format!("parameter {key} must be a number, got {other:?}")),
        }
    }

    pub fn get_str(&self, key: &str) -> Result<&str, String> {
        match self.get(key)? {
            ParamValue::Str(s) => Ok(s),
            other => Err(format!("parameter {key} must be a string, got {other:?}")),
        }
    }
}

/// Default `train_size` of CONTRACT 6.2: min(N, 256 * k), but never below k.
pub fn default_train_size(n: usize, k: usize) -> usize {
    n.min(256 * k).max(k)
}
