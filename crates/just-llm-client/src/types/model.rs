use serde::{Deserialize, Serialize};

/// Normalized model catalog returned by an LLM client backend.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelCatalogResponse {
    /// Models currently exposed by the backend.
    pub data: Vec<ModelInfo>,
}

/// Minimal normalized model metadata.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelInfo {
    /// Provider model identifier.
    pub id: String,
    /// Optional provider object label from the upstream API.
    pub object: Option<String>,
    /// Optional owner or namespace metadata from the upstream API.
    pub owned_by: Option<String>,
}
