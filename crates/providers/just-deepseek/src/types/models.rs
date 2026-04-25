//! DeepSeek model-listing DTOs.
#![allow(missing_docs)]

use serde::{Deserialize, Serialize};

/// Wire DTO returned by `GET /models`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ListModelsResponse {
    pub object: String,
    pub data: Vec<Model>,
}

/// Wire DTO for one listed model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Model {
    pub id: String,
    pub object: String,
    pub owned_by: String,
}
