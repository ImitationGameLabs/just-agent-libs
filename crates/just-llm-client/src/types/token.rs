use serde::{Deserialize, Serialize};

use crate::types::{chat::Usage, prepared::PreparedChatRequest};

/// How trustworthy a token estimate is.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TokenEstimateKind {
    /// Exact provider or protocol-derived value.
    Exact,
    /// Best-effort local estimate.
    Approximate,
    /// Count reported by the provider after execution.
    ProviderReported,
}

/// Token accounting metadata derived before or after request execution.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenEstimate {
    /// Estimated or reported prompt token count.
    pub prompt_tokens: u32,
    /// Estimated or reported total token count when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
    /// Trust level for this estimate.
    pub kind: TokenEstimateKind,
    /// Name of the estimator or upstream source when useful for diagnostics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

impl TokenEstimate {
    /// Creates a token estimate with the supplied metadata.
    pub fn new(
        prompt_tokens: u32,
        total_tokens: Option<u32>,
        kind: TokenEstimateKind,
        source: Option<String>,
    ) -> Self {
        Self { prompt_tokens, total_tokens, kind, source }
    }

    /// Creates an approximate estimate from the canonical prepared-request payload.
    ///
    /// This intentionally estimates on the serialized payload because that is the exact body the
    /// provider backend will execute, even when provider-specific fields are present.
    pub fn approximate_from_prepared_text(
        request: &PreparedChatRequest,
        estimator: impl FnOnce(&str) -> u32,
        source: impl Into<String>,
    ) -> Self {
        Self::new(
            estimator(&request.request_body_text()),
            None,
            TokenEstimateKind::Approximate,
            Some(source.into()),
        )
    }

    /// Creates a provider-reported estimate from response usage data.
    pub fn from_usage(usage: &Usage, source: impl Into<String>) -> Self {
        Self::new(
            usage.prompt_tokens,
            Some(usage.total_tokens),
            TokenEstimateKind::ProviderReported,
            Some(source.into()),
        )
    }
}
