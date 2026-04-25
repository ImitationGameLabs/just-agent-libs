use serde::{Deserialize, Serialize};

/// Normalized balance snapshot returned by a backend.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BalanceSnapshot {
    /// Whether the provider reports the account as currently available for use.
    pub is_available: bool,
    /// Per-currency balance entries.
    pub entries: Vec<BalanceEntry>,
}

/// Normalized balance entry for a specific currency.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BalanceEntry {
    /// Currency for the balance values.
    pub currency: Currency,
    /// Total balance reported by the provider.
    pub total_balance: String,
    /// Portion of the balance granted by the provider.
    pub granted_balance: String,
    /// Portion of the balance added through top-up.
    pub topped_up_balance: String,
}

/// Currency codes currently normalized by the client layer.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
#[serde(rename_all = "PascalCase")]
pub enum Currency {
    /// Chinese yuan.
    Cny,
    /// United States dollars.
    Usd,
    /// A currency not explicitly listed above.
    Other(String),
}
