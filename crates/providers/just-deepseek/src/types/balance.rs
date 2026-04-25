//! DeepSeek balance DTOs.
#![allow(missing_docs)]

use serde::{Deserialize, Serialize};

/// Wire DTO returned by `GET /user/balance`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GetUserBalanceResponse {
    pub is_available: bool,
    pub balance_infos: Vec<BalanceInfo>,
}

/// One balance entry in the DeepSeek balance response.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BalanceInfo {
    pub currency: Currency,
    pub total_balance: String,
    pub granted_balance: String,
    pub topped_up_balance: String,
}

/// Currencies currently returned by the DeepSeek balance endpoint.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Currency {
    #[serde(rename = "CNY")]
    Cny,
    #[serde(rename = "USD")]
    Usd,
}
