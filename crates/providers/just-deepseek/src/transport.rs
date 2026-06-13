//! Re-exported transport utilities from the `just-common` crate.
//!
//! Users who want to send requests with a raw `reqwest::Client` can use these
//! helpers directly without depending on `just-common` separately.

pub use just_common::error::TransportError;
pub use just_common::transport::http::{build_client, endpoint_url, ensure_success, parse_json};
pub use just_common::transport::sse::JsonEventStream;
