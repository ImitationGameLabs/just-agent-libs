//! The API key, held so that it is hard to leak by accident.

use std::fmt;

use zeroize::{Zeroize, ZeroizeOnDrop};

/// A type-safe API key.
///
/// The key is wiped from memory when the value is dropped, and nothing in this crate prints it:
/// [`Debug`] shows a placeholder, and there is deliberately no `Display`, `Serialize` or
/// `PartialEq`. The only way to read it is [`ApiKey::expose`], which keeps every use easy to find
/// and review.
///
/// Wiping is best effort. The HTTP stack keeps its own copy of the `Authorization` header while a
/// client is alive, and that copy is outside this type's control.
///
/// # Examples
///
/// ```
/// use just_common::ApiKey;
///
/// let key = ApiKey::new(String::from("example-key\n")).unwrap();
/// assert_eq!(key.expose(), "example-key");
/// assert_eq!(format!("{key:?}"), "ApiKey([REDACTED])");
/// ```
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct ApiKey(String);

impl ApiKey {
    /// Wraps a key, trimming the surrounding whitespace that a file or a pipe tends to leave.
    ///
    /// The string is taken by value so that the caller's copy is consumed; the original buffer is
    /// wiped before this returns.
    ///
    /// # Errors
    ///
    /// Fails when the key is empty after trimming, or contains anything other than visible ASCII
    /// characters, which could not be sent in an HTTP header. The error never contains the key.
    pub fn new(mut key: String) -> Result<Self, InvalidApiKey> {
        let trimmed = key.trim().to_owned();
        key.zeroize();

        let mut candidate = Self(trimmed);
        if candidate.0.is_empty() {
            return Err(InvalidApiKey::Empty);
        }
        if !candidate.0.bytes().all(|byte| byte.is_ascii_graphic()) {
            candidate.zeroize();
            return Err(InvalidApiKey::InvalidCharacters);
        }
        Ok(candidate)
    }

    /// The key itself. Call this only at the point where the key is handed to the HTTP stack, and
    /// never pass the result to a log, an error or a formatter.
    #[must_use]
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "ApiKey([REDACTED])")
    }
}

/// Why a string was rejected as an API key. The message never contains the key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum InvalidApiKey {
    /// The key was empty, or only whitespace.
    #[error("the API key is empty")]
    Empty,
    /// The key held a space, a control character or a non-ASCII character.
    #[error("the API key contains characters that cannot be sent in an HTTP header")]
    InvalidCharacters,
}

#[cfg(test)]
mod tests {
    use super::{ApiKey, InvalidApiKey};

    const SENTINEL: &str = "sentinel-key-0123456789";

    #[test]
    fn debug_never_shows_the_key() {
        let key = ApiKey::new(SENTINEL.to_owned()).unwrap();

        for rendered in [
            format!("{key:?}"),
            format!("{key:#?}"),
            format!("{:?}", Some(&key)),
        ] {
            assert!(!rendered.contains(SENTINEL), "{rendered}");
            assert!(rendered.contains("[REDACTED]"), "{rendered}");
        }
    }

    #[test]
    fn trims_the_whitespace_a_file_or_pipe_leaves_behind() {
        let key = ApiKey::new(format!("  {SENTINEL}\r\n")).unwrap();

        assert_eq!(key.expose(), SENTINEL);
    }

    #[test]
    fn rejects_an_empty_key() {
        let error = ApiKey::new(String::new()).unwrap_err();

        assert_eq!(error, InvalidApiKey::Empty);
    }

    #[test]
    fn rejects_a_whitespace_only_key() {
        let error = ApiKey::new(" \n\t".to_owned()).unwrap_err();

        assert_eq!(error, InvalidApiKey::Empty);
    }

    #[test]
    fn rejects_keys_that_cannot_go_in_an_http_header() {
        let cases = [
            "sentinel key",  // a space is not a header-safe byte
            "sentinel\nkey", // control bytes break header parsing
            "sentinel-ключ", // non-ASCII cannot go in a header value
        ];

        for input in cases {
            let error = ApiKey::new(input.to_owned()).unwrap_err();

            assert_eq!(error, InvalidApiKey::InvalidCharacters, "{input:?}");
            assert!(!error.to_string().contains("sentinel"), "{error}");
            assert!(!format!("{error:?}").contains("sentinel"), "{error:?}");
        }
    }
}
