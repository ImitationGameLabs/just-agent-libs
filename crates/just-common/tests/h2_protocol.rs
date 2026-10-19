//! Protocol-level tests for the HTTP/2 path: a plaintext h2c server proves the terminator path
//! skips the drain, and a TLS server whose ALPN list decides between h2 and the HTTP/1.1 fallback
//! proves default negotiation.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, StreamBody};
use hyper::body::{Bytes, Frame};
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::{TokioExecutor, TokioIo};
use just_common::transport::sse::JsonEventStream;
use serde_json::Value;
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;
use tokio_rustls::rustls;
use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer};

/// A stream that ends later than this did not skip the drain: the drain budget is 100 ms, so the
/// ceiling only has to separate "ended on the terminator" from "waited out the budget".
const SKIP_CEILING: Duration = Duration::from_millis(50);

type SseBody = BoxBody<Bytes, Infallible>;

/// An SSE response that sends one data event and the terminator, then holds the stream open: a
/// consumer that stops at the terminator is only released by the drain budget unless the drain is
/// skipped.
fn sse_response() -> Response<SseBody> {
    let events = futures_util::stream::iter([
        Ok(Frame::data(Bytes::from_static(b"data: {\"n\":1}\n\n"))),
        Ok(Frame::data(Bytes::from_static(b"data: [DONE]\n\n"))),
    ])
    .chain(futures_util::stream::pending::<
        Result<Frame<Bytes>, Infallible>,
    >());

    Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .body(BodyExt::boxed(StreamBody::new(events)))
        .expect("sse response")
}

async fn serve_sse(
    _request: Request<hyper::body::Incoming>,
) -> Result<Response<SseBody>, Infallible> {
    Ok(sse_response())
}

/// Serves the SSE response over h2c (prior knowledge, no upgrade) and counts accepted connections.
async fn spawn_h2c_server() -> (SocketAddr, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test server");
    let addr = listener.local_addr().expect("test server address");
    let accepted = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&accepted);

    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                return;
            };
            counter.fetch_add(1, Ordering::SeqCst);
            tokio::spawn(async move {
                let io = TokioIo::new(stream);
                let _ = hyper::server::conn::http2::Builder::new(TokioExecutor::new())
                    .serve_connection(io, service_fn(serve_sse))
                    .await;
            });
        }
    });

    (addr, accepted)
}

/// A self-signed certificate for `localhost` and `127.0.0.1`, generated per call: the repository
/// stores no certificate or key material.
fn tls_material() -> (CertificateDer<'static>, PrivateKeyDer<'static>) {
    let key_pair = rcgen::KeyPair::generate().expect("key pair");
    let params =
        rcgen::CertificateParams::new(vec!["localhost".to_string(), "127.0.0.1".to_string()])
            .expect("certificate params");
    let certificate = params
        .self_signed(&key_pair)
        .expect("self-signed certificate");

    (
        CertificateDer::from(certificate.der().to_vec()),
        PrivateKeyDer::Pkcs8(key_pair.serialize_der().into()),
    )
}

/// Serves the SSE response over TLS with the given ALPN list, choosing the server protocol from
/// what was actually negotiated.
async fn spawn_tls_server(alpn: &[&str]) -> (SocketAddr, CertificateDer<'static>) {
    let (certificate, key) = tls_material();

    let mut config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::aws_lc_rs::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .expect("default protocol versions")
    .with_no_client_auth()
    .with_single_cert(vec![certificate.clone()], key)
    .expect("server config");
    config.alpn_protocols = alpn
        .iter()
        .map(|protocol| protocol.as_bytes().to_vec())
        .collect();

    let acceptor = TlsAcceptor::from(Arc::new(config));
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test server");
    let addr = listener.local_addr().expect("test server address");

    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                return;
            };
            let acceptor = acceptor.clone();
            tokio::spawn(async move {
                let Ok(tls) = acceptor.accept(stream).await else {
                    return;
                };
                let negotiated = tls.get_ref().1.alpn_protocol().map(<[u8]>::to_vec);
                let io = TokioIo::new(tls);
                let _ = match negotiated.as_deref() {
                    Some(b"h2") => {
                        hyper::server::conn::http2::Builder::new(TokioExecutor::new())
                            .serve_connection(io, service_fn(serve_sse))
                            .await
                    }
                    _ => {
                        hyper::server::conn::http1::Builder::new()
                            .serve_connection(io, service_fn(serve_sse))
                            .await
                    }
                };
            });
        }
    });

    (addr, certificate)
}

fn h2c_client() -> reqwest::Client {
    reqwest::Client::builder()
        .http2_prior_knowledge()
        .read_timeout(Duration::from_secs(5))
        .build()
        .expect("client")
}

fn tls_client(certificate: CertificateDer<'static>) -> reqwest::Client {
    reqwest::Client::builder()
        .add_root_certificate(
            reqwest::Certificate::from_der(certificate.as_ref()).expect("root certificate"),
        )
        .read_timeout(Duration::from_secs(5))
        .build()
        .expect("client")
}

/// Consumes a stream to its end and returns the number of data events it carried.
async fn collect_events(response: reqwest::Response) -> usize {
    let mut stream = JsonEventStream::<Value>::from_response(response).expect("sse stream");

    let mut events = 0;
    while let Some(event) = stream.next().await {
        event.expect("event");
        events += 1;
    }
    events
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn h2c_ends_on_the_terminator_without_draining() {
    let (addr, accepted) = spawn_h2c_server().await;
    let client = h2c_client();
    let url = format!("http://{addr}/v1/chat/completions");

    let response = client.get(&url).send().await.expect("request");
    assert_eq!(
        response.version(),
        reqwest::Version::HTTP_2,
        "the fixture must speak h2c for this test to mean anything"
    );

    let started = Instant::now();
    let events = collect_events(response).await;
    let elapsed = started.elapsed();
    eprintln!("h2c terminator: consume {elapsed:?}");

    assert_eq!(events, 1);
    assert!(
        elapsed < SKIP_CEILING,
        "an HTTP/2 stream must end on the terminator, not on the 100 ms drain budget: {elapsed:?}"
    );

    let response = client.get(&url).send().await.expect("second request");
    assert_eq!(collect_events(response).await, 1);
    assert_eq!(
        accepted.load(Ordering::SeqCst),
        1,
        "the connection must stay reusable after ending on the terminator"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tls_alpn_negotiates_h2_and_falls_back_to_http_1_1() {
    let (addr, certificate) = spawn_tls_server(&["h2", "http/1.1"]).await;
    let client = tls_client(certificate);
    let response = client
        .get(format!("https://{addr}/v1/chat/completions"))
        .send()
        .await
        .expect("request");
    assert_eq!(
        response.version(),
        reqwest::Version::HTTP_2,
        "a server advertising h2 must be spoken to over h2 by default"
    );
    assert_eq!(collect_events(response).await, 1);

    let (addr, certificate) = spawn_tls_server(&["http/1.1"]).await;
    let client = tls_client(certificate);
    let response = client
        .get(format!("https://{addr}/v1/chat/completions"))
        .send()
        .await
        .expect("request");
    assert_eq!(
        response.version(),
        reqwest::Version::HTTP_11,
        "without h2 in the ALPN list the client must fall back to HTTP/1.1"
    );
    assert_eq!(collect_events(response).await, 1);
}
