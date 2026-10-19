//! Regression tests for the SSE terminator path: after a `[DONE]` event the
//! parser keeps reading the response body within a bounded budget, so a client
//! can reuse the connection when the framing end arrives in time — and gives up
//! without stalling when it does not, and the stream ends cleanly when the
//! connection drops mid-body.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use just_common::transport::sse::JsonEventStream;
use serde_json::Value;

const RESPONSE_HEAD: &str =
    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";

/// A stream that ends later than this did not end on the drain budget: the
/// budget plus the scheduling slack a loopback test may spend around it.
const BUDGET_CEILING: Duration = Duration::from_millis(250);

/// What the server sends once the `[DONE]` event is on the wire.
#[derive(Clone, Copy)]
enum Tail {
    /// One more data event, then the terminating chunk, after a delay.
    After(Duration),
    /// Half of a chunked frame, then the connection closes.
    HalfFrameThenClose,
    /// Nothing at all, with the connection left open.
    Silence,
}

/// One chunked-transfer frame carrying one SSE event.
fn chunk(payload: &str) -> String {
    format!("{:x}\r\n{payload}\r\n", payload.len())
}

/// Serves the SSE response on every accepted connection and every request on
/// that connection, counting the accepted connections.
fn spawn_server(tail: Tail) -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
    let addr = listener.local_addr().expect("test server address");
    let accepted = Arc::new(AtomicUsize::new(0));

    let counter = Arc::clone(&accepted);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { return };
            counter.fetch_add(1, Ordering::SeqCst);
            std::thread::spawn(move || serve(stream, tail));
        }
    });

    (format!("http://{addr}/v1/chat/completions"), accepted)
}

fn serve(mut stream: TcpStream, tail: Tail) {
    let mut buf = [0u8; 1024];

    loop {
        // Read one request head; the test sends bodyless GETs.
        let mut request = Vec::new();
        loop {
            match stream.read(&mut buf) {
                Ok(0) | Err(_) => return,
                Ok(n) => {
                    request.extend_from_slice(&buf[..n]);
                    if request.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
            }
        }

        if stream.write_all(RESPONSE_HEAD.as_bytes()).is_err() {
            return;
        }
        let _ = stream.write_all(chunk("data: {\"n\":1}\n\n").as_bytes());
        let _ = stream.write_all(chunk("data: [DONE]\n\n").as_bytes());
        let _ = stream.flush();

        match tail {
            Tail::After(delay) => {
                std::thread::sleep(delay);
                let _ = stream.write_all(chunk("data: {\"n\":2}\n\n").as_bytes());
                let _ = stream.write_all(b"0\r\n\r\n");
                let _ = stream.flush();
            }
            // Half a frame, then the socket drops: the client sees the response end
            // mid-body, which the drain must treat as a reason to stop, not to fail.
            Tail::HalfFrameThenClose => {
                let _ = stream.write_all(b"20\r\n");
                let _ = stream.flush();
                return;
            }
            // Hold the connection open without sending a byte: a peer that never
            // finishes the framing must not keep the consumer waiting.
            Tail::Silence => std::thread::sleep(Duration::from_secs(10)),
        }
    }
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .read_timeout(Duration::from_secs(5))
        .build()
        .expect("client")
}

/// Consumes a stream to its end, returning the chunk count and how long that took.
async fn consume_to_end(client: &reqwest::Client, url: &str) -> (usize, Duration) {
    let started = Instant::now();
    let response = client.get(url).send().await.expect("request");
    let mut events = JsonEventStream::<Value>::from_response(response).expect("sse stream");

    let mut chunks = 0;
    while let Some(event) = events.next().await {
        event.expect("event");
        chunks += 1;
    }

    (chunks, started.elapsed())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_trailer_inside_the_budget_leaves_the_connection_reusable() {
    let (url, accepted) = spawn_server(Tail::After(Duration::from_millis(25)));
    let client = client();

    let (chunks, first) = consume_to_end(&client, &url).await;
    assert_eq!(chunks, 1);

    // The pool insertion runs on the connection task and cannot be awaited on from the client,
    // so yield with a margin far larger than a loopback round trip before reusing the client.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let (chunks, second) = consume_to_end(&client, &url).await;
    assert_eq!(chunks, 1);

    let connections = accepted.load(Ordering::SeqCst);
    eprintln!(
        "trailer at 25ms: first consume {first:?}, second {second:?}, connections {connections}"
    );

    assert_eq!(
        connections, 1,
        "a trailer that arrives inside the drain budget must leave the connection reusable"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_trailer_past_the_budget_ends_the_stream_on_the_budget() {
    let (url, accepted) = spawn_server(Tail::After(Duration::from_millis(300)));
    let client = client();

    let (chunks, elapsed) = consume_to_end(&client, &url).await;
    assert_eq!(chunks, 1);
    assert!(
        elapsed < BUDGET_CEILING,
        "the stream must end on the drain budget, not on the trailer: {elapsed:?}"
    );

    assert_eq!(consume_to_end(&client, &url).await.0, 1);
    let connections = accepted.load(Ordering::SeqCst);
    eprintln!("trailer at 300ms: consume {elapsed:?}, connections {connections}");

    assert_eq!(
        connections, 2,
        "a drain that gives up leaves the connection closed, so the next request opens a new one"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_silent_peer_does_not_stall_the_terminator_path() {
    let (url, _accepted) = spawn_server(Tail::Silence);
    let client = client();

    let (chunks, elapsed) = consume_to_end(&client, &url).await;
    assert_eq!(chunks, 1);
    eprintln!("silent peer: consume {elapsed:?}");

    assert!(
        elapsed < BUDGET_CEILING,
        "a peer that sends nothing after [DONE] must not hold the consumer past the budget: {elapsed:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_connection_lost_during_the_drain_ends_the_stream_without_an_error() {
    let (url, _accepted) = spawn_server(Tail::HalfFrameThenClose);
    let client = client();

    let (chunks, elapsed) = consume_to_end(&client, &url).await;
    assert_eq!(chunks, 1);
    eprintln!("peer dropped mid-frame: consume {elapsed:?}");

    // `consume_to_end` fails on any error item, so reaching this point means the
    // connection loss ended the stream instead of turning it into an error.
    assert!(
        elapsed < BUDGET_CEILING,
        "a connection lost during the drain must end the stream, not wait it out: {elapsed:?}"
    );
}
